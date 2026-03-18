#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
FP4 quant + FP4 GEMM reference: bf16 A, MXFP4 B -> MXFP4 per-1x32 quant A -> gemm_a4w4 -> bf16 C.
Quant logic follows aiter op_tests/test_gemm_a4w4.py (get_triton_quant(QuantType.per_1x32)).
"""
import torch
import triton
import triton.language as tl
from task import input_t, output_t

# Cache B_scale across calls — B is fixed weight, only quantize once per unique tensor
_b_scale_cache: dict = {}


# =============================================================================
# LOOP STRUCTURE OVERVIEW
# =============================================================================
#
#  PARALLEL loops  (each iteration runs on a separate CU simultaneously):
#    [P1] K-split loop  — NUM_KSPLIT chunks of K, each on separate CUs
#    [P2] M-tile loop   — ceil(M / BLOCK_M) tiles of M rows
#    [P3] N-tile loop   — ceil(N / BLOCK_N) tiles of N cols
#    These three are FUSED into a single 1-D grid: pid in [0, NUM_KSPLIT*Mtiles*Ntiles)
#
#  SEQUENTIAL loops  (run serially within one CU / wavefront):
#    [S1] Inner K loop  — iterates SPLITK_BLOCK // BLOCK_K times per CU
#                         accumulates partial dot products into acc
#
#  HARDWARE (not a loop — single MFMA instruction per iteration):
#    [H]  tl.dot_scaled — computes BLOCK_M x BLOCK_N x BLOCK_K fp4 matmul tile
#
#  REDUCTION (separate kernel, parallel over M and N tiles):
#    [R]  _reduce_kernel — sums NUM_KSPLIT partial [M,N] results → final [M,N]
# =============================================================================


@triton.jit
def _mxfp4_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    a_scales_ptr, b_scales_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_ck, stride_cm, stride_cn,
    stride_asm, stride_ask,
    stride_bsn, stride_bsk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    SPLITK_BLOCK: tl.constexpr,  # packed K elements assigned to each K-split CU
    SCALE_GROUP: tl.constexpr,   # = 16 bytes per scale (32 fp4 elements; BLOCK_K is in packed bytes)
):
    # -------------------------------------------------------------------------
    # [P1][P2][P3]  PARALLEL: decode which (K-split, M-tile, N-tile) this CU owns
    # -------------------------------------------------------------------------
    num_m = tl.cdiv(M, BLOCK_M)
    num_n = tl.cdiv(N, BLOCK_N)
    pid   = tl.program_id(axis=0)  # one CU per (k_split, m_tile, n_tile) triple

    pid_k  = pid % NUM_KSPLIT       # [P1] which K-split chunk
    pid_mn = pid // NUM_KSPLIT      # [P2][P3] which (M, N) output tile

    # Standard Triton L2 swizzle (handles num_m < GROUP_M correctly)
    num_pid_in_group = GROUP_M * num_n
    group_id    = pid_mn // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.where(num_m - first_pid_m < GROUP_M,
                             num_m - first_pid_m, GROUP_M)
    pid_in = pid_mn % num_pid_in_group
    pid_m  = first_pid_m + pid_in % group_size_m   # [P2] M-tile index
    pid_n  = pid_in // group_size_m                 # [P3] N-tile index

    # -------------------------------------------------------------------------
    # Compute base pointers for this CU's tile of A, B, and their scales
    # -------------------------------------------------------------------------
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # M row indices
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # N col indices
    offs_k = tl.arange(0, BLOCK_K)                    # packed K indices within one BLOCK_K

    # Starting packed-K offset for this split chunk
    k_split_start = pid_k * SPLITK_BLOCK

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + (k_split_start + offs_k[None, :]) * stride_ak
    b_ptrs = b_ptr + (k_split_start + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn

    # Scale pointers: one scale per SCALE_GROUP packed elements
    offs_ks    = tl.arange(0, BLOCK_K // SCALE_GROUP)
    ks_start   = pid_k * (SPLITK_BLOCK // SCALE_GROUP)
    a_sc_ptrs  = a_scales_ptr + offs_m[:, None] * stride_asm + (ks_start + offs_ks[None, :]) * stride_ask
    b_sc_ptrs  = b_scales_ptr + offs_n[:, None] * stride_bsn + (ks_start + offs_ks[None, :]) * stride_bsk

    # -------------------------------------------------------------------------
    # [S1]  SEQUENTIAL: inner K loop — iterate over BLOCK_K tiles within this split chunk
    # -------------------------------------------------------------------------
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(SPLITK_BLOCK // BLOCK_K):   # [S1] sequential K tiles
        a_scales = tl.load(a_sc_ptrs, mask=offs_m[:, None] < M, other=0)
        b_scales = tl.load(b_sc_ptrs, mask=offs_n[:, None] < N, other=0)
        a = tl.load(a_ptrs, mask=offs_m[:, None] < M, other=0)
        b = tl.load(b_ptrs, mask=offs_n[None, :] < N, other=0)

        # [H] HARDWARE: single MFMA instruction — not a loop
        acc = tl.dot_scaled(a, a_scales, "e2m1", b, b_scales, "e2m1", acc)

        # Advance pointers to next BLOCK_K tile
        a_ptrs    += BLOCK_K * stride_ak
        b_ptrs    += BLOCK_K * stride_bk
        a_sc_ptrs += (BLOCK_K // SCALE_GROUP) * stride_ask
        b_sc_ptrs += (BLOCK_K // SCALE_GROUP) * stride_bsk

    # -------------------------------------------------------------------------
    # Write this CU's partial result into c_partial[pid_k, :, :]
    # (when NUM_KSPLIT == 1 this is the final output)
    # -------------------------------------------------------------------------
    # Compile-time branch: keep float32 for split-K partials, convert to bf16 for direct output
    if NUM_KSPLIT > 1:
        c = acc          # float32 → float32 partial buffer
    else:
        c = acc.to(tl.bfloat16)  # float32 → bf16 final output
    offs_cm = offs_m.to(tl.int64)
    offs_cn = offs_n.to(tl.int64)
    c_ptrs = (c_ptr
              + pid_k * stride_ck
              + offs_cm[:, None] * stride_cm
              + offs_cn[None, :] * stride_cn)
    tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


@triton.jit
def _reduce_kernel(
    c_in_ptr, c_out_ptr,
    M, N,
    stride_k, stride_in_m, stride_in_n,
    stride_out_m, stride_out_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,  # must be next_power_of_2 for tl.sum
):
    """
    [R] REDUCTION (parallel over output tiles, sequential sum over NUM_KSPLIT):
    Sum c_partial[NUM_KSPLIT, M, N] → c_out[M, N].
    Each CU owns one (M-tile, N-tile) and sums all K-split partial results.
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, NUM_KSPLIT)

    # Load all K-split partials for this (M-tile, N-tile)
    ptrs = (c_in_ptr
            + offs_k[:, None, None] * stride_k
            + offs_m[None, :, None] * stride_in_m
            + offs_n[None, None, :] * stride_in_n)
    partials = tl.load(ptrs, mask=((offs_m[None, :, None] < M) & (offs_n[None, None, :] < N)), other=0.0)
    out = tl.sum(partials, axis=0).to(tl.bfloat16)

    out_ptrs = (c_out_ptr
                + offs_m[:, None] * stride_out_m
                + offs_n[None, :] * stride_out_n)
    tl.store(out_ptrs, out, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def mxfp4_gemm(A_q, B_q, A_scale, B_scale, M, N, K, num_ksplit=1,
               BLOCK_M=16, BLOCK_N=64, BLOCK_K=128):
    """
    Launch the tiled MXFP4 GEMM.

    Grid = [P1] × [P2] × [P3] = NUM_KSPLIT × ceil(M/BLOCK_M) × ceil(N/BLOCK_N) CUs in parallel.
    Each CU runs [S1] = SPLITK_BLOCK // BLOCK_K sequential K-tile iterations.
    If NUM_KSPLIT > 1, a separate reduction kernel sums the partial results.

    Args:
        A_q:     [M, K//2]  fp4x2 (2 fp4 values packed per byte)
        B_q:     [N, K//2]  fp4x2  — passed in transposed to the kernel as [K//2, N]
        A_scale: [M, K//32] e8m0
        B_scale: [N, K//32] e8m0
        M, N, K: logical dimensions (K = number of fp4 elements along reduction axis)
    """
    K_packed    = K // 2        # number of bytes in K dimension (2 fp4 per byte)
    SCALE_GROUP = 16            # bytes per scale group (32 fp4 elements / 2 fp4 per byte)

    # Compute actual split-K: round up to multiple of BLOCK_K (NOT next_power_of_2,
    # which can inflate SPLITK_BLOCK beyond K_packed causing OOB reads)
    splitk_block = triton.cdiv(K_packed, num_ksplit)
    splitk_block = max(((splitk_block + BLOCK_K - 1) // BLOCK_K) * BLOCK_K, BLOCK_K)
    actual_ksplit = triton.cdiv(K_packed, splitk_block)

    use_splitk = actual_ksplit > 1

    # Allocate output: partial buffer for split-K, or final buffer directly
    # Pad to next_power_of_2 so reduce kernel's tl.arange(0, NUM_KSPLIT) stays in bounds
    if use_splitk:
        padded_ksplit = triton.next_power_of_2(actual_ksplit)
        c_partial = torch.zeros((padded_ksplit, M, N), dtype=torch.float32, device=A_q.device)
        c_buf = c_partial
    else:
        c_out = torch.empty((M, N), dtype=torch.bfloat16, device=A_q.device)
        c_buf = c_out

    # B is stored [N, K//2]; we pass it transposed so kernel sees [K//2, N]
    B_t = B_q.T.contiguous()

    # Triton's type_canonicalisation_dict on this server doesn't include float4_e2m1fn_x2;
    # view as uint8 — same bits, compatible pointer type
    A_u8 = A_q.view(torch.uint8)
    B_u8 = B_t.view(torch.uint8)

    # [P1][P2][P3] — total parallel CUs
    grid = (actual_ksplit * triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    _mxfp4_gemm_kernel[grid](
        A_u8, B_u8, c_buf,
        A_scale, B_scale,
        M, N, K_packed,
        A_u8.stride(0),  A_u8.stride(1),
        B_u8.stride(0),  B_u8.stride(1),
        c_buf.stride(0) if use_splitk else 0,  # stride_ck: stride between K-split slices
        c_buf.stride(-2), c_buf.stride(-1),    # stride_cm, stride_cn
        A_scale.stride(0), A_scale.stride(1),
        B_scale.stride(0), B_scale.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_M=8,
        NUM_KSPLIT=actual_ksplit,
        SPLITK_BLOCK=splitk_block,
        SCALE_GROUP=SCALE_GROUP,
        num_warps=4, num_stages=2,
    )

    if use_splitk:
        # [R] reduction kernel: parallel over (M-tile, N-tile), sequential sum over K-splits
        c_out = torch.empty((M, N), dtype=torch.bfloat16, device=A_q.device)
        grid_reduce = (triton.cdiv(M, 16), triton.cdiv(N, 64))
        _reduce_kernel[grid_reduce](
            c_partial, c_out, M, N,
            c_partial.stride(0), c_partial.stride(1), c_partial.stride(2),
            c_out.stride(0), c_out.stride(1),
            BLOCK_M=16, BLOCK_N=64,
            NUM_KSPLIT=padded_ksplit,  # must equal c_partial.shape[0] (power-of-2)
        )

    return c_out


# =============================================================================
# Entry points
# =============================================================================

def custom_kernel_orig(data: input_t) -> output_t:
    """Original: aiter.gemm_a4w4 (compiled ASM/CK kernel), kept for reference."""
    import aiter
    from aiter import dtypes
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    from aiter.utility.fp4_utils import e8m0_shuffle

    def _quant_mxfp4(x, shuffle=True):
        x_fp4, bs_e8m0 = dynamic_mxfp4_quant(x)
        if shuffle:
            bs_e8m0 = e8m0_shuffle(bs_e8m0)
        return x_fp4.view(dtypes.fp4x2), bs_e8m0.view(dtypes.fp8_e8m0)

    A, B, B_q, B_shuffle, B_scale_sh = data
    A = A.contiguous()
    B = B.contiguous()
    A_q, A_scale_sh = _quant_mxfp4(A, shuffle=True)
    return aiter.gemm_a4w4(
        A_q, B_shuffle, A_scale_sh, B_scale_sh,
        dtype=dtypes.bf16, bpreshuffle=True,
    )


def custom_kernel(data: input_t) -> output_t:
    """New: hand-written Triton kernel with all loops (parallel + sequential) visible."""
    from aiter import dtypes
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B_q.shape[0]

    # Quantize A to MXFP4 — standard (non-shuffled) layout for the basic kernel
    A_q_u8, A_scale = dynamic_mxfp4_quant(A)
    A_q = A_q_u8.view(dtypes.fp4x2)
    # tl.dot_scaled expects scales as uint8 (e8m0 represented as uint8)
    A_scale = A_scale.view(torch.uint8)

    # B_scale: cache with object identity check (both id() and data_ptr() can be
    # reused by Python GC / PyTorch allocator for different tensors)
    b_key = id(B)
    cached = _b_scale_cache.get(b_key)
    if cached is None or cached[0] is not B:
        _b_scale_cache.clear()
        _, B_scale = dynamic_mxfp4_quant(B)
        _b_scale_cache[b_key] = (B, B_scale.view(torch.uint8))
    B_scale = _b_scale_cache[b_key][1]

    # Compute Split-K factor to saturate ~1024 CUs on MI355X
    TILE_M, TILE_N = 16, 64
    output_tiles = triton.cdiv(m, TILE_M) * triton.cdiv(n, TILE_N)
    num_ksplit = min(16, max(1, 1024 // output_tiles))

    return mxfp4_gemm(
        A_q.view(torch.uint8), B_q.view(torch.uint8), A_scale, B_scale, m, n, k,
        num_ksplit=num_ksplit,
        BLOCK_M=TILE_M, BLOCK_N=TILE_N, BLOCK_K=128,
    )
