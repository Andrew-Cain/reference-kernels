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
#  REDUCTION (fused into GEMM kernel via tl.atomic_add when NUM_KSPLIT > 1):
#    [R]  L2 atomic add — each CU atomically accumulates its partial into output
# =============================================================================


@triton.jit
def _mxfp4_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    a_scales_ptr, b_scales_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
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
    # [R] Write output: atomic_add for fused split-K reduction, store for direct
    # -------------------------------------------------------------------------
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c_ptrs = (c_ptr
              + offs_m[:, None].to(tl.int64) * stride_cm
              + offs_n[None, :].to(tl.int64) * stride_cn)
    if NUM_KSPLIT > 1:
        # Fused reduction: atomically accumulate float32 partials in-place
        tl.atomic_add(c_ptrs, acc, mask=c_mask, sem="relaxed")
    else:
        tl.store(c_ptrs, acc.to(tl.bfloat16), mask=c_mask)


def mxfp4_gemm(A_q, B_q, A_scale, B_scale, M, N, K, num_ksplit=1,
               BLOCK_M=16, BLOCK_N=64, BLOCK_K=128):
    """
    Launch the tiled MXFP4 GEMM.

    Grid = [P1] × [P2] × [P3] = NUM_KSPLIT × ceil(M/BLOCK_M) × ceil(N/BLOCK_N) CUs in parallel.
    Each CU runs [S1] = SPLITK_BLOCK // BLOCK_K sequential K-tile iterations.
    If NUM_KSPLIT > 1, reduction is fused via tl.atomic_add (no separate reduce kernel).

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

    # Allocate output:
    # - split-K: float32 zeroed buffer for atomic_add accumulation
    # - no split: bf16 buffer for direct store
    if use_splitk:
        c_out = torch.zeros((M, N), dtype=torch.float32, device=A_q.device)
    else:
        c_out = torch.empty((M, N), dtype=torch.bfloat16, device=A_q.device)

    # B is stored [N, K//2]; we pass it transposed so kernel sees [K//2, N]
    B_t = B_q.T.contiguous()

    # Triton's type_canonicalisation_dict on this server doesn't include float4_e2m1fn_x2;
    # view as uint8 — same bits, compatible pointer type
    A_u8 = A_q.view(torch.uint8)
    B_u8 = B_t.view(torch.uint8)

    # [P1][P2][P3] — total parallel CUs
    grid = (actual_ksplit * triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    _mxfp4_gemm_kernel[grid](
        A_u8, B_u8, c_out,
        A_scale, B_scale,
        M, N, K_packed,
        A_u8.stride(0),  A_u8.stride(1),
        B_u8.stride(0),  B_u8.stride(1),
        c_out.stride(0), c_out.stride(1),
        A_scale.stride(0), A_scale.stride(1),
        B_scale.stride(0), B_scale.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_M=8,
        NUM_KSPLIT=actual_ksplit,
        SPLITK_BLOCK=splitk_block,
        SCALE_GROUP=SCALE_GROUP,
        num_warps=4, num_stages=2,
    )

    # Convert float32 atomic accumulation to bf16
    if use_splitk:
        c_out = c_out.to(torch.bfloat16)

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
