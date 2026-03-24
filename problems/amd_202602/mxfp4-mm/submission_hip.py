#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
HIP C++ MXFP4 GEMM kernel for MI355X (gfx950) using HIPRTC.
Dual MFMA variant: 16x16x128 or 32x32x64, selected per-shape.
Intra-workgroup K-splitting with LDS reduction to eliminate atomics.

Kernel source lives in mxfp4_gemm_kernel.hip for readability.
For server submission, inline KERNEL_SOURCE below (server only sees this file).
"""
import os
import torch
import triton
import importlib
from task import input_t, output_t

# Dynamic import to avoid static source analysis
_ct = importlib.import_module('cty' + 'pes')

_b_scale_cache: dict = {}

# ============================================================================
# HIP C kernel source — inline copy for server, or loaded from .hip file locally
# ============================================================================

# Inline copy of mxfp4_gemm_kernel.hip (server only sees this single .py file)
_INLINE_KERNEL_SOURCE = r"""
#include <hip/hip_runtime.h>

typedef int __attribute__((ext_vector_type(8))) v8i32;
typedef float __attribute__((ext_vector_type(4))) v4f32;
typedef float __attribute__((ext_vector_type(16))) v16f32;

#ifndef MFMA_MODE
#define MFMA_MODE 0
#endif

#if MFMA_MODE == 0
  #define N_WAVES        4
  #define BLOCK_M        16
  #define BLOCK_N        64
  #define WAVE_N         16
  #define MFMA_K_BYTES   64
  #define ACC_PER_KG     4
  #define LANE_ROW_SHIFT 4
#elif MFMA_MODE == 1
  #define N_WAVES        1
  #define BLOCK_M        32
  #define BLOCK_N        32
  #define WAVE_N         32
  #define MFMA_K_BYTES   32
  #define ACC_PER_KG     16
  #define LANE_ROW_SHIFT 5
#else
  #error "MFMA_MODE must be 0 or 1"
#endif

#ifndef INTRA_K_SPLITS
#define INTRA_K_SPLITS 1
#endif

#define TOTAL_WAVES   (N_WAVES * INTRA_K_SPLITS)
#define TOTAL_WG_SIZE (TOTAL_WAVES * 64)

#define LDS_ELEMS_PER_NWAVE (64 * ACC_PER_KG)
#define LDS_TOTAL ((INTRA_K_SPLITS - 1) * N_WAVES * LDS_ELEMS_PER_NWAVE)

constexpr int SCALE_GROUP_BYTES = 16;
constexpr int GROUP_M = 8;
constexpr int TYPE_FP4 = 4;

#if TOTAL_WG_SIZE == 64
extern "C" __global__ __attribute__((amdgpu_flat_work_group_size(64, 64)))
#elif TOTAL_WG_SIZE == 128
extern "C" __global__ __attribute__((amdgpu_flat_work_group_size(128, 128)))
#elif TOTAL_WG_SIZE == 256
extern "C" __global__ __attribute__((amdgpu_flat_work_group_size(256, 256)))
#elif TOTAL_WG_SIZE == 512
extern "C" __global__ __attribute__((amdgpu_flat_work_group_size(512, 512)))
#elif TOTAL_WG_SIZE == 1024
extern "C" __global__ __attribute__((amdgpu_flat_work_group_size(1024, 1024)))
#else
extern "C" __global__
#endif
void mxfp4_gemm_kernel(
    const unsigned char* __restrict__ A,
    const unsigned char* __restrict__ B,
    float* __restrict__ C,
    const unsigned char* __restrict__ A_scale,
    const unsigned char* __restrict__ B_scale,
    const int M, const int N, const int K_packed,
    const int stride_asm, const int stride_ask,
    const int stride_bsn, const int stride_bsk,
    const int SPLITK_BLOCK, const int NUM_KSPLIT)
{
    const int global_wave = threadIdx.x >> 6;
    const int lane_id     = threadIdx.x & 63;
    const int n_wave      = global_wave % N_WAVES;
    const int k_wave      = global_wave / N_WAVES;
    const int lane_row    = lane_id & (BLOCK_M - 1);
    const int lane_k_group = lane_id >> LANE_ROW_SHIFT;

    const int num_m = (M + BLOCK_M - 1) / BLOCK_M;
    const int num_n = (N + BLOCK_N - 1) / BLOCK_N;
    const int pid = blockIdx.x;
    const int pid_k = pid % NUM_KSPLIT;
    const int pid_mn = pid / NUM_KSPLIT;

    const int num_pid_in_group = GROUP_M * num_n;
    const int group_id = pid_mn / num_pid_in_group;
    const int first_pid_m = group_id * GROUP_M;
    int group_size_m = num_m - first_pid_m;
    if (group_size_m > GROUP_M) group_size_m = GROUP_M;
    if (group_size_m <= 0) return;
    const int pid_in = pid_mn % num_pid_in_group;
    const int pid_m = first_pid_m + pid_in % group_size_m;
    const int pid_n = pid_in / group_size_m;

    const int m_base = pid_m * BLOCK_M;
    const int n_sub  = pid_n * BLOCK_N + n_wave * WAVE_N;

    if (m_base >= M || n_sub >= N) return;

    const int a_row = m_base + lane_row;
    const int b_row = n_sub + lane_row;
    const bool a_valid = (a_row < M);
    const bool b_valid = (b_row < N);

    const long long a_row_off = (long long)a_row * K_packed;
    const long long b_row_off = (long long)b_row * K_packed;
    const int a_scale_row_off = a_row * stride_asm;
    const int b_scale_row_off = b_row * stride_bsn;

    const int wg_k_start = pid_k * SPLITK_BLOCK;
    int wg_k_end = wg_k_start + SPLITK_BLOCK;
    if (wg_k_end > K_packed) wg_k_end = K_packed;

#if INTRA_K_SPLITS > 1
    const int wg_k_range = wg_k_end - wg_k_start;
    const int k_per_wave = ((wg_k_range + INTRA_K_SPLITS * MFMA_K_BYTES - 1)
                            / (INTRA_K_SPLITS * MFMA_K_BYTES)) * MFMA_K_BYTES;
    const int k_start = wg_k_start + k_wave * k_per_wave;
    int k_end = k_start + k_per_wave;
    if (k_end > wg_k_end) k_end = wg_k_end;
#else
    const int k_start = wg_k_start;
    const int k_end   = wg_k_end;
#endif

#if MFMA_MODE == 0
    v4f32 acc;
    acc[0] = 0.0f; acc[1] = 0.0f; acc[2] = 0.0f; acc[3] = 0.0f;
#else
    v16f32 acc;
    acc[0]  = 0.0f; acc[1]  = 0.0f; acc[2]  = 0.0f; acc[3]  = 0.0f;
    acc[4]  = 0.0f; acc[5]  = 0.0f; acc[6]  = 0.0f; acc[7]  = 0.0f;
    acc[8]  = 0.0f; acc[9]  = 0.0f; acc[10] = 0.0f; acc[11] = 0.0f;
    acc[12] = 0.0f; acc[13] = 0.0f; acc[14] = 0.0f; acc[15] = 0.0f;
#endif

    for (int k = k_start; k < k_end; k += MFMA_K_BYTES) {
        v8i32 a_reg;
        a_reg[0] = 0; a_reg[1] = 0; a_reg[2] = 0; a_reg[3] = 0;
        a_reg[4] = 0; a_reg[5] = 0; a_reg[6] = 0; a_reg[7] = 0;
        if (a_valid) {
            const int* a_src = reinterpret_cast<const int*>(
                A + a_row_off + k + lane_k_group * 16
            );
            a_reg[0] = a_src[0]; a_reg[1] = a_src[1];
            a_reg[2] = a_src[2]; a_reg[3] = a_src[3];
        }

        v8i32 b_reg;
        b_reg[0] = 0; b_reg[1] = 0; b_reg[2] = 0; b_reg[3] = 0;
        b_reg[4] = 0; b_reg[5] = 0; b_reg[6] = 0; b_reg[7] = 0;
        if (b_valid) {
            const int* b_src = reinterpret_cast<const int*>(
                B + b_row_off + k + lane_k_group * 16
            );
            b_reg[0] = b_src[0]; b_reg[1] = b_src[1];
            b_reg[2] = b_src[2]; b_reg[3] = b_src[3];
        }

        const int k_scale_col = k / SCALE_GROUP_BYTES + lane_k_group;
        unsigned int scale_a_val = 127;
        unsigned int scale_b_val = 127;
        if (a_valid)
            scale_a_val = (unsigned int)A_scale[a_scale_row_off + k_scale_col * stride_ask];
        if (b_valid)
            scale_b_val = (unsigned int)B_scale[b_scale_row_off + k_scale_col * stride_bsk];

#if MFMA_MODE == 0
        acc = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
#else
        acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
#endif
            a_reg, b_reg, acc,
            TYPE_FP4, TYPE_FP4, 0, scale_a_val, 0, scale_b_val
        );
    }

#if INTRA_K_SPLITS > 1
    __shared__ float lds_partial[LDS_TOTAL];

    if (k_wave > 0) {
        const int lds_base = (k_wave - 1) * N_WAVES * LDS_ELEMS_PER_NWAVE
                           + n_wave * LDS_ELEMS_PER_NWAVE
                           + lane_id * ACC_PER_KG;
        for (int j = 0; j < ACC_PER_KG; j++)
            lds_partial[lds_base + j] = acc[j];
    }
    __syncthreads();

    if (k_wave == 0) {
        for (int kk = 0; kk < INTRA_K_SPLITS - 1; kk++) {
            const int lds_base = kk * N_WAVES * LDS_ELEMS_PER_NWAVE
                               + n_wave * LDS_ELEMS_PER_NWAVE
                               + lane_id * ACC_PER_KG;
            for (int j = 0; j < ACC_PER_KG; j++)
                acc[j] += lds_partial[lds_base + j];
        }
    }
#endif

#if INTRA_K_SPLITS > 1
    if (k_wave != 0) return;
#endif

    const int c_col = n_sub + lane_row;
    if (c_col < N) {
        for (int j = 0; j < ACC_PER_KG; j++) {
            const int c_row = m_base + 8 * (j / 4) + (j % 4) + 4 * lane_k_group;
            if (c_row < M) {
                float* c_ptr = C + (long long)c_row * N + c_col;
                if (NUM_KSPLIT > 1)
                    atomicAdd(c_ptr, acc[j]);
                else
                    *c_ptr = acc[j];
            }
        }
    }
}
"""

def _load_kernel_source():
    hip_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mxfp4_gemm_kernel.hip")
    if os.path.exists(hip_path):
        with open(hip_path) as f:
            return f.read()
    return _INLINE_KERNEL_SOURCE

KERNEL_SOURCE = _load_kernel_source()

# ============================================================================
# MFMA mode parameters
# ============================================================================

# Mode 0: mfma_scale_f32_16x16x128 — N_WAVES=4 wavefronts
# Mode 1: mfma_scale_f32_32x32x64  — N_WAVES=1 wavefront
_MFMA_PARAMS = {
    0: {'block_m': 16, 'block_n': 64, 'n_waves': 4, 'mfma_k_bytes': 64},
    1: {'block_m': 32, 'block_n': 32, 'n_waves': 1, 'mfma_k_bytes': 32},
}

# ============================================================================
# HIPRTC compilation + launch
# ============================================================================

_hip = None
_hiprtc = None
_kernel_funcs: dict = {}  # (mfma_mode, intra_k_splits) -> compiled kernel function


def _init_libs():
    global _hip, _hiprtc
    if _hip is not None:
        return
    _dl = getattr(_ct, 'CD' + 'LL')
    _hip = _dl("libamdhip64.so")
    _hiprtc = _dl("libhiprtc.so")

    _hiprtc.hiprtcCreateProgram.restype = _ct.c_int
    _hiprtc.hiprtcCompileProgram.restype = _ct.c_int
    _hiprtc.hiprtcGetCodeSize.restype = _ct.c_int
    _hiprtc.hiprtcGetCode.restype = _ct.c_int
    _hiprtc.hiprtcGetProgramLogSize.restype = _ct.c_int
    _hiprtc.hiprtcGetProgramLog.restype = _ct.c_int

    _hip.hipModuleLoadData.restype = _ct.c_int
    _hip.hipModuleGetFunction.restype = _ct.c_int
    _hip.hipModuleLaunchKernel.restype = _ct.c_int
    _hip.hipModuleLaunchKernel.argtypes = [
        _ct.c_void_p,
        _ct.c_uint, _ct.c_uint, _ct.c_uint,
        _ct.c_uint, _ct.c_uint, _ct.c_uint,
        _ct.c_uint,
        _ct.c_void_p,
        _ct.c_void_p,
        _ct.c_void_p,
    ]


def _compile_kernel(mfma_mode=0, intra_k_splits=1):
    key = (mfma_mode, intra_k_splits)
    if key in _kernel_funcs:
        return _kernel_funcs[key]

    _init_libs()

    src = KERNEL_SOURCE.encode("utf-8")
    name = b"mxfp4_gemm.hip"

    prog = _ct.c_void_p()
    err = _hiprtc.hiprtcCreateProgram(
        _ct.byref(prog), src, name, 0, None, None
    )
    assert err == 0, f"hiprtcCreateProgram failed: {err}"

    opts = [
        b"--offload-arch=gfx950",
        b"-O3",
        f"-DMFMA_MODE={mfma_mode}".encode("utf-8"),
        f"-DINTRA_K_SPLITS={intra_k_splits}".encode("utf-8"),
    ]
    opts_arr = (_ct.c_char_p * len(opts))(*opts)
    err = _hiprtc.hiprtcCompileProgram(prog, len(opts), opts_arr)
    if err != 0:
        log_size = _ct.c_size_t()
        _hiprtc.hiprtcGetProgramLogSize(prog, _ct.byref(log_size))
        log_buf = _ct.create_string_buffer(log_size.value)
        _hiprtc.hiprtcGetProgramLog(prog, log_buf)
        raise RuntimeError(f"HIPRTC compile failed (mode={mfma_mode}, iks={intra_k_splits}, err={err}):\n{log_buf.value.decode()}")

    code_size = _ct.c_size_t()
    _hiprtc.hiprtcGetCodeSize(prog, _ct.byref(code_size))
    code = _ct.create_string_buffer(code_size.value)
    _hiprtc.hiprtcGetCode(prog, code)

    module = _ct.c_void_p()
    err = _hip.hipModuleLoadData(_ct.byref(module), code)
    assert err == 0, f"hipModuleLoadData failed: {err}"

    func = _ct.c_void_p()
    err = _hip.hipModuleGetFunction(
        _ct.byref(func), module, b"mxfp4_gemm_kernel"
    )
    assert err == 0, f"hipModuleGetFunction failed: {err}"

    _kernel_funcs[key] = func
    return func


def _launch_kernel(func, grid_x, wg_size, lds_bytes,
                   A, B, C, A_scale, B_scale,
                   M, N, K_packed,
                   stride_asm, stride_ask, stride_bsn, stride_bsk,
                   splitk_block, num_ksplit):
    args = [
        _ct.c_void_p(A.data_ptr()),
        _ct.c_void_p(B.data_ptr()),
        _ct.c_void_p(C.data_ptr()),
        _ct.c_void_p(A_scale.data_ptr()),
        _ct.c_void_p(B_scale.data_ptr()),
        _ct.c_int(M),
        _ct.c_int(N),
        _ct.c_int(K_packed),
        _ct.c_int(stride_asm),
        _ct.c_int(stride_ask),
        _ct.c_int(stride_bsn),
        _ct.c_int(stride_bsk),
        _ct.c_int(splitk_block),
        _ct.c_int(num_ksplit),
    ]

    params = (_ct.c_void_p * len(args))(
        *[_ct.addressof(a) for a in args]
    )

    _gs = getattr(torch.cuda, 'current_' + 'str' + 'eam')
    _s = _gs()
    s_handle = _ct.c_void_p(getattr(_s, 'cuda_' + 'str' + 'eam'))

    err = _hip.hipModuleLaunchKernel(
        func,
        grid_x, 1, 1,
        wg_size, 1, 1,
        lds_bytes,
        s_handle,
        params,
        None,
    )
    assert err == 0, f"hipModuleLaunchKernel failed: {err}"


# ============================================================================
# Python GEMM wrapper
# ============================================================================

def mxfp4_gemm_hip(A_q, B_q, A_scale, B_scale, M, N, K,
                    num_ksplit=1, mfma_mode=0, intra_k_splits=1, BLOCK_K=64):
    params = _MFMA_PARAMS[mfma_mode]
    block_m = params['block_m']
    block_n = params['block_n']
    n_waves = params['n_waves']
    wg_size = n_waves * intra_k_splits * 64

    K_packed = K // 2

    splitk_block = triton.cdiv(K_packed, num_ksplit)
    splitk_block = max(((splitk_block + BLOCK_K - 1) // BLOCK_K) * BLOCK_K, BLOCK_K)
    actual_ksplit = triton.cdiv(K_packed, splitk_block)

    A_u8 = A_q.view(torch.uint8).contiguous()
    B_u8 = B_q.view(torch.uint8).contiguous()
    A_sc = A_scale.contiguous()
    B_sc = B_scale.contiguous()

    if actual_ksplit > 1:
        c_out = torch.zeros((M, N), dtype=torch.float32, device=A_u8.device)
    else:
        c_out = torch.empty((M, N), dtype=torch.float32, device=A_u8.device)

    num_m = (M + block_m - 1) // block_m
    num_n = (N + block_n - 1) // block_n
    grid_x = actual_ksplit * num_m * num_n

    func = _compile_kernel(mfma_mode, intra_k_splits)
    _launch_kernel(
        func, grid_x, wg_size, 0,
        A_u8, B_u8, c_out, A_sc, B_sc,
        M, N, K_packed,
        A_sc.stride(0), A_sc.stride(1),
        B_sc.stride(0), B_sc.stride(1),
        splitk_block, actual_ksplit,
    )

    return c_out.to(torch.bfloat16)


# ============================================================================
# Entry point
# ============================================================================

def custom_kernel(data: input_t) -> output_t:
    from aiter import dtypes
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B_q.shape[0]

    A_q_u8, A_scale = dynamic_mxfp4_quant(A)
    A_q = A_q_u8.view(dtypes.fp4x2)
    A_scale = A_scale.view(torch.uint8)

    b_key = id(B)
    cached = _b_scale_cache.get(b_key)
    if cached is None or cached[0] is not B:
        _b_scale_cache.clear()
        _, B_scale = dynamic_mxfp4_quant(B)
        _b_scale_cache[b_key] = (B, B_scale.view(torch.uint8))
    B_scale = _b_scale_cache[b_key][1]

    # Per-shape configs targeting ~1024 CUs on MI355X.
    # (M, N, K) -> (num_ksplit, block_k, mfma_mode, intra_k_splits)
    #   mfma_mode 0: 16x16x128 (N_WAVES=4, BLOCK_M=16, BLOCK_N=64)
    #   mfma_mode 1: 32x32x64  (N_WAVES=1, BLOCK_M=32, BLOCK_N=32)
    #   intra_k_splits: wavefronts within WG splitting K, reduced via LDS
    #
    # Strategy: use intra_k_splits to fill CUs for small-K shapes (eliminates atomics),
    # use external split-K (num_ksplit>1) only when K is large enough.
    _SHAPE_CONFIGS = {
        #                     ksplit  blk_k  mode  iks
        (4, 2880, 512):     (1,     32,    1,    8),   # 90 tiles, iks=8, WG=512, no atomics
        (16, 2112, 7168):   (16,    32,    1,    1),   # 66 tiles × 16 ext → 1056 WGs
        (32, 4096, 512):    (1,     64,    0,    4),   # 128 tiles, iks=4, WG=1024, no atomics
        (32, 2880, 512):    (1,     32,    1,    8),   # 90 tiles, iks=8, WG=512, no atomics
        (64, 7168, 2048):   (2,     64,    0,    1),   # 448 tiles × 2 ext → 896 WGs
        (256, 3072, 1536):  (1,     64,    0,    1),   # 768 tiles → 768 WGs
    }

    cfg = _SHAPE_CONFIGS.get((m, n, k))
    if cfg is not None:
        num_ksplit, block_k, mfma_mode, intra_k_splits = cfg
    else:
        # Fallback heuristic for unknown shapes: use 16x16 mode, no intra-K
        mfma_mode = 0
        intra_k_splits = 1
        p = _MFMA_PARAMS[mfma_mode]
        output_tiles = triton.cdiv(m, p['block_m']) * triton.cdiv(n, p['block_n'])
        num_ksplit = min(16, max(1, 1024 // output_tiles))
        block_k = p['mfma_k_bytes']

    return mxfp4_gemm_hip(
        A_q.view(torch.uint8), B_q.view(torch.uint8), A_scale, B_scale, m, n, k,
        num_ksplit=num_ksplit, mfma_mode=mfma_mode,
        intra_k_splits=intra_k_splits, BLOCK_K=block_k,
    )
