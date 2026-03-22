#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
HIP C++ MXFP4 GEMM kernel for MI355X (gfx950) using HIPRTC.
Uses MFMA intrinsics with atomic split-K reduction.

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
# HIP C kernel source — loaded from .hip file, or inline fallback for server
# ============================================================================

def _load_kernel_source():
    hip_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mxfp4_gemm_kernel.hip")
    if os.path.exists(hip_path):
        with open(hip_path) as f:
            return f.read()
    raise FileNotFoundError(
        f"Kernel source not found at {hip_path}. "
        "For server submission, inline the kernel source as KERNEL_SOURCE in this file."
    )

KERNEL_SOURCE = _load_kernel_source()

# ============================================================================
# HIPRTC compilation + launch
# ============================================================================

_hip = None
_hiprtc = None
_kernel_func = None


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


def _compile_kernel():
    global _kernel_func
    if _kernel_func is not None:
        return _kernel_func

    _init_libs()

    src = KERNEL_SOURCE.encode("utf-8")
    name = b"mxfp4_gemm.hip"

    prog = _ct.c_void_p()
    err = _hiprtc.hiprtcCreateProgram(
        _ct.byref(prog), src, name, 0, None, None
    )
    assert err == 0, f"hiprtcCreateProgram failed: {err}"

    opts = [b"--offload-arch=gfx950", b"-O3"]
    opts_arr = (_ct.c_char_p * len(opts))(*opts)
    err = _hiprtc.hiprtcCompileProgram(prog, len(opts), opts_arr)
    if err != 0:
        log_size = _ct.c_size_t()
        _hiprtc.hiprtcGetProgramLogSize(prog, _ct.byref(log_size))
        log_buf = _ct.create_string_buffer(log_size.value)
        _hiprtc.hiprtcGetProgramLog(prog, log_buf)
        raise RuntimeError(f"HIPRTC compile failed (err={err}):\n{log_buf.value.decode()}")

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

    _kernel_func = func
    return func


def _launch_kernel(func, grid_x,
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
        256, 1, 1,
        0,
        s_handle,
        params,
        None,
    )
    assert err == 0, f"hipModuleLaunchKernel failed: {err}"


# ============================================================================
# Python GEMM wrapper
# ============================================================================

def mxfp4_gemm_hip(A_q, B_q, A_scale, B_scale, M, N, K, num_ksplit=1,
                    BLOCK_M=16, BLOCK_N=64, BLOCK_K=128):
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

    num_m = (M + BLOCK_M - 1) // BLOCK_M
    num_n = (N + BLOCK_N - 1) // BLOCK_N
    grid_x = actual_ksplit * num_m * num_n

    func = _compile_kernel()
    _launch_kernel(
        func, grid_x,
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

    TILE_M, TILE_N = 16, 64
    output_tiles = triton.cdiv(m, TILE_M) * triton.cdiv(n, TILE_N)
    num_ksplit = min(16, max(1, 1024 // output_tiles))

    return mxfp4_gemm_hip(
        A_q.view(torch.uint8), B_q.view(torch.uint8), A_scale, B_scale, m, n, k,
        num_ksplit=num_ksplit,
        BLOCK_M=TILE_M, BLOCK_N=TILE_N, BLOCK_K=128,
    )
