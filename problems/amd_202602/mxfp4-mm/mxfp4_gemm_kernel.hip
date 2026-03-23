// ============================================================================
// MXFP4 GEMM kernel for MI355X (gfx950) — dual MFMA variant
//
// Computes C[M,N] = A[M,K] @ B[N,K]^T using MFMA FP4 intrinsics.
// A and B are MXFP4-quantized: E2M1 data packed 2 values per byte, with
// E8M0 per-block scales (one scale per 32 elements along K).
//
// Compile-time MFMA_MODE selects the instruction variant:
//
//   MFMA_MODE 0 — mfma_scale_f32_16x16x128_f8f6f4
//     4 wavefronts (256 threads), BLOCK_M=16, BLOCK_N=64
//     128 FP4 per MFMA = 64 packed bytes, 4 float accumulator per thread
//     Best for: high-K shapes where CUs are already well saturated
//
//   MFMA_MODE 1 — mfma_scale_f32_32x32x64_f8f6f4
//     1 wavefront (64 threads), BLOCK_M=32, BLOCK_N=32
//     64 FP4 per MFMA = 32 packed bytes, 16 float accumulator per thread
//     Best for: small-K shapes where finer K-splitting fills more CUs
//
// Split-K: multiple workgroups can share the same output tile, each computing
// a partial sum over a K-chunk. Partials are accumulated via atomicAdd.
// ============================================================================

#include <hip/hip_runtime.h>

// ---- Vector types for MFMA operands and accumulators ----
typedef int __attribute__((ext_vector_type(8))) v8i32;
typedef float __attribute__((ext_vector_type(4))) v4f32;
typedef float __attribute__((ext_vector_type(16))) v16f32;

// ---- Compile-time configuration based on MFMA_MODE ----

#ifndef MFMA_MODE
#define MFMA_MODE 0
#endif

#if MFMA_MODE == 0
  // 16x16x128: 4 wavefronts per workgroup
  #define WG_SIZE        256
  #define BLOCK_M        16
  #define BLOCK_N        64
  #define WAVE_N         16
  #define MFMA_K_BYTES   64      // 128 FP4 / 2 = 64 bytes per MFMA
  #define ACC_PER_KG     4       // accumulator elements per K-group
  #define LANE_ROW_SHIFT 4       // log2(16) for lane_k_group = lane_id >> shift
#elif MFMA_MODE == 1
  // 32x32x64: 1 wavefront per workgroup
  #define WG_SIZE        64
  #define BLOCK_M        32
  #define BLOCK_N        32
  #define WAVE_N         32
  #define MFMA_K_BYTES   32      // 64 FP4 / 2 = 32 bytes per MFMA
  #define ACC_PER_KG     16      // accumulator elements per K-group
  #define LANE_ROW_SHIFT 5       // log2(32) for lane_k_group = lane_id >> shift
#else
  #error "MFMA_MODE must be 0 (16x16x128) or 1 (32x32x64)"
#endif

constexpr int SCALE_GROUP_BYTES = 16;   // 32 FP4 / 2 = 16 bytes per E8M0 scale group
constexpr int GROUP_M = 8;              // L2 swizzle group size along M
constexpr int TYPE_FP4 = 4;             // MFMA cbsz/blgp selector for E2M1 FP4

// ---- Kernel entry point ----

#if MFMA_MODE == 0
extern "C" __global__ __attribute__((amdgpu_flat_work_group_size(256, 256)))
#else
extern "C" __global__ __attribute__((amdgpu_flat_work_group_size(64, 64)))
#endif
void mxfp4_gemm_kernel(
    const unsigned char* __restrict__ A,        // [M, K_packed] row-major, FP4 packed 2/byte
    const unsigned char* __restrict__ B,        // [N, K_packed] row-major, FP4 packed 2/byte
    float* __restrict__ C,                      // [M, N] row-major output
    const unsigned char* __restrict__ A_scale,  // [M, K/32] E8M0 block scales
    const unsigned char* __restrict__ B_scale,  // [N, K/32] E8M0 block scales
    const int M, const int N, const int K_packed,
    const int stride_asm, const int stride_ask, // A_scale strides (row, col)
    const int stride_bsn, const int stride_bsk, // B_scale strides (row, col)
    const int SPLITK_BLOCK, const int NUM_KSPLIT)
{
    // ---- Thread identity ----
    // wave_id: which wavefront in the workgroup (0 for 32x32, 0..3 for 16x16)
    // lane_row: which row of the MFMA tile this thread handles
    // lane_k_group: which K-group this thread belongs to (loads a K-slice)
    const int wave_id = threadIdx.x >> 6;
    const int lane_id = threadIdx.x & 63;
    const int lane_row = lane_id & (BLOCK_M - 1);
    const int lane_k_group = lane_id >> LANE_ROW_SHIFT;

    // ---- Block ID decomposition with L2 swizzle ----
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

    // ---- Tile coordinates ----
    const int m_base = pid_m * BLOCK_M;
    const int n_sub = pid_n * BLOCK_N + wave_id * WAVE_N;

    if (m_base >= M || n_sub >= N) return;

    // ---- Per-thread data pointers ----
    const int a_row = m_base + lane_row;
    const int b_row = n_sub + lane_row;
    const bool a_valid = (a_row < M);
    const bool b_valid = (b_row < N);

    const long long a_row_off = (long long)a_row * K_packed;
    const long long b_row_off = (long long)b_row * K_packed;
    const int a_scale_row_off = a_row * stride_asm;
    const int b_scale_row_off = b_row * stride_bsn;

    // ---- Split-K range ----
    const int k_start = pid_k * SPLITK_BLOCK;
    int k_end = k_start + SPLITK_BLOCK;
    if (k_end > K_packed) k_end = K_packed;

    // ---- Accumulator ----
    // 16x16 → 4 floats per thread (v4f32)
    // 32x32 → 16 floats per thread (v16f32)
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

    // ---- Main K loop ----
    // Each thread loads 16 bytes (4 × int32 = 32 FP4 values) from its K-group slice.
    // Both MFMA variants use v8i32 operands; only the lower 4 ints carry data.
    for (int k = k_start; k < k_end; k += MFMA_K_BYTES) {

        // Load A data: A[a_row, k + lane_k_group*16 .. +16]
        v8i32 a_reg;
        a_reg[0] = 0; a_reg[1] = 0; a_reg[2] = 0; a_reg[3] = 0;
        a_reg[4] = 0; a_reg[5] = 0; a_reg[6] = 0; a_reg[7] = 0;
        if (a_valid) {
            const int* a_src = reinterpret_cast<const int*>(
                A + a_row_off + k + lane_k_group * 16
            );
            a_reg[0] = a_src[0];
            a_reg[1] = a_src[1];
            a_reg[2] = a_src[2];
            a_reg[3] = a_src[3];
        }

        // Load B data: B[b_row, k + lane_k_group*16 .. +16]
        v8i32 b_reg;
        b_reg[0] = 0; b_reg[1] = 0; b_reg[2] = 0; b_reg[3] = 0;
        b_reg[4] = 0; b_reg[5] = 0; b_reg[6] = 0; b_reg[7] = 0;
        if (b_valid) {
            const int* b_src = reinterpret_cast<const int*>(
                B + b_row_off + k + lane_k_group * 16
            );
            b_reg[0] = b_src[0];
            b_reg[1] = b_src[1];
            b_reg[2] = b_src[2];
            b_reg[3] = b_src[3];
        }

        // Load E8M0 block scales.
        // Each scale covers 32 FP4 = 16 packed bytes. The number of scale groups
        // per MFMA depends on the variant (4 for 16x16, 2 for 32x32), and each
        // lane_k_group handles one group. Default 127 = 2^0 (neutral scale).
        const int k_scale_col = k / SCALE_GROUP_BYTES + lane_k_group;

        unsigned int scale_a_val = 127;
        unsigned int scale_b_val = 127;
        if (a_valid)
            scale_a_val = (unsigned int)A_scale[a_scale_row_off + k_scale_col * stride_ask];
        if (b_valid)
            scale_b_val = (unsigned int)B_scale[b_scale_row_off + k_scale_col * stride_bsk];

        // Execute MFMA with block scaling. cbsz=4, blgp=4 select E2M1 FP4.
#if MFMA_MODE == 0
        acc = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
#else
        acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
#endif
            a_reg, b_reg, acc,
            /*cbsz*/ TYPE_FP4,
            /*blgp*/ TYPE_FP4,
            /*op_sel_a*/ 0,
            scale_a_val,
            /*op_sel_b*/ 0,
            scale_b_val
        );
    }

    // ---- Store output ----
    // Output mapping (transposed, interleaved in groups of 4 rows):
    //   acc[j] → C[m_base + 8*(j/4) + (j%4) + 4*lane_k_group, n_sub + lane_row]
    //
    // For 16x16 (j=0..3): simplifies to row = 4*lane_kg + j (verified empirically)
    // For 32x32 (j=0..15): interleaves — kg=0 gets rows {0-3,8-11,16-19,24-27},
    //                                     kg=1 gets rows {4-7,12-15,20-23,28-31}
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
