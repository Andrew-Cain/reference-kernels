// ============================================================================
// MXFP4 GEMM kernel for MI355X (gfx950) — dual MFMA variant with LDS reduction
//
// Computes C[M,N] = A[M,K] @ B[N,K]^T using MFMA FP4 intrinsics.
// A and B are MXFP4-quantized: E2M1 data packed 2 values per byte, with
// E8M0 per-block scales (one scale per 32 elements along K).
//
// Compile-time parameters (passed via -D flags):
//
//   MFMA_MODE: selects the MFMA instruction variant
//     0 — mfma_scale_f32_16x16x128 (N_WAVES=4, BLOCK_M=16, BLOCK_N=64)
//     1 — mfma_scale_f32_32x32x64  (N_WAVES=1, BLOCK_M=32, BLOCK_N=32)
//
//   INTRA_K_SPLITS: number of K-chunks reduced within the workgroup via LDS
//     1 — no intra-WG reduction (uses external split-K with atomicAdd)
//     2,4 — wavefronts split K, reduce via LDS, write without atomics
//
// Workgroup layout: (N_WAVES × INTRA_K_SPLITS) wavefronts of 64 threads.
// n_wave selects which N sub-tile; k_wave selects which K chunk.
// After the K-loop, k_wave>0 stores partials to LDS, k_wave=0 reduces and writes.
// ============================================================================

#include <hip/hip_runtime.h>

typedef int __attribute__((ext_vector_type(8))) v8i32;
typedef float __attribute__((ext_vector_type(4))) v4f32;
typedef float __attribute__((ext_vector_type(16))) v16f32;

// ---- MFMA mode configuration ----

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

// ---- Intra-workgroup K-splitting configuration ----

#ifndef INTRA_K_SPLITS
#define INTRA_K_SPLITS 1
#endif

#define TOTAL_WAVES   (N_WAVES * INTRA_K_SPLITS)
#define TOTAL_WG_SIZE (TOTAL_WAVES * 64)

// LDS for intra-WG reduction: (INTRA_K_SPLITS-1) partials per N-wave per thread
// k_wave=0 keeps its result in registers; only k_wave>=1 store to LDS.
#define LDS_ELEMS_PER_NWAVE (64 * ACC_PER_KG)
#define LDS_TOTAL ((INTRA_K_SPLITS - 1) * N_WAVES * LDS_ELEMS_PER_NWAVE)

constexpr int SCALE_GROUP_BYTES = 16;
constexpr int GROUP_M = 8;
constexpr int TYPE_FP4 = 4;

// ---- Kernel entry point ----
// Use explicit WG size values since macros in attributes can be fragile.

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
    // ---- Thread identity ----
    // Wavefronts are organized as: n_wave (tiles N) × k_wave (tiles K)
    const int global_wave = threadIdx.x >> 6;
    const int lane_id     = threadIdx.x & 63;
    const int n_wave      = global_wave % N_WAVES;         // 0..N_WAVES-1
    const int k_wave      = global_wave / N_WAVES;         // 0..INTRA_K_SPLITS-1
    const int lane_row    = lane_id & (BLOCK_M - 1);
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
    const int n_sub  = pid_n * BLOCK_N + n_wave * WAVE_N;

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

    // ---- K range for this wavefront ----
    // External split-K determines the workgroup's K range.
    // Intra-WG splitting further divides it among k_waves.
    const int wg_k_start = pid_k * SPLITK_BLOCK;
    int wg_k_end = wg_k_start + SPLITK_BLOCK;
    if (wg_k_end > K_packed) wg_k_end = K_packed;

#if INTRA_K_SPLITS > 1
    const int wg_k_range = wg_k_end - wg_k_start;
    // Round up to MFMA_K_BYTES so each wave gets aligned chunks
    const int k_per_wave = ((wg_k_range + INTRA_K_SPLITS * MFMA_K_BYTES - 1)
                            / (INTRA_K_SPLITS * MFMA_K_BYTES)) * MFMA_K_BYTES;
    const int k_start = wg_k_start + k_wave * k_per_wave;
    int k_end = k_start + k_per_wave;
    if (k_end > wg_k_end) k_end = wg_k_end;
#else
    const int k_start = wg_k_start;
    const int k_end   = wg_k_end;
#endif

    // ---- Accumulator ----
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
            /*cbsz*/ TYPE_FP4, /*blgp*/ TYPE_FP4,
            /*op_sel_a*/ 0, scale_a_val,
            /*op_sel_b*/ 0, scale_b_val
        );
    }

    // ---- LDS reduction (when INTRA_K_SPLITS > 1) ----
    // k_wave=0 keeps its partial in registers.
    // k_wave>=1 stores to LDS, then k_wave=0 reads and accumulates.
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

    // ---- Store output ----
    // Only k_wave=0 writes (other waves have already contributed via LDS).
    // Output mapping (transposed, interleaved in groups of 4 rows):
    //   acc[j] → C[m_base + 8*(j/4) + (j%4) + 4*lane_k_group, n_sub + lane_row]
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
