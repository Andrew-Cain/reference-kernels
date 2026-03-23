// ============================================================================
// MXFP4 GEMM kernel for MI355X (gfx950)
//
// Computes C[M,N] = A[M,K] @ B[N,K]^T using the MFMA 16x16x128 FP4 intrinsic.
// A and B are MXFP4-quantized: E2M1 data packed 2 values per byte, with
// E8M0 per-block scales (one scale per 32 elements along K).
//
// Workgroup layout: 256 threads = 4 wavefronts of 64 threads each.
// Each wavefront computes one 16x16 output sub-tile via a single MFMA.
// The 4 wavefronts tile across the N dimension: BLOCK_N = 4 * 16 = 64.
//
// Split-K: multiple workgroups can share the same output tile, each computing
// a partial sum over a K-chunk. Partials are accumulated via atomicAdd.
// ============================================================================

#include <hip/hip_runtime.h>

// Vector types required by the MFMA intrinsic on ROCm 7.1.
// The builtin demands v8i32 operands (256 bits = 64 FP4 values per thread),
// even though only the lower 4 ints (128 bits) carry data per lane_k_group.
typedef int __attribute__((ext_vector_type(4))) v4i32;
typedef int __attribute__((ext_vector_type(8))) v8i32;
typedef float __attribute__((ext_vector_type(4))) v4f32;

extern "C"
__global__ __attribute__((amdgpu_flat_work_group_size(256, 256)))
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
    // ---- Tile dimensions ----
    constexpr int BLOCK_M = 16;             // M-tiles per workgroup (matches MFMA M)
    constexpr int BLOCK_N = 64;             // N-tiles per workgroup (4 wavefronts * 16)
    constexpr int WAVE_N = 16;              // N per wavefront (matches MFMA N)
    constexpr int MFMA_K_BYTES = 64;        // Bytes consumed per MFMA: 128 FP4 values / 2 = 64 bytes
    constexpr int SCALE_GROUP_BYTES = 16;   // Bytes per scale group: 32 FP4 values / 2 = 16 bytes
    constexpr int GROUP_M = 8;              // L2 swizzle group size along M
    constexpr int TYPE_FP4 = 4;             // MFMA cbsz/blgp selector for E2M1 FP4

    // ---- Thread identity within the workgroup ----
    // 256 threads = 4 wavefronts. wave_id selects which 16-col sub-tile along N.
    // Within a wavefront (64 threads), the MFMA maps threads as:
    //   lane_row    = lane_id % 16  -> which row of the 16x16 tile
    //   lane_k_group = lane_id / 16 -> which of 4 K-groups (each loads 16 bytes)
    const int wave_id = threadIdx.x >> 6;       // 0..3
    const int lane_id = threadIdx.x & 63;       // 0..63
    const int lane_row = lane_id & 15;          // 0..15: row within 16x16 tile
    const int lane_k_group = lane_id >> 4;      // 0..3: K-group index

    // ---- Block ID decomposition with L2 swizzle ----
    // 1D grid is linearized as: pid = pid_k * (num_m * num_n) + pid_mn
    // L2 swizzle groups GROUP_M consecutive M-tiles together to improve
    // cache locality for B accesses across adjacent M-tiles.
    const int num_m = (M + BLOCK_M - 1) / BLOCK_M;
    const int num_n = (N + BLOCK_N - 1) / BLOCK_N;
    const int pid = blockIdx.x;
    const int pid_k = pid % NUM_KSPLIT;         // which K-split chunk
    const int pid_mn = pid / NUM_KSPLIT;         // which output tile

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
    const int m_base = pid_m * BLOCK_M;                     // first M-row of this tile
    const int n_sub = pid_n * BLOCK_N + wave_id * WAVE_N;   // first N-col for this wavefront

    if (m_base >= M || n_sub >= N) return;

    // ---- Per-thread data pointers ----
    // Each thread is responsible for one row of A (row = m_base + lane_row)
    // and one row of B (row = n_sub + lane_row). "Row" in B corresponds to
    // a column of the logical B^T, i.e. we compute A @ B^T.
    const int a_row = m_base + lane_row;
    const int b_row = n_sub + lane_row;
    const bool a_valid = (a_row < M);
    const bool b_valid = (b_row < N);

    const long long a_row_off = (long long)a_row * K_packed;   // byte offset to start of A row
    const long long b_row_off = (long long)b_row * K_packed;   // byte offset to start of B row
    const int a_scale_row_off = a_row * stride_asm;            // scale offset for A row
    const int b_scale_row_off = b_row * stride_bsn;            // scale offset for B row

    // ---- Split-K range ----
    const int k_start = pid_k * SPLITK_BLOCK;
    int k_end = k_start + SPLITK_BLOCK;
    if (k_end > K_packed) k_end = K_packed;

    // ---- Accumulator ----
    // MFMA 16x16 produces 4 float outputs per thread (v4f32).
    v4f32 acc;
    acc[0] = 0.0f; acc[1] = 0.0f; acc[2] = 0.0f; acc[3] = 0.0f;

    // ---- Main K loop ----
    // Each iteration feeds one MFMA instruction: 128 FP4 values = 64 packed bytes.
    // Each thread loads 16 bytes (32 FP4 values) from its lane_k_group slice.
    for (int k = k_start; k < k_end; k += MFMA_K_BYTES) {

        // Load 16 bytes of A data for this thread's row and K-group.
        // Layout: A[row, k + lane_k_group*16 .. +16] as 4 x int32.
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

        // Load 16 bytes of B data similarly.
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
        // Each scale covers 32 FP4 values = 16 packed bytes. The MFMA consumes
        // 128 FP4 = 4 scale groups, and each lane_k_group handles one group.
        // Default 127 = E8M0 exponent for 2^0 = 1.0 (neutral scale for OOB).
        const int k_scale_col = k / SCALE_GROUP_BYTES + lane_k_group;

        unsigned int scale_a_val = 127;
        unsigned int scale_b_val = 127;
        if (a_valid)
            scale_a_val = (unsigned int)A_scale[a_scale_row_off + k_scale_col * stride_ask];
        if (b_valid)
            scale_b_val = (unsigned int)B_scale[b_scale_row_off + k_scale_col * stride_bsk];

        // Execute the MFMA: 16x16 matmul over 128 FP4 elements with block scaling.
        // cbsz=4, blgp=4 select E2M1 FP4 for both A and B operands.
        acc = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
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
    // MFMA 16x16x128 output mapping (gfx950 FP4 variant):
    //   acc[j] of thread (lane_row, lane_k_group) -> C[4*lane_k_group + j, lane_row]
    // This is TRANSPOSED from the standard expectation of C[lane_row, 4*lane_k_group + j].
    // Discovered empirically — the M and N roles are swapped in the output register layout.
    const int c_col = n_sub + lane_row;
    if (c_col < N) {
        for (int j = 0; j < 4; j++) {
            const int c_row = m_base + 4 * lane_k_group + j;
            if (c_row < M) {
                float* c_ptr = C + (long long)c_row * N + c_col;
                if (NUM_KSPLIT > 1)
                    atomicAdd(c_ptr, acc[j]);   // multiple K-splits accumulate atomically
                else
                    *c_ptr = acc[j];            // single K-split: direct write
            }
        }
    }
}
