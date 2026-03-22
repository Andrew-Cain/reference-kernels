#include <hip/hip_runtime.h>

typedef int __attribute__((ext_vector_type(4))) v4i32;
typedef int __attribute__((ext_vector_type(8))) v8i32;
typedef float __attribute__((ext_vector_type(4))) v4f32;

extern "C"
__global__ __attribute__((amdgpu_flat_work_group_size(256, 256)))
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
    constexpr int BLOCK_M = 16;
    constexpr int BLOCK_N = 64;
    constexpr int WAVE_N = 16;
    constexpr int MFMA_K_BYTES = 64;
    constexpr int SCALE_GROUP_BYTES = 16;
    constexpr int GROUP_M = 8;
    constexpr int TYPE_FP4 = 4;  // A/B type selector: E2M1 FP4

    const int wave_id = threadIdx.x >> 6;
    const int lane_id = threadIdx.x & 63;
    const int lane_row = lane_id & 15;
    const int lane_k_group = lane_id >> 4;

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
    const int n_sub = pid_n * BLOCK_N + wave_id * WAVE_N;

    if (m_base >= M || n_sub >= N) return;

    const int a_row = m_base + lane_row;
    const int b_row = n_sub + lane_row;
    const bool a_valid = (a_row < M);
    const bool b_valid = (b_row < N);

    const long long a_row_off = (long long)a_row * K_packed;
    const long long b_row_off = (long long)b_row * K_packed;
    const int a_scale_row_off = a_row * stride_asm;
    const int b_scale_row_off = b_row * stride_bsn;

    const int k_start = pid_k * SPLITK_BLOCK;
    int k_end = k_start + SPLITK_BLOCK;
    if (k_end > K_packed) k_end = K_packed;

    v4f32 acc;
    acc[0] = 0.0f; acc[1] = 0.0f; acc[2] = 0.0f; acc[3] = 0.0f;

    for (int k = k_start; k < k_end; k += MFMA_K_BYTES) {
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

        const int k_scale_col = k / SCALE_GROUP_BYTES + lane_k_group;

        unsigned int scale_a_val = 127;
        unsigned int scale_b_val = 127;
        if (a_valid)
            scale_a_val = (unsigned int)A_scale[a_scale_row_off + k_scale_col * stride_ask];
        if (b_valid)
            scale_b_val = (unsigned int)B_scale[b_scale_row_off + k_scale_col * stride_bsk];

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

    // MFMA 16x16x128 output mapping: acc[j] -> D[4*lane_kg + j, lane_row]
    // (transposed relative to the standard M-row, N-col expectation)
    const int c_col = n_sub + lane_row;
    if (c_col < N) {
        for (int j = 0; j < 4; j++) {
            const int c_row = m_base + 4 * lane_k_group + j;
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
