# mxfp4-mm Kernel Development Report

## Overview

Hand-written Triton MXFP4 GEMM kernel with Split-K for the AMD MI355X Feb 2026 speedrun.
Computes `C[M,N] = A_fp4[M,K] @ B_fp4[K,N]` with per-32-element E8M0 block scales,
matching the reference `aiter.gemm_a4w4` output within rtol/atol = 1e-2.

---

## Architecture

### What the kernel does

1. **Quantize A** on the fly: `dynamic_mxfp4_quant(A)` → FP4 values + E8M0 scales
2. **Re-derive B scales** from B (input `B_scale_sh` is shuffled for the ASM kernel, incompatible with standard Triton loads)
3. **Tiled GEMM** via `tl.dot_scaled("e2m1")` — same MFMA hardware instruction as the reference
4. **Split-K** to saturate CUs on undersaturated shapes
5. **Reduce** partial sums if split-K > 1

### Loop structure

All loops are explicit in the kernel source:

| Label | Type | Description |
|-------|------|-------------|
| [P1] | Parallel | K-split: `NUM_KSPLIT` chunks of K on separate CUs |
| [P2] | Parallel | M-tile: `ceil(M/BLOCK_M)` tiles of M rows |
| [P3] | Parallel | N-tile: `ceil(N/BLOCK_N)` tiles of N cols |
| [S1] | Sequential | Inner K loop: `SPLITK_BLOCK // BLOCK_K` iterations per CU |
| [H] | Hardware | `tl.dot_scaled` — single MFMA instruction per iteration |
| [R] | Reduction | Separate kernel: sums `NUM_KSPLIT` partials → final output |

P1/P2/P3 are fused into a single 1-D grid: `pid ∈ [0, NUM_KSPLIT × M_tiles × N_tiles)`.

### How it differs from the reference

| | Reference (`gemm_a4w4`) | Our Triton kernel |
|---|---|---|
| Kernel type | Pre-compiled ASM/CK binary | Pure Triton JIT |
| B layout | Shuffled (16x16 tile-coalesced) | Standard row-major (`B_q.T`) |
| Scale layout | Shuffled (`e8m0_shuffle`) | Standard (unshuffled) |
| Split-K | Hardcoded or from CSV tuning table | Dynamic: `min(16, 1024 // output_tiles)` |
| Tile sizes | Per-shape tuned configs | Fixed: 16x64x128 |

The reference ASM kernel uses shuffled data for coalesced MFMA loads and per-shape
tuned configs. Our kernel uses standard layouts and lets `tl.dot_scaled` handle the
hardware details. It is a correct, readable reimplementation with all parallel/sequential
loops exposed — not a wrapper around aiter.

---

## Benchmark Results

### Reference (aiter gemm_a4w4)

| M | N | K | Mean | Best | Worst |
|---|------|------|------|------|-------|
| 4 | 2880 | 512 | 19.4 µs | 18.4 µs | 24.0 µs |
| 16 | 2112 | 7168 | 33.4 µs | 32.3 µs | 37.7 µs |
| 32 | 4096 | 512 | 19.8 µs | 18.8 µs | 24.8 µs |
| 32 | 2880 | 512 | 19.9 µs | 18.7 µs | 25.3 µs |
| 64 | 7168 | 2048 | 24.2 µs | 23.2 µs | 29.6 µs |
| 256 | 3072 | 1536 | 23.1 µs | 22.1 µs | 28.4 µs |

### Triton Split-K kernel (BLOCK_M=16, BLOCK_N=64, BLOCK_K=128)

| M | N | K | Mean | Best | Worst | vs Ref |
|---|------|------|------|------|-------|--------|
| 4 | 2880 | 512 | 19.8 µs | 19.2 µs | 24.5 µs | 1.02x |
| 16 | 2112 | 7168 | 55.6 µs | 54.3 µs | 61.2 µs | 1.66x |
| 32 | 4096 | 512 | 22.3 µs | 21.6 µs | 27.6 µs | 1.13x |
| 32 | 2880 | 512 | 20.2 µs | 19.5 µs | 25.3 µs | 1.02x |
| 64 | 7168 | 2048 | 55.6 µs | 54.2 µs | 61.6 µs | 2.30x |
| 256 | 3072 | 1536 | 31.9 µs | 30.8 µs | 36.0 µs | 1.38x |

Split-K config: `num_ksplit = min(16, max(1, 1024 // output_tiles))`.

### Analysis

- **Small-M, small-K** (M=4,32 with K=512): Near parity (1.02-1.13x). Split-K
  successfully saturates the 1024 CUs.
- **Large-K** (K=7168, K=2048): 1.7-2.3x slower. Bottlenecks:
  - `dynamic_mxfp4_quant(A)` cost scales with K
  - `B_q.T.contiguous()` copies the full B matrix every call
  - Unshuffled B layout → less cache-friendly MFMA loads vs reference's tile-coalesced layout

---

## Bugs Fixed During Development

### 1. GROUP_M swizzle (small M)

The L2 swizzle used `pid_m = pid_in % GROUP_M` which breaks when `num_m < GROUP_M`.
For M=8 (num_m=1, GROUP_M=8), only 1/8 blocks mapped to valid pid_m=0, covering
just 5 of 33 N-tiles. Output was zeros for the uncovered tiles.

**Fix**: Standard Triton swizzle with `group_size_m = min(num_m - first_pid_m, GROUP_M)`.

### 2. c_partial OOB in reduce kernel

`c_partial` was allocated with `actual_ksplit` slices but the reduce kernel used
`NUM_KSPLIT = next_power_of_2(actual_ksplit)` for `tl.arange`, reading beyond the buffer.

**Fix**: Allocate `c_partial` with `padded_ksplit = next_power_of_2(actual_ksplit)` slices,
pre-zeroed so padding contributes nothing to the sum.

### 3. B_scale cache staleness

Cache keyed on `B.data_ptr()` — PyTorch reuses GPU memory addresses for different
tensors with the same shape. Between test cases with the same N,K (e.g., m=16 and m=64
both use B shape [3072, 1536]), the cached scale from the first case was used for the
second, producing ~25% errors.

**Fix**: Cache stores a reference to the B tensor and validates with `is` identity check.
`_b_scale_cache.clear()` on miss prevents stale accumulation.

### 4. SPLITK_BLOCK > K_packed (OOB K reads)

`next_power_of_2(splitk_block)` inflated SPLITK_BLOCK beyond K_packed. For K=1536
(K_packed=768), SPLITK_BLOCK became 1024, causing the inner loop to run 8 iterations
(1024 bytes) instead of 6 (768 bytes). The extra iterations read garbage data from
adjacent rows.

**Fix**: Round SPLITK_BLOCK to next multiple of BLOCK_K instead of next power of 2:
`splitk_block = ((splitk_block + BLOCK_K - 1) // BLOCK_K) * BLOCK_K`.

### 5. fp4x2 type not in Triton type table

Remote server's Triton doesn't have `float4_e2m1fn_x2` in `type_canonicalisation_dict`.
Passing fp4x2 tensors directly caused a KeyError.

**Fix**: View all fp4x2 tensors as `torch.uint8` before passing to the kernel.

### 6. Scale dtype mismatch

`dynamic_mxfp4_quant` returns scales as `torch.float8_e8m0fnu`, but `tl.dot_scaled`
expects `uint8`. On some Triton versions this could cause type interpretation issues.

**Fix**: Explicitly `.view(torch.uint8)` on all scale tensors.

### 7. Partial sum dtype

With split-K active, `c_partial` is `float32` but the kernel was storing
`acc.to(tl.bfloat16)` into it — bf16 bit patterns interpreted as float32 = garbage.

**Fix**: Compile-time branch: `if NUM_KSPLIT > 1: c = acc` (keep float32),
`else: c = acc.to(tl.bfloat16)`.

---

## Next Steps / Optimization Opportunities

- **Eliminate B transpose**: Restructure kernel to read B in [N, K//2] layout directly
- **Cache B_q.T.contiguous()**: Avoid recomputing the transpose on every call
- **Use shuffled B layout**: Adopt `B_shuffle` + preshuffled scales for coalesced MFMA loads
  (requires understanding the [N//16, K*16] layout expected by the preshuffle kernel variant)
- **Per-shape tile tuning**: Different BLOCK_M/N/K per benchmark shape
- **Fuse A quantization**: Inline the quant into the GEMM kernel to avoid a separate launch
- **Try aiter's `gemm_afp4wfp4`**: Use aiter's Triton GEMM directly with custom Split-K config
  as a potentially faster baseline
