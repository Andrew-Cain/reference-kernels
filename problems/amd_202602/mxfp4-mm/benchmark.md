# CU saturation analysis (BLOCK_M=16, BLOCK_N=64, MI355X = 1,024 CUs)

| Shape (M×N×K) | M tiles | N tiles | Output tiles | vs 1,024 CUs | Saturation |
|----------------|---------|---------|--------------|--------------|------------|
| 4 × 2880 × 512 | 1 | 45 | 45 | 4.4% | Very low |
| 16 × 2112 × 7168 | 1 | 33 | 33 | 3.2% | Very low |
| 32 × 4096 × 512 | 2 | 64 | 128 | 12.5% | Low |
| 32 × 2880 × 512 | 2 | 45 | 90 | 8.8% | Low |
| 64 × 7168 × 2048 | 4 | 112 | 448 | 43.8% | Medium |
| 256 × 3072 × 1536 | 16 | 48 | 768 | 75.0% | Decent |

Every shape is undersaturated — especially the small-M cases where M=4 or M=16
produces only 1 row of M-tiles. Split-K is critical: it multiplies the grid by
NUM_KSPLIT to fill more CUs, at the cost of an atomic reduction.


# mxfp4-mm Reference Benchmark (aiter gemm_a4w4 on MI355X)

| M | N | K | Mean | Best | Worst |
|---|------|------|------|------|-------|
| 4 | 2880 | 512 | 19.4 µs | 18.4 µs | 24.0 µs |
| 16 | 2112 | 7168 | 33.4 µs | 32.3 µs | 37.7 µs |
| 32 | 4096 | 512 | 19.8 µs | 18.8 µs | 24.8 µs |
| 32 | 2880 | 512 | 19.9 µs | 18.7 µs | 25.3 µs |
| 64 | 7168 | 2048 | 24.2 µs | 23.2 µs | 29.6 µs |
| 256 | 3072 | 1536 | 23.1 µs | 22.1 µs | 28.4 µs |

# Triton Split-K Kernel Benchmark (submission.py, BLOCK_M=16, BLOCK_N=64, BLOCK_K=128)

| M | N | K | Mean | Best | Worst | vs Reference |
|---|------|------|------|------|-------|-------------|
| 4 | 2880 | 512 | 19.8 µs | 19.2 µs | 24.5 µs | 1.02x |
| 16 | 2112 | 7168 | 55.6 µs | 54.3 µs | 61.2 µs | 1.66x |
| 32 | 4096 | 512 | 22.3 µs | 21.6 µs | 27.6 µs | 1.13x |
| 32 | 2880 | 512 | 20.2 µs | 19.5 µs | 25.3 µs | 1.02x |
| 64 | 7168 | 2048 | 55.6 µs | 54.2 µs | 61.6 µs | 2.30x |
| 256 | 3072 | 1536 | 31.9 µs | 30.8 µs | 36.0 µs | 1.38x |

Split-K config: num_ksplit = min(16, max(1, 1024 // output_tiles)).
Small-M shapes (M=4, 32) are near reference. Large-K shapes (K=7168, 2048) are 1.7-2.3x slower.

# Fused Atomic Reduce (tl.atomic_add, sem="relaxed") — REJECTED

Replaced separate `_reduce_kernel` with `tl.atomic_add` in the GEMM kernel.
All K-splits atomically accumulate float32 into a single [M,N] buffer, then
convert to bf16 via `c_out.to(torch.bfloat16)` in Python.

| M | N | K | Mean | Best | Worst | vs 2-kernel | vs Reference |
|---|------|------|------|------|-------|-------------|--------------|
| 4 | 2880 | 512 | 20.3 µs | 19.7 µs | 25.3 µs | +2.5% | 1.05x |
| 16 | 2112 | 7168 | 57.2 µs | 55.8 µs | 61.7 µs | +2.9% | 1.71x |
| 32 | 4096 | 512 | 25.1 µs | 24.4 µs | 29.4 µs | +12.6% | 1.27x |
| 32 | 2880 | 512 | 22.3 µs | 21.6 µs | 27.0 µs | +10.4% | 1.12x |
| 64 | 7168 | 2048 | 62.6 µs | 61.1 µs | 67.9 µs | +12.6% | 2.59x |
| 256 | 3072 | 1536 | 32.0 µs | 31.0 µs | 35.2 µs | +0.3% | 1.38x |

**Result: slower across all split-K shapes (2.5–12.6%).** M=256 unchanged (no split-K).

**Why it's slower:**
1. Atomic contention — even 2 K-splits serialize L2 atomic adds per output element
2. `c_out.to(torch.bfloat16)` is itself a kernel launch, replacing the reduce kernel
   with an equally expensive dtype conversion kernel
3. Net effect: traded reduce kernel overhead for atomic overhead + conversion overhead

# HIP C++ MFMA Kernel (submission_hip.py, HIPRTC + atomic split-K)

Uses `__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4` directly via HIPRTC.
BLOCK_M=16, BLOCK_N=64, 4 wavefronts per workgroup, atomic float32 split-K.

| M | N | K | Mean | Best | Worst | vs Reference | vs Triton |
|---|------|------|------|------|-------|-------------|-----------|
| 4 | 2880 | 512 | 20.4 µs | 19.8 µs | 25.2 µs | 1.05x | 1.03x |
| 16 | 2112 | 7168 | 27.5 µs | 26.5 µs | 32.3 µs | 0.82x | 0.49x |
| 32 | 4096 | 512 | 21.3 µs | 20.4 µs | 30.0 µs | 1.08x | 0.96x |
| 32 | 2880 | 512 | 20.8 µs | 19.9 µs | 29.1 µs | 1.05x | 1.03x |
| 64 | 7168 | 2048 | 39.5 µs | 38.2 µs | 44.2 µs | 1.63x | 0.71x |
| 256 | 3072 | 1536 | 36.4 µs | 35.2 µs | 43.4 µs | 1.58x | 1.14x |

## HIP kernel CU saturation with split-K

Split-K formula: `num_ksplit = min(16, max(1, 1024 // output_tiles))`,
then aligned to BLOCK_K=128 bytes. Grid = actual_ksplit × output_tiles.

| Shape (M×N×K) | Output tiles | Requested K-split | Actual K-split | Grid size | vs 1,024 CUs | Saturation |
|----------------|-------------|-------------------|----------------|-----------|--------------|------------|
| 4 × 2880 × 512 | 45 | 16 | 2 | 90 | 8.8% | Very low |
| 16 × 2112 × 7168 | 33 | 16 | 14 | 462 | 45.1% | Medium |
| 32 × 4096 × 512 | 128 | 8 | 2 | 256 | 25.0% | Low |
| 32 × 2880 × 512 | 90 | 11 | 2 | 180 | 17.6% | Low |
| 64 × 7168 × 2048 | 448 | 2 | 2 | 896 | 87.5% | Good |
| 256 × 3072 × 1536 | 768 | 1 | 1 | 768 | 75.0% | Decent |

Small-K shapes (K=512) can only split to 2 because K_packed=256 aligned to
BLOCK_K=128 gives at most 2 chunks. The 16×2112×7168 shape benefits most from
split-K (14-way), which explains why it beats the reference despite low base tiles.

**Key observations:**
- Beats Triton significantly on high-K shapes (16×2112×7168: 2x faster, 64×7168×2048: 1.4x faster)
- 16×2112×7168 actually beats the aiter reference (0.82x) — high K-split fills CUs well
- Slower on large-tile shapes (M=256) — no split-K benefit, naive global memory access pattern

# Dual MFMA Kernel (16x16x128 + 32x32x64, per-shape dispatch)

Uses both MFMA variants compiled via HIPRTC with `-DMFMA_MODE=N`:
- Mode 0: `mfma_scale_f32_16x16x128` (256 threads, BLOCK_M=16, BLOCK_N=64)
- Mode 1: `mfma_scale_f32_32x32x64`  (64 threads,  BLOCK_M=32, BLOCK_N=32)

32x32 output mapping: interleaved 4-row groups — `row = 8*(j/4) + (j%4) + 4*lane_kg`.
This is a unified formula: for 16x16 (j=0..3) it simplifies to `4*kg + j`.

| M | N | K | Mode | Mean | Best | Worst | vs Reference | vs v1 HIP |
|---|------|------|------|------|------|-------|-------------|-----------|
| 4 | 2880 | 512 | 32x32 | 20.4 µs | 19.6 µs | 25.2 µs | 1.05x | 1.00x |
| 16 | 2112 | 7168 | 16x16 | 28.7 µs | 27.5 µs | 33.3 µs | 0.86x | 1.04x |
| 32 | 4096 | 512 | 32x32 | 23.3 µs | 22.4 µs | 29.4 µs | 1.18x | 1.09x |
| 32 | 2880 | 512 | 32x32 | 22.2 µs | 21.4 µs | 27.3 µs | 1.12x | 1.07x |
| 64 | 7168 | 2048 | 16x16 | 39.3 µs | 38.2 µs | 43.9 µs | 1.62x | 0.99x |
| 256 | 3072 | 1536 | 16x16 | 36.2 µs | 35.1 µs | 41.3 µs | 1.57x | 1.00x |

## Dual kernel CU saturation

Per-shape config with optimal MFMA mode and split-K:

| Shape (M×N×K) | Mode | Tiles | K-split | Grid | vs 1,024 CUs | Saturation |
|----------------|------|-------|---------|------|--------------|------------|
| 4 × 2880 × 512 | 32x32 | 90 | 8 | 720 | 70.3% | Good |
| 16 × 2112 × 7168 | 16x16 | 33 | 28 | 924 | 90.2% | Excellent |
| 32 × 4096 × 512 | 32x32 | 128 | 8 | 1024 | 100% | Full |
| 32 × 2880 × 512 | 32x32 | 90 | 8 | 720 | 70.3% | Good |
| 64 × 7168 × 2048 | 16x16 | 448 | 2 | 896 | 87.5% | Excellent |
| 256 × 3072 × 1536 | 16x16 | 768 | 1 | 768 | 75.0% | Good |

**Result:** The 32x32 mode didn't improve timing for K=512 shapes despite much higher CU
saturation (70-100% vs 9-25%). The 32x32 shapes are actually slightly slower than v1.
This suggests the bottleneck is NOT CU saturation but rather atomic contention or memory
bandwidth per-output-element. With 8 K-splits, each output element gets 8 atomic adds,
which may serialize in L2 cache. The 16x16 mode shapes are unchanged as expected.

# LDS Intra-Workgroup Reduction (v3, 16x16x128 + INTRA_K_SPLITS)

Replaces external split-K (multiple WGs + atomicAdd) with intra-workgroup K-splitting
where multiple wavefronts within a single WG each process a K-chunk, then reduce via LDS.
Only k_wave=0 writes to global memory — no atomics needed when external NUM_KSPLIT=1.

Compile-time define: `-DINTRA_K_SPLITS=N` (1, 2, or 4).
WG size = N_WAVES × INTRA_K_SPLITS × 64 threads.

| M | N | K | IKS | Ext-K | Mean | Best | Worst | vs Reference | vs v2 Dual |
|---|------|------|-----|-------|------|------|-------|-------------|------------|
| 4 | 2880 | 512 | 4 | 1 | 19.0 µs | 18.2 µs | 25.6 µs | 0.98x | 0.93x |
| 16 | 2112 | 7168 | 1 | 28 | 28.4 µs | 27.4 µs | 35.2 µs | 0.85x | 0.99x |
| 32 | 4096 | 512 | 4 | 1 | 19.6 µs | 18.8 µs | 27.1 µs | 0.99x | 0.84x |
| 32 | 2880 | 512 | 4 | 1 | 19.6 µs | 18.7 µs | 27.9 µs | 0.98x | 0.88x |
| 64 | 7168 | 2048 | 1 | 2 | 39.5 µs | 38.3 µs | 44.3 µs | 1.63x | 1.01x |
| 256 | 3072 | 1536 | 1 | 1 | 36.4 µs | 35.2 µs | 41.0 µs | 1.58x | 1.01x |

## LDS kernel CU saturation

| Shape (M×N×K) | Mode | Tiles | IKS | Ext-K | WG threads | Grid | Saturation |
|----------------|------|-------|-----|-------|------------|------|------------|
| 4 × 2880 × 512 | 16x16 | 45 | 4 | 1 | 1024 | 45 | 4.4% |
| 16 × 2112 × 7168 | 16x16 | 33 | 1 | 28 | 256 | 924 | 90.2% |
| 32 × 4096 × 512 | 16x16 | 128 | 4 | 1 | 1024 | 128 | 12.5% |
| 32 × 2880 × 512 | 16x16 | 90 | 4 | 1 | 1024 | 90 | 8.8% |
| 64 × 7168 × 2048 | 16x16 | 448 | 1 | 2 | 256 | 896 | 87.5% |
| 256 × 3072 × 1536 | 16x16 | 768 | 1 | 1 | 256 | 768 | 75.0% |

**Key findings:**
- K=512 shapes improved 16-19% vs v2 Dual (32x32+atomics), now matching reference
- LDS reduction eliminates all atomics for K=512 shapes (ext-K=1, single write)
- Low CU saturation (4-13%) for K=512 shapes doesn't hurt — confirms the bottleneck
  was atomic contention, not CU fill. 45 WGs × 1024 threads still keeps CUs busy
- High-K shapes (K=7168, 2048) unchanged — still using external split-K with atomics
- M=256, M=64 still 1.58-1.63x slower than reference — different bottleneck (likely
  memory access pattern or occupancy)

# LDS + 32x32 Mode for Selected Shapes (v4)

Switched 4×2880×512, 16×2112×7168, and 32×2880×512 to 32x32 mode with LDS reduction
(no atomic contention since LDS handles the intra-WG reduce).

| M | N | K | Mode | IKS | Ext-K | Mean | Best | Worst | vs Reference | vs v3 LDS |
|---|------|------|------|-----|-------|------|------|-------|-------------|-----------|
| 4 | 2880 | 512 | 32x32 | 8 | 1 | 19.2 µs | 18.5 µs | 29.6 µs | 0.99x | +1% |
| 16 | 2112 | 7168 | 32x32 | 1 | 16 | 27.7 µs | 26.6 µs | 32.8 µs | 0.83x | -2.5% |
| 32 | 4096 | 512 | 16x16 | 4 | 1 | 19.7 µs | 18.8 µs | 26.9 µs | 0.99x | 0% |
| 32 | 2880 | 512 | 32x32 | 8 | 1 | 22.1 µs | 22.1 µs | 22.2 µs | 1.11x | +13% |
| 64 | 7168 | 2048 | 16x16 | 1 | 2 | 39.1 µs | 38.0 µs | 46.8 µs | 1.62x | -1% |
| 256 | 3072 | 1536 | 16x16 | 1 | 1 | 36.2 µs | 35.1 µs | 41.7 µs | 1.57x | -1% |

**Result:** Mixed. 16×2112×7168 improved (27.7 vs 28.4 µs, now 0.83x reference).
32×2880×512 regressed (22.1 vs 19.6 µs). The 16x16 MFMA processes 2x more K per
instruction (64 bytes vs 32 bytes), so for K=512 shapes it's more compute-efficient.
32x32 mode benefits the high-K shape where more external K-splits fill CUs better
(66 tiles × 16 ext-K = 1056 WGs vs 33 tiles × 28 ext-K = 924 WGs).

# Config Tuning v5 (revert regressions + iks=2 for large shapes)

Reverted 32×2880×512 and 4×2880×512 to 16x16 mode with iks=4 (was 32x32 iks=8).
Added iks=2 for 64×7168×2048 (eliminates atomics from ext-K=2) and 256×3072×1536.

| M | N | K | Mode | IKS | Ext-K | Mean | Best | Worst | vs Reference | vs v4 |
|---|------|------|------|-----|-------|------|------|-------|-------------|-------|
| 4 | 2880 | 512 | 16x16 | 4 | 1 | 18.9 µs | 18.0 µs | 26.6 µs | 0.97x | -2% |
| 16 | 2112 | 7168 | 32x32 | 1 | 16 | 27.6 µs | 26.5 µs | 34.1 µs | 0.83x | 0% |
| 32 | 4096 | 512 | 16x16 | 4 | 1 | 19.8 µs | 19.0 µs | 27.8 µs | 1.00x | 0% |
| 32 | 2880 | 512 | 16x16 | 4 | 1 | 19.5 µs | 18.6 µs | 25.3 µs | 0.98x | -12% |
| 64 | 7168 | 2048 | 16x16 | 2 | 1 | 36.7 µs | 35.4 µs | 47.8 µs | 1.52x | -6% |
| 256 | 3072 | 1536 | 16x16 | 2 | 1 | 34.8 µs | 33.6 µs | 39.1 µs | 1.51x | -4% |

**Key findings:**
- Reverting 32×2880×512 to 16x16 mode recovered from 22.1→19.5 µs (matches v3 LDS)
- iks=2 for 64×7168×2048 improved 39.1→36.7 µs by eliminating atomic contention
- iks=2 for 256×3072×1536 improved 36.2→34.8 µs
- Both large shapes still 1.5x slower than reference — memory access pattern bottleneck

# B_shuffle Coalesced Loads (v6)

B_shuffle uses (16,16) tile-coalesced layout where each 16-row × 16-byte sub-tile
occupies a contiguous 256-byte block. Eliminates strided memory access for B loads.
Compile flag: `-DUSE_SHUFFLE=1`. B_scale still uses unshuffled row-major format.

Shuffle disabled for 16×2112×7168 (mode 1/32x32) where it regressed (+5%).

| M | N | K | Mode | Shuf | Mean | Best | Worst | vs Reference | vs v5 |
|---|------|------|------|------|------|------|-------|-------------|-------|
| 4 | 2880 | 512 | 16x16 | yes | 18.7 µs | 17.9 µs | 24.7 µs | 0.96x | -1% |
| 16 | 2112 | 7168 | 32x32 | no | 27.8 µs | 26.8 µs | 32.4 µs | 0.83x | 0% |
| 32 | 4096 | 512 | 16x16 | yes | 19.5 µs | 18.7 µs | 28.0 µs | 0.98x | -2% |
| 32 | 2880 | 512 | 16x16 | yes | 19.2 µs | 18.3 µs | 26.1 µs | 0.96x | -2% |
| 64 | 7168 | 2048 | 16x16 | yes | 33.4 µs | 32.3 µs | 43.6 µs | 1.38x | -9% |
| 256 | 3072 | 1536 | 16x16 | yes | 31.3 µs | 30.3 µs | 35.7 µs | 1.36x | -10% |

**Key findings:**
- Large shapes (M=64, M=256) improved ~9-10% from coalesced B loads
- K=512 shapes see small 1-2% improvement (already memory-efficient with small K)
- 32x32 mode (16×2112×7168) regressed with shuffle — possibly because 32-row tiles
  span two 16-row shuffle groups, reducing coalescing benefit
- Remaining 1.36-1.38x gap vs reference is likely from: output write pattern,
  A load striding, or reference's superior tile scheduling
