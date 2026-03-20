Let's compare with our mxfp4-mm benchmark shapes:

┌───────────────────┬──────────────────────┬────────────────┬────────────┐
│   Shape (MxNxK)   │ Output tiles (32x32) │ vs 1,024 cores │ Saturation │
├───────────────────┼──────────────────────┼────────────────┼────────────┤
│ 4 × 2880 × 512    │ 1 × 90 = 90          │ 8.8%           │ Very low   │
├───────────────────┼──────────────────────┼────────────────┼────────────┤
│ 16 × 2112 × 7168  │ 1 × 66 = 66          │ 6.4%           │ Very low   │
├───────────────────┼──────────────────────┼────────────────┼────────────┤
│ 32 × 4096 × 512   │ 1 × 128 = 128        │ 12.5%          │ Low        │
├───────────────────┼──────────────────────┼────────────────┼────────────┤
│ 32 × 2880 × 512   │ 1 × 90 = 90          │ 8.8%           │ Low        │
├───────────────────┼──────────────────────┼────────────────┼────────────┤
│ 64 × 7168 × 2048  │ 2 × 224 = 448        │ 43.8%          │ Medium     │
├───────────────────┼──────────────────────┼────────────────┼────────────┤
│ 256 × 3072 × 1536 │ 8 × 96 = 768         │ 75%            │ Decent     │
└───────────────────┴──────────────────────┴────────────────┴────────────┘

Every shape is undersaturated — especially the small-M cases where M=4 or M=16 produces only 1 row of tiles. The chip is mostly idle.

This is why:
1. Our benchmarks are 2x slower than the tuned reference — the default kernel doesn't compensate for this
2. Split-K is critical for these shapes — it creates extra blocks by parallelizing the K reduction, trading more blocks for a final reduction step
3. For M=4, K=512: splitting K into 8 chunks gives 8 × 90 = 720 blocks instead of 90


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
