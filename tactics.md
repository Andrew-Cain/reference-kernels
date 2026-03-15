# GPU Kernel Optimization Tactics

## Programming Hierarchy

From highest to lowest level of control:

```
PyTorch            Call torch ops, torch.compile — no kernel control
    ↓
aiter              AMD's optimized kernel library (pre-built CK assembly)
    ↓
Triton             Write tile-level GPU programs in Python
    ↓
HIP C++            Explicit thread/wavefront/block control
    ↓
CK                 AMD's Composable Kernel template framework
    ↓
CDNA assembly      Raw ISA — full instruction scheduling control
```

**aiter is not a kernel language.** It's a library of pre-compiled kernels (like cuBLAS for NVIDIA). You call it; you don't write kernels in it. To beat aiter, you write your own kernels in Triton or HIP C++.

| Level | Parallelism | Pipelining | Wavefront Specialization |
|-------|------------|------------|--------------------------|
| PyTorch | Implicit | No | No |
| Triton | Block-level (tile programs) | Automatic software pipelining | No |
| HIP C++ | Thread/wavefront/block explicit | Manual double-buffering | Yes |
| CK | Full template control | Explicit pipeline stages | Yes |
| Assembly | Individual instruction | Explicit `s_waitcnt` overlap | Full control |

---

## Loop Nest Structure

Every GEMM-based kernel is a loop nest. For the MoE kernel:

```
for token m in M:                    # batch dimension
  for expert e in top_k:             # 9 experts per token
    for tile_n in N/BLOCK_N:         # output dimension tiles
      for tile_k in K/BLOCK_K:       # reduction dimension tiles
        accumulate(A_tile, B_tile)   # MFMA instructions
```

On a GPU, **loop order = what's parallel vs. what's sequential**:

| Loop | GPU mapping | What it means |
|------|------------|---------------|
| Outermost parallel loops | `blockIdx` / `program_id` | Each GPU block computes one tile |
| Inner parallel loops | `threadIdx` / wavefront lanes | Threads within a block cooperate |
| Sequential loops | Explicit `for` in kernel code | The reduction dimension (K) |

**Loop interchange** on a GPU = changing which dimensions are mapped to blocks (parallel) vs. which are explicit for-loops (sequential). This changes memory access patterns and occupancy.

---

## Loop Transformations

### Loop Interchange
Change which dimension is parallel vs. sequential:
```python
# Default: M and N parallel, K sequential
pid_m = tl.program_id(0)  # parallelize M
pid_n = tl.program_id(1)  # parallelize N
for k in range(0, K, BLOCK_K):  # sequential reduction
    ...

# Interchanged: M and K parallel, N sequential (split-K)
pid_m = tl.program_id(0)
pid_k = tl.program_id(1)  # parallelize K
for n in range(0, N, BLOCK_N):  # sequential over N
    ...
```

### Loop Tiling
Change tile sizes to match hardware:
```python
BLOCK_M: tl.constexpr = 128   # tokens per block
BLOCK_N: tl.constexpr = 128   # output cols per block
BLOCK_K: tl.constexpr = 32    # reduction tile (matches MXFP4 block scale size)
```

Considerations:
- Larger tiles → more register pressure, fewer waves, but better data reuse
- Smaller tiles → more waves (better occupancy), but less reuse
- BLOCK_K=32 aligns naturally with MXFP4's 32-element block scaling

### Loop Fusion (Inter-stage)
Combine two kernel launches into one:
```
Before: kernel1(gate_up GEMM + SwiGLU) → write intermediate → kernel2(down GEMM)
After:  single_kernel(gate_up GEMM → SwiGLU → down GEMM in registers)
```
Eliminates the intermediate buffer write/read — a full global memory round-trip.

### Loop Unrolling
- Triton auto-unrolls most loops
- Use `tl.static_range()` for compile-time unrolling when needed
- In HIP C++: `#pragma unroll`

---

## Triton Kernel Structure

```python
@triton.jit
def gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # === Parallel loops (mapped to GPU blocks) ===
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # === Sequential loop (reduction dimension) ===
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        a = tl.load(A_ptr + offs_m[:, None] * K + offs_k[None, :])
        b = tl.load(B_ptr + offs_k[:, None] * N + offs_n[None, :])
        acc += tl.dot(a, b)

    tl.store(C_ptr + offs_m[:, None] * N + offs_n[None, :], acc)

# Launch: each (pid_m, pid_n) pair is one tile — runs in parallel
grid = (M // BLOCK_M, N // BLOCK_N)
gemm_kernel[grid](A, B, C, M, N, K, BLOCK_M=128, BLOCK_N=128, BLOCK_K=32)
```

To apply transformations:
- **Interchange**: swap what's a `program_id` vs. what's a `for` loop
- **Tile**: change `BLOCK_M`, `BLOCK_N`, `BLOCK_K`
- **Fuse**: put two GEMMs in the same kernel, keep intermediate in `acc` registers
- **Split-K**: add a `program_id(2)` for K dimension, reduce partial sums after

---

## AMD MI355X Architecture

### Execution Model
- **Compute Units (CUs)**: the parallel processors on the chip
- **Wavefront**: 64 threads executing in lockstep (AMD's equivalent of NVIDIA's warp of 32)
- **Multiple wavefronts per CU**: hardware schedules wavefronts to hide latency
- **Occupancy**: ratio of active wavefronts to maximum — higher occupancy hides memory latency

### Memory Hierarchy
```
Registers          ← fastest, per-thread, limited
    ↓
LDS (Local Data Share)  ← shared within a workgroup (block), ~64 KB per CU
    ↓
L2 Cache           ← shared across CUs
    ↓
HBM (Global)       ← highest bandwidth bulk memory, highest latency
```

### MFMA Instructions
Matrix Fused Multiply-Add — AMD's tensor core equivalent:
- Operate on matrix tiles (e.g., 16x16, 32x32)
- Support bf16, fp16, fp8, fp4 input types
- Accumulate in fp32
- These are what Triton's `tl.dot()` and CK's GEMM compile down to

### Key Optimization Levers
- **Vectorized loads**: load 128 bits at a time from global memory
- **LDS bank conflicts**: arrange data to avoid threads hitting the same LDS bank
- **Async global→LDS copies**: overlap data movement with compute (HIP C++ only)
- **Register pressure**: fewer registers per thread → more wavefronts → better latency hiding

---

## Pipelining

### Concept
Overlap memory loads with compute so the GPU is never idle waiting for data:

```
Without pipelining:
  [Load tile 0] [Compute tile 0] [Load tile 1] [Compute tile 1] ...

With double-buffering:
  [Load tile 0] [Load tile 1   ] [Load tile 2   ] ...
                [Compute tile 0 ] [Compute tile 1 ] ...
```

### In Triton
Triton applies software pipelining automatically via `num_stages`:
```python
# Triton handles double/triple buffering internally
# Tune num_stages in the kernel config
```

### In HIP C++
Manual double-buffering with async copies:
```cpp
// Two LDS buffers
__shared__ half A_lds[2][BLOCK_M][BLOCK_K];
__shared__ half B_lds[2][BLOCK_K][BLOCK_N];

// Prefetch first tile
async_load(A_lds[0], A_global + 0);
async_load(B_lds[0], B_global + 0);
__syncthreads();

for (int k = 0; k < K; k += BLOCK_K) {
    int curr = (k / BLOCK_K) % 2;
    int next = 1 - curr;

    // Prefetch next tile while computing current
    if (k + BLOCK_K < K) {
        async_load(A_lds[next], A_global + k + BLOCK_K);
        async_load(B_lds[next], B_global + k + BLOCK_K);
    }

    // Compute on current tile
    mfma_compute(A_lds[curr], B_lds[curr], acc);
    __syncthreads();
}
```

---

## Wavefront Specialization

Only available in HIP C++ or lower. Assign different roles to different wavefronts within a workgroup:

```cpp
// Producer wavefronts: load data from global → LDS
// Consumer wavefronts: compute MFMA on LDS data

int wavefront_id = threadIdx.x / 64;

if (wavefront_id < NUM_PRODUCERS) {
    // This wavefront only does memory loads
    while (has_work) {
        async_load_tile_to_lds(...);
        signal_consumers();
        wait_for_consumers();
    }
} else {
    // This wavefront only does compute
    while (has_work) {
        wait_for_producers();
        mfma_compute_from_lds(...);
        signal_producers();
    }
}
```

Why it helps:
- Producer wavefronts are memory-latency bound — hardware can schedule many of them
- Consumer wavefronts are compute-bound — they saturate MFMA units
- Decoupling avoids the "load then compute then load" serialization

**Not possible in Triton** — Triton's single-program model treats all threads identically.

---

## Quantization Tactics

### MXFP4 (4-bit)
- **4x bandwidth savings** over bf16 — loads 4x less data from HBM
- Block size = 32 elements per E8M0 scale (power-of-2 exponent)
- FP4 format: E2M1, values {0, 0.5, 1, 1.5, 2, 3, 4, 6}
- Packed as fp4x2 (2 values per byte)

### FP8 (8-bit)
- **2x bandwidth savings** over bf16
- Per-tensor scalar scale
- MI355X uses e4m3fnuz variant

### Fusing Quantization into GEMM
The reference quantizes activations in a **separate pass** before the GEMM:
```
[Quant kernel: read A from HBM, write A_fp4 to HBM] → [GEMM kernel: read A_fp4 from HBM]
```

Fusing quant into the GEMM prologue eliminates one HBM round-trip:
```
[GEMM kernel: read A_bf16, quantize in registers, compute]
```

### Dequantization Fusion
For attention (MLA): instead of dequantizing KV cache to bf16 then doing attention, load quantized KV tiles and dequantize in registers/LDS immediately before the matmul.

---

## Fusion Opportunities (MoE Kernel)

1. **Activation quantization fusion** — fuse dynamic MXFP4 quant of activations into Stage 1 GEMM prologue. Saves one global memory round-trip.

2. **Inter-stage fusion** — fuse Stage 1 (gate_up GEMM + SwiGLU) and Stage 2 (down GEMM) into a single kernel. Keep intermediate results in registers instead of writing to global memory.

3. **Shared expert fusion** — the shared expert is always selected for all tokens. Compute it as a dense GEMM (no routing overhead) and fuse with the routed expert reduction.

4. **Expert-parallel wave scheduling** — with 257 experts but only 9 active per token, use work-stealing or compact-dispatch to avoid wasted wavefronts on empty expert slots.

5. **Split-K for large M** — for bs=512+ with EP-on (E=33, d_expert=2048), GEMMs are large enough for split-K parallelism within each expert.

---

## Profiling

```bash
# Torch profiler trace (shows CUDA/HIP kernel times)
popcorn-cli submit --no-tui --mode profile submission.py

# Benchmark with L2 cache clearing and statistical convergence
popcorn-cli submit --no-tui --mode benchmark submission.py
```

Profile mode runs `torch.profiler` with CPU + CUDA activity tracing and returns a table sorted by `self_cuda_time_total`. Use this to identify which operations dominate runtime before optimizing.
