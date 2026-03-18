# MXFP4 (16,16) Shuffle Layout — 64x64 FP4 Matrix

## Setup

A 64x64 matrix in FP4 is stored as **[64, 32]** in fp4x2 (2 FP4 values packed per byte).
Each byte `(n, k)` holds logical FP4 values at columns `2k` and `2k+1`.

## Shuffle Parameters

From `shuffle_weight(x, layout=(16, 16))`:
```python
IN, IK = 16, 16
BN = 16          # tile height (rows)
BK = 32          # tile width (bytes) = IK * 2
K  = 16          # sub-tile width (bytes) = 16 // element_size(uint8)

x_ = x.view(-1, N//16, 16, K_bytes//32, 2, 16)
#              [N_tiles, n,  K_tiles, k_sub, k]
x_ = x_.permute(0, 1, 3, 4, 2, 5)
#              [N_tiles, K_tiles, k_sub, n, k]
```

The permute swaps the `n` and `k_sub` dimensions — instead of iterating
"for each row, all sub-tiles", it iterates "for each sub-tile, all rows".

## Before Shuffle (Row-Major)

Each row's 32 bytes are contiguous. `(n, k)` = byte at row n, byte-column k.

```
Memory     Contents
Address
─────────────────────────────────────────────────────────────────────────
         ┌── Tile 0 (rows 0-15) ────────────────────────────────────────┐
  0-31   │ (0,0)  (0,1)  ... (0,15) │ (0,16) (0,17) ... (0,31)        │
 32-63   │ (1,0)  (1,1)  ... (1,15) │ (1,16) (1,17) ... (1,31)        │
 64-95   │ (2,0)  (2,1)  ... (2,15) │ (2,16) (2,17) ... (2,31)        │
   ...   │  ...                      │  ...                             │
480-511  │ (15,0) (15,1) ... (15,15)│ (15,16)(15,17)... (15,31)       │
         └──────────────────────────────────────────────────────────────┘
         ┌── Tile 1 (rows 16-31) ───────────────────────────────────────┐
512-543  │ (16,0) (16,1) ... (16,15)│ (16,16)(16,17)... (16,31)       │
   ...   │  ...                      │  ...                             │
         └──────────────────────────────────────────────────────────────┘
         ┌── Tile 2 (rows 32-47) ───────────────────────────────────────┐
   ...   │  ...                      │  ...                             │
         └──────────────────────────────────────────────────────────────┘
         ┌── Tile 3 (rows 48-63) ───────────────────────────────────────┐
   ...   │  ...                      │  ...                             │
1984-2015│ (63,0) (63,1) ... (63,15)│ (63,16)(63,17)... (63,31)       │
         └──────────────────────────────────────────────────────────────┘

Total: 64 rows x 32 bytes = 2048 bytes
```

**Problem**: To load a 16x16 sub-tile (e.g., rows 0-15, byte-cols 0-15),
a wavefront must skip 16 bytes per row (the cols 16-31 gap). Reads are strided.

## After Shuffle (Tile-Coalesced)

The permute groups all bytes of each 16x16 sub-tile contiguously:

```
Memory     Contents
Address
─────────────────────────────────────────────────────────────────────────
         ┌── Tile 0, Sub-tile 0 (rows 0-15, byte-cols 0-15) ──────────┐
  0- 15  │ (0,0)  (0,1)  (0,2)  ... (0,15)                            │
 16- 31  │ (1,0)  (1,1)  (1,2)  ... (1,15)                            │
 32- 47  │ (2,0)  (2,1)  (2,2)  ... (2,15)                            │
   ...   │  ...                                                         │
240-255  │ (15,0) (15,1) (15,2) ... (15,15)                           │
         └─────────────────────────────────────────── 256 bytes ────────┘
         ┌── Tile 0, Sub-tile 1 (rows 0-15, byte-cols 16-31) ─────────┐
256-271  │ (0,16) (0,17) (0,18) ... (0,31)                            │
272-287  │ (1,16) (1,17) (1,18) ... (1,31)                            │
   ...   │  ...                                                         │
496-511  │ (15,16)(15,17)(15,18)... (15,31)                           │
         └─────────────────────────────────────────── 256 bytes ────────┘
         ┌── Tile 1, Sub-tile 0 (rows 16-31, byte-cols 0-15) ─────────┐
512-527  │ (16,0) (16,1) (16,2) ... (16,15)                           │
   ...   │  ...                                                         │
752-767  │ (31,0) (31,1) (31,2) ... (31,15)                           │
         └─────────────────────────────────────────── 256 bytes ────────┘
         ┌── Tile 1, Sub-tile 1 (rows 16-31, byte-cols 16-31) ────────┐
768-783  │ (16,16)(16,17)(16,18)... (16,31)                           │
   ...   │  ...                                                         │
1008-1023│ (31,16)(31,17)(31,18)... (31,31)                           │
         └─────────────────────────────────────────── 256 bytes ────────┘
         ┌── Tile 2, Sub-tile 0 (rows 32-47, byte-cols 0-15) ─────────┐
1024-1039│ (32,0) (32,1) (32,2) ... (32,15)                           │
   ...   │  ...                                                         │
         └─────────────────────────────────────────── 256 bytes ────────┘
         ┌── Tile 2, Sub-tile 1 (rows 32-47, byte-cols 16-31) ────────┐
   ...   │  ...                                                         │
         └─────────────────────────────────────────── 256 bytes ────────┘
         ┌── Tile 3, Sub-tile 0 (rows 48-63, byte-cols 0-15) ─────────┐
   ...   │  ...                                                         │
         └─────────────────────────────────────────── 256 bytes ────────┘
         ┌── Tile 3, Sub-tile 1 (rows 48-63, byte-cols 16-31) ────────┐
   ...   │  ...                                                         │
1984-2015│ (63,16)(63,17)(63,18)... (63,31)                           │
         └─────────────────────────────────────────── 256 bytes ────────┘

Total: 4 tiles x 2 sub-tiles x 256 bytes = 2048 bytes (same size)
```

## Side-by-Side Comparison (Tile 0)

```
BEFORE (row-major):                    AFTER (tile-coalesced):

addr  0: (0,0) ...(0,15)(0,16)...(0,31)    addr  0: (0,0) ...(0,15)
addr 32: (1,0) ...(1,15)(1,16)...(1,31)    addr 16: (1,0) ...(1,15)
addr 64: (2,0) ...(2,15)(2,16)...(2,31)    addr 32: (2,0) ...(2,15)
  ...                                         ...
addr 480:(15,0)...(15,15)(15,16)...(15,31)  addr 240:(15,0)...(15,15)
                                            ─── sub-tile boundary ───
Each row = 32 bytes (whole row)             addr 256:(0,16)...(0,31)
Sub-tile data is interleaved                addr 272:(1,16)...(1,31)
with the other sub-tile.                      ...
                                            addr 496:(15,16)...(15,31)

                                            Each sub-tile = 256 contiguous bytes
```

## Why This Matters for MFMA

The MFMA instruction on MI355X consumes a 16x16 tile of data. A wavefront has 64 threads.

**Before shuffle**: To load a 16x16 sub-tile, threads must read with stride 32
(skip the other sub-tile's bytes every row). Memory requests scatter across
cache lines.

```
Thread 0 reads addr 0   (row 0, col 0)
Thread 1 reads addr 32  (row 1, col 0)  ← 32-byte gap
Thread 2 reads addr 64  (row 2, col 0)  ← 32-byte gap
...
Cache lines loaded: many, partially used
```

**After shuffle**: The entire 16x16 sub-tile is 256 contiguous bytes.
Threads read sequential addresses — fully coalesced.

```
Thread 0 reads addr 0   (row 0, col 0)
Thread 1 reads addr 1   (row 0, col 1)  ← adjacent!
Thread 2 reads addr 2   (row 0, col 2)  ← adjacent!
...
Cache lines loaded: few, fully utilized
```

## Key Takeaway

The shuffle does NOT change data values or separate mantissa/exponent fields.
It rearranges whole fp4x2 bytes so that each 16x16 sub-tile occupies a
contiguous 256-byte block in memory, enabling coalesced loads by MFMA wavefronts.

```
Before: [...row 0 full...][...row 1 full...][...row 2 full...]
After:  [...sub-tile 0: all 16 rows, cols 0-15...][...sub-tile 1: all 16 rows, cols 16-31...]
```
