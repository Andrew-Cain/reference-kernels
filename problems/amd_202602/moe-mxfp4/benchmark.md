# moe-mxfp4 Reference Benchmark (aiter fused_moe on MI355X)

All cases: d_hidden=7168, n_shared_experts=1, n_experts_per_token=8

| Batch Size | d_expert | n_routed_experts | Mean | Best | Worst |
|-----------|----------|-----------------|------|------|-------|
| 16 | 256 | 256 | 139 µs | 135 µs | 144 µs |
| 128 | 256 | 256 | 218 µs | 211 µs | 229 µs |
| 512 | 256 | 256 | 250 µs | 245 µs | 258 µs |
| 16 | 512 | 32 | 94.7 µs | 90.8 µs | 100 µs |
| 128 | 512 | 32 | 129 µs | 126 µs | 138 µs |
| 512 | 512 | 32 | 214 µs | 212 µs | 218 µs |
| 512 | 2048 | 32 | 353 µs | 345 µs | 363 µs |
