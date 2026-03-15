# Submission Guide

## 1. Install popcorn-cli

```bash
curl -fsSL https://raw.githubusercontent.com/gpu-mode/popcorn-cli/main/install.sh | bash
source ~/.zshrc
```

Verify: `popcorn-cli --version`

## 2. Register

```bash
popcorn-cli register discord   # or: popcorn-cli register github
```

Opens a browser for OAuth. Once complete, verify:

```bash
cat ~/.popcorn.yaml
# Should show: cli_id: <your-id>
```

If auth is broken:
```bash
popcorn-cli reregister discord
```

## 3. POPCORN Directives

Add these to the top of your `submission.py` so you can submit without extra flags:

**mxfp4-mm:**
```python
#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
```

**moe-mxfp4:**
```python
#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
```

**mixed-mla:**
```python
#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
```

## 4. Submit

Navigate to the problem directory, then run:

```bash
# Test correctness (free, no ranking impact)
popcorn-cli submit --no-tui --mode test submission.py

# Benchmark timing (free, no ranking impact)
popcorn-cli submit --no-tui --mode benchmark submission.py

# Official ranked submission
popcorn-cli submit --no-tui --mode leaderboard submission.py
```

Or with explicit flags (skips reading directives):
```bash
popcorn-cli submit --no-tui --gpu MI355X --leaderboard amd-mxfp4-mm --mode test submission.py
```

### Problem Directories

```
amd_gpu_mode_e2e_model_speedrun/reference-kernels/problems/amd_202602/
  mxfp4-mm/submission.py     ← simplest (MXFP4 GEMM)
  moe-mxfp4/submission.py    ← medium  (fused MoE)
  mixed-mla/submission.py    ← hardest (MLA decode)
```

### Submission Modes

| Mode | What it does | Affects ranking? |
|------|-------------|-----------------|
| `test` | Correctness check against reference | No |
| `benchmark` | Timing with L2 cache clearing | No |
| `leaderboard` | Correctness + timing, re-checks with different seeds | Yes |
| `profile` | Torch profiler trace (kernel-level timing breakdown) | No |

Always run `test` first to verify correctness before submitting to `leaderboard`.

## 5. Manage Submissions

```bash
popcorn-cli submissions list --leaderboard amd-mxfp4-mm
popcorn-cli submissions show <ID>
popcorn-cli submissions delete <ID>
```

## 6. Interpreting Results

- `check: pass` — correctness ok, safe to do leaderboard submission
- `check: fail` — fix your kernel before submitting to leaderboard
- `benchmark.N.mean` — mean kernel time in nanoseconds for benchmark case N
- `benchmark.N.best` — best observed time
- Ranking is by **geometric mean** of all benchmark case times

## 7. Troubleshooting

| Problem | Solution |
|---------|----------|
| `aiter` import errors locally | Normal — `aiter` is only on MI355X. Submit remotely to test. |
| `popcorn-cli: command not found` | Re-run install script or `source ~/.zshrc` |
| "already has valid account" on register | You're already registered — just submit |
| Auth broken | `popcorn-cli reregister discord` |
| Timeout | mxfp4-mm: 420s, moe-mxfp4: 540s, mixed-mla: 900s. Check for infinite loops. |
| "Leaderboard does not exist" | Check leaderboard name matches `amd_202602.yaml` (prefix with `amd-`) |
| "Device not configured" error | Use `--no-tui` flag — TUI doesn't work in non-interactive shells |
