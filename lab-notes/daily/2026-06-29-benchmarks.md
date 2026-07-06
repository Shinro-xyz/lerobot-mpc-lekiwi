---
title: "2026-06-29 (Late Late Evening) — JIT + Numba Benchmarks"
date: 2026-06-29
tags:
  - lekiwi
  - mpc
  - journal
  - daily
  - benchmarks
  - jit
  - numba
category: lekiwi
status: live
---

## 2026-06-29 (Late Late Evening) — JIT + Numba Benchmarks

**Context:** Wanted to see if Python 3.14.3's experimental JIT or Numba could speed up the simulation loop.

### Setup

| Config | Python | Env |
|--------|--------|-----|
| 3.12.3 plain | 3.12.3 | lebrobot |
| 3.12.3 + Numba | 3.12.3 + `numba` | lebrobot |
| 3.14.3 JIT | 3.14.3 `--enable-experimental-jit` | lrtest |

All deps pinned to identical versions (numpy 2.2.6, scipy 1.17.1, mujoco 3.9.0, etc.).

### JIT Benchmark Results

| Test | 3.12.3 | 3.14.3 JIT | Speedup |
|------|--------|------------|---------|
| Chaos test suite (143 tests) | 32.0s | 32.8s | 0.97x |
| Arm demo (--gif --fast) | 22.2s | 25.0s | 0.88x |
| Base demo (--gif --fast) | 1m11s | 1m10s | 1.02x |
| Pure compute (matmul + mujoco) | 1.4s | 1.6s | 0.87x |
| **Arm IK (1000 calls)** | **4.7s** | **3.5s** | **1.36x 🏆** |

### Numba Benchmark Results

| Test | 3.12.3 plain | 3.12.3+Numba | 3.14.3 JIT | Winner |
|------|-------------|--------------|------------|--------|
| DLS kernel (10000 calls) | 134.5ms | **13.4ms** | 80.6ms | **Numba 10x 🏆** |
| Full mujoco_ik (1000 calls) | 3494.3ms | **3114.0ms** | 3843.0ms | **Numba 1.12x** |
| numpy matmul (100x 500x500) | 220.6ms | — | 501.9ms | 3.12.3 plain 2.3x |
| numpy solve (50x 500x500) | 197.0ms | — | **111.1ms** | **3.14 JIT 1.77x** |
| numpy SVD (20x 500x500) | 3000.0ms | — | **835.2ms** | **3.14 JIT 3.6x 🏆** |

### Profiling: Where does the time go?

**Per-call breakdown of `mujoco_ik()`:**
| Operation | Time | % of total |
|-----------|------|-----------|
| `mj_jac` (Jacobian) | 9.2μs | 3% |
| DLS solve (numpy) | 32.7μs | 9% |
| **`mj_forward` (FK update)** | **317.7μs** | **88%** |

**Full simulation loop:**
| Layer | Wall-clock per step | vs Real-time (20ms) |
|-------|-------------------|-------------------|
| Bare physics (`engine.step()`) | 0.23ms | **85x faster** |
| Full loop (IK + step) | 0.52ms | **38x faster** |
| Demo with plot + GIF | 73-79ms | **3.7x slower** 😭 |

### Key Takeaways

1. **The simulation is already 85x faster than real-time.** The bottleneck is matplotlib rendering and GIF capture, not the physics or IK.
2. **The entire stack is C under the hood** — MuJoCo, numpy, scipy, osqp are all C libraries. Python is just glue code. JIT compilers solve a problem that doesn't exist here.
3. **Numba crushes the DLS kernel (10x)** but only helps 12% on the full IK because `mj_forward` (C, not JIT-able) dominates at 88% of runtime.
4. **3.14 JIT is inconsistent** — faster on SVD (3.6x) and solve (1.77x) but slower on matmul (0.48x) due to ABI compatibility shims.
5. **IK iteration sweet spot is 5** — error drops to 0.0007m (sub-millimeter) and going to 20 iterations doesn't improve accuracy, just costs time.

**Verdict:** Not worth switching to 3.14 JIT or adding Numba for this project. The gains are marginal (12% best case) and the bottleneck is matplotlib, not Python bytecode.
