---
title: "2026-06-29 (Late Evening) — DRY Refactor + Chaos Tests + Detached HEAD"
date: 2026-06-29
tags:
  - lekiwi
  - mpc
  - journal
  - daily
  - refactor
  - tests
  - git
category: lekiwi
status: live
---

## 2026-06-29 (Late Evening) — DRY Refactor + Chaos Tests + Detached HEAD

**Commits pushed:** `6f3a094..668ec0e` (9 orphaned commits rescued from detached HEAD)

### DRY Refactor of `armrobot.py`

Eliminated 5 duplication sites:

| What | Before | After |
|---|---|---|
| `self.state = np.zeros(6)` | Set twice in `__init__` | Once |
| `get_state()` | Two identical `return self.state.copy()` branches | One line |
| EE position `xpos[ee_body_id][0..2]` | Repeated in `physics_engine()` and `step()` | `_get_ee_pos()` helper |
| Jacobian alloc + `mj_jac` + column slice | Repeated in `step()` and `mujoco_ik()` | `_get_ee_jacobian()` helper |
| `arm_jac_start` logic | Repeated in `step()` and `mujoco_ik()` | Cached in `physics_engine()` |

**`step()` now delegates to `mujoco_ik()`** — integrates `u[:3] * dt` to a target EE position, then calls `mujoco_ik()` for damped least-squares convergence. Both paths converge to the same IK solver.

### Chaos Tests

**143/143 passed** — threw every degenerate input at all concrete implementations:

- **ArmRobot (43 tests):** NaN, Inf, wrong shapes, empty arrays, complex numbers, strings, negative dt, zero dt, 1000-step stress tests, sawtooth patterns
- **HolonomicMobileRobot (25 tests):** Same battery — all caught `__init__` signature mismatch (missing `radius_robots`, `gamma`, `radius_wheels` params)
- **PIDController (23 tests):** NaN targets, zero/negative/huge gains, scalar vs vector, saturation with unequal limits, only-min/only-max clamping
- **MPC_LTI (21 tests):** NaN/Inf states, horizon=0, horizon=100, singular A, zero B, zero Q, zero R, mismatched dims, no constraints
- **MPC_LTI_DeltaU (8 tests):** NaN/Inf, no u_prev, wrong u_prev shape, zero/huge delta penalty
- **LuenbergerObserver (23 tests):** NaN/Inf, zero/negative gain, singular A, zero C, partial observation, 2D/6D systems
- **Cross-component (5 tests):** Arm+PID loop, Base+PID loop, Base+MPC loop, Base+Observer+MPC loop, Arm+Observer+PID loop

### Detached HEAD Bug

The lerobot-mpc-lekiwi repo was in detached HEAD state — the auto-commit script (`ai_commit.py`, runs every 30 min) was committing locally but `git push` silently failed because there was no branch to push from. **9 commits** were orphaned between `6f3a094` and `2af827e`.

**Root cause:** A `git checkout <hash>` during the session left the repo in detached HEAD. The auto-commit script never checked what branch it was on.

**Fix:** Patched `ai_commit.py` to detect detached HEAD and checkout `main` before committing:
```python
branch = run_git_command("git rev-parse --abbrev-ref HEAD", cwd=repo_path)
if not branch or branch == "HEAD":
    run_git_command("git checkout main", cwd=repo_path)
```

### Tracking Error Bug

`get_state()` was returning `self.state.copy()` — a cache only updated inside `step()`. Since the demo calls `mujoco_ik()` directly (not `step()`), the cache was frozen at the home position. The plot showed ~0.04m+ error when the arm was actually tracking at ~0.0004m.

**Fix:** `get_state()` now reads directly from MuJoCo's `xpos` when an engine is attached.

### Future Idea: Simulation Loop Abstraction

The simulation loop boilerplate is duplicated across `capture_gif.py` and `demo_base_movement.py`:

```
setup engine + plants + controllers + plot
for step in range(total_steps):
    compute control
    step physics
    log data
    update plot
    capture frame
teardown
```

Could be abstracted into a `SimulationRunner` class where demos just configure controllers, loggers, and plot layout — ~50 lines instead of 300+. Not urgent, but the pattern is ripe for extraction when a third demo appears.
