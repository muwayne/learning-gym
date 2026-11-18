# PR1 consensus delta — executable review candidate

## Status and scope

Prepared against PR #1 head `5aa289b0562c831951c3b288bb1a72f5d9839e66`.
This is a **review candidate**, not a new review approval. Publication to the existing PR was explicitly authorized by Wayne after local validation.
Wayne requested code implementing the agreed direction for reviewers to inspect.
No other agent was contacted. No learner records, training permission, or course structure changed.

## PRD

Users: a learner stepping through TD3 and a reviewer auditing gradient/state boundaries.

- Given a fixed replay batch, stepping forward shows target → Critic forward → backward → step. Backward creates gradients, not parameter updates.
- Given Actor backward, show the Q curve at fixed state, local action slope, action-loss gradient and Actor parameter gradients. Q1 parameter gradients are absent. The action point remains fixed until Actor step.
- Given a paused executable reference, snapshot → continue and snapshot → restore → continue produce bitwise-identical state in the same CPU/PyTorch environment.
- Given a terminal intervention, run on a separate batch copy and leave original replay intact.
- Given changed hyperparameters, create an explicit comparison fork from initialization, invalidating all derived state rather than silently mixing configurations.

Output contract: the HTML remains standalone/offline, with no package downloads or backend calls. The Python reference exposes `next`, `back`, `snapshot`, `restore`, `fork`, `run`. Its default CLI prints measured gradients as JSON. It is reference code supplied by AI, **not a learner-authored solution or learner Pass evidence**.

## Implementation map

- `training-debugger.html`: 18 computed display frames, separated Critic/Actor forward and backward, per-row contributions to one shared Critic action weight, Q1 action curve and tangent, stationary action during backward. SGD stays the only optimizer.
- `reference/debugger.py`: real PyTorch autograd and SGD; CPU/float64; two Critic updates; delayed Actor and target sync; explicit phase cursor; deep in-process snapshots, copied `.grad`, optimizer state/configuration, target parameters, counters, replay, selected batch, fixed base noise and purpose-separated generator states.
- `reference/test_debugger.py`: phase-by-phase restore, back/re-run, deep-copy isolation, frozen-Q gradient boundary, finite difference, shared-gradient sum, terminal copy isolation, config validation and JS/PyTorch numeric agreement.
- `reference/check_browser.cjs`: dependency-free Node numeric checks against the HTML's actual inline engine.

### Restore semantics

If next phase is `critic_step` or `actor_step`, restore saved parameter gradients and step directly; no graph reconstruction. If next phase is a backward phase, lazily reconstruct forward using the **same** model, batch, target and base noise. This does not advance any RNG. `requires_grad` flags are restored too. Snapshots exclude live autograd graphs and navigation history; `back()` owns the separate history stack. Restore is intended for trusted objects within this process, not external pickle files.

### Parameter-change invalidation boundary

The reference deliberately uses one conservative, auditable rule: all config edits fork from initialization. There is no selective cache reuse.

| Change | Mathematical dependency | Implemented invalidation / comparison boundary |
| --- | --- | --- |
| gamma | Y, Critic loss/gradients, later Actor/targets | Fresh fork; keep original replay and base z |
| noise_std / noise_clip | smoothed target action, Y and downstream | Fresh fork; re-scale/re-clip **same z**, no resample |
| critic_lr | Critic step and downstream Actor/targets | Fresh fork, recompute earlier phases too |
| actor_lr | Actor step and targets | Fresh fork, recompute earlier phases too |
| tau | target sync and later labels | Fresh fork, recompute earlier phases too |
| delay | Actor/sync scheduling | Fresh fork, reset Critic update counter |
| terminal_override | Y and downstream | Fresh fork with changed **batch copy only** |
| model architecture, action bounds, replay collection | outside this bounded fixture | Not exposed as changeable settings |

The base fixture uses z=[0.3, -0.3], noise_std=1, clip=0.2. These are **preselected teaching draws**, not a claim of random sampling in this run. Generators are separated by replay/exploration/target purpose and snapshotted, but unused in this fixed trace. A general replay sampler and interactive collection remain future work.

### Golden fixture (now verified)

Initial Actor W=[0.2,-0.1], b=0.1; Q1 W=[0.1,-0.05,0.6,-0.4], b=0.1; Q2 W=[0.08,0.02,0.4,-0.35], b=0.02. Targets copy online parameters. Actor=tanh(affine(p,v)); Critic=affine(p,v,a,a²). The README defines both transitions.

gamma=0.9, tau=0.1, critic_lr=0.1, actor_lr=0.12, delay=2. SGD has no momentum, dampening or weight decay; its state dictionary is empty but still saved through the generic optimizer interface.

- Initial Y[0]: 0.05882929126052877 (PyTorch float64).
- Q2 target bias after sync: 0.03188794513945641.
- Actor at s=[0.3,0]: 0.1586485042974989 → 0.22074740221392233.
- Actor backward dQ1/da at first state: 0.4997707434824863.
- Actor backward dloss/da at first state: -0.24988537174124315 (B=2).

JS/PyTorch comparison uses absolute tolerance 1e-14, not cross-runtime bitwise equality. Bitwise identity is tested only within the same PyTorch CPU execution environment.

## Validation

From repository root, with Python 3.12 + PyTorch 2.10.0 and Node installed:

```sh
python modules/t3-sim-to-rl/td3/reference/debugger.py
python -m unittest discover -s modules/t3-sim-to-rl/td3/reference -v
node modules/t3-sim-to-rl/td3/reference/check_browser.cjs
git diff --check
```

The developer used the existing `ego` environment (no dependencies installed). Chrome headless checked all 18 phase buttons, code highlights, action-point invariance at backward, movement at step, previous navigation, mobile viewport and absence of JS exceptions. This is local browser verification, **not Feishu app rendering verification**.

## Review boundaries / remaining work

- The HTML still computes an analytic trace eagerly. Navigation selects computed frames; it is **not connected to the PyTorch executor** and must not be described as restoring a framework graph. Browser parity tests ensure matching default final parameters, not a shared runtime.
- Config forks, real state restore and terminal overrides are Python APIs, not yet browser controls. This candidate supplies executable implementation for review; it does not claim the complete interactive product is finished.
- Python begins with the documented fixed replay already prepared. No online environment collection, generalized replay sampling, disk checkpoint format, long-run convergence, GPU determinism, or Adam implementation.
- Confirm whether conservative full-restart invalidation is the desired first UI behavior before adding selective recomputation. A more granular cache would require new tests.
- Reviewers should focus on snapshot completeness, no RNG advancement on graph reconstruction, loss reduction factors, frozen-Q action gradient and the teaching/production boundary. No fresh reviewer approval is claimed.
