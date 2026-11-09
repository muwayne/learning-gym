# Suggested replacement PR1 description (not published)

## Summary

- Establish a TD3 teaching home with problem/mechanism-first narrative, complete training pseudocode, adjacent numbered block walkthroughs, DPG/DDPG/TD3 comparison and inference.
- Define the slider environment, reward, return and one-step TD target before using the numeric trace; distinguish historical and proposed actions.
- Include the A1 autograd companion for shared parameters, mean/min/clip and graph-routing questions.
- Add separated forward/backward/SGD-step visualization and a real PyTorch reference with graph-aware snapshot restore and comparison forks.
- Record teaching-feedback coverage in `PR1-teaching-feedback-audit.md`, implementation boundaries and exact fixtures in `PR1-consensus-review.md`.

## Verification

- Nine reference tests: exact same-environment restore/re-run at every phase, back/next identity, gradient boundaries, finite differences, per-sample shared-gradient sum, terminal-copy isolation, deep snapshots/config validation and JS/PyTorch agreement.
- Default target 0.05882929126052877, synced Q2 target bias 0.03188794513945641; full configuration documented.
- Local Chrome: all 18 phase buttons and code highlights, backward stationary action, step movement, previous navigation, mobile viewport, no page exceptions.
- File-link / pseudocode checks and `git diff --check`.

## Review and evidence boundaries

The earlier independent simulated teaching trial is recorded as such; it is not attributed to the external group bots. Subsequent group agreement concerns the revised design, not approval of this new code. No fresh reviewer or human-learning-effect approval is claimed.

The HTML is an offline analytic trace, not yet a frontend for PyTorch restore/config controls. The Python reference is a fixed-batch teaching/audit implementation, not a production trainer or learner-authored exercise. No real-task training, learner Pass edits, GPU determinism or Adam support. No merge implied.
