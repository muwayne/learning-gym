# PR1 teaching-feedback coverage audit

## Scope and evidence

Audit of the teaching feedback recorded in `retro/td3-first-lesson-feedback.md`, the tutor guide and the group consensus supplied by Wayne. Checked against GitHub PR1 head `5aa289b` and the local candidate. No unseen chat or new reviewer approval is inferred.

**Important:** “In PR1” below means the existing GitHub head. “Candidate” means local changes that have not been pushed. The remote PR description still describes the initial short lesson and the old HTML-build failure; it does not accurately summarize the newer files. A replacement description is supplied locally in `PR1-description.md` for publication when authorized.

## Teaching feedback → concrete coverage

| Feedback / concern | Existing PR1 evidence | Candidate action / disposition |
| --- | --- | --- |
| Start with challenge, map and evolution; show whole algorithm before block detail | README §0–2 gives environment/mechanism/full pseudocode; detailed DPG/DDPG/TD3 comparison is at the end | Added a compact reading map and lineage near the opening; detailed comparison stays optional |
| Explain terms before use; examples must answer a question | README defines transition, reward, return, TD target and Y in order; block titles/explanations give purposes | Retained; first-read map explicitly makes weights/tensors optional |
| Explain environment/reward completely | README §0 gives start, motion, action bounds, reward and termination; debugger repeats these rules | Retained and checked against reference replay |
| Distinguish immediate reward, future estimate and fixed label | README §0 and TD-target block; debugger Target detail separates recorded motion/reward from estimated continuation | Retained; terminal override reference changes a batch copy, never historical replay |
| Show whole loop and adjacent code excerpts, consistent names/line numbers | README §2 and §3; debugger code highlighting | Corrected stale debugger step references after adding phases; now use stable phase names |
| Explain standard algorithm versus teaching choices/variants, cite original | README paper/Algorithm 1, implementation-differences table, TD3+BC and DPG/DDPG/TD3 comparison | Retained; full SGD fixture configuration and evidence limits in candidate note |
| Make network learning and shared parameters visible | Debugger already shows inputs, shapes, weights and before/after values, but Critic update combines phases | Split Critic forward/backward/step; show each sample's contribution to a single shared Q1 weight |
| Freeze Q weights without cutting Actor gradient | README Actor block, tutor question, A1 autograd companion | Added simultaneous local Q slope, action-loss gradient, Actor gradients, absent Q1 parameter gradients and curve; action remains fixed until step |
| Teach mean/min/clip and general autograd separately | A1 `autograd-primer.md/.html`, scalar engine and backward-path demo already in PR1 | Retained companion, linked from debugger; no need to duplicate the full autograd lesson inside TD3 |
| Explain target sync as parameter interpolation, not optimizer LR | README Target-sync block gives paired parameters, tau definition, bias example, next-target comparison | Numeric claims rechecked against JS and PyTorch; both agree within 1e-14 |
| Give a deployment endpoint and stable material home | `td3/README.md`, `training-debugger.html`, `tutor.md`; README §5 and final debugger frame | Retained; reference code stays under `td3/reference/` |
| Mechanism first, code locates it, example resolves difficulty | Recorded in teaching retro; README §1 mechanism before §2 complete code and §3 worked blocks | Added reading map; do not replace narrative with the review APIs or tests |
| Keep tutoring bounded and learner-owned | Tutor says one question at a time, fixed context, natural stopping points and hint marking | Added explicit reference-solution boundary: reviewer code is not learner implementation evidence |
| Avoid declaring toy results, AI agreement or prompted answers as mastery | README/tutor/retro already state evidence limits | Retained; no Pass, ledger or learner status edits |

## New execution consensus: implemented versus future UI

SGD-only, graph-aware restore, snapshot gradients/state, fixed base z, separated RNG state, config forks and terminal-copy experiments are implemented in the Python reference and tested. Browser phase/gradient visualization is implemented. **The HTML does not yet expose config-fork controls or call the PyTorch executor.** This is a code-bearing review candidate, not a claim of full interactive-product completion.

## Findings resolved during this audit

1. Opening material did not yet put the evolutionary map up front: added a short map, without moving the full comparison into the first-read path.
2. Splitting execution phases invalidated old debugger step references: replaced affected references with stable phase names.
3. Tutor review/build status could be mistaken for current status: added a dated clarification, preserving the historical record.
4. GitHub PR description is stale: prepared a replacement locally; no remote write or review-agent notification performed.

## Remaining verification limits

Checked recorded-feedback coverage, file presence, local links, pseudocode alignment, executable fixtures and local browser behavior. This is an editorial self-audit, not a fresh independent review or a human teaching trial. The prior HTML-build failure remains historical; no claim is made that all reading editions were regenerated here. Feishu app rendering and learning effectiveness remain unverified.
