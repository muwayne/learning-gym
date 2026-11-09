# TD3-first learner feedback · 2026-11-09

After reading the [original PR #1 lesson](https://github.com/muwayne/learning-gym/blob/364aaeccd95142061ad5c94969ff76f346ed20f1/modules/t3-sim-to-rl/td3-first-lesson.md), Wayne said it gave him a reasonable initial explanation of TD3. The spoken example helped him form a specific question: if the Critic's weights are fixed, does `-Q1(s, actor(s)).mean()` still give the Actor a gradient?

| Learner feedback | Design change to try in future lessons |
| --- | --- |
| Start with a map, the challenge, and the algorithm's evolution. Then show the complete high-level pseudocode before diving into modules. | Open with the problem, the direct lineage, and what changes relative to the predecessor. Keep data collection, target construction, both optimizer steps, and target sync visible in one algorithm block. Move implementation details into named sections. |
| “慢速副本” was hard to understand, and “一次能算清的例子” did not say what it helped explain. | Keep searchable terms such as target network and Polyak averaging. Define a term by what changes it and when. Name examples by the question they answer, and show the expected observation before the arithmetic. |
| The main missing mental picture was how the Actor and Critic networks get trained step by step. | Use one small runnable trace from initial parameters through experience, target values, backward, optimizer updates, and a second iteration. Display actual inputs, outputs, tensor shapes and changed parameter values. |
| The handout omitted inference and the algorithm needs a stable home for its materials. | End each algorithm lesson with a deployment forward pass. Keep the handout, runnable visual, and tutor notes in one algorithm directory. |
| The mechanical-arm example helped surface the difference between freezing Critic weights and differentiating with respect to its input action. | Use one concrete state across Critic regression and Actor optimization. Ask the learner to point to the gradient path and the parameters that actually change. |

The updated [TD3 lesson](../modules/t3-sim-to-rl/td3/README.md) and [Training Debugger](../modules/t3-sim-to-rl/td3/training-debugger.html) implement this design for a fixed slider task. The learner's report describes the original handout and a question raised during live explanation. It is not evidence that the revised lesson has been tried, that the learner has passed, or that the toy demonstrates control performance.

## Follow-up on the revised lesson

Wayne asked what target networks store, whether a mean preserves gradients, and how to locate each debugger step in code. This points to three concrete design changes. Define the model copy, the replay records, and the regression target before introducing the update. Annotate each operation and show how a two-element mean sends a gradient of one half to each input, with the sign from the loss. Keep the current code lines next to their tensors, label the axes and feature order, and let readers open parameter details when they need them.

During the HTML walkthrough, Wayne lost the thread at “05 Target 1” and found the visual too thin. Begin that step with one actual replay transition on the rail. Trace the recorded reward and the estimated continuation separately, then join them at the number the online Critic will fit in the next step. Allow switching to a terminal transition so the learner can see the continuation term drop out. Keep smoothing details expandable until the two actions and their roles are clear.

## Review the lesson as a first-time reader

Wayne asked for one teaching principle. Explain the mechanism first, then use code to locate it and a worked example to resolve a specific difficulty in the execution. Keep the detailed trace in the debugger. Put the relevant code excerpt next to its explanation so the learner can follow it without scrolling back to the full loop.

After drafting, reread the lesson from the beginning as a first-time student. Check whether each new term has a purpose and a definition before it is used, each block answers a question the reader already has, and each example resolves that question. Revise places where the reader would have to guess a dependency or search another section for the relevant lines. Check code and numeric evidence separately from this reading pass.
