# TD3 第一篇 · 导师问答与后续学习思路

> 配套[学员讲义](README.md)。本稿含导师答案，授课时只展示当前场景和题面。后续深度可按学员进展选择。现场反馈见[教学复盘](../../../retro/td3-first-lesson-feedback.md#follow-up-on-live-tutoring)。

## 开场与讲解

先给一句目的：今天看懂一次 TD3 update，分清历史动作、训练 label 和 Actor 的梯度。无需完整 foundations、MDP 推导或 P0 测试。

先讲连续动作挑战和 DDPG → TD3 的改动，再展示完整训练流程。用 replay 表和相邻的编号伪代码走读一次 update，Debugger 可用于查看具体数值和梯度。先定义变量，再解释它在哪一步被读取或写回，保留 loss、no_grad、backward/step。

基础只补到解决当前阻塞。答完疑问明确返回当前代码行，暂不需要的理论记为可选后续，不自动延长课程。需要图时只画当前的数据或梯度路径，不附加一份概念大全。

## 问答 · 沿同一个 replay batch 集中作答

先说明本组要检查的能力，例如读出一次 update 的输入与参数写回。每题标出正在执行的 block、已知数据、改动项和回答形式。把几道关联的问题一次给齐，允许学员简答或标记不确定。先反馈答错、答漏和不确定的地方，再选一个贴近卡点的变式。若上下文缺失，重给场景和题面。

术语统一为 **task goal**（环境任务要达到的条件）、**TD target**（Critic 这次拟合的数值，batch 中记作 `Y`）和 **target network**（经同步更新、参与计算 TD target 的参数副本）。

### 可直接展示的题组

滑块的 state 是 `[位置, 速度]`，action 是推力，task goal 是位置达到 `0.95`。环境此前产生了下表中的两条经验。当前从 replay 取出这批数据，开始第 2 次 Critic update，`policy_delay = 2`。

| 经验 | `S` | 实际执行的 `A` | `S_next` | reward `R` | `terminated` |
| --- | --- | --- | --- | --- | --- |
| ① 还在路上 | `[0.3, 0]` | `0.6` | `[0.6, 0.3]` | `−0.1` | `False` |
| ② 到达终点 | `[0.6, 0.3]` | `0.5` | `[1.0, 0.4]` | `+1` | `True` |

1. **算 TD target。** 对经验①，`Actor_target(S_next)` 已产生下一步 action 并完成 smoothing，两个 target Critic 对同一 `(S_next, A_next)` 的估值分别约为 `0.32` 和 `0.18`，`γ = 0.9`。对经验②，环境已终止。分别写出两条经验的 TD target 算式。只看进入最终数值的项。
2. **更新 online Critics。** TD target 已算完。以经验①为例，写出 `Q1` 回归用的 `(S, A)`，指出用哪个数算预测误差，并说明 `Q2` 是否使用同一个 TD target。
3. **更新 Actor 并同步参数。** 两个 online Critic 已完成本次 optimizer step。现在 `Actor` 对经验①的原状态 `S` 提议 `0.2`。写出 `Q1` 给它评分时用的 state/action，再把 online `Actor`、`Actor_target`、`Q1_target`、`Q2_target` 分配到 optimizer step 或用 `τ` 插值的步骤。

### 导师答案与按需追问

1. 用题中的舍入值，经验①为 `−0.1 + 0.9 × 0.18 = 0.062`，经验②为 `+1`。batch forward 可以先算出终止行的估值，terminal mask 使它对该行最终数值的贡献为零。
2. 两个 online Critics 都读 `([0.3, 0], 0.6)`，各自的预测和同一行 TD target `0.062` 比较。若问整批 optimizer step 后某行预测的方向，要交代 batch 内的样本共享 Critic 参数。
3. Actor loss 使用 `Q1([0.3, 0], 0.2)`；固定 `Q1` 参数，保留对 action 的梯度。如果误写 `Q1(S, Actor(S_next))`，请学员指出是哪一步把两个 state 混在了一起。第 2 次 Critic update 后先用 optimizer 更新 online Actor，再用 `τ` 把三个 target networks 分别朝对应 online 参数插值。之后每隔 `policy_delay` 次 Critic update 重复。

比较机制时，先说清观察范围。若固定经验①和 `A_next`，只把 `min(Q1_target, Q2_target)` 改成取 `Q1_target`，就问这条经验在 optimizer step 前的 TD target 如何改变。讨论控制效果时另设有训练与独立评测的对照；两者各自记录。

### 把流程重新组织起来

学员可查公式与 API，用几行中文或伪代码串起 replay → TD target → Critic update → delayed Actor update → target sync，说明每步读什么、写谁的参数。若用骨架、提示、重述题面或事后纠正，记录提供了什么帮助。学员可在完成本次走读后暂停，日后按需要用新变式校准或进入实现。

## 从理解进入工程能力：后续四个自然停止点

1. **实现核心**：AI 搭固定 batch/harness，学员写 target、Critic/Actor loss 与 delay/Polyak。对拍 noise/action clipping、terminal target、梯度隔离及计数顺序；不要求从零写环境和日志。
2. **读真实 update**：锁定公开实现 revision/config，追踪一条 transition 到 loss/optimizer，能解释 shape、动作身份和一处实现差异；可行时同 batch 对拍。只学完 TD3 即可读相关路径，不等 PPO/SAC 全部完成。
3. **接上控制任务并诊断**：先校准参考实现预算，再运行学员实现。出现低 return 时，同时查看 action saturation、Q 与回报估计、更新计数和参考对照；不预设低 return 就是代码错误。训练预算、seeds、评测与停止条件按 P4/P2 准备规则另定，不在讲义中凭空保证成绩。
4. **小改进**：从已有证据提出一个局部解释，先预测、只改一个主要变量、比较效果和成本、检查回归。可选先在固定 batch 把 target min 换为 Q1：若 Q1 已是较小值，两者相同；选 Q1 较大的样本才能看到差别。不能由此断言真实训练 return 必然提高或下降。

按 [Pass completion gates](../../../learning-modes.md#pass-completion-gates) 区分问答、独立复原和实际实现的证据。记录学员作答与获得的帮助，验收实现时看学员自己的 runnable artifact 和对照结果。

## 后续学习顺序

按学员目标选择后续课次。可带着已经走读的 TD3 update 去读 PPO 的 rollout、固定 advantage 和 log-prob ratio，或读 SAC 的 stochastic policy、entropy 与 soft TD target。回到 TD3 做工程练习时，从上面的自然停止点选一项。

## PR #1 原始版本的编辑与审阅记录

本稿根据群内明确反馈整理。独立 BOT 模拟试教已完成，过程与修改见下；Orion/Atlas 的群内审阅尚未收到，不能把本次代理审阅冒称为两者批准。验证不包括训练、真实实现对拍或 Wayne 的学习效果。

本轮本地检查：固定数值与 Actor 输出梯度有限差分核对通过；本地文件链接、代码围栏与 `git diff --check` 通过。离线 HTML 构建已尝试，因本机缺少 pandoc 失败；未安装依赖，先交付 Markdown。未检查外链可达性，未运行训练。

## Independent simulated teaching review · 2026-11-08

Participants: Collie (editor), native `learner_trial` (simulated learner), native `tutor_trial` (tutor / technical reviewer). These are not Orion or Atlas. The questions and responses below summarize an actual exchange between the two independent agents; they do not impersonate the human learner.

1. **Learner:** How can the Critic evaluate an action never executed? Could the Actor exploit it? **Tutor:** Function approximation permits a prediction, not a reliability guarantee. Actor optimization can exploit estimation errors; TD3 mitigates rather than eliminates this. Return to `loss_a`: it increases the current estimate, not necessarily true return.
2. **Learner:** With random networks and generated labels, where does learning begin? **Tutor:** Observed rewards and actual transitions anchor the signal. The future term remains an estimate. The label is fixed only for the current BP and is recomputed in subsequent updates.

The simulated learner reported that both answers addressed the questions. Both reviewers recommended Go conditional on adding these short explanations, retaining explicit target formulas, and keeping the first-read focus on `y → loss → BP`. A proposed opaque noisy-action helper was rejected during the exchange and not adopted. The editor applied the agreed changes in §2, including an optional first-read hint without removing the exact smoothing calculation.

Routing review also found that the old README entry still started at P0. README and curriculum now explicitly offer TD3-first, preserve the full revision route and course Exit Gates, and scope the local code-reading prerequisite to TD3's own core and fixed-batch verification.

Outcome: both independent reviewers agreed to publish a material PR with these revisions; no unresolved blocking finding. This is editorial and simulated teaching evidence only, not evidence of learner mastery or successful training.

## Current review candidate · 2026-11-09

The earlier review/build paragraphs above describe the original November 8 version, not the current readiness state. The group subsequently agreed on the revised SGD-first design; that agreement does not constitute review approval of the new implementation. See the [teaching-feedback audit](PR1-teaching-feedback-audit.md) and [executable candidate](PR1-consensus-review.md).

Before teaching from the candidate, follow the forward → backward → step split. Ask the learner to predict whether the action point or a parameter will move, then show the corresponding phase. At Actor backward, distinguish Q1's nonzero action derivative from absent Q1 parameter gradients; only the later Actor step moves the policy action. In the Critic example, sum the two samples' contributions to one shared weight instead of treating each output as an independently adjustable table entry.

The runnable reference contains complete AI-written update code for reviewer audit. Keep it separate from the learner task: the learner should still write the designated target/loss/delay/Polyak core before comparing against the reference. Browser navigation selects analytic trace frames; actual framework snapshot restoration is demonstrated by the Python reference, not by browser frame selection.
