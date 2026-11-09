# TD3 第一篇 · 导师问答与后续学习思路

> 配套[学员短讲义](td3-first-lesson.md)。本稿含答案；不要先把它整篇发给学员。目标是理解、实现、诊断与有证据的小改进，不要求发明算法或证明收敛。此稿是待试用设计，不是学习完成记录。

## 开场与讲解

先给一句目的：今天看懂一次 TD3 update，分清历史动作、训练 label 和 Actor 的梯度。无需完整 foundations、MDP 推导或 P0 测试。

先展示整条短流程，再读两个 update 代码块。变量定义在使用前；中文注释解释用途，保留 loss、no_grad、backward/step。不要展开环境 API、框架配置或网络层定义；也不要用不透明的 update() 代替核心学习步骤。

基础只补到解决当前阻塞。答完疑问明确返回当前代码行，暂不需要的理论记为可选后续，不自动延长课程。需要图时只画当前的数据或梯度路径，不附加一份概念大全。

## 问答：一次只出一题

### 第一轮 · 已展示例子的单变量变式

题：只将讲义 fixture 的 terminal 从 0 改成 1，y 变成多少？为什么？

导师答案：1；真正终止后不接未来估值。无需同时追加 truncation、n-step 或 GAE 问题。先仅反馈一致／不一致；学员请求解释时再指出第一处分歧。

### 第二轮 · 动作身份

题：保留那条 replay 经验，但当前 Actor 已输出不同动作。Critic loss 中的 a 要换成新动作吗？

导师答案：不换，已观察到的奖励和后继属于实际动作。Actor loss 另用当前 Actor 动作。若需要提示“再看奖励属于谁”，记录该提示，不将答案算作独立完成。

### 第三轮 · 梯度路径

先回看讲义 Actor 例子。题：如果在 `Q1(s, actor(s))` 整体外加 no_grad，Actor 还能靠这个 loss 更新吗？

导师答案：不能；计算图被截断。固定 Q1 参数不等于切断 Q1 对动作的梯度。可画一条 `Actor parameters → action → Q score → loss` 的图，只解释这条路径。

### 第四轮 · 把流程重新组织起来

收起完整 update，保留变量说明、公式和 API 查询。请学员组织：构造 target → Critics BP → 延迟 Actor BP → 同步 targets，并解释一次动作身份与梯度边界。

允许查公式，不考默写。只填模板不够；导师给出下一步、变量或位置均记提示。无需一次做完四轮；疲劳时在当前理解点停下，下一次用新变式接续。

## 从理解进入工程能力：后续四个自然停止点

1. **实现核心**：AI 搭固定 batch/harness，学员写 target、Critic/Actor loss 与 delay/Polyak。对拍 noise/action clipping、terminal target、梯度隔离及计数顺序；不要求从零写环境和日志。
2. **读真实 update**：锁定公开实现 revision/config，追踪一条 transition 到 loss/optimizer，能解释 shape、动作身份和一处实现差异；可行时同 batch 对拍。只学完 TD3 即可读相关路径，不等 PPO/SAC 全部完成。
3. **接上控制任务并诊断**：先校准参考实现预算，再运行学员实现。出现低 return 时，同时查看 action saturation、Q 与回报估计、更新计数和参考对照；不预设低 return 就是代码错误。训练预算、seeds、评测与停止条件按 P4/P2 准备规则另定，不在讲义中凭空保证成绩。
4. **小改进**：从已有证据提出一个局部解释，先预测、只改一个主要变量、比较效果和成本、检查回归。可选先在固定 batch 把 target min 换为 Q1：本篇给定 Q1=3、Q2=5 时两者相同；交换两者后才出现差别。不能由此断言真实训练 return 必然提高或下降。

前三个问答只建立基础，不自动达到 Implement/Improve。工程练习分次完成，沿用现有 Pass/record/ledger；不新增 CSV 字段，不改历史记录。用 `[独立]` / `[提示: 原话]` 标记 session 过程；是否写入 CSV 备注仍未决定。随后某课的新变式独立完成才作为掌握证据，刚获提示后原题重做不算。

## 延伸顺序是建议，不是新门槛

先用本篇试读确认负担，再完成 TD3 的首个可运行核心。随后可选择 PPO 对照其 rollout、固定 advantage 和 log-prob ratio，或 SAC 对照随机策略、entropy 与 soft target。后续顺序由实际理解与任务需要调整，不为了凑算法清单同时学习三个。

## 编辑与审阅记录

本稿根据群内明确反馈整理。独立 BOT 模拟试教已完成，过程与修改见下；Orion/Atlas 的群内审阅尚未收到，不能把本次代理审阅冒称为两者批准。验证不包括训练、真实实现对拍或 Wayne 的学习效果。

本轮本地检查：固定数值与 Actor 输出梯度有限差分核对通过；本地文件链接、代码围栏与 `git diff --check` 通过。离线 HTML 构建已尝试，因本机缺少 pandoc 失败；未安装依赖，先交付 Markdown。未检查外链可达性，未运行训练。

## Independent simulated teaching review · 2026-11-08

Participants: Collie (editor), native `learner_trial` (simulated learner), native `tutor_trial` (tutor / technical reviewer). These are not Orion or Atlas. The questions and responses below summarize an actual exchange between the two independent agents; they do not impersonate the human learner.

1. **Learner:** How can the Critic evaluate an action never executed? Could the Actor exploit it? **Tutor:** Function approximation permits a prediction, not a reliability guarantee. Actor optimization can exploit estimation errors; TD3 mitigates rather than eliminates this. Return to `loss_a`: it increases the current estimate, not necessarily true return.
2. **Learner:** With random networks and generated labels, where does learning begin? **Tutor:** Observed rewards and actual transitions anchor the signal. The future term remains an estimate. The label is fixed only for the current BP and is recomputed in subsequent updates.

The simulated learner reported that both answers addressed the questions. Both reviewers recommended Go conditional on adding these short explanations, retaining explicit target formulas, and keeping the first-read focus on `y → loss → BP`. A proposed opaque noisy-action helper was rejected during the exchange and not adopted. The editor applied the agreed changes in §2, including an optional first-read hint without removing the exact smoothing calculation.

Routing review also found that the old README entry still started at P0. README and curriculum now explicitly offer TD3-first, preserve the full revision route and course Exit Gates, and scope the local code-reading prerequisite to TD3's own core and fixed-batch verification.

Outcome: both independent reviewers agreed to publish a material PR with these revisions; no unresolved blocking finding. This is editorial and simulated teaching evidence only, not evidence of learner mastery or successful training.
