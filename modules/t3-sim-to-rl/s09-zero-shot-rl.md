# T3·S09 · Zero-Shot RL：从 FB 到人形 BFM

> 选修专题 · 一个 Lesson 内的八个学习单元，不要求一次 session 学完。
> **先通读 [完整讲义 HTML](zero-shot-rl-primer.html)**（[Markdown 源稿](zero-shot-rl-primer.md)），再开始下列活动。讲义准备与学员完成分开记录。
> 2026-10-16 已整合用户的 [FB / BFM 学习笔记精编](../../references/fb-bfm-notes-digest.md)。笔记是 routing / claim-audit 资产，不自动登记 Pass；本章仍为八个单元。
> 通读正文以机制与例子为主，常见误读集中于 §10.2。§4.7 的 CPU / seed=0 / 12,000-update neural toy 是 AI 准备的教学参考；运行版本、代码修订和证据包身份见精编。本轮没有 learner experiment，不据参考运行追加 prediction ledger 或个人 progress。

跨课温习可先读 [PPO / TD3 / FB 与 DexLab](rl-control-review-primer.html)，再用 [P5 的 task-inference / coverage 对照](rl-control-practice.md)进入本章。复用下列八单元，不重复登记参考演示或 learner evidence。

## 开课地图与 routing

前置：T2·S01 的 MDP / Bellman / policy improvement；T3·S05 的 offline RL 与 data coverage。单元 8 可回查 T3·S03/S04 的 DR 与 privileged information。可从相应 RL 基础讲义知识点补齐，不要求先完成整条 PPO 线。

目标：解释 `successor measure → F/B/z → policy → reward / goal / motion inference`，并明确数据、有限秩和测试时优化的边界。QP 暂不进入。

| Pass | 产物 | 必须由学员拥有的部分 |
|---|---|---|
| Understand | 一个机制判读 + revised mental model | 用自己的话说明因果与证据边界 |
| Implement | 最小 SR / SF / FB 接口的 runnable verification + 一轮 P-T-I-E | occupancy TD / task encoding 的核心逻辑、时间合同、预测与干预意图 |
| Improve | failure hypothesis、区分性 control、held-out / regression evidence | 选择 falsifier、解释结果并限制结论 |

Wayne 沿用最终深度 Improve；本次先准备全章阅读材料，不据此登记任何 Pass。其他学员按自己的 baseline routing。AI 可搭 harness、录入学员口述与生成图表，不能代答核心判断；亲手实现、口述驾驶或填空抓错均可。

## Understand · 八个单元的逐步陪练入口

**一次只问一行。** 每次先重述该单元的场景和必要定义，再提一个问题；答对推进，卡点回查讲义，不机械重复数值计算。下列是陪练提示，不是一次性交给学员的考试。

| 单元 / 讲义 | 一个主问题 | 自然停止点 |
|---|---|---|
| 1 · §1 SR | 固定策略的未来访问表已学好，换奖励后可以直接改变什么，还不能据此声称什么？请用走廊目标从 4 换到 2 解释。 | 明确 evaluation 与 optimization 的差别 |
| 2 · §2 SF / GPI | 两个旧策略没有完整执行过“到中间停住”，为什么 action-level GPI 仍可能组合出这个行为？ | 说清第一步任意动作与逐步重选 |
| 3 · §3 FB | 某 actor 已经最大化 `Fᵀz`，但 F 表示另一个策略的 occupancy。为什么“对自己的 Q 贪心”的定理不能直接使用？ | 接上 self-consistency；再按卡点进入坐标、ρ 或投影 |
| 4 · §4 loss | replay 的 `s'` 与独立抽样的 `s+` 各承担什么角色？如果把二者永远设成同一个样本，训练的对象怎样改变？ | 解释正转移锚定与全终点递推，不要求求导 |
| 5 · §5 2023 | 在同一个 replay buffer 上，FB 得到 TD3 归一化 return 的 85%。它支持怎样的结论，还需什么才能谈 success rate？ | metric、分母与证据边界准确 |
| 6 · §6 扩展 | 两段示范含相同状态但顺序相反，整段 average 编码相同。要表达局部 tracking，推断接口需要怎样改变？ | 区分 occupancy、滚动窗口与 tracking 保证 |
| 7 · §7 Motivo | motion dataset 没有 action labels，FB-CPR 的 action-conditioned transition 从哪里获得？ | state-only 示范与 online replay 的来源分开 |
| 8 · §8–9 BFM-Zero / VLA | 网络冻结，但在模拟器中搜索更好的 z 后才部署。这条路径应怎样与直接 prompt 的结果分别报告？ | 区分 adaptation，再对照 task z / state token 接口 |

至少累计一条 AI claim audit：从讲义 §10.2 选一条，依据 [rubric](../../ai-review-rubric.md) 分别给 Conceptual Verdict 与 Evidence Status。论文证据可支持文献结论，不代替学员自己实现的验证。

## 笔记整合后的增量问题池

已理解前述主问题时，只选当前卡点对应的一题，不为整合笔记重走全部课程：

| 讲义位置 | 一个可选择的增量判读 |
|---|---|
| §2.3 GPI 时间线 | 为什么“第一步 stay，之后跟随 πR”得 1.5，而每一步重新 GPI 后实际得到 2？指出当前 reward、续行策略与重复决策的角色 |
| §3.6–3.7 坐标 / 分布 | B 未完全正交且 reward samples 换成另一分布时，为什么不能仍把简单平均直接当作普通回归系数？ |
| §3.8 投影失败 | 给定 B 的精确投影里，actor 已做到 r̂ 最优，原任务仍失败。它支持哪种归因，还不能说明什么？ |
| §4.6 batch | 拿同一条真实 transition 配不同 z，哪些部分是数据事实，哪些部分必须随 πz 重新计算？ |
| §6.1 FB-IL | 专家在两处按 60% / 40% 占用，为什么最大化固定 density-ratio reward 可能偏向一处？ |
| §6.1 / §8.2 推断 | BCFB 只在既有示范上优化 z，与在 simulator 中 CEM 搜索 z，额外数据与控制合同哪里不同？ |
| §9 IL / VLA | 有交互的 DAgger 为什么可以仍是监督式 imitation，而无 critic 的 REINFORCE 为什么仍是 RL？ |

问题都以讲义中已经解释的机制为前置。完整参考代码只作为审查材料，不据阅读参考实现补记 learner experiment。

## Implement · 在阅读与预测之后安排

从以下活动选一个能闭合当前卡点的最小验证，不提前生成整套参考答案。已有可核验的等价资产可替代重复实现。

- **SR TD 与直接求解对拍**：五格环境、current-state reward、无 terminal。学员实现一行 occupancy TD 或口述每个量的角色；AI 搭 deterministic transitions、seed、solver 和 Bellman residual。检查 occupancy 的行和、relabeling 与固定策略价值。
- **SF 与 task encoding 接口**：学员实现 feature aggregation / second-moment regression，明确 `ρ` 与 normalization；AI 提供矩阵求解。验证 feature span 内与 span 外两个独立任务，记录 reward reconstruction 与实际 policy return 两种指标。
- **FB self-consistency 接口**：先使用精确 occupancy 与显式 B 做代数验证，检查 `Fᵀz` 与 occupancy × reward 的一致性。随后若选择 neural FB，再由具体 session 冻结 data、dimension、更新预算和独立 verifier；不把显式基底求解说成 learned FB。

新增可选 control：固定给定 B，用 second-moment correction 与未经校正的编码做接口对拍；或者固定 actor / B，只改变 task-inference sampling distribution。开始前由学员选变量并锁 prediction，不照搬 §3.8 / §4.7 已经公开的参考结果。附录 neural trainer 已补零范数与 projection 修正，代码审查可围绕目标网络、时间索引和有限秩残差展开；参考运行不替代学员自己的实现与干预。

学员必须负责：时间索引、核心累积 / 编码逻辑、预测、单变量 intervention 和 Explain；不以 AI 一次性写完整实现作为 learner evidence。参考讲义的三个交互已展示基础场景，因此实践使用**新的 control / task**，不能把看到过的结果再记成事前预测。

## Improve · 先冻结 design，再锁 prediction

可选起点：某任务的 zero-shot return 低。分别考虑表示投影丢失任务信息、state-action coverage 不足、actor 未满足 greedy、reward inference 采样不足。选其中两个 competing hypotheses，设计只改变一个关键变量的区分性检查。

示例设计方向：固定 representation 与 policy，用同一组可寻址 reward states 改变 task-inference sample budget；独立 reward function 评估原任务 return，并保留一个已能完成任务的 regression control。**本文不预告这个 intervention 的结果**；实际开始前再展示完整环境、预算、falsifier 与 stopping rule，由学员锁定 prediction 后运行。

若要训练可引用 checkpoint 或运行真实人形模型，回根仓库 task / training authorization 流程；本练习单没有为此授予资源或部署许可。当前没有 learner experiment，也不追加 prediction ledger。

## 记录与重建

课次记录使用 [session-log 模板](../../templates/session-log.md)，保存讲到哪个单元、已作答与待验证的区别。只有实际达到 [learning-modes gate](../../learning-modes.md) 后才更新个人 progress。

讲义 HTML 的可复现生成命令（仓库根运行，写入阅读版）：

```sh
python3 tools/build_reading_edition.py modules/t3-sim-to-rl/zero-shot-rl-primer.md \
  --eyebrow "T3 · S09 · Zero-Shot RL 通读讲义" \
  --subtitle "用同一个走廊理解未来访问、任务编码与条件控制，再走向人形 BFM"
```

公式由 pandoc 生成为 MathML；图与交互位于 `reading-assets/zero-shot-rl-primer/`。先改 Markdown / assets 再重建，不手改生成的 HTML。
