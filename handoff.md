# Gym AI Handoff

> 本文件是没有原始聊天记录时的最小完整接力上下文：为什么存在 Gym v2、哪些决定已定、哪些仍开放。执行状态看 `learners/<name>/progress.md` 与 [progress.md](progress.md)（学员总览），正式项目事实只看 `tpc_manip` 权威来源。

## 1. 面向的学员与真实问题

Gym 服务于团队多名学员——长期工程与 ML 经验、概念层参差但不弱的研究者。**每位学员的具体背景、既有资产与节奏记录在 `learners/<name>/baseline.md`，AI 陪练前必读。**

课程立足于一个跨学员反复验证的模式：

```text
输入型学习（问答、推导、审计）——强，且有闭环
输出型闭环（亲手训到 work → 干预 → 拥有结果）——系统性缺失
```

**没有训练闭环锚定的概念，保质期很短**——具体例证见各学员的 baseline 与 records。

因此课程主题是：**给每个已有概念的系统补上训练闭环**，而不是输入新知识。

## 2. 已定决策（新 AI 不要重新发明）

- 双线结构：A 线机制（A1 Transformer → A2 生成式建模与动作头 → A3 Latent 融合），T 线训练（T0 训练闭环 → T1 训练阶段 → T2 RL 从零 → T3 Sim-to-RL，T4 训练工程随行）；共 45 节，含 3 节选修，既有编号保留；
- RL（T2+T3）17/45 ≈ 37.8%，保持在 1/3–40% 区间；新增或合并课次后仍须重算比例；
- 教程、自检、追问、实验与固化是可选 activity toolbox，不要求每个 Lesson 或 Pass 五段齐全；固定顺序只有“先给完整 context 再提问”和“先展示完整实验设计再锁 prediction”；
- **档案 ≠ 现役**：引用既有档案是 routing evidence，不自动等于现役能力；用最小 calibration 验证即可，已有现役 evidence 时不机械重复；
- 按 session 计数，不按日历周；
- 多人共用：共享材料走 PR 迭代，个人状态只进 `learners/<name>/`；先行者每过一课顺手修材料；
- **诉求按 session routing**：课表共同核心不按职业拆版本；每次按 [learning modes](learning-modes.md) 选择 `Understand（看懂）/ Implement（做出）/ Improve（改好）`。新学员每课询问，有稳定默认值者不重复询问；role 只提供案例背景，不决定深度；
- 一切载体在本目录内，不引入外部平台；
- 高级模型（WAM、Cosmos、GR00T 等）只学底层原理与接线方式（A3），不做深度专题；
- T3·S06 扩为 expressive-policy value maximization：EXPO（selection + bounded edit）与 DPPO / RECAP 并列为三条路线，EXPO 段把 base 当黑盒 sampler、不要求 A2；不改课次数（2026-10-04，源自 wayne 的 RLT / EXPO-FT online RL 复现工作）；
- RL 基础讲义（`modules/t3-sim-to-rl/rl-foundations-primer`）承担 T2·S01、T2·S07、T3·S02/S05/S06/S07 的 `Understand` 层；T3 这四课的 `Implement` / `Improve` 层可用 toy 或学员在根仓库已授权工作中的真实复现证据（同样要求 P-T-I-E）二选一；T2·S02–S06 on-policy 线不变；课次数与 RL 占比不变（2026-10-05，源自 wayne：论文机制以真实训练学习、课程补系统基础）；
- T3·S09 以 [Zero-Shot RL 完整讲义](modules/t3-sim-to-rl/zero-shot-rl-primer.html)先行，通读后按 [练习入口](modules/t3-sim-to-rl/s09-zero-shot-rl.md)逐单元提问、实践与答疑。先用 SR / SF 的走廊例子建立机制，再看 2021 FB、精读 2023，最后到 Meta Motivo / BFM-Zero；QP 暂不进入。三个离线交互是显式基底与精确求解的教学参考，不是 learned FB 或 learner experiment；本次仅准备材料（2026-10-13）；
- S09 于 2026-10-16 整合用户的 [FB / BFM 学习笔记](references/fb-bfm-notes-digest.md)：补 GPI 时间线、task coordinates / ρ、训练 batch、FB-IL 四类与 IL / VLA 接口，八单元与课次数不变；随后根据审阅将正文改为正面机制、纠错集中至 §10.2，并核对 BFM 术语出处。给定 B 的投影已复算，修订版 neural toy 完成 CPU / seed=0 / 12k 单次实测；过程记录与证据身份在精编，未登记 learner Pass；
- 2026-10-17 增 [机器人 RL 统一温习讲义](modules/t3-sim-to-rl/rl-control-review-primer.html)与 [六段实践](modules/t3-sim-to-rl/rl-control-practice.md)，连接新增 RL 基础 / Zero-Shot 专题与 DexLab HORA / AnyRotate。T3·S02 明确为 TD3 主线 / SAC 对照；T3·S03/S04 补 teacher PPO、history distillation、control / sensing / bank / evaluation。P1–P6 是跨课 routing，不新增 Lesson；本次只准备材料与验证阅读版，未启动 learner experiments、未补记 Pass。DexLab pin / scope 以 [digest](references/dexlab-sim-rl-digest.md)为准。On/off-policy 与 zero-shot 属于不同分类维度，FB task vector 与 HORA adaptation latent 不互换。
- 统一温习按 Wayne 的反馈以基础知识为主，DexLab 源码追踪降为可选；P2 开课前先验证参考配置和预算，低 return 与实现错误分开诊断；zero-shot / few-shot / adaptation 沿用 S09 定义。FB 的 z 搜索对照报告额外交互成本，搜索未改进仍保留搜索能力不足这一解释（2026-10-17）。
- 2026-11-06 重排 [RL 学习与实践](curriculum.md#rl-基础讲义与课次的对应t2t3-共用)：三种经典算法先重建伪代码与对拍，PPO、TD3 默认做最小闭环，SAC 复用 replay 练更新差异。P7 对照生成器、残差、候选与 support，再读真实代码；P8 作为 S10 可微动力学选修，独立验证梯度经过的时间轴。P0 是批量恢复题；阶段门槛与实施归 [课表](curriculum.md) 和 [分段实践](modules/t3-sim-to-rl/rl-control-practice.md)。
- RLT 作为 `frozen foundation policy → compact state interface → small online learner → guarded deployment handoff` 的系统案例；官方 paper/blog 是 mechanism 证据，π0.5 / RoboDojo 迁移是待验证 hypothesis。用户提供的工程 draft 只作 claim-audit case，不构成训练授权；
- A 线终点能力：对任何前沿架构画出 latent-interface 图（什么 latent 过界、谁训谁冻、损失落谁头上）；
- T 线终点产物：Action Expert selective-RL preregistered proposal（沿用 v1 的真实项目终点）；
- 课程本身不授权模型训练；可引用 checkpoint 必须回根仓库审批流程。

## 3. 固定研究循环

需要实验 evidence 的 `Implement / Improve` 活动遵循 P-T-I-E：先完整展示 design，再写带机制的 prediction 与反证条件，然后跑最小训练或等价验证，一次只改一个变量，最后对照结果修正 mental model。AI 承担脚手架与机械验证；人拥有核心判断、实现意图、prediction、intervention 与三档判定（Correct / Plausible-but-vague / Conceptually-wrong，判据见 [ai-review-rubric.md](ai-review-rubric.md)）。

## 4. 真实项目终点（保留自 v1）

> 在尽量保留 π0.5 general representation 的前提下，仅或主要更新 Action Expert，RL 是否能改善一个明确 RoboTwin task 的 closed-loop failure distribution，同时不造成不可接受的行为漂移？

进入真实 pilot 前必须闭合的机制链、以及 flow/denoising loss 不得当作 `log π` 等 fatal red flags，见 [ai-review-rubric.md](ai-review-rubric.md)。T3·S08 的 proposal 以此为审查标准。

## 5. 仍然开放的问题

- 新课提案「多任务优化几何：梯度尺度与冲突」（wayne，2026-08-30，源自其 G2.0 D0b 梯度诊断与 T0·S02 课上建议）——新增 RLT 后 RL 占比已高于 1/3 下界；可并入 T1·S03（π0.5 配方解剖）扩容，或单列后重新核算 RL 比例，走 PR 议定；
- 各 Lesson 的三种 Pass map 尚待随材料迭代逐步补齐；未单列时由 AI 按 `learning-modes.md` 在开课地图中给出本课 routing；
- T3 pilot 的 task/checkpoint/reward/estimator 选择（沿用 v1 开放问题清单，见 rubric 与 T3 grounding 材料）；
- 各学员的 session 节奏由可用时间与当下诉求决定，不设统一日历进度。

## 6. 新 AI 的冷启动顺序

如果当前学员还没有 `learners/<name>/`，先按 [onboarding.md](onboarding.md) 建立账户；不要从其他学员目录复制答案、prediction 或完成状态。

1. [README.md](README.md)：目标、双线地图、协作方式；
2. 本文件：背景、已定决策、开放问题；
3. `learners/<当前学员>/baseline.md` 与 `progress.md`：这个人已有什么、走到哪；
4. [curriculum.md](curriculum.md) 中当前模块；
5. 当前 session 材料（`modules/.../sXX-*.md`）及其 grounding 链接。

开始陪练前，AI 应先输出：本课要减少的不确定性、学员 baseline 中与本课相关的已有资产、三个 Pass 在本课分别到哪里、当前采用的 Pass、自然停止点、哪些环节必须学员亲手完成、是否涉及训练授权。若学员没有持久默认值，给完地图后再询问本次 Pass。

## 7. 维护原则

- 共享材料（curriculum/modules/templates/rubric）改动一律走 PR；
- 通读讲义的 HTML 阅读版由 `tools/build_reading_edition.py` 从 Markdown 源稿生成（图与交互示例放在同目录 `reading-assets/<讲义名>/`），先改 Markdown 再重建，不手改 HTML；
- 个人状态只进 `learners/<name>/`；不写他人目录；
- 背景或已定决策变化时更新本文件；
- 正式项目事实仍回 `tpc_manip` 权威位置，Gym 不做第二事实源。
