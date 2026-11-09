# Learning Gym · AI 辅助技术学习工坊

> **Start here.** Learning Gym 是一个公开的技术学习与练习仓库，涵盖 Transformer、机器人强化学习与训练诊断。它的主题只有一句话：**给每个你已有概念的系统，补上属于你自己的训练闭环。**

从[课表](curriculum.md)选一节课，从[协作说明](onboarding.md)开始练习。想讨论 AI 怎么帮人学，先读 [retro/](retro/README.md) 的案例与复盘。

HTML 讲义随仓库提供。克隆后可在浏览器直接打开，例如 [RL 基础讲义](modules/t3-sim-to-rl/rl-foundations-primer.html)。

## 为什么是"闭环"而不是"知识"

这套课程面向已经具备长期工程与 ML 经验的研究者。多数参与者的问题不是没学过——attention、PPO、Bellman、flow matching 都问答过、推导过，甚至局部实现过——而是缺少"亲手把它训到 work、看着它学、干预它、解释它"的最后一环。没有这一环锚定的概念，几个月后就退化成"似是而非"。

因此所有动手型 Pass 遵循统一循环 **P-T-I-E**：

1. **Predict**：动手前写下带机制的预测和推翻条件；
2. **Train / Verify**：跑通最小训练或等价验证；
3. **Intervene**：一次只改一个关键变量；
4. **Explain**：对照预测解释结果，记录 surprise 与修正后的 mental model。

**档案 ≠ 现役**：过去学过、问答过，只是 routing evidence，不自动等于现在会。需要时用最小 reactivation 验证；已有现役 evidence 时可以缩短或替代，不为走流程重复作业。

同一个 Lesson 不要求一次学到底。课内深度分为 `Understand（看懂）→ Implement（做出）→ Improve（改好）`，可以分多次短 session 逐层完成；新学员每课选择，已有稳定偏好者可直接沿用默认值。完整规则见 [learning modes](learning-modes.md)。

## 双线课表（共 45 节）

| | A 线 · 机制 | T 线 · 训练 |
|---|---|---|
| 起步 | — | T0 训练闭环解剖（3） |
| 第一段 | A1 现代 Decoder-only Transformer（7） | T1 训练阶段架构（4） |
| 第二段 | A2 生成式建模与动作头（5） | T2 RL 从零到 PPO（7，含 1 选修） |
| 第三段 | A3 Latent-Space 融合机制（7） | T3 Sim-to-RL 与机器人 RL（10，含 Zero-Shot RL、可微动力学选修） |
| 随行 | — | T4 训练工程实践（2，位置灵活） |
| 终点能力 | 对任何前沿架构画出 latent-interface 图：什么 latent 过界、谁训谁冻、损失落在谁头上 | 一份可提交审批（或有据拒绝）的 Action Expert selective-RL preregistered proposal |

RL（T2+T3）共 17/45 节，约占总课次 37.8%。两线在 T3·S08 汇合；T3·S09/S10 是独立选修。唯一课表见 [curriculum.md](curriculum.md)。

**想先学懂一个算法：** 可直接从 [TD3 第一篇](modules/t3-sim-to-rl/td3-first-lesson.md)进入；不要求先通读 foundations、完成 P0 或学完 PPO/SAC。先读第 1–4 节，再按需做问答与 P4 实践；旧路线保留供系统复习。

系统复习路线从[恢复练习 P0](modules/t3-sim-to-rl/rl-control-practice.md#p0--恢复价值学习)进入，先用[基础讲义](modules/t3-sim-to-rl/rl-foundations-primer.html)与[统一温习讲义](modules/t3-sim-to-rl/rl-control-review-primer.html)打通 PPO、TD3、SAC 的更新，再进入生成策略后训练与可微动力学。DexLab、FB 和综合诊断按原课次选用。各阶段亲手做什么、怎样验收，见[课表](curriculum.md#rl-基础讲义与课次的对应t2t3-共用)。

课程先提供完整通读讲义（Markdown 源稿生成离线 HTML，图与交互内置），再逐步提问、实践和答疑。新增 [Zero-Shot RL 讲义](modules/t3-sim-to-rl/zero-shot-rl-primer.html)用同一个走廊串起 SR / SF / FB，再精读 2023 并走到 Meta Motivo 与 BFM-Zero；阅读与材料准备不自动记为课次完成。

教学对话与材料采用 **English key terms first**：首次出现可用中文解释，之后保持与论文、代码、config 和报告一致的英文写法；中文类比不替代正式术语。统一写法见 [Gym terminology guide](glossary.md)。

## 快速开始（新学员）

完整开户步骤、目录结构和可直接发送给陪练 AI 的第一条 prompt 见 [learner onboarding](onboarding.md)。最短路径是：填写 baseline → 初始化个人 progress 与 ledger → 在学员总览登记 → 先看本课地图，再选择 `Understand / Implement / Improve`。

## 多人协作与迭代方式

- **共享材料**（curriculum、modules、templates、rubric）：改动一律走 PR；
- **个人空间**（`learners/<name>/`）：自己的进度、预测账本、session 记录、练习代码；不写别人的目录；
- **先行者条款**：先学的人每完成一课，顺手修材料中不顺处（错误、缺提示、时长失真），PR 合并后后来者直接受益；
- **Dojo 角色进 PR review**：同事以 Skeptic 身份审你的 Explain 段与 claim audit，比开会轮换更轻。
- 本目录是唯一载体：不再引入外部平台或额外内容输入，一切迭代通过本仓库的提交完成。

## AI 与人的分工

AI 承担：开课地图与 Pass routing、脚手架代码、harness、日志图表、机械验证、按材料陪练追问。全体学员共用[可视化教学原则](curriculum.md#learning-activity-toolbox)：经常以清晰、可交互的图示解释抽象机制，并配合简短讲解与动手验证。

人拥有：手算自检、核心实现（每课明确标出哪几十行必须自己写）、预测、干预设计、Correct / Plausible-but-vague / Conceptually-wrong 判定（见 [ai-review-rubric.md](ai-review-rubric.md)）、是否申请真实训练授权。

AI 给你搭的每份脚手架本身就是审查对象：每课至少 1 条 claim audit，顺手完成。

## 历史案例与证据

课程从机器人研究工作区演化而来。历史笔记中标为「内部资料」的来源需要原项目权限。公开读者可以直接使用通用讲义、synthetic toy 与公开论文，按课堂记录区分示例、学员证据与正式工程结论。

## 非目标

- 不重教已系统学过的理论（引用学员既有档案，不重写教材）；
- 不追 SOTA 复现，toy 结论不外推为真实系统结论；
- 不保存 checkpoint、dataset、训练日志或大型 rollout；
- 实际机器人或正式模型训练遵守各自项目的授权与证据记录流程；
- 课程原稿与练习以本仓版本为准。
