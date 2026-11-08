# Gym learning modes · 看懂、做出、改好

> 本文件定义 Gym 的课内深度与 session 节奏。角色只提供案例背景；真正决定本次教学路径的是学员当下的学习诉求。

## 三个单位不要混用

- **Lesson**：课表中的一个稳定知识主题，例如 T0·S03；
- **Pass**：这次希望在该主题上达到的深度；
- **Session**：一次实际学习时段。一个 Pass 可以跨多个 sessions，一个 Lesson 也可以多次进入。

因此“一节课”不再默认等于“一次坐完一套固定流程”。教程、自检、追问、实验与固化只是可按需要组合的 activities，不是必须集齐的五段。

## 三种 Pass

| Pass | 直观目标 | 本次重点 |
|---|---|---|
| `Understand` | **看懂**：能参与交流和判断 | 完整概念地图、English key terms、worked example、证据边界 |
| `Implement` | **做出**：理解原理并亲手实现 | 必要公式、data flow、runnable implementation、一次单变量 intervention |
| `Improve` | **改好**：能诊断 failure 并验证改善 | instrumentation、competing hypotheses、falsifier、controls、improvement 与 regression |

不规定 Pass 或 Session 时长。优先让一次学习尽快产生独立收获，到自然 checkpoint 就停止；`Improve` 表示最终深度，不表示必须一次完成。

## 开课 routing

AI 每次先给出本课地图：本课减少什么不确定性、共同核心、三个 Pass 在本课分别会做到哪里、会产出什么。学员先有 context，之后才选择 Pass。

- 新学员默认“每课询问”；没有明确选择时，从 `Understand` 开始；
- 有持久默认值的学员直接沿用，AI 在地图里声明即可，不重复询问；
- 学员临时指定只覆盖当前 session，不修改持久默认；
- 只有学员明确要求长期改变时，才更新 `baseline.md` 中的默认值；
- 已完成较浅 Pass 后再次进入，只做短 reactivation 与增量内容，不完整重讲；
- baseline 是 routing evidence，不是免做凭证。AI 可用 1–2 个有上下文的 calibration tasks 调整节奏。

推荐开场格式：

```text
Lesson：T0·S03
共同核心：offline loss 不保证 closed-loop SR
Understand：能判断现有 evidence 支持什么
Implement：能运行并解释 execution-horizon intervention
Improve：能定位 first divergence 并设计 recovery falsification
本次 Pass：沿用默认 Improve（或：请在三者中选择）
自然停止点：完成 Arm G 判读
```

## Pass completion gates

### Understand · 看懂

- 听过完整 lesson map，而不是从无上下文问题开始；
- 能用自己的话复述核心 mechanism 和关键 evidence boundary；
- 完成至少 1 个有上下文的判读或解释；
- 写下一句 revised mental model。

`Understand` 不强制安装环境、写长代码或填写实验 ledger。首次接触的概念不设盲猜 prediction。

### Implement · 做出

- 已满足 `Understand`，或在本次用短 reactivation 验证等价基础；
- 能把必要公式、shape、units 或 data flow 对应到实现；
- 产出最小 runnable artifact；
- 完成至少一次完整 P-T-I-E：实验设计先展示、prediction 后锁定、单变量 intervention、结果解释；
- ledger 至少追加 1 条可判定 prediction。

### Improve · 改好

- 已满足 `Implement`，或有可核验的等价实现资产；
- 从具体 failure 或 capability gap 建立 competing hypotheses；
- 增加能区分解释的 instrumentation；
- 定义并执行至少一个 falsifier 或 negative control；
- 尝试 improvement，并检查 held-out result、uncertainty 与 regression；
- 明确结论的 evidence strength，不把 directional evidence 写成 proof。

AI claim audit 按 **Lesson** 累计至少 1 条，不要求每个 Pass 为凑数再造一道 audit。现场出现会影响决策的 claim 时立即审查并留痕。

## Spiral learning · 一课多遍

推荐路径是：

```text
Understand（看懂） → Implement（做出） → Improve（改好）
```

学员可以在第一次快速建立 breadth，之后因兴趣或工作需要回来加深。已有足够 evidence 的学员也可以直接进入 `Implement` 或 `Improve`；高层 Pass 必须覆盖或引用其前置 evidence，但不要求机械重走。

记录最高层时必须保留粒度：

```text
T0·S03 · U✅ I✅ M▶
```

个人总览可同时显示：

```text
接触主题：12 / 42 · Understand 12 · Implement 7 · Improve 3
```

“接触主题”表示至少完成 `Understand`，不等于模块 mastery。模块和全课程的能力声明仍须通过原有 Exit Gate；Exit Gate 可以要求 `Implement` 或 `Improve` evidence。

## 已有记录兼容

- 旧记录中的 plain `✅` 不批量改写；根据现有 evidence 至少映射到 `Implement`，必要时人工标为 `Improve`；
- 同一 Lesson 的后续 Pass 可以追加到原 record，也可以新建带日期的 record；不得覆盖早期 prediction 与 mental model；
- progress 显示最高已完成 Pass，record 保存每次 session 的具体过程。

一句话原则：

> **先用 Understand 快速建立地图，再用 Implement 做出闭环，最后按需要用 Improve 诊断和提升；深度与单次课时长度彼此独立。**

## 已落地的授课案例

[Summer 的 A1·S01–S03 学习流程](learners/summer/learning-workflow.md) 将充分的可视化预习、知识点问答、少量手算、口述编程、运行调试、改动判断和整体回顾接成小闭环。这是已约定的个人教学路径；Pass 深度、完成证据与其他学员的 routing 仍按本文件。
