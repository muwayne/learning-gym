# T3·S06 · Expressive policy 上的 value maximization：EXPO 的 select + edit

> 模块 T3 第 6 节 · **先通读 [讲义（HTML 阅读版）](s06-primer.html)**（[Markdown 源稿](s06-primer.md)）· 本文件是练习单 · [Pass 规则](../../learning-modes.md) · [交互演示独立版](demos/expo-select-edit.html) · CPU harness：[s06_expo_toy.py](../../learners/wayne/code/t3/s06_expo_toy.py) · 记录用 [session-log 模板](../../templates/session-log.md)
>
> 前置：T3·S07 actor–critic 基础衔接（单元 1–3）或等价的 Q / TD / twin-min 现役理解。本课 EXPO 段把 base policy 当作黑盒 sampler，不要求 A2；讲义知识点 10 中 DPPO / RECAP 路线的 estimator 合法性对照表需要 A2 的 flow sampling 背景，留到 A2 之后补完。
>
> 本课只做 toy implementation 与 evidence audit，不授权 π0.5、ManiSkill、RoboDojo 或真机训练。真实运行仍按根仓库 pi-expo-sim 计划（内部资料） 的授权与 Stop 条件执行。

## 0 · 本课地图

本课的全部知识先在[讲义](s06-primer.html)中一次讲完（29 个知识点：RL 基础重激活 → expressive policy 的难点 → EXPO → EXPO-FT → RLT 对照与诊断地图 → toy 走读 → 易混淆概念）。本文件只放练习：问答、判读、口述实现、实验与迁移审计。练习不引入讲义之外的新概念；卡住时回读对应知识点。

本课共同问题：

> 给定一个冻结的 expressive policy，怎样用 learned Q 改善闭环成功率，同时不让 Q 的误差把策略带出可信区域？Q 究竟在估计哪个 policy 的价值？

为什么现在学：你在 pi-expo-sim（内部资料） 中已经完成 X2 SFT 基线、X3.7/X3.8 完整在线 EXPO（未验证出收益）以及 X3.9/X3.10 的 critic 与失败诊断，复现中的问题整理在[复现案例集](expo-ft-casebook.md)；而 RLT 线（内部资料） 的诊断已经指出"backup 中的下一步动作决定学到什么"与"actor 沿 ∇Q 走出包络"两个核心问题。本课把这两点放到一个可以秒级重跑的 toy 上亲手验证。

### 三个 Pass

| Pass | 本课做到哪里 | 自然停止点 |
|---|---|---|
| `Understand` | 通读讲义；在 §1 问答中分清各条 expressive-policy RL 路线、EXPO 三个组件各自解决什么、Q 是哪个 policy 的价值 | 完成 §1 选定的问答与 §2 判读题 + 一句 revised mental model |
| `Implement` | 口述 OTF selection、backup next action、TD target 三个函数，跑通 EXPO-lite baseline，并完成一次自选的单变量 intervention | 一个 arm 的 P-T-I-E 与 ledger 1 行 |
| `Improve` | 在弱 executor 上找出一个 failure，用 pre-training gate 和一个 control 区分竞争假设；把结论迁移到 EXPO-FT 代码与 X3 设计的监控项 | failure hypothesis + falsifier + control 结果 + transfer audit |

默认 Pass：沿用 Wayne 的持久默认 `Improve`，按自然停止点分多次进入。

### 两天的建议节奏（按 session，不按日历）

| Session | 内容 | 产出 |
|---|---|---|
| 0 · 通读 | 从头到尾读一遍[讲义](s06-primer.html)，交互示例都动手拖一拖；记下读不懂的词或段落 | 疑问清单（可以为空）；阅读本身不记通过 |
| 1 · Understand | 先处理疑问清单，再按 §1 索引一次一问；最后做 §2 判读 | 问答原答与修订、判读答案、mental model 一句 |
| 2 · Implement | §3 口述三函数 → baseline → 自选一个 intervention | ledger 1 行；可复现命令 |
| 3 · Improve | §4 弱 executor：gate 测量 → 预测 → 运行 → control；§5 迁移审计 | failure 卡片；EXPO-FT 代码核对 2 条；X3 监控项映射 |
| 之后（可选） | 接 [T3·S07 短 session B](s07-rl-token-online-adaptation.md#短-session-bcritic-的乐观与-actor-的偏离)：同一视角看 RLT 的 actor-Q 偏离 | — |

学员必须亲手完成：三函数的意图口述（§3 任务 I-1～I-3）、每次运行前的 prediction、intervention 的选择、结果解释与 claim audit 判定。AI 负责 harness、运行、图表和机械核对。

## 1 · 问答入口索引（Session 1）

下表只是索引，不要一次全部发出。每次只问一个问题：先贴出对应知识点的原文要点与必要公式，学员用 1–3 句话回答，AI 反馈后再进下一题。已经答得清楚的部分直接跳过，不为填表重考。公式表在讲义附录 A 与各知识点内，开卷。

| 讲义知识点 | 问答入口 |
|---|---|
| 2 · Qπ | 同一个 `(s,a)`，后续换成更强的 policy，Q 值会怎样变？为什么说 Q 之前要先说"谁的 Q"？ |
| 3–4 · TD target 与 mask | 超时被当作终止（`mask=0`）在什么条件下是自洽的？什么时候会误导 critic？ |
| 7–8 · Winner's curse 与 ensemble min | 候选从 8 个加到 64 个，被选中动作的"评分 − 真实"会怎样变？min of 2 起什么作用，代价是什么？ |
| 9–10 · Expressive policy 与四条路线 | 为什么拿不到便宜的 `log π` 会挡住 PPO，却不挡 EXPO？ |
| 12 · Edit 与 OTF | edit 有上限 β，又有挑选，二者各管什么？如果正确动作离 base 分布很远，EXPO 能不能到达？ |
| 13 · Backup identity | 为什么 EXPO 在 TD backup 里也用 OTF，而不只在 rollout 时用？只在 rollout 用会怎样？ |
| 14 · Selection space | 什么样的 VLA 会让 selection space 接近 0？这时 EXPO 还能贡献什么？ |
| 15 · 挑选 vs 梯度 | 同一个在数据外有虚假高峰的 critic，分别交给 RLT 的 actor 和 EXPO 的挑选，结果会有什么不同？ |
| 17 · Chunk 与 n-step | 为什么 chunk 越长，Q 越偏向"采数据那个 policy"的价值？ |
| 19 · 系统层与模型层 | EXPO-FT 报告的 30/30 属于哪一层？要回答"RL 是否提升了 VLA 本身"还缺什么评测？ |
| 25 · 诊断地图 | "选中 edit 的比例很高且大多贴着 β 上限"可能对应哪两种不同情况？各该先查什么？ |

## 2 · Understand Pass

### 判读任务（Session 1 自然停止点）

- 类型：判读
- 位置与目的：通读讲义、完成 §1 问答后，确认"Q 是谁的价值"和"selection 的收益上限"已经能用于读监控
- 期望产出：2–3 句话

场景：X3 运行中，固定探针 snapshot 上每个 block 比较一次 Q 与 MC 回报（早期 RoboDojo 设计 §7 的监控项之一）。某次运行的 rollout 使用 OTF selection（N=8 + edit），但 backup 被误配为"只采 1 个 VLA 样本"。监控显示：rollout 成功率持续上升，而 Q 在探针上系统性低于 MC 回报约 0.1。请判断：这个 Q 低估更可能说明 critic 学坏了，还是另有解释？如果另有解释，Q 此时在估计哪个 policy 的价值？

## 3 · Implement Pass：EXPO-lite toy

### Toy contract

完整合同在讲义知识点 26–28（环境、冻结 VLA、agent 部件与配置、评测与仪表）。发布本节任何任务或锁定 prediction 之前，AI 在对话中重贴这三点的合同要点；学员不需要回翻讲义。规模：CPU 单线程，每个 seed 约 2–20 秒，无 checkpoint 输出。

运行入口（只看合同、不训练）：

```bash
uv run learners/wayne/code/t3/s06_expo_toy.py
```

### 学员任务（口述驾驶 B：你说意图，AI 代笔并当场跑）

三个函数在 harness 中保留为 TODO，`--run` 在补齐前拒绝启动。每个任务独立发布，AI 每轮重贴本任务全文。

**I-1 `otf_select(cand, scores)`**
- 类型：口述实现（1–2 句意图）
- 规格：`cand`、`scores` 形状均为 `[B, M]`（M = 2N 个候选）；返回 `(chosen_action [B], chosen_index [B])`
- 验收：对 `scores=[[0.1, 0.7, 0.3]]` 返回 index 1；不得修改 `cand`

**I-2 `backup_next_action(agent, x2, t2, rng)`**
- 类型：口述实现（说明两种模式各取什么动作，以及为什么 OTF 模式要用 target critic 打分）
- 规格：返回 `[B]` 的下一步动作 `a'`；`agent.cfg["backup"]` 为 `"otf"` 或 `"sample"`；可用工具 `agent.act(x2, t2, rng, net=agent.q_target)`（OTF 选择，返回三元组，第一个是动作）与 `base_sample(x2, 1, agent.cfg, rng)`（单个 base 样本，形状 `[B, 1]`）
- 验收：两种模式输出形状均为 `[B]`；`"sample"` 模式不调用 critic

**I-3 `td_target(r, done, q_next, gamma)`**
- 类型：口述实现（一行）
- 规格：输入都是形状 `[B]` 的 tensor；`done=1` 覆盖成功、掉落、超时三种 terminal
- 验收：`r=1, done=1` 时 `y=1`；`r=0, done=0, q_next=0.8, γ=0.99` 时 `y=0.792`

### Baseline 与 intervention

1. 先跑 baseline `--arm expo --seeds 0 1 2`，看真实成功率、Q 与 MC 的差、edit 选中比例。baseline 不设赌局（首次接触）。
2. 从下面选**一个** intervention，写下带机制的 prediction 与推翻条件，再运行。每个 arm 相对 baseline 只改一项：

| Arm | 只改什么 | 回答的问题 |
|---|---|---|
| `base-only` | N=1、无 edit | 不用 critic 时的地板（设计 A0） |
| `n1` | N=1，保留 edit | 改进来自 selection 还是 edit |
| `no-edit` | 关掉 edit | edit 的边际贡献 |
| `backup-sample` | backup 改为单个 base 样本 | Q 的身份改变后，成功率与 Q 校准各怎样变（EXPO 原文 Fig 5 的 toy 版） |
| `single-q` | 1 个 Q、不取 min | ensemble min 在这个 toy 里是否起作用 |

```bash
uv run learners/wayne/code/t3/s06_expo_toy.py --run --arm <arm> --seeds 0 1 2
```

Implement 收口：一个 arm 的完整 P-T-I-E、ledger 1 行、一句 revised mental model。其余 arm 可在之后的 session 继续，不为凑表一次跑完。

## 4 · Improve Pass：executor 太弱时会怎样

### 问题

EXPO-FT 要求 VLA 先 SFT 到约 40% 再开始 RL；X2 计划以 `success_once ≥ 40%` 作为进入 EXPO-FT 的门槛，早期 RoboDojo 设计 §5 另有 G2"选择空间 ≥ 10 pp"。实际复现中，起点越过了 40% 门槛，在线 EXPO 仍未见收益（[复现案例集](expo-ft-casebook.md)案例 17）。这些门槛在 toy 上意味着什么？一个弱 executor 加更宽的 selection（更大 N）能不能补回来？会不会出现新的 failure？

### 步骤

1. **Pre-training gate（不训练）**：对 `expo`、`mid-n8`、`weak-n8`、`weak-n32` 测 base 成功率与 selection space：

   ```bash
   uv run learners/wayne/code/t3/s06_expo_toy.py --gate --arm weak-n8
   ```

   `weak-*` 把 good mode 概率从 0.6 降到 0.05（单次采样成功率约 7%）；`mid-n8` 为 0.15。
2. **Prediction**：只根据 gate 数字，预测各 arm 训练后的成功率排序，以及哪个 arm 最可能出现 seed 间不稳定；写下机制与推翻条件。
3. **运行**：`weak-n8`、`weak-n32` 至少 6 个 seeds（`--seeds 0 1 2 3 4 5`），全部报告。
4. **Instrumentation**：如果出现异常 seed，用 `--out` 保存 JSON，查看逐 episode 成功序列、探针上的 Q 评分与选中 edit 比例；先列出至少两个竞争假设，再选区分性检查。
5. **Control**：harness 提供 `weak-n32-demo`（训练前在 replay 放 10 条成功示教，对应早期 RoboDojo 设计 §3.2 的 replay 初始化）。你也可以提出别的 control；一次只改一个变量，并说明它能区分哪两个假设。
6. **Regression**：control 不应在 `expo` 默认设定上损害成功率；需要时补一次对照运行。

Improve 收口：failure hypothesis、falsifier、control 结果与 regression evidence 各一段；ledger 至少 1 行。

## 5 · 迁移审计：toy → EXPO-FT → X3

toy 与官方 EXPO-FT 的逐项差别见讲义知识点 29；这张表本身也是 claim audit 对象。

### 审计任务（Session 3 后半）

1. 在 EXPO-FT 代码核对（内部资料） 引用的 `expo_ft/agents/alg/expo_ft.py` 中确认：rollout 时 OTF 对 ensemble 用哪种聚合；backup 的候选由当前 VLA 还是 target VLA 采样。把结论与讲义知识点 29 的差别表对照，判断 toy 的哪条结论可能因此不迁移。
2. 把你在 §3/§4 用过的两个仪表（Q − MC、选中 edit 比例）映射到 X3 实际记录的监控字段（W&B 指标对照（内部资料））与[复现案例集](expo-ft-casebook.md)案例 17–20，各写一句：在 X3 中看到什么数值模式时，应先怀疑哪一层（L0 executor / L1 reward / L3 critic 排序 / L4 策略改进，见 RLT 方法迭代参考 §2（内部资料））。
3. 对本课 AI 讲解或 harness 至少做 1 条 claim audit（Correct / Plausible-but-vague / Conceptually-wrong，判据见 [ai-review-rubric.md](../../ai-review-rubric.md)）。

toy 结论不外推为 π0.5、StackCube 或 RoboDojo 的能力证据；它只用于校准"该看什么、先怀疑什么"。

## 6 · Pass completion

- `Understand`：§1 地图讲完、§2 判读完成、一句 revised mental model；
- `Implement`：三函数由学员口述意图并通过验收、baseline 跑通、一个 intervention 的完整 P-T-I-E、ledger ≥1；
- 替代路径：`Implement` / `Improve` 也可用学员在根仓库已授权 EXPO-FT 复现工作中的真实证据完成，要求与 toy 相同的 P-T-I-E 结构（见[课表中的对应说明](../../curriculum.md#rl-基础讲义与课次的对应t2t3-共用)）；
- `Improve`：§4 的 failure hypothesis + falsifier + control + regression，§5 的迁移审计 1–2 项，claim audit ≥1。

## 7 · 与相邻课次的关系

- **T3·S07（RLT）**：同一问题的另一种答案。完成本课后再看 S07 短 session B，可以直接比较"actor 沿 `∇Q` 走"与"Q 只做选择"的偏离风险。
- **T3·S02（SAC）**：本课只用到 edit policy 需要的那部分 SAC（ensemble critic、熵正则的位置）；auto-α 推导与完整 SAC 实现仍属 S02，本课不计为 S02 完成。
- **T3·S08（Capstone）**：本课提供 selective-RL proposal 的一个候选路线及其前提（executor 成功率、selection space、critic 表征）。

## 来源链接

- EXPO：[arXiv 2507.07986](https://arxiv.org/abs/2507.07986)；EXPO-FT：[arXiv 2605.25477](https://arxiv.org/abs/2605.25477)；官方代码 [pd-perry/expo-ft](https://github.com/pd-perry/expo-ft)
- 仓库内：EXPO 系列回顾（内部资料）、EXPO-FT 代码核对（内部资料）、pi-expo-sim 设计（内部资料）、RLT 方法迭代参考（内部资料）
