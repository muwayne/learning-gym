# T0·S01 · Loss 作为接口：第一次完整训练闭环

> 模块 T0 第 1 节 · 前置：无 · [Pass 规则](../../learning-modes.md) · 记录用 [session-log 模板](../../templates/session-log.md)
> 本课重制自 gym v1 的 W1D1：保留其 formulation 转译内核，按"教程先行"重排；干预组全部留给 S02。

## 0 · 本课在地图的位置

T 线从这里开始，而不是从 GPT 或 RL 开始，原因只有一个：**第一次完整闭环应该发生在没有任何架构神秘感的系统上**。MLP 二分类里，formulation、loss、梯度、数据分布各自看得清清楚楚；等到 A1·S03 训 tiny GPT 时，未知数就只剩 attention 本身。

本课两个产出：一份 formulation 转译（把显式规则"翻译"成 learning system contract）；一个 training baseline。S02 将在这个 baseline 上做 five-arm intervention，所以 seed 和 config 必须存档。

## ① 上手：learning system contract 六要素

一个 learned system 是由六个选择"编写"出来的：

```text
Observation → Target → Representation → Objective → Data distribution → Inductive bias
```

**已做好的转译样例**（电机过热保护，照着这个格式做④的练习）：

传统规则：

```cpp
overheat = temp > 90 && current > 10 && duration > 5s;
```

| 要素 | 选择 | 为什么 |
|---|---|---|
| Observation | 最近 10s 的 temp/current 滑窗统计 | 规则里的 `duration` 暴露了这是时序问题，瞬时观测装不下"持续" |
| Target | 未来 10s 内是否触发热保护 | 预测比复述阈值有用；标签可由后见数据自动生成 |
| Representation | 窗口统计特征 + 小 MLP | 最小够用；时序结构已被特征吸收 |
| Objective | BCE | 直接优化平均分类似然；**不保证**少数类召回、校准或提前量 |
| Data distribution | 必须含边界样本：89° 长时 / 91° 瞬时 / 双高但短暂 | 决策边界由边界样本决定，不由典型样本决定 |
| Inductive bias | 特征选择保留"温度电流相关"的知识；阈值与组合逻辑交给学习 | 显式知识退到特征层，规则本体消失 |

punchline：**没有手写规则，但每一行选择都是在编程。**

## ② 手算自检（不回看答案）

在 2 参数 logistic 回归上手算一步梯度下降：

```text
x = (1.0, 2.0)   y = 1   w = (0.5, −0.5)   b = 0   lr = 0.1
```

依次写出：z = w·x → σ(z) → BCE loss → ∂L/∂z = σ−y → ∂L/∂w → w′ → 用 w′ 重算 z 与 loss。

要点不是数值，是两个"必须能说出"：loss 为什么必然下降；每个分量的调整方向为什么对。答案在文末折叠段。

## ③ 追问（先自答一句，再让 AI 判）

1. BCE"直接优化什么、没保证什么"？（提示：平均似然 vs 少数类召回 / 校准 / 时序提前量）
2. Target 换成"未来 1 秒是否 slip"，六要素中哪些跟着变？
3. 原规则的三个阈值去哪了？（inductive bias 的迁移路径）
4. 类别 90/10 失衡时，总体 accuracy 为什么没有信息量？
5. 举一个本仓库的真实 proxy 错位例子（提示：offline loss 下降 vs 闭环成功率）
6. v1 原题：没有手写抓稳规则，但我们通过哪些选择"编写"了系统？（现在你有六要素表了）

其中至少 1 条 AI 回答按 [rubric](../../ai-review-rubric.md) 记 claim audit。

## ④ 动手（P-T-I-E）

**第一步 · 转译练习**：把 v1 的抓稳规则自己转译，禁止新增阈值，按六要素表写进 session log：

```cpp
stable = normal_force > F && abs(slip_velocity) < S && pose_error < P;
```

至少说出一条与电机样例的关键差异（提示：抓稳的 Target 有三个候选——当前是否稳 / 未来 1s 是否 slip / 任务最终是否完成——选哪个，各付什么代价？）。

**第二步 · baseline 训练**。先锁预测：

- P1：平衡 + 标准化下，两类 recall 差 < 5 个百分点；
- P2：输出概率呈双峰（堆在 0/1 附近）而不是摊平——先写出你认为的机制；
- P3：第一层 gradient norm——首次接触该概念者不设赌局：先玩[配套演示](demos/gradient-norm-dashboard.html)（1-8-1 toy 的 loss/两层梯度范数交互仪表盘），再复述"两条力拔河"的机制；已有先验者照常押走势

实现约定：

- 数据：AI 按 spec 写生成器——x = (normal_force, slip_velocity, pose_error)，阈值规则 + 标签噪声，三类边界样本占比明确，类别平衡，标准化；**spec 与 seed 写死存档**；
- 模型 / loss / 训练循环核心约 30 行**自己写**（小 MLP + BCE + Adam；A 亲手 / B 口述驾驶 / C 填空抓错三种模式均可，见课表「亲手的三种合法模式」）；AI 只搭数据生成、画图与 assert；
- 记录：loss 曲线、per-class recall、输出概率直方图、各层 gradient norm；
- 存档：seed / config / 初始与最终权重——S02 五组必须在同一 seed 同一初始化上跑。

**Intervention**（本课只做一个最小的）：去掉 normalization 重跑一次，其余不变，看 early loss 与 convergence speed——这是 S02 Arm B（force×1000）的预告。

**Explain**：逐条判 P1–P3，写 surprise；没有就写"最接近翻车的一步"。

## ⑤ 固化

- 按本次 Pass 保存必要 evidence；`Understand` 可在建立 mental model 后停止，`Implement / Improve` 再记录 P1–P3 与 ledger；
- 一句 revised mental model（句式参考："训练一个分类器，实际是在编写____"）；
- 代码入 `learners/<你>/code/t0/`；
- **S02 预告**：同一 baseline 上做 Arm B（force×1000）/ C（90-10 class imbalance）/ D（freeze first layer + Δ=0）/ E（width=2）四个 interventions，每个 Arm 先预测。别删 seed 和 weights。

## Implement Exit

手算全对（或纠错后复算全对）；转译表六要素齐且含一条与样例的关键差异；baseline 四项记录齐；P1–P3 有判定；≥1 条 audit 入档。

---

<details>
<summary>② 参考答案（先算再展开）</summary>

```text
z  = 0.5·1.0 + (−0.5)·2.0 = −0.5
σ(−0.5) ≈ 0.378
BCE = −ln(0.378) ≈ 0.97
∂L/∂z = σ − y = −0.622
∂L/∂w = (σ−y)·x = (−0.622, −1.244)
w′ = w − lr·∇ = (0.562, −0.376)
新 z = 0.562·1.0 + (−0.376)·2.0 = −0.19 → σ ≈ 0.453 → BCE ≈ 0.79 ✓ 下降
```

方向检查：y=1 而 σ<0.5，需要 z 变大；x 两个分量都是正数，所以 w 两个分量都应增大——梯度下降给出的更新（+0.062, +0.124）正是如此，且 x₂ 更大所以 w₂ 调得更多。
</details>
