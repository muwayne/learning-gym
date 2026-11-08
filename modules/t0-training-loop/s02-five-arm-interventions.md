# T0·S02 · Optimization geometry 与 training distribution：five-arm intervention

> 模块 T0 第 2 节 · 前置：T0·S01（复用其 baseline、数据与 seed）· [Pass 规则](../../learning-modes.md) · 记录用 [session-log 模板](../../templates/session-log.md) · 术语见 [glossary](../../glossary.md)

## 0 · 本课在地图的位置（白话版）

上节课你把一个模型训好了。这节课在同一个 baseline（同数据、同 seed、同初始化）上建立五个 experimental arms，每个 Arm 只改变一个主要变量。中文短名只帮助记忆，实验身份始终使用 `Arm A/B/C/D/E`：

| Arm | 中文提示 | 唯一改动 | 你工作里的对应事故 |
|---|---|---|---|
| Arm A (control) | baseline | 无（= 上节课的 baseline） | 正常训练 |
| Arm B | input-scale mismatch | force 列 ×1000 且不做 normalization | 传感器单位没对齐 / 忘了 normalization |
| Arm C | class imbalance | training set 改 90% stable / 10% unstable（test set 保持平衡） | 采集偏差：成功样本远多于失败样本 |
| Arm D | freeze verification | 第一层 freeze，训后验证参数 Δ=0 | freeze VLM，只训 action expert |
| Arm E | capacity bottleneck | 结构砍到 3-2-1（单 hidden layer、width=2） | model capacity 装不下任务结构 |

Intervention discipline：**一次只跑一个 Arm，判完预测再开下一个**。五个 arms 共用同一 seed、data generator 和 evaluation set。

**开课确认（地图看完再答；负载前确认）**——本课全程反复用到上节课的三样工具，先各自唤醒，说不全当场补：

1. **手算工具**（②手算段与 Arm B 都靠它）："梯度正比于 input scale"（∂L/∂w = (σ−y)·x）这句话你现在还能推吗？
2. **对照锚点**（五个 arms 都要和它比）：上节课你选定的 baseline 是哪个结构、关键 metrics 各多少？
3. **读图工具**（每张 loss 曲线的第一眼）：EMA 平滑线怎么读——什么时候看它、什么时候必须回头看原始细线？

## ① 上手：化验单怎么读

每个 Arm 跑完看同一套 metrics（上节课全部见过）：

- **loss 曲线**（EMA 粗线看趋势）：下降速度、最终地板；
- **per-class recall**：两类各自多少、差多少；
- **σ(z) 直方图**：两端质量 vs 中间质量；
- **per-layer gradient norm**：Arm B 的主指标——force 那一列会发生什么；直观图景见配套演示[逐层梯度对比](demos/layer-gradnorm-mixer.html)；
- **parameter delta Δ**（仅 Arm D）：frozen layer 训前训后逐元素相减，必须全零。

判读锚点（上节课已建立）：理论噪声地板 ≈ 0.20；baseline（3-16-16-16-1）末段 loss ≈ 0.26、recall gap 8.6pp、中间质量 8.2%。

## ② 手算自检（开卷）

**本课公式参考表**（随时回翻，不考背诵）：

| 名字 | 公式 | 白话 |
|---|---|---|
| 打分 | z = w₁x₁ + w₂x₂ | 加权求和 |
| 读数 | σ(z) = 1/(1+e⁻ᶻ) | 分数翻译成概率 |
| 损失 | BCE = −[y·ln σ + (1−y)·ln(1−σ)] | 答得越离谱罚越重 |
| 坍缩（卡 3） | ∂L/∂z = σ − y | 误差 = 概率减标签 |
| 本课主角 | ∂L/∂wᵢ = (σ − y) · xᵢ | 每路权重的梯度 = 误差 × 那一路的输入 |
| 更新 | wᵢ ← wᵢ − lr · ∂L/∂wᵢ | 往梯度反方向挪一步 |

重激活上节课的卡 3（∂L/∂w = (σ−y)·x）。单样本 logistic 回归：

```text
原版：x = (1.0, 2.0),  w = (0.5, −0.5) → z = −0.5, σ ≈ 0.38, y = 1
B 版：x′ = (1000, 2.0)，同时把 w₁ 缩成 0.0005（保持 z 不变，公平对比）
```

手算两问：①两版的梯度 ∂L/∂w 各是多少？第 1 个分量差几倍？②B 版按 lr=0.1 更新一步后 w₁ 变成多少、下一步 z 会怎样？（答案在文末折叠段）

## ③ 追问（先自答，再与 AI 过）

1. Arm B 用的是 Adam——它的 per-parameter scaling 会不会把 input-scale mismatch 自动救回来？救到什么程度？
2. Arm C 的 **training loss** 会比 Arm A 好看还是难看？为什么 loss 与实际 utility 在这里分家？
3. Arm D 怎么证明参数真的 frozen？只看 `requires_grad=False` 够吗？（提示：optimizer parameter groups、训前训后 Δ、checksum）
4. Arm E（讲解题，不设赌局）：抓稳规则拆开是 4 个 decision surfaces，width=2 的 hidden layer 装不下 4 个面——先理解 `capacity = representation budget`。哪个 slice 先退化留作运行后的观察题。

## ④ 任务书（在前）→ 逐 Arm 下注（在后）

**实现约定**：复用 `code/t0/` 的数据与 seed。AI 搭统一 runner（`t0_arms.py --arm B/C/D/E`，输出全套 metrics + parameter delta 检查）；**每个 Arm 的 config difference 由你口述**，AI 照打。

**每组开跑前锁 1–2 条预测**（模板，口述即可）：

- **Arm B**：early loss 相比 Arm A 如何？force 对应的 gradient component 相比其他列如何？最终能不能追平 Arm A？
- **Arm C**：training loss vs Arm A？balanced test set 上的 unstable recall 大概落到什么区间？输出概率整体往哪偏？
- **Arm D**：freeze 后最终 metrics 比 Arm A 差多少——随机的第一层 projection 是否保留足够信息？（Δ=0 是工具性检查，不算预测）
- **Arm E**：loss floor vs Arm A？中间质量回升到多少？哪类 boundary slice 的 recall 先崩？

**Explain**：每个 Arm 判完预测立刻写一句 mechanism explanation；五个 arms 跑完，区分 **optimization geometry**（Arm B）、**training distribution**（Arm C）与 **model capacity**（Arm E）。

## ⑤ 固化

- 按本次 Pass 保存必要 evidence；`Understand` 可在机制判读后停止，`Implement / Improve` 再为执行的 Arms 追加 ledger；
- 一句 revised mental model；
- **Exit Gate（即 T0 模块出口）**：能用本课证据区分 model capacity / optimization geometry / training distribution；Arm D 有 Δ=0 的实证记录。

---

<details>
<summary>② 参考答案（先算再看）</summary>

```text
两版 z 相同（0.5·1−1 = 0.0005·1000−1 = −0.5），σ−y 相同 ≈ −0.622
原版梯度 = (σ−y)·x  = (−0.62, −1.24)
B 版梯度 = (σ−y)·x′ = (−622, −1.24)      ← 第 1 分量 ×1000
B 版更新：w₁ ← 0.0005 + 0.1×622 ≈ 62.2   ← 一步飞出天际
下一步 z = 62.2×1000 − 1 ≈ 62199 → σ 饱和 → 梯度归零假死
```

这是 optimization geometry 被单一 feature scale 主导的最小版本；Adam 能救多少，由 Arm B 实验决定。
</details>
