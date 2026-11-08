# A1·S02 · Causal mask + multi-head attention + mini decoder

> 模块 A1 第 2 节 · 前置：A1·S01（复用其 attention 函数与手算数值）· [Pass 规则](../../learning-modes.md) · 记录用 [session-log 模板](../../templates/session-log.md) · 术语见 [glossary](../../glossary.md)
>
> 基础阅读（审阅稿）：[S01–S03 汇总课表](s01-s03-overview.md)与[入门讲义](s01-s03-primer.md)（[HTML 阅读版](s01-s03-primer.html)）。首次学习先讲完整地图与基础概念；Summer 按 [预习—问答—手算—口述编程—调试—回顾](../../learners/summer/learning-workflow.md) 推进，先审阅材料再现场练习。
>
> 开课确认（负载前确认，AI 先问；抓获 #13：每问须自带作答上下文——问③前先重贴 S01 toy 的 X 与 S/√2 全表，不得让学员凭空回忆）：①上节课 mask 的正确操作是什么、为什么不能乘？②"自回归 / autoregressive"这个词你现在怎么理解？③S01 的 L=3 toy 的 P 矩阵三行还记得大概长什么样吗？

## 0 · 本课在地图的位置（白话版）

上节课你的 attention 是"全场随便看"。这节课做两件事，然后把积木拼成整块：

1. **Causal mask**：预测下一个 token 时，不允许读取 future positions；实现上是固定的 lower-triangular allow mask；
2. **Multi-head attention (MHA)**：把 projected Q/K/V reshape into heads，每个 attention head 独立产生一张 attention map，再 merge heads；
3. **Transformer block**：attention + FFN + residual connection + LayerNorm。你日常训的 π0.5 主干就是这种 block × N layers。

终点：一个能前向的 mini decoder。下一课（S03）给它接上 embedding 和位置信息，**训到能生成**——那是你的档案里缺的最后一环。

## ① 上手

**Causal mask 就是一张下三角允许表**（L=3）：

```text
        看0  看1  看2
token0 [ 1    0    0 ]     ← 只能看自己
token1 [ 1    1    0 ]     ← 能看 0 和自己
token2 [ 1    1    1 ]     ← 全能看
```

**已算样例**：把它套在 S01 的 toy 上（S/√2 第 1 行 = [0, 2.83, 1.41]）：

```text
token1 被禁看 2 → 有效分数 [0, 2.83] → softmax(e⁰=1, e^2.83=16.9) ≈ [0.06, 0.94, 0]
token0 被禁看 1、2 → 只剩自己 → 权重必然 [1, 0, 0]，输出 = 自己的 V
token2 无禁令 → 与上节课相同 [1/3, 1/3, 1/3]
```

**MHA shape path**（d=4、H=2，head dimension d_h=d/H=2）：

```text
X (L,4) → Q/K/V (L,4) → reshape into heads (2,L,2)
each head: S(L,L) → P(L,L) → O_h(L,2)
merge heads / concat → (L,4) → output projection W_O → (L,4)
```

Reshape into heads **不增加** QKV 参数（W 仍是 d×d）；output projection W_O 是额外的 d×d matrix。每个 head 的 attention logits 除以 √d_h，不是整体的 √d。

配套交互演示：[demos/multihead-and-block.html](demos/multihead-and-block.html)——展示 reshape 的对象是 projected Q/K/V、不同 attention maps，以及 Pre-LN block；含 causal 开关。

**Block 的分工**（你的 Transformer 精编里那四句）：

```text
Attention = 交流（token 之间搬运信息）      残差 = 高速路（信息不丢）
FFN       = 各自消化（逐 token 变换，d→4d→d） LN  = 稳压器（能堆深）
Pre-LN 顺序：X + Attn(LN(X)) → 再 + FFN(LN(·))
```

## ② 手算自检

用 S01 的数值（S/√2 全表你手上有），手算 **causal 版 P 的完整 3×3**（第 0、1 行上面演示了思路，独立复算并补第 2 行），再算 **O 的第 0 行**。要点：说出"为什么第 0 行必然是 [1,0,0]"。

## ③ 追问（先自答）

1. 为什么 causal 下 token0 的输出必然等于它自己的 V？这对"句子的第一个词"意味着什么？
2. 8 个 d_h=8 的头 vs 1 个 d=64 的头：参数量一样，那多头**多买到了什么**？（提示：每个头有自己独立的 P 矩阵）
3. LayerNorm 放 attention 前（Pre-LN）还是后（Post-LN）？现代 decoder 为什么默认前者？（一句话级别即可，S05 深挖）
4. FFN 为什么要先放大到 4d 再缩回？它和 attention 谁在"跨 token"谁在"原地加工"？

## ④ 任务书（在前）→ 下注（在后）

**要实现的清单**（numpy，续写在 S01 文件旁；核心自己写/口述驾驶，AI 搭初始化、assert 与实验脚本）：

```python
causal_mask(L)                       # 下三角允许表
split_heads(X, H) / merge_heads(...) # (L,d) ↔ (H,L,d_h)
mha(X, WQ, WK, WV, WO, H, causal=False)   # 多头 attention，内部复用 S01 的单头逻辑
block(X, params)                     # Pre-LN：X + Attn(LN(X)) + FFN(LN(·))
```

**跑四个检查**：

| 检查 | 做什么 | 看什么 |
|---|---|---|
| A · 退化对拍 | H=1、无 causal、W_O=单位阵 | 与 S01 的输出完全一致 |
| B · causal mask 生效 | L=3 toy + causal | P 是下三角、行和=1、第 0 行=[1,0,0] |
| C · MHA shape path | 随机 X（L=5, d=8, H=2） | split/merge 每步 shape assert 全过 |
| D · 顺序实验 | 打乱输入行，分别在无 causal / 有 causal 下比较输出 | 输出是否还"跟着打乱"（置换等变还在吗） |

**下注（口述"成立/不成立 + 一句机制"）**：

- **P1**（对应 A）：H=1 退化版与 S01 输出完全一致；
- **P2**（对应 B）：causal 后 P 的第 0 行 = [1, 0, 0]；
- **P3**（对应 D）：上节课验证了 permutation equivariance。**加上 causal mask 后，这个性质还在吗？**——mask 按 position 固定，input token permutation 后 allow relation 是否随之 permutation？

**Explain**：判三注；P3 无论结果如何，写一句它对"位置信息从哪来"的启示——这句话就是 S03 开场的门票。

## ⑤ 固化

- 按本次 Pass 保存必要 evidence；`Understand` 可在 attention/block 地图与判读后停止，`Implement / Improve` 再记录 runnable checks 与 ledger；
- 一句 revised mental model；
- 代码入 `learners/<你>/code/a1/`；
- **S03 预告**：embedding + 位置信息 + 交叉熵，把这块积木训到能生成——训练闭环收官战。

## 附 · 卡点问答（先行者场次沉淀，2026-09-01；配套演示见 [demos/multihead-and-block.html](demos/multihead-and-block.html)）

- **Reshape into heads 的对象是什么？** 是 projected Q/K/V 的 columns，不是把 raw input features 分给不同 heads。每个 projection column 都读取全部 d 维输入；各 head 在自己学习出的 d_h-dimensional subspace 中打分。
- **等宽 heads 是否合理？** 等分的是 representation slots；内容分配由 projection matrices 学习，重要信息可以同时进入多个 heads。等宽便于组成 `(H,L,d_h)` tensor 做 batched computation。
- **MHA 贵在哪？** attention matmul FLOPs 与单头同阶（H·L²·d_h=L²·d）；增加的是每个 head 的 softmax elements 与 attention-map memory。
- **Norm 三兄弟**：BN 固定一维、跨 batch 样本（依赖 batch 统计，序列推理靠不住）；LN 每个 token 自己的 d 维、与 L 无关；RMSNorm 是 LN 砍掉减均值、只除 RMS 只留 γ（S05 换装；Llama/Gemma 系主干在用）。
- **残差是两条路并联**：输出 = 恒等直通路 + 加工路；反传时恒等路梯度系数恒为 1，深网络因此训得动（ResNet 的遗产）。
- **Pre-LN vs Post-LN**：Post-LN 的 LN 站在主干道上，恒等路每层被打断，深了不稳、要 warmup 伺候；Pre-LN 恒等路裸奔到顶，开箱即稳——原版 2017 与 BERT 是 Post-LN，GPT-2 起 Pre-LN 成默认。
- **FFN 为什么 d→4d→d**：放大买的是非线性加工的工作台；中间激活是命门——没有它，两个线性层连乘塌缩回一个 d×d 线性映射，扩容白扩。
- **Output projection W_O 有什么价值？** Merge/concat 只把 H 个 head outputs 并排放回 model dimension；W_O 让每个输出维度成为所有 heads 的 learned linear combination。
