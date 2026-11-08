# A1·S03 · Tiny GPT training loop：从 forward 到 generation

> 模块 A1 第 3 节 · 前置：A1·S01、A1·S02 · [Pass 规则](../../learning-modes.md) · [术语表](../../glossary.md) · 记录用 [session-log 模板](../../templates/session-log.md)
>
> 基础阅读（审阅稿）：[S01–S03 汇总课表](s01-s03-overview.md)与[入门讲义](s01-s03-primer.md)（[HTML 阅读版](s01-s03-primer.html)）。Embedding、loss、gradient 与两个 loops 的前置解释见讲义；Summer 按 [预习—问答—手算—口述编程—调试—回顾](../../learners/summer/learning-workflow.md) 推进，先审阅材料再现场练习。
>
> 本课使用小型 character-level language model 闭合训练，不追求语言质量或 benchmark；重点是 objective、gradient path、evaluation 与 autoregressive generation 是否形成可审计的闭环。
>
> Wayne checkpoint 1：[single-batch overfit implementation](../../learners/wayne/code/a1/s03_tiny_gpt.py)。

## 0 · 本课地图

A1·S01 造出了单头 attention，A1·S02 把 causal MHA、Pre-LN block 与 FFN 接成可 forward 的 decoder block。现在还缺四样东西：离散 token 怎样进入网络、每个 position 监督什么、parameters 怎样因 loss 改变，以及训练好的模型怎样把一个 prefix 延长成新 sequence。

本课只抓一条主线：

> `token IDs → hidden states → next-token logits → cross-entropy → gradients → updated parameters`，然后把同一个 forward 放进 autoregressive generation loop。

三个 Pass 的边界：

- `Understand（看懂）`：能沿 shape 解释 forward、shifted targets、teacher forcing 与 generation 的差别；
- `Implement（做出）`：用 PyTorch 写出 tiny GPT，先通过 single-batch overfit，再在小 corpus 上训练并生成；
- `Improve（改好）`：定位一个“loss 看似正常但 objective/behavior 错了”的 failure，完成单变量 intervention 与 regression check。

Wayne 默认采用 `Improve`，允许跨多个 session。自然停止点依次是：讲清两个 loops；通过 single-batch overfit；跑完 baseline；完成一次 intervention。

## 1 · Model contract：从 token IDs 到 logits

### Character-level tokenization

为把变量压到最少，本课使用 English corpus，但不引入 BPE。每个 unique character——包括字母、空格、换行和标点——对应一个 integer ID：

```text
text → encode → token IDs (B, T)
```

`B` 是 batch size，`T` 是 context length。模型不会直接读取字符；它只读取 ID，并通过 embedding table 查出向量。这里的 `character-level tokenization` 是刻意选择：训练内容使用英文，但暂不把 BPE vocabulary、merge rules 与 unknown words 混进 Transformer training-loop 实验。

### Input representation

每个位置的初始表示为：

\[
X_0 = E_{token}[x] + E_{position}[0:T],
\]

shape 为 `(B, T, D)`。`token embedding` 表示“它是什么”，`position embedding` 表示“它在哪里”。causal mask 规定 information 只能从 past 流向 future，但不应把它与显式 position signal 混为一件事。

### Decoder and language-model head

`X_0` 依次经过 `N` 个 A1·S02 的 Pre-LN Transformer blocks，再经过 final LayerNorm 与 linear `lm_head`：

\[
H_N \in \mathbb{R}^{B\times T\times D}
\quad\longrightarrow\quad
logits \in \mathbb{R}^{B\times T\times V},
\]

其中 `V` 是 vocabulary size。`logits[b,t,:]` 是模型在 position `t` 看完允许看到的 prefix 后，对“下一个 token”的未归一化分数。

`lm_head` 不是 classifier 只给整句一个答案；它在全部 `B×T` positions 上同时给出一个 `V`-way prediction。

## 2 · Training example：为什么 input 和 target 必须错开一格

从 corpus 取连续 `T+1` 个 token：

```text
sequence:  [c0, c1, c2, c3, c4]
input x:   [c0, c1, c2, c3]
target y:  [c1, c2, c3, c4]
```

因此：

\[
y_t=x_{t+1}.
\]

在 position `t`，模型输入中确实包含 `x_t`，但标签是 future token `x_{t+1}`。causal mask 保证该 position 的 hidden state 不能读取输入里的 later positions。

如果误写成 `y=x`，模型只需从 residual path 把当前 token 复制到 `lm_head`，loss 可能下降得更快，却学的不是 next-token prediction。这是本课最重要的 silent objective bug：**低 loss 只说明模型完成了你实际写下的 objective，不说明你写的是想要的任务。**

## 3 · Cross-entropy 到底在平均什么

对每个 position：

\[
p(v\mid x_{\le t})=\operatorname{softmax}(logits_t)_v,
\]

\[
\mathcal L
=-\frac{1}{BT}\sum_{b,t}\log p(y_{b,t}\mid x_{b,\le t}).
\]

PyTorch 中通常把 logits reshape 为 `(B*T, V)`，targets reshape 为 `(B*T)` 后调用 `cross_entropy`。这里的 flatten 只是在收集独立 prediction positions，不改变每个 position 的 vocabulary 维度。

一个可解释的辅助量是 perplexity：

\[
PPL=\exp(\mathcal L).
\]

它近似表示模型在每一步还像在多少个 equally likely choices 中犹豫；但它仍是 offline next-token metric，不等于生成内容一定好。

## 4 · 两个 loops：训练并行，生成串行

### Training loop · teacher forcing

训练时，完整正确 prefix 已经放在 `x` 中。causal mask 让所有 positions 可以在一次一次 forward 中并行计算，但 position `t` 只能读取 `x_≤t`：

```text
sample (x,y)
→ forward all positions
→ cross-entropy over B×T targets
→ zero_grad
→ backward
→ optimizer.step
→ parameters change
```

这叫 `teacher forcing`：训练时每一步看到的是 corpus 中真实的 past tokens，而不是模型自己上一步生成的 token。

### Autoregressive generation loop

生成时没有 future ground truth。模型先读当前 context，只取最后一个 position 的 logits，sample/argmax 出一个 token，把它 append 回 context，再重新 forward：

```text
prefix → next-token distribution → sample one token → append → repeat
```

因此训练阶段虽然并行，deployment generation 仍然是 closed-loop sequence generation：模型自己的输出会成为下一步输入。它与 T0·S03 的共同点是 policy-induced context；不同点是这里的 dynamics 只是“append 一个 token”，不是外部物理环境。

context 超过 `block_size` 时，只把最后 `block_size` 个 tokens 喂回模型。`temperature` 改变 distribution sharpness；`top-k` 截掉低概率尾部。它们只能改变 sampling，不能修复训练 objective、mask 或 target shift。

## 5 · 一个完整的 training loop

顺序采用：

1. `model.train()`，sample training batch；
2. forward 得到 `(logits, loss)`；
3. `optimizer.zero_grad(set_to_none=True)`；
4. `loss.backward()`；
5. 检查关键 parameters 的 gradient finite 且非零；
6. `optimizer.step()`；
7. 定期切到 `model.eval()` + `torch.no_grad()` 估计 train/validation loss；
8. 恢复 train mode；保存 metrics，并用固定 prompt/seed 生成样本。

`zero_grad` 必须发生在下一次 backward 前，否则 gradients 默认累积。`eval()` 不会关闭 autograd；`no_grad()` 也不会切换 dropout 等 module behavior，两者作用不同。

## 6 · 先证明“能学”，再讨论“学得好”

正式 baseline 前先做 `single-batch overfit test`：固定一个很小的 `(x,y)` batch，反复更新同一批数据。一个容量足够且接线正确的 tiny GPT 应把该 batch 的 loss 压得很低，并复现其 continuation。

这个 test 同时检查 embedding、position、blocks、lm_head、target shift、gradient 与 optimizer 能否闭合。它不能证明 generalization；但如果连固定 batch 都记不住，继续扩大 dataset 或训练时长没有诊断价值。

随后才跑 train/validation split：

- train loss 与 validation loss 都下降：模型学到可迁移 pattern；
- train 继续下降、validation 转为上升：overfitting；
- loss 下降、sample 仍异常：先查 objective、generation path 与 sampling，不先归咎“模型太小”；
- loss 不动：查 parameter registration、gradient、learning rate、mask、dtype 和 labels。

## 7 · Baseline 与 checks

本课 baseline 使用小型 English character corpus、learned absolute position embedding、2 个 Pre-LN blocks、causal MHA 与 AdamW。train/validation 在句子层面先 split、再分别重复，避免同一句子的副本泄漏到两侧。具体维度在实现前一次展示并冻结；目标不是追求最佳配置，而是让 CPU/MPS 上短时间可跑、曲线与 behavior 可解释。device contract 使用 `MPS if available, otherwise CPU`，同时保留 CUDA compatibility；A1·S03 不以迁移到独立 GPU 机器为前置。

四个 checks 在正式训练前后都保留：

| Check | Predicate | 能排除什么 |
|---|---|---|
| Shape | logits=`(B,T,V)`，loss scalar | embedding/head/flatten 接线错误 |
| Causal invariance | 只改 future input，earlier logits 不变 | mask 泄漏或 axis 错误 |
| Parameter delta | 至少一个核心 weight 在 step 后 `Δ>0` | optimizer 未连接、参数被冻结 |
| Single-batch overfit | 固定 batch loss 显著下降，continuation 可复现 | 大量 implementation / optimization failures |

训练报告至少同时保存：train/validation loss、single-batch test、固定 prompt samples、parameter/gradient health。正式结论使用统一的 true next-token validation metric；generated sample 只作 qualitative sanity check，不以主观“质量”代替合同指标。只有一条 loss curve 不足以证明完整闭环。

## 8 · Improve intervention

baseline 跑通后只改一个变量。优先候选是 `shifted target → unshifted target` negative control：它可能得到更漂亮的 loss，却在 next-token evaluation 与 generation 上失败，直接复现“metric 优化成功、task formulation 错误”。

第二候选才是移除 position embedding，观察 order-sensitive corpus 上的 validation loss 与 generation。这个 intervention 更贴近 A1·S02 留下的“causal direction 不等于完整 position information”，但解释时要承认 causal mask 本身可能提供部分隐式 position cue。

每次只选一个 candidate，先写 prediction 与 falsifier，再运行。不要同时改 model width、learning rate、corpus 和 objective。

## 9 · 本课公式参考表（开卷）

| Formula | 白话用途 |
|---|---|
| `X0 = E_token[x] + E_pos[0:T]` | 把 token identity 与 position 变成同维 hidden state |
| `logits = lm_head(LN(blocks(X0)))` | 每个 position 输出 vocabulary scores |
| `y_t = x_(t+1)` | 监督 next token，而不是复制当前 token |
| `CE = -mean(log p(y_t | x_≤t))` | 对 batch 与全部 positions 平均 next-token error |
| `PPL = exp(CE)` | 把平均 NLL 转成较直观的有效选择数 |
| `θ ← θ - η·∇θL` | backward 产生 gradient，optimizer 真正改变 parameters |

## 10 · Pass completion

- `✅ U`：能讲清 `(B,T)→(B,T,D)→(B,T,V)`、shifted targets，以及 training/generation 两个 loops；完成一个带上下文的 bug 判读；
- `✅ I`：核心实现可运行；shape、causal、parameter delta、single-batch overfit 四个 checks 通过；有 baseline loss 与 fixed-prompt sample；
- `✅ M`：完成一个预注册 single-variable intervention，能解释 offline loss、next-token metric 与 generated behavior 为什么可能不一致，并检查 baseline regression。

下一课 A1·S04 会保持训练好的 model 不变，只改 inference path：把每一步重复计算 past K/V，换成可验证等价的 KV cache。
