# Transformer 精编（学习档案浓缩版）

> 源自 wayne 的个人学习档案（2025-03 学习弧：DETR/ViT 切入 → 亲手实现 Vanilla Transformer + 单元测试 → BERT/T5），2026-08-29 浓缩入库。完整原档留学员本地；本文由档案所有者发起 PR 维护。
>
> 用途：两套 15 问 checklist 是 A1/A2 Exit Gate 的正式工具；缺口表是 A1 课程顺序的依据。

## 通用 15 问 debug checklist（A1 Exit Gate 工具）

看到任何 Transformer 代码，依次回答：

1. 输入 token 是什么？
2. 每个 token 的 embedding dimension 是多少？
3. Q 从哪里来？
4. K 从哪里来？
5. V 从哪里来？
6. Self-Attention 还是 Cross-Attention？
7. attention score 的 shape 是多少？
8. mask 是什么？
9. 每个 head 的 dimension 是多少？
10. FFN 对哪个 dimension 做变换？
11. residual 两边 shape 是否一致？
12. position 信息在哪里加入？
13. 输出是 token、latent、bbox、language 还是 action？
14. loss 对什么做 supervision？
15. inference 是否 autoregressive？

## 机器人扩展 15 问（A2 Exit Gate 工具）

1. observation 如何 tokenize？
2. image 是否经独立 vision encoder？
3. proprioception 怎么编码？
4. language token 与 visual token 什么时候融合？
5. action 是 continuous 还是 discrete？
6. 是否 action chunking？
7. 是否 causal decoder？
8. 是否 diffusion / flow matching？
9. policy horizon 多长？
10. 是否含 world model prediction？
11. attention context 是否跨时间？
12. 是否使用 history window？
13. inference rate 多高？
14. action head 是否独立？
15. Transformer 是 backbone 还是 action expert？

## 模块分工 mental model

```text
Attention = Communication（token 之间交换信息）
FFN       = Computation（逐 token 的特征变换）
Residual  = Information Highway（可堆几十层的通路）
Norm      = Optimization Stabilizer
```

两条防误解：不要机械把 head 解释成"语法头/语义头"（对应关系不固定）；learnable query 学的是 slot prior 与 attention pattern，不是"专门认某类目标"。

## 关键 shape 语义

```text
X: B×L×D → 拆头 (B,H,L,d_h)，d_h = D/H
scores = QK^T/√d_k: B×H×L×L = batch × head × query token × key token
softmax 后每行和为 1；AV → (B,H,L,d_h) → 拼回 B×L×D
mask：不许看的位置 S_ij → −∞（实现常用 masked_fill(mask==0, -1e9)）
```

## Position 三代

- BERT：learnable absolute embedding（E_token + E_position + E_segment）；
- T5：relative bias——按 i−j 给 attention score 加偏置；
- RoPE：直接旋转 Q/K，点积自然携带相对位置（A1·S05 亲手换装）。

## DETR object query 要点（A3·S02 latent query 的直系前身）

- 一个 learnable parameter matrix Q_obj ∈ R^{N×D}：随梯度更新，推理时固定，不是 encoder 输出；
- 与 Faster R-CNN proposal 的区别：没有显式 (x,y,w,h) 空间语义，是 latent slot；
- Hungarian matching（分类 + L1 + gIoU cost 的最优二分匹配）使 unordered set prediction 可训；
- 同一思想延伸：Perceiver / Q-Former 的 latent query、机器人 action query。

## 档案自列缺口 → 课程落点

| 缺口（原档 §52/§57/§58） | 落点 |
|---|---|
| decoder-only 系统拆解（prefill/decode/autoregressive） | A1·S02–S04 |
| causal mask | A1·S02 |
| KV cache | A1·S04 |
| GQA / MQA | A1·S04 追问段 |
| RoPE / RMSNorm / SwiGLU | A1·S05 |
| FlashAttention（tiling + online softmax，概念级） | A1·S05 追问段 |
| Transformer × diffusion / flow matching（DiT、action expert） | A2 |
| world-model Transformer | A3·S07 |

## 公式速查

```text
Attention(Q,K,V) = softmax(QK^T/√d_k)·V
MHA = Concat(head_1..head_H)·W^O
Pre-LN block: X' = X + Attn(LN(X));  Y = X' + FFN(LN(X'))
FFN(经典) = W2·σ(W1·x)，D→4D→D
SwiGLU: FFN(x) = W_o(SiLU(W_g x) ⊙ W_v x)
RMSNorm(x) = g · x / RMS(x)，不减 mean
KV cache：第 t 步只算 q_t,k_t,v_t，attend([K_past;k_t],[V_past;v_t])
```
