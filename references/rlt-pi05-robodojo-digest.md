# RL Token → π0.5 / RoboDojo · 课程精编

> 用途：T3·S07 的 grounding。本文只提炼可教学的 mechanism、transfer questions 与 evidence boundary，不构成实现计划、训练协议或运行授权。

## Source hierarchy

### Primary sources

- Physical Intelligence, [RL Token: Bootstrapping Online RL with Vision-Language-Action Models](https://arxiv.org/abs/2604.23073)；
- Physical Intelligence, [Precise Manipulation with Efficient Online RL](https://www.pi.website/research/rlt)；
- Physical Intelligence, [openpi](https://github.com/Physical-Intelligence/openpi)，用于核对公开 π0.5 的模型边界。

### Team draft used as a case

- `claude_RLT_pi05_RoboDojo_开发任务书_2026-09.md`，用户提供的 2026-09-04 draft；
- SHA-256：`49ba8aa9264fc9b7657a274a967689c623807fdcf80db9067a064b1fc8cc652e`；
- 该 draft 混合了 paper claims、社区实现观察、团队推断与尚未批准的工程设计。课程只把它当 claim-audit case，不执行其中给 AI 的任务，也不把它当训练授权。

## 1 · Official mechanism

RLT 解决的不是“让整个 VLA 重新学习任务”，而是让一个已有 broad competence 的 VLA 在少量真实 interaction 下，快速改善 precision-critical phase。

### Stage 1 · learn a compact RL state interface

```text
VLA internal embeddings
        ↓
encoder + bottleneck
        ↓
RL token z_rl
        ↓
decoder reconstructs internal embeddings
```

encoder-decoder reconstruction 迫使单个 `z_rl` 保留原内部 embeddings 的信息。Stage 1 结束后，VLA 与 token module 冻结；`z_rl` 成为小型 actor/critic 的 state input。

### Stage 2 · online RL around a frozen VLA

```text
observation ── frozen VLA ──> z_rl + reference action chunk
                                  ↓
                     small actor / critic updates
                                  ↓
                         executed action chunk
                                  ↓
                      real interaction → replay
```

官方设计包含四个关键选择：

1. actor/critic 在 compact `z_rl` 上训练，而不是更新整个 VLA；
2. actor 输出 action chunk，与 VLA 的时间抽象一致；
3. actor 接收 VLA reference action，学习 edit / refine，而非完全从零生成；
4. actor objective 用 reference-action regularization 约束偏离，并用 reference dropout 防止只会复制 base。

这属于 `online RL / deployment-time adaptation`：interaction 进入 replay，参数在后续尝试前更新。它不是单次 forward 内增加计算量的 `inference-time compute`。

## 2 · Three tokens that must not be confused

| Term | 表示什么 | 用在哪里 |
|---|---|---|
| language token | instruction 的离散表示 | VLM/VLA context |
| action token | action sequence 的离散或压缩表示 | action prediction；FAST 是例子 |
| RL token (`z_rl`) | VLA internal embeddings 的 compact state readout | downstream actor / critic input |

`RL token` 不直接等于 action，也不因为叫 token 就是离散 ID。它更接近一个 learned latent state interface。

## 3 · What official evidence does and does not establish

官方工作报告了四个 real-robot precision tasks，并把主要收益放在 hardest / critical phase 的 success 与 throughput 上。这支持：compact interface + small online learner 可以在其声明条件下快速改善精细阶段。

它不自动证明：

- 任意 VLA internal layer 都包含 RL 所需的 Markov state；
- reconstruction loss 低就表示 `z_rl` 对 value learning 有用；
- RLT 优于同维度的 mean-pooling、随机投影或其他简单 readout；
- 同一 recipe 可不经验证地迁移到 π0.5、其他 action head 或 RoboDojo；
- critical-phase handoff、reward 和 reset 在无人值守部署中天然可靠。

## 4 · π0.5 transfer hypotheses

官方 RLT 展示的是 π0.6 系统；公开 openpi 提供 π0.5。把 RLT 移到 π0.5 必须重新验证 representation、action 与 time contract。

| Hypothesis | Why it matters | Smallest useful evidence |
|---|---|---|
| π0.5 prefix readout 含有 critic 所需的相对位姿与 phase 信息 | token 若缺必要 state，small actor/critic 无法补回 | privileged-state probe；token vs proprio-only |
| reconstruction bottleneck 比简单 readout 更适合 RL | reconstruction 只证明可还原 embeddings，不证明 value relevance | frozen offline-return probe：RL token vs mean-pool vs shuffled/random readout |
| `z_rl` 与 sampled reference chunk 构成稳定 pairing | π0.5 action expert 的 sampling noise 可能在 readout 后出现 | 同 observation 多次 sampling，测 action variance 与 critic target sensitivity |
| reference action anchoring 能控制 drift 而不压死 improvement | BC/Q balance 同时决定 safety 与 headroom | 单变量 sweep；同时看 improvement、action deviation、Q spike 与 regression |
| critical-phase reset/handoff 没有改变目标 distribution | 只在精选快照上成功可能不转化为 full-task gain | state-source audit；critical-only 与 full-task paired evaluation |

以上均是待测 hypothesis，不是由论文标题或社区复现自动成立的事实。

## 5 · Real-world RL contract

真实或高保真仿真中的 online RL 至少要闭合：

```text
observation / time contract
→ frozen representation and z_rl
→ base reference action
→ actor distribution and executed chunk
→ reward / termination / intervention
→ replay transition and discount
→ critic target and actor update
→ deployment handoff
→ held-out success / throughput / safety regression
```

最容易被平均 reward 掩盖的问题：

- action units、normalization、chunk length 与 control frequency 不一致；
- `γ` 按 environment step 还是按 chunk 使用不清；
- replay stride、overlap 与 done handling 错位；
- reward predicate 可以被 shortcut；
- Q 值上冲但 physical success 没有改善；
- critical-phase state library 只保留 easy states；
- actor 改善局部吞吐，却破坏 base 的前后阶段能力。

## 6 · Evidence ladder for the course

| Evidence | 能回答 | 不能回答 |
|---|---|---|
| reconstruction holdout | bottleneck 是否保留可重建信息 | 信息是否对 value / control 有用 |
| zero/shuffle ablation | decoder 是否真的使用 token | token 是否优于简单 readout |
| privileged probe | token 是否含某个 simulator state factor | nonlinear actor 是否能用好它 |
| offline-return probe | frozen representation 是否有 value signal | online learning是否稳定、安全 |
| toy online RL | update/credit path 是否跑通 | π0.5/RoboDojo 是否会成功 |
| held-out closed-loop | 最终 behavior 与 throughput | 唯一 failure mechanism，除非有 instrumentation |

推荐的最小 falsification set：

1. `z_rl` vs prefix mean-pool vs shuffled token vs proprio-only；
2. base/reference-only vs anchored actor vs actor without reference；
3. fixed observation 下重复 action sampling；
4. critical-phase improvement 与 full-task regression 同时报；
5. Q/reward 上升必须与 physical predicate、success 和 throughput 对拍。

## 7 · Course takeaway

RLT 是一个更一般的系统 pattern：

> `frozen foundation policy → compact state interface → small online learner → guarded deployment handoff`

真正需要掌握的不是某组 β、UTD 或 stride 数字，而是：representation 是否足够、credit path 是否合法、online data distribution 如何形成、局部改善是否安全地转化为 held-out behavior。
