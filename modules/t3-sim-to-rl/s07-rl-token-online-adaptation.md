# T3·S07 · Online adaptation：RL Token 与 critical-phase real-world RL

> 模块 T3 第 7 节 · 前置：T0·S03、T3·S02、T3·S06；推荐已完成 A3·S05 · [Pass 规则](../../learning-modes.md) · [课程精编](../../references/rlt-pi05-robodojo-digest.md) · [两阶段系统图](demos/rl-token-system.html) · 记录用 [session-log 模板](../../templates/session-log.md)
>
> 本课只做 toy implementation 与 evidence audit，不授权 π0.5、RoboDojo、Isaac 或真机训练。真实运行仍须回到根仓库的 training spec、registry 与审批边界。

## 0 · 本课地图

一个 generalist VLA 可能已经会完成任务的大部分阶段，却在最后几毫米的接触、对齐或插入上又慢又不稳定。直接对整个 VLA 做 online RL 很贵，也可能破坏已有能力。RLT 的解法是：让 frozen VLA 输出 compact `RL token`，再让一个小 actor/critic 在 critical phase 根据真实 interaction 快速学习。

本课共同问题：

> 怎样让一个大而通用的 frozen policy，在部署过程中通过小型 online learner 安全地改善局部精细能力？

三个 Pass：

- `Understand（看懂）`：分清 RL token、action token、inference-time compute、online RL 与 real-world RL，能画出 data/update path；
- `Implement（做出）`：在 precision-phase toy 上接入 frozen representation、reference action 与 small off-policy actor/critic；
- `Improve（改好）`：判断 token 是否真的提供 value-relevant information，并审计 π0.5 transfer、handoff、reward hacking 与 regression。

自然停止点可以是：画完系统图、跑通 toy baseline、完成一个 representation ablation，或交付一份 transfer audit；不要求一次完成整课。

## 1 · 先划清四个概念

### RL token is a state interface, not an action token

- `action token` 表示要执行的动作或 action chunk；
- `RL token (z_rl)` 是从 VLA internal embeddings 读出的 compact latent state，供 actor/critic 使用；
- `reference action chunk` 是 frozen VLA 原本会执行的动作建议；small actor 学的是 refine / edit。

### Online RL is not ordinary inference

```text
ordinary inference:
observation → fixed parameters → action

online RL / deployment-time adaptation:
interaction → replay → parameter update → later interaction changes
```

RLT 可以发生在 deployment 期间，但它有跨尝试的 parameter update，因此不应简写成“单次 inference 时多算一会儿”。

### Critical-phase handoff is part of the policy

如果 base 负责长任务的 broad phase，RL actor 只负责 precision phase，那么“何时交接、谁判断交接、交接 state 来自什么 distribution”都属于 system contract，而不是外围工程细节。

### Real-world RL optimizes throughput under constraints

只看 binary SR 会漏掉速度、reset 成本和安全代价。真实部署至少要同时看 success、steps-to-success / throughput、action health、intervention/reset cost 与 base-capability regression。

## 2 · Stage 1：训练 compact RL state

RLT 在 VLA internal embeddings 上增加 encoder/bottleneck/decoder：

```text
embeddings Z ── encoder ──> z_rl ── decoder ──> reconstructed Z_hat
```

课程抽象使用：

\[
\mathcal L_{recon}=\frac{1}{N}\sum_i\lVert \bar z_i-\hat z_i(z_{rl},\bar z_{<i})\rVert_2^2,
\]

其中 reconstruction target stop-gradient。这个 objective 只证明 bottleneck 保留了“可重建 internal embeddings 的信息”，不证明信息对 return、Q 或 control 有用。

所以 Stage 1 不能只看 train reconstruction loss；至少需要 held-out reconstruction、zero/shuffle ablation，以及 downstream value/state probes。

## 3 · Stage 2：围绕 reference action 做 small online RL

课程中的 actor 输入为：

\[
x_t=(z_{rl,t},\;q_t,\;\tilde a_t),
\]

其中 `q_t` 是 proprioception，`\tilde a_t` 是 frozen VLA 的 reference chunk。actor 输出同一 action space 中实际候选 chunk `a_t`。

一个便于审计的 actor objective 抽象为：

\[
\mathcal L_{actor}
=-\lambda_Q\min(Q_1(x_t,a_t),Q_2(x_t,a_t))
+\beta\lVert a_t-\tilde a_t\rVert_2^2.
\]

- Q term 提供 improvement pressure；
- BC/reference term 限制 policy drift；
- `β` 太大可能只会复制 base，太小可能出现 unsafe deviation 或 reward hacking；
- reference dropout 检查 actor 是否拥有独立 pathway，而不是把 reference 原样透传。

若一个 RL action 包含 `C` 个 environment steps，则 TD target 的时间单位必须写清：

\[
y_t=R_t^{(C)}+\gamma^C(1-d_t)\min_j Q_j^{target}(x_{t+C},a_{t+C}).
\]

例如 `γ=0.99, C=10` 时，bootstrap discount 是 `0.99^{10}≈0.904`，不是 `0.99`。chunk reward aggregation、done 位置与 replay stride 必须使用同一时间合同。

### 本课公式参考表（开卷）

| Formula | 白话用途 |
|---|---|
| `L_recon = mean(||Z_stopgrad - Z_hat(z_rl)||²)` | 检查 compact bottleneck 能否重建 VLA information；不能单独证明 value relevance |
| `L_actor = -λ_Q min(Q1,Q2) + β ||a-ref||²` | 一边推动高 value action，一边限制 policy 偏离 base reference |
| `R_t^(C) = Σ_(k=0)^(C-1) γ^k r_(t+k)` | 把一个 action chunk 内的 environment-step rewards 聚合到同一 transition |
| `y_t = R_t^(C) + γ^C (1-d_t) min(Q1_target,Q2_target)` | 以 chunk 为 replay 单位时，bootstrap 也必须跨过 `C` 个 environment steps |
| `throughput = successes / wall-clock time` | 把速度、失败重试和 reset 成本纳入部署结果，而不只看 binary SR |

## 4 · Understand Pass

### Worked case

已知系统：

- VLA frozen；
- `z_rl` module frozen；
- actor/critic 每轮从 replay 更新；
- actor 收到 reference chunk，并在 critical phase 接管。

完成一张 data/update-path 图，明确：

1. observation 怎样进入 `z_rl`；
2. reference action 从哪里来；
3. reward 怎样进入 critic target；
4. 哪些 parameters 会跨 rollout 改变；
5. handoff predicate 是否属于 evaluation distribution 的一部分。

判读门槛：能够说明“reconstruction loss 下降”为什么不能推出“online RL 会学得更好”。

## 5 · Implement Pass：precision-phase toy

### Toy contract

环境只模拟一个已经到目标附近的精细对齐阶段：

- latent physical state：relative pose `(dx, dy, dθ)`；
- proprioception 只含 robot joints 的低维 proxy，不直接给 object-target relative pose；
- frozen synthetic VLA encoder 把 relative pose、proprioception 与 distractors 映射成高维 embeddings；
- base reference policy 能靠近目标，但在最后对齐存在固定 bias；
- sparse reward 只在 pose 进入 success tolerance 时为 1；dense pose error 只作 diagnostic，不进入主 objective。

复用 T3·S02 的 replay、twin critic 和 actor update，不重新发明整套 SAC。AI 负责环境、frozen encoder、plots 与 asserts；学员负责定义 actor inputs、chunk TD target 和唯一 intervention。

### Minimal arms

| Arm | Actor input | 问题 |
|---|---|---|
| A · base | reference chunk only，不训练 | base headroom 是多少？ |
| B · proprio | proprio + reference | 不看视觉 latent 能否改善？ |
| C · mean-pool | pooled frozen embeddings + proprio + reference | 简单 readout 是否已经足够？ |
| D · RL token | `z_rl` + proprio + reference | bottleneck 是否带来额外 sample efficiency？ |

先展示完整 seed、reward、chunk、update 与 evaluation design，再选择一个 representation comparison 锁 prediction。必须同时报告 learning curve、held-out SR / throughput、action deviation 与 Q health。

Toy 只验证 mechanism chain 和 instrumentation；它不能证明 π0.5/RoboDojo 会受益。

## 6 · Improve Pass：π0.5 transfer audit

以 [课程精编](../../references/rlt-pi05-robodojo-digest.md) 和用户提供的工程 draft 为 case，依次检查：

1. **Representation**：π0.5 选定 layer 是否包含 relative pose、phase 与 failure information？
2. **Action contract**：`z_rl`、sampled reference chunk、actual executed prefix 的 noise/time pairing 是否明确？
3. **Credit**：chunk reward、`γ^C`、done 与 replay stride 是否一致？
4. **Handoff**：critical-phase entry 是人工、oracle predicate 还是 learned head？其 errors 怎样进入 evaluation？
5. **Safety**：BC/reference anchoring、clipping、intervention 与 rollback 能否限制 exploration？
6. **Evidence**：token 是否胜过 mean-pool、random/shuffled token 与 proprio-only？
7. **Regression**：局部 throughput 上升时，full-task SR 与 base 前后阶段是否保持？

最小 falsification：固定同一批 states，对 `{RL token, mean-pool, shuffled token, proprio-only}` 做 offline-return probe；若 RL token 不优于 mean-pool，就不能把 encoder-decoder bottleneck 当作进入昂贵 online pilot 的必要条件。

### Claim audit

| Claim | Conceptual verdict | Evidence status | 最小补证据 |
|---|---|---|---|
| reconstruction holdout 下降，说明 token 已适合 RL | Conceptually wrong | Refuted by objective mismatch | downstream value/control probe |
| π0.5 readout 与 action-expert sampling noise 解耦会伤害 critic | Plausible but vague | Untested | fixed-observation repeated sampling + target-sensitivity test |
| sim oracle 可以无损替代 reward、intervention 与 handoff 三个人类角色 | Plausible but vague | Untested | 分别定义三种 contract，再做 oracle-error / distribution audit |

## 7 · Pass completion

- `✅ U`：能画对两阶段 data/update path，分清三个 token 与 online/inference 边界，并解释 reconstruction 的证据上限；
- `✅ I`：toy 至少三 Arms 跑通；有一次预注册 representation intervention、ledger 判定和 held-out behavior evidence；
- `✅ M`：完成 π0.5 transfer audit，给出至少一个区分性 falsifier、一个 safety/contract negative control 和 full-task regression plan。
- 替代路径：`✅ I` / `✅ M` 也可用学员在根仓库已授权 RLT 工作中的真实证据完成，要求与 toy 相同的 P-T-I-E 结构；`Understand` 层可由 [RL 基础讲义](rl-foundations-primer.md) 知识点 30–33 承担（见[课表中的对应说明](../../curriculum.md#rl-基础讲义与课次的对应t2t3-共用)）。

## 8 · 连接 Capstone

本课只提供一种 selective adaptation pattern，不把 RLT 预设为最终方案；同一问题的另一种答案（Q 只在冻结 VLA 的候选中做选择）见 [T3·S06 EXPO select + edit](s06-expressive-policy-value-maximization.md)。T3·S08 必须比较：更新 Action Expert、外挂 small actor/critic、offline advantage-conditioned distillation 或拒绝进入 RL pilot。选择哪条路线由 task failure distribution、estimator legality、deployment constraints 与最小证据共同决定。

## 9 · 从真实实现问题切入的 session routing

如果 actor / critic / Q-function / TD3 的角色关系还不清晰，先用 [Actor–critic foundations](actor-critic-foundations.md) 完成基础衔接；不要直接发布 chunk TD target 或 Oracle replay 诊断题。

学员已有相关实现或运行记录时，可以先进入本课的局部诊断 session；这不等于跳过或完成 T3·S02 / S06 的前置能力。先讲清所需的 replay、Q、bootstrap 概念，再做对应的小验证；完整算法实现与 Pass gate 仍按上文执行。

### 三个维度，不是一条阶段序列

- Inference-time compute：在参数固定时增加候选采样、搜索或规划等计算；不因为用到了已训练 critic 就等于在线 RL。
- Test-time / deployment-time RL：测试或部署期间从反馈构造学习信号并更新参数；必须说明哪些参数更新、更新是否跨尝试保留、评测是否关闭更新。该名称有多种具体实现，不能直接等同于某一篇 TTRL 方法。
- Real-world RL：强调 interaction 来自真实物理系统，可在专门训练期发生，也可发生在部署期；sim 中 online RL 不因此变成 real-world RL。
- RLT 的 online 阶段冻结 VLA 与已训练 token module，更新 small actor/critic；后续 fixed-checkpoint evaluation 可关闭更新。Online describes data collection/update timing; off-policy describes reuse of experience collected under other behavior policies. 两者可以同时成立。

来源：[PI RLT](https://www.pi.website/research/rlt)；[TTRL 原论文](https://arxiv.org/abs/2504.16084)作为 test-time parameter update 的另一类例子，不作为机器人实现配方。

### 短 session A：一条真实 transition 的学习信号

目的：区分 reference action、actor proposal、executed action、correction BC target；弄清 reward 属于哪段执行，以及 TD target 应跨过多久。

AI 先展示一条动作与更新时间线，逐个定义上述字段。学员口述 replay 记录及 TD target；AI 只提供 CPU harness、显示结果与验收。先给完整设计，再锁预测，不要求学员先重写 actor-critic。

可用合成例：actor 预测 C10、只执行2步，随后 Oracle 从新的实际 observation 执行纠正。两段必须成为各自真实的 transition；不能把后段 Oracle 的 action 或 reward 回填为前段 actor 自主执行的事实。

最小验证设计：固定 gamma=0.99、首段2步的 rewards=[0,0]、nonterminal、next_Q=0.8；比较按 valid_length=2 与错误地按 predicted_chunk_length=10 计算的 bootstrap。再验证 terminal transition 不 bootstrap。该例为合成教学数据，不复写 PR 的运行证据，也不预判现有实现存在这个 bug。

自然停止点：学员能解释数值差异，说明哪些字段用于 critic、哪些字段用于 BC。一次小验证不等于本课 Implement 全部完成。

### 短 session B：critic 的乐观与 actor 的偏离

在 A 的 Q/return 定义清楚后，用固定 replay 与 actor/reference 候选检查 replay-Q 与 actor-Q 的区别。若真实 reward 每 episode 最多一次1且无其他 reward 项，非负 discounted return 有上界1；超过上界的 learned Q 是诊断信号，但数值不越界并不证明估值准确。学员选择一个 negative control，再观察 Q term 与 BC anchoring 如何共同改变更新方向。

### 短 session C：纠正数据、可观测性与任务可达性

区分 assisted success、Oracle-OFF actor success、动作拟合误差和几何 proxy。先审查同一状态的纠正目标、执行后果与 held-out control，再选择 representation ablation 或数据覆盖验证；不把所有失败都解释为 token 无效或 Oracle 正确。真实 mesh / simulator 限制只引用原报告，不从静态图直接推出动态必败。
