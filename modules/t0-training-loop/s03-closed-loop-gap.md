# T0·S03 · Closed-loop gap：为什么 loss 一直降，SR 仍不及预期

> 模块 T0 第 3 节 · 前置：T0·S01/S02（复用 P-T-I-E loop 与 Arm C 的 distribution mismatch 经验）· [Pass 规则](../../learning-modes.md) · 记录用 [session-log 模板](../../templates/session-log.md) · 术语见 [glossary](../../glossary.md)
>
> 本节来历：真实工作暴露的课表覆盖缺口——VLA post-training 出现 `offline loss↓ / closed-loop SR=0`，而 `covariate shift`、`first divergence` 与 `recovery` 没有任何一节课正面教过。本节把这条诊断链造成一个可以亲手拨弄的最小系统。

## 0 · 本课在地图的位置（白话版）

S01/S02 的 classification toy 中，每个 sample 独立；当前 prediction 不会改变下一个 input。VLA 的 current action 会改变 next observation，因此需要区分两种 evaluation distribution：

- **offline holdout loss**：在 expert demonstrations 的 states 上比较 policy action 与 expert action；
- **closed-loop rollout**：真正执行 policy，后续 observations 由 policy actions 与 environment dynamics 共同产生。

两者可以反向变化，因为 offline holdout 只覆盖 `expert-state distribution`；closed-loop 中 policy 逐渐产生 `policy-induced states`。离开 `expert-state support` 后，动作误差可能扩大并继续改变 observation，形成 `covariate shift + compounding error`。白话类比是：offline 的题由 expert 产生，closed-loop 的下一道题由 policy 自己产生。

本课就造一个最小的"开车"系统亲眼看这件事，然后做三种干预。真实对应：G2.0 报告里那句"offline holdout 主要测专家状态附近的动作拟合……policy 第一次偏离后 observation 来自自身错误，offline holdout 不会惩罚恢复失败"——上完本课，这句话你可以指着自己的图讲给别人听。

**开课确认（地图看完再答；负载前确认）**：① Arm C 的 `training distribution ≠ test distribution` 机制还能讲吗？② 你工作里一次 rollout evaluation 从头到尾有哪些 components？③ `prediction horizon H` 与 `execution horizon E`（如 H16/E16）各指什么？说不出也没关系，①上手段会先补。

## ① 上手：2D corridor toy environment

**先玩演示再读字**：[闭环鸿沟演示](demos/closed-loop-gap.html)（浏览器直接打开）。它是机制卡通，不是训练出来的模型——课中 ④ 段你训的真 BC 会替代它。

Toy system（④ 段实现用）：

- **environment**：2D 平面上一条弯曲走廊；从左端到右端为 success，撞墙或 timeout 为 failure；
- **expert controller**：沿 centerline 驾驶；N 条 expert trajectories 形成 `(state, action chunk)` demonstrations；
- **expert-state support**：expert trajectories 在 corridor 中形成一条窄带，它近似表示 policy 训练时见过的 states；
- **policy**：小 MLP 输入 state、输出 action chunk，用 MSE 做 `behavior cloning (BC)`；
- **prediction horizon H / execution horizon E**：policy 一次预测 H actions，controller 只执行前 E actions 再 replan。H16/E1 每步 replan；H16/E16 执行完整 chunk 后才重新观察。

Metrics（③④ 段共用；中文列只解释直觉）：

| Metric | 直觉解释 | 它回答什么 |
|---|---|---|
| offline holdout loss | 卷面分 | 专家路面上模仿得像不像 |
| closed-loop success rate (SR) | 路考通过率 | policy 自己执行能不能到终点 |
| XY nearest-training-state distance | 离训练轨迹多远 | 当前 `(x,y)` 离最近 training state 多远；这是 OOD proxy，不是完整 state-space distance |
| first XY-support exit | 首次离开窄带 | XY distance 首次超过 threshold τ 的 timestep；不得自动写成 first divergence |
| return-to-support rate | 偏离后是否回来 | 已 exit 的 trajectories 中有多少再次进入 threshold |
| per-horizon holdout MSE | action chunk 每个位置的误差 | chunk tail 是否比 early actions 更差 |
| raw/executed action health | 输出与实际执行体检 | finite、range、mean-abs 与 clipping fraction 是否异常 |

### loss↓ / SR↓ debugging map：只记四层

这不是新增一串需要背诵的 metrics。需要记住的只有因果顺序：

```text
Metric → Contract → Distribution → Evaluation
```

| Layer | 只问一个问题 | 需要时查的 metrics / controls |
|---|---|---|
| Metric | loss 实际平均了什么？ | frozen holdout；per-task / per-phase / per-horizon；P50/P90；train 与 holdout 分开 |
| Contract | model output 是否按声明被执行？ | finite/range/saturation；normalize round-trip；joint order/unit/sign；raw chunk vs executed prefix；tracking MAE；expert-action replay |
| Distribution | policy 是否进入 demonstrations 没覆盖的 states？ | nearest-training-state distance；first out-of-support step；state strata；return-to-support / recovery |
| Evaluation | SR 是否被正确且稳定地测量？ | stage/subgoal coverage；scorer predicate；reset determinism；paired initial states；`k/n + Wilson interval` |

最短分流规则：

- `expert-action replay` 也失败：先查 Contract / Evaluation，不归因 model；
- expert states 上 action 正常，但真正执行后很快偏离：查 controller、E、latency 与 dynamics gap；
- support 内正常，exit 后无法 return：查 Distribution 与 recovery data；
- 所有 checkpoints 都是 `SR=0`：存在 floor effect，先看 stage coverage 和连续 diagnostics，不能用 0/1 SR 排 checkpoint。

本课 runner 只实现 toy 中确实可测的部分：per-horizon loss、action health、XY-support exit 和 return-to-support。真实 VLA 的 joint mapping、camera slots、controller tracking、scorer 与 exact reset 只在应用作业中审计，不伪装成 toy 已验证。

推荐 diagnosis order：interface/contract audit → fixed-input counterfactual → 约 5 条 instrumented closed-loop rollouts → single-variable intervention → 最后扩大 rollout 分母。**Large-N rollout 用于估计 SR，不用于最初找错。**

**本课公式参考表**（随时回翻，不考背诵；手算段开卷）：

| 名字 | 公式 | 白话 |
|---|---|---|
| 一步走偏 | x_{t+1} = x_t + δ | 每步动作带固定小偏差 δ，没人纠正就往同一边挪 |
| 纯累加 | x_E = E · δ | 闭眼开 E 步，偏差不折不扣加 E 次 |
| 每步纠正 | x_{t+1} = (1−k)·x_t + δ | 睁眼时先拉回一部分（k 是纠正力度），再留 δ 残差 |
| 纠正后的稳态 | x_∞ = δ / k | 拉回与新偏差打平的位置；k=1 时就是 δ 本身 |
| support-exit 判据 | \|x\| > support half-width w | 离开 expert-state support |
| OOD action error | 误差 ≈ 5δ、方向随机 | toy 对 support 外 generalization failure 的明确假设 |
| SR interval | Wilson 95% interval for k/n | 0/25 的上界 ≈13%；没有观测到 success 不等于真实 SR 恰为 0 |
| XY nearest-state distance | d_t = min over training state s′ of ‖(x_t,y_t)−s′‖ | 首次 d_t>τ 记为 first XY-support exit；不包含 heading θ |

## ② 手算自检

1D 简化：位置 x，专家永远把 x 拉回 0；模型学得不错但每步动作带固定小偏差 δ = 0.1（方向相同）。

手算两问（答案在文末折叠段）：

1. **E=1**（每步观察、纠正能力完美，只是纠正完仍留 δ 残差）：开 100 步后，偏移大概停在什么量级？**E=16**（16 步内没有任何纠正，偏差累加）：第 16 步末偏移多大？
2. 现在假设 expert-state support 的 half-width 为 1.0；support 外 action error 变成 5δ 且方向随机。E=16 的第 17 步会发生什么？用一句话描述之后的走势。

## ③ 追问（先自答，再与 AI 过）

1. offline holdout loss 在哪个 state distribution 上算？closed-loop 中实际 state distribution 由什么决定？
2. G2.0 报告说 N17 "holdout 持续改善 + 450 条全零，不支持闭环 overfitting"——用两场考试的语言解释：为什么这个组合不像过拟合？它更像什么？
3. 0/25 成功意味着真实成功率就是 0 吗？报告里"0/25 的 Wilson 95% 上界 13.32%"是在防哪种错误结论？（直觉版：25 次没抽中 ≠ 不存在）
4. 为什么报告把 `first divergence / first out-of-support event` 排在更多 rollout 前面？event 前后的 frames，诊断价值差在哪？

## ④ 任务书（在前）→ three-arm intervention（在后）

**实现约定**：AI 搭 corridor environment、expert controller、BC training 与 rollout runner（`t0_gap.py`）；**每个 Arm 的 config difference 由学员口述**。先跑 baseline：N 条 expert trajectories → 训练 BC → offline holdout loss + H16/E1 closed-loop SR。

- **Arm F · instrumentation（不是赌局）**：记录 XY nearest-training-state distance、first support exit、return to support、per-horizon MSE 与 action health；先看 5 条 failure trajectories。
- **Arm G · execution horizon**：同一 policy 在 paired initial states 上比较 E=1/4/16，各 20 rollouts。锁预测：first support exit 与 SR 随 E 如何变化，是否线性？
- **Arm H · recovery data**：采集时注入 perturbation，并记录 expert recovery actions；同 data budget 重训。锁两条预测：closed-loop SR 如何变化？原 expert holdout loss 如何变化？这只是 DAgger idea 的最小近似，不是完整 iterative DAgger。

**Explain**：每个 Arm 判完立刻写一句 mechanism explanation；最后用 `Metric → Contract → Distribution → Evaluation` 标注 loss、holdout、sentinel 与 formal rollout 各能、不能回答什么。

## ⑤ 固化

- 按本次 Pass 保存必要 evidence；`Understand` 可在 closed-loop gap 判读后停止，`Implement / Improve` 再为实际执行的 Arms 追加 ledger；
- **应用作业（计入 T0 Exit Gate）**：带着本课概念重读 G2.0 integrated review 的 §4–§5，给每条 claim 标注 `contract evidence / offline evidence / closed-loop evidence / untested`，再放进 `Metric → Contract → Distribution → Evaluation` 的相应 layer；标注稿入个人 records。

---

<details>
<summary>② 参考答案（先算再看）</summary>

```text
问 1：
E=1：每步纠正后残差 δ=0.1，不累积 → 偏移始终 ≈ 0.1 量级（100 步后仍是 0.1，不是 10）。
E=16：无纠正，δ 逐步累加 → 16 步末偏移 ≈ 16×0.1 = 1.6。

问 2：
1.6 > support half-width 1.0，第 16 步末已 outside expert-state support。第 17 步虽然重新观察了，
但此刻 state 没有被 demonstrations 覆盖 → action error 5δ 且方向随机 → correction 不可靠，
之后是"越偏 → 越没见过 → 动作越差 → 更偏"的正反馈雪崩。
这就是 compounding error：E=1 里被每步纠正掐灭的火苗，E=16 给了它 16 步自由燃烧。
Re-observation 本身不保证恢复，因为 recovery capability 只在 support 内被训练过。
```

</details>
