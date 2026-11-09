# 第一篇 · TD3：让动作越来越值得执行

> 这是直接进入一个 RL 算法的短讲义，不要求先读完 foundations。第一次只读到第 4 节；练习另开一轮。属于 T3·S02 / P4 的入口，不新增 Lesson，也不代表完成 Pass。

## 1 · 先知道这几个对象在做什么

想象一个连续控制任务：根据当前状态，决定施加多大的力。

- `actor(s)` 输出动作；它是可训练网络，不是规则表。
- `Q(s, a)` 输出执行动作后、再按当前策略继续行动的预期折扣回报。Critic 学这个估计，不只预测眼前奖励。
- 一条经验是 `(s, a, r, s2, terminal)`：旧状态、实际执行动作、奖励、新状态、任务是否真正结束。Replay 保存这些经验，训练时抽一批复用。
- `actor_targ`、`Q1_targ`、`Q2_targ` 是对应网络的慢速副本，用来算较稳定的训练目标；不通过自己的 loss 做 BP。

Actor 和两个 Critics 起初可以随机初始化；target networks 起初复制它们。接下来反复：**带探索采经验 → 从 replay 取 batch → 学 Critics → 按频率学 Actor 并同步 targets → 再采经验**。Replay 足够前先积累数据，不从空 buffer 训练。

你熟悉的监督学习在这里仍然存在：Critic 做回归，只是 label 由奖励和当前的未来估计临时构造。Actor 则不模仿一个动作标签，而是借 Critic 的梯度寻找评分更高的动作。

## 2 · 核心 update：在脑中执行一次

下面所有 `s, a, r, s2, terminal` 都来自同一个 replay batch；不是刚采到的单条经验。`γ` 是折扣系数。`.mean()` 表示对 batch 求平均。

先算固定 label，再训练两个 Critics：

```text
# target policy smoothing：下一动作附近也应该值得执行
with no_grad:
    ε  = clip(normal_noise(), -noise_clip, noise_clip)
    a2 = clip(actor_targ(s2) + ε, action_low, action_high)
    y  = r + γ * (1 - terminal) * min(Q1_targ(s2, a2), Q2_targ(s2, a2))
    # 同一个 a2，由两份 target Critics 打分；y 固定，不回传

loss_c = ((Q1(s, a) - y)^2 + (Q2(s, a) - y)^2).mean()
opt_critics.zero_grad()
loss_c.backward(); opt_critics.step()  # 只改两个 Critics；a 是历史实际动作
```

再按较慢频率训练 Actor，并同步 target networks：

```text
if critic_updates % 2 == 0:            # 这里采用常见的 delay=2
    freeze_parameters(Q1)            # 不改 Q1 参数，仍保留它对动作的梯度
    loss_a = -Q1(s, actor(s)).mean()   # 当前 Actor 重新提出动作，不是 replay 的 a
    opt_actor.zero_grad()
    loss_a.backward(); opt_actor.step()
    unfreeze_parameters(Q1)

    for online, target in network_pairs:
        target = (1 - τ) * target + τ * online  # 参数平均，不做 BP
```

这里 `network_pairs` 只含 Actor、Q1、Q2 各自与其 target 的三对参数；`τ` 是新参数混入慢副本的比例。计数按 Critic optimizer updates，而不是环境步。伪代码省略框架建图、初始化和日志，不省略学习对象及梯度边界。

**两种动作不能换：** Critic 用实际执行的 `a`，因为这条奖励和后继状态属于它；Actor 用当前 `actor(s)`，因为现在要改的是当前策略。Actor BP 经过 Q1 对动作的计算，但不经过环境，也不更新 Q1 参数。

## 3 · 一次能算清的例子

只看 batch 中的一条经验：`r=1, γ=0.9, terminal=0`。Target Actor 提出 `0.95`，给定噪声 `0.30`，noise clip 为 `0.20`，动作范围为 `[-1,1]`。

- 噪声截成 `0.20`，动作 `0.95+0.20` 截成 `a2=1.0`。
- 两个 target Critics 对这个动作打分为 `3`、`5`，取较小的 `3`。
- `y = 1 + 0.9 × 3 = 3.7`。
- 在线 Critics 对 replay 动作的当前预测若为 `2`、`4`，这条样本贡献的 Critic loss 为 `(2−3.7)²+(4−3.7)²=2.98`。

BP 改的是网络参数，不是把输出直接赋值为 3.7。共享参数改变后，其他状态动作的评分也可能变化。表格版本可以直接把一个表项往 target 移；神经网络版本通过 loss 实现逼近，不保证一步到达。

另看 Actor 的梯度：固定 Critic 为 `Q1(s,a)=−(a−0.6)²`，当前 Actor 输出 `0.2`。Actor loss 是 `(actor(s)−0.6)²`，对输出动作的导数为 `−0.8`；梯度下降在动作这个局部方向上推向 `0.6`。实际网络参数更新的大小还取决于 Actor 的导数和 optimizer，不保证每个状态的动作都同向改善。

## 4 · TD3 的三个选择，各解决什么

- **Twin Critics**：target 取两份估计的较小值，缓解过高估值被反复放大；不意味着消灭偏差，也不意味着取平均。
- **Target policy smoothing**：算 target 时在下一动作附近加有界噪声，减少利用狭窄的错误高分峰；它和采经验时的探索噪声不是同一用途。
- **Delayed Actor / target updates**：Critics 更频繁学习，Actor 与慢副本较慢更新，减少策略追着不成熟评分变化；不是 Actor 永远等待 Critic 收敛。

到这里可以停。你应该能沿一次 update 指出：谁提供数据、谁造 label、谁做 BP、谁只是参数平均。不必今天就证明它能收敛或跑出高 return。

## 5 · 下一轮只做一个小练习

保持第 3 节所有数字不变，只把 `terminal` 改成 `1`。运行前先说 y 会怎样变、为什么；再算一次核对。最后用自己的话解释：为什么 Actor loss 不能直接复用 replay 的动作？

题目只改变一个条件；答案及后续提问放在[导师问答稿](td3-first-lesson-tutor.md)，第一次阅读不必打开。真正 terminal 时不 bootstrap；如果只是外部时间限制打断仍可继续的任务，则从 reset 前的 final observation bootstrap，不能把 reset 的 observation 当 s2。

后续按[已有 P4](rl-control-practice.md#p4--td3旧数据怎样给-actor-一个方向)逐层进入：亲手实现上述核心 → 固定输入与梯度检查 → 校准参考训练 → 控制任务诊断 → 一次小改进与回归对照。**读懂本篇不是工程能力毕业证**；这些实践可以分多次 session，不作为本篇的入场考试。已有材料的 PPO-first 路线继续保留，本篇先作为 TD3-first 的试读入口，不要求先完成其他两个算法。

## 参考与证据边界

算法依据：[TD3 原论文](https://arxiv.org/abs/1802.09477)、[Spinning Up TD3](https://spinningup.openai.com/en/latest/algorithms/td3.html)。这里采用常见的 Actor 与 targets 同频延迟更新约定。以上数字为教学 fixture，不是训练结果；未据此声称算法在某控制任务上有效。PPO 的 Actor 更新不是沿 Q 对动作求导，不能直接套用本篇的 Actor loss。
