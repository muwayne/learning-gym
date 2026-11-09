# TD3 · 从一次交互到可部署的 Actor

**TD3 原论文** · Fujimoto、van Hoof 和 Meger，ICML 2018。[论文页面](https://proceedings.mlr.press/v80/fujimoto18a.html) · [Algorithm 1 与式 (15)，PDF 第 6 页](https://proceedings.mlr.press/v80/fujimoto18a/fujimoto18a.pdf#page=6)

机械臂能用任意大小的合法推力。**TD3 要解决的是，怎样从试过的推力和环境反馈里，学出下一次该怎么推。** Critic 先学会估计一次推力的回报，Actor 再用这个估计调整自己的输出。讲义先讲这两者怎样合作，再用代码和滑块例子解释为什么训练目标需要控制变化速度。

### Reading map · challenge → lineage → execution

The challenge is to learn a useful continuous action from real transitions, even though the Actor has no supervised “best-action” label. The Critic learns a return estimate; its action derivative then supplies the Actor's local improvement direction. An estimated improvement is not a guarantee of better real return.

**DPG → DDPG → TD3:** DPG supplies the deterministic policy gradient; DDPG uses neural Actor/Critic networks, replay and target networks; TD3 retains that setup and adds twin-target minimum, target policy smoothing and delayed Actor/target updates. The [full comparison](#dpgddpg-和-td3-对照) is optional on a first read.

Read the environment and mechanism first (§0–1), then the complete pseudocode (§2), then only the block answering your current question (§3). Use the debugger for the detailed numeric trace and finish with inference (§5). No need to memorize all weights or study every tensor on the first pass.

## 0 · 先看环境和一步 TD 更新

机械臂沿一条直线推滑块。这里的**环境（environment）**就是模拟滑块的程序。训练程序选择一次推力，把它交给 `env.step(a)`。环境按运动规则算出滑块的新位置和速度，再按奖励规则给出这一步的奖励，并报告这局是否结束。新一局开始时，`env.reset()` 提供起始观察。

| 量 | 在滑块里指什么 |
| --- | --- |
| `s` | 施加推力前的观察，即滑块的位置和速度 |
| `a` | 这次实际施加的推力 |
| `s_next` | 环境执行推力后返回的观察 |
| `r` | 环境为这次移动给出的**即时奖励（reward）** |
| `terminated` | 任务是否真正结束。这里指是否到达终点 |

这五项组成一条 **transition**，存入 **replay buffer**。训练时可以反复抽取同一条记录。一次抽取的多条记录叫 **batch**。

### 滑块环境的完整设定

| 项目 | 规则 |
| --- | --- |
| 起点 | `s = [p, v] = [0.3, 0]`。`p` 是位置，`v` 是速度。 |
| 动作 | `a ∈ [−1, 1]`，是这个 toy 里的归一化推力。正值朝终点方向推。 |
| 每步怎么动 | 先算 `v_next = 0.5 × v + 0.5 × a`，再算 `p_next = p + v_next`。环境返回 `s_next = [p_next, v_next]`。 |
| 终止和 reward | 若 `p_next ≥ 0.95`，本步给 `+1` 并终止。其余每步给 `−0.1` 并继续。这个例子没有额外的步数上限。 |

速度决定下一位置，位置决定有没有越过终点。**只要还没到终点，无论离终点多近、速度多快，当步的 reward 都是 `−0.1`。** 这笔每步成本让更快到达终点的轨迹获得更高回报。

Debugger 用两次预先选定的推力演示这个规则。第一次推 `0.6`，从 `[0.3, 0]` 到 `[0.6, 0.3]`，所以得到 `−0.1`。第二次推 `0.5`，到 `[1.0, 0.4]`，越过终点，得到 `+1`。这两条经验就是后面抽出的 batch。

### Critic 在预测什么

把这一次的奖励和之后按策略继续行动能拿到的奖励加起来，越往后的奖励权重越小。这叫 **return（回报）**。Critic 的输出 `Q(s, a)` 回答「在观察到 `s` 时先做 `a`，之后按策略继续，预计能拿到多少回报？」

### 一步 TD 目标从哪里来

TD(0) 用一次实际交互更新价值预测。先拿环境已经给出的奖励 `r`，再加上从下一状态开始的回报估计，乘折扣因子 `γ`。状态价值 `V(s)` 表示从状态 `s` 开始按策略走的预期回报。预测 `V` 时，这一步的训练目标是 `r + γ × V(s_next)`。用这个目标减去当前预测 `V(s)`，得到 **TD error**，再按这个差值修正 `V(s)`。本例 `γ = 0.9`，到达终点时，训练目标只保留这次的 `r`。

TD3 用的是同样的 **一步 TD 目标（one-step TD target）**，但要训练的是状态与动作的价值 `Q(s, a)`。现在估计未来时，还要指定在下一状态准备做哪个动作。把下一状态和这个动作交给评分网络，得到后续回报的估值，再接到已知的 `r` 后面。代码把一批经验的一步 TD 目标记作 `Y`。

| 阶段 | 谁做什么 |
| --- | --- |
| **02/03 交互** | 环境实际执行动作，返回观察、即时奖励和终止标记。训练程序把经验存进 replay。 |
| **05 Target 1** | 训练程序把记录中的 reward 和从下一状态开始的估值合成一步 TD 目标。 |
| **Critic 1 · forward / backward / step** | 两个 Critic 对原来执行的 `(s, a)` 做预测，用上一步算出的 TD 目标做回归。 |

读完原理后，可以用[完整训练流程](#2--完整训练流程)定位代码，再按[逐块走读](#3--逐个-block-走读)跟一条实际经验。想了解三代算法的演进，可以单独读[DPG、DDPG 和 TD3 对照](#dpgddpg-和-td3-对照)。[TD3 Training Debugger](training-debugger.html) · [P4 实践与验收](../rl-control-practice.md#p4--td3旧数据怎样给-actor-一个方向)

## 1 · 原理 · Critic 怎样教 Actor 选动作

### Critic 从环境反馈学评分

一次交互能确定当步 reward。要训练 Critic 预测整段回报，就用上一节的一步 TD 目标，把当步 reward 接到下一步的估值上。Critic 对比的是**同一条经验的两个数**。一个是它对原状态和原动作的预测，另一个是根据这条经验构造的回归目标。

### Actor 用评分的斜率调整推力

知道推力 `0.6` 的预测回报有多高，还需要知道**把推力往哪个方向微调，会让这个预测更高**。确定性 Actor 为每个状态输出一个动作。训练时，用它在同一个出发状态重新提议的动作来问 Critic「如果推力稍大一点，你的评分会升还是会降？」这个斜率就是在 Actor 刚提议的动作上求出的 `∂Q/∂a`。训练通过链式法则把这个方向传回 Actor 权重。这个更新规则叫 **deterministic policy gradient**。 [DPG 原论文第 3.2 节](https://proceedings.mlr.press/v32/silver14.pdf#page=4)

### TD3 控制错误估值怎样传下去

如果 Critic 给某个动作报了虚高的分数，Actor 可能被这个高分吸引。这个高估还可能进入下一轮的一步 TD 目标，把误差传给更早的动作。

训练时，把 optimizer 直接更新的 Actor 和 Critic 称为 **online networks**。DDPG 另存它们的参数副本，叫 **target networks**。副本负责给出下一动作和后续估值，每次同步只往 online 参数靠一点。这样，Critic 刚改了自己的预测，下一次用来训练它的目标也不会立即跟着大幅跳动。TD3 保留这些副本，再在三个位置限制错误估值的传播。 [DDPG 原论文第 3 节](https://arxiv.org/pdf/1509.02971#page=4)

| 机制 | 改在哪 | 为什么改 |
| --- | --- | --- |
| **Clipped Double Q-learning** | 对同一个下一动作，用两个 target Critic 评分，取较小值。 | 一个 Critic 的偶然高分更难抬高训练目标。 |
| **Target policy smoothing** | 给用于估值的下一动作加有界扰动。 | 让目标反映动作附近的表现，降低对狭窄高分点的依赖。 |
| **Delayed policy updates** | Critic 更新几次，Actor 和 target networks 才更新一次。 | 先给 Critic 几次拟合机会，再让 Actor 根据它的评分调整，同时限制负责出标签的副本变化频率。 |

三项改动分别落在[一步 TD 目标](#td-target)和[更新间隔](#delay-gate)。[TD3 原论文第 4.2、5.2–5.3 节](https://proceedings.mlr.press/v80/fujimoto18a/fujimoto18a.pdf#page=4)

## 2 · 完整训练流程

这是为本课编写的 **vanilla TD3 教学伪代码**。核心更新来自原论文 Algorithm 1。论文用数学式写 Critic 回归、Actor 的 deterministic policy gradient 和 target 参数平均，这里把它们展开成代码块。每个 block 的第一行说明它要完成什么。下方完整训练流程和第 5 节推理代码的行号与 Debugger 对应。 [TD3 原论文 Algorithm 1 和式 (15)](https://proceedings.mlr.press/v80/fujimoto18a/fujimoto18a.pdf#page=6)

| Block | 训练伪代码行号 | Debugger 步骤 |
| --- | --- | --- |
| [Initialize](#initialize) | 1–6 | 01 初始化 |
| [Collect](#collect) | 8–16 | 02/03 交互 |
| [Sample](#sample) | 18–21 | 04 抽取 batch |
| [TD target](#td-target) | 22–27 | Target 1 / Target 2 |
| [Critic update](#critic-update) | 29–34 | Critic forward → backward → step（两轮） |
| [Delay gate](#delay-gate) | 36–37 | Delay gate · 跳过 Actor |
| [Actor update](#actor-update) | 38–44 | Actor forward → backward |
| [Actor step](#actor-step) | 46–48 | Actor step · 写回 |
| [Target sync](#target-sync) | 50–55 | Target sync · 同步 targets |

第 21 行从 replay 抽一批经验。`S, A` 的同一行是 Critic 要打分的原状态和实际动作，`R, S_next, D` 的同一行是这次动作得到的 reward、到达状态和终止标记。同一条经验在这五组变量中的行号一致。

```text
 1 │ # Initialize · 建立可训练网络、用于算标签的副本和经验库
 2 │ Actor, Q1, Q2 = init_online_networks()                 # 创建由 optimizer 更新的三个网络
 3 │ Actor_target, Q1_target, Q2_target = copy_networks(Actor, Q1, Q2)  # 各复制一份结构和权重
 4 │ replay = empty_replay()                                 # 准备存放环境交互记录
 5 │ critic_updates = 0                                      # 按 Critic optimizer.step 的次数计数
 6 │ s = env.reset()                                         # 获得当前 episode 的初始观察
 7 │
 8 │ repeat:                                                 # 每次先采一条经验，再考虑更新
 9 │     # Collect · 在环境里执行一次动作，把真实反馈存进 replay
10 │     if warmup:                                          # 初期先收集足够的交互
11 │         a = random_action()                             # 从合法动作范围抽一个实际推力
12 │     else:                                               # warmup 结束后由 online Actor 提议
13 │         a = clip(Actor(s) + exploration_noise, action_bounds)  # 适度探索，并限制实际推力
14 │     s_next, r, terminated, truncated = env.step(a)       # 执行 a，记录奖励、后继观察与结束原因
15 │     replay.add(s, a, r, s_next, terminated)               # 存实际执行的 a 和 reset 前的 s_next
16 │     s = env.reset() if terminated or truncated else s_next  # episode 结束就重置，否则沿后继状态继续
17 │
18 │     # Sample · 经验够了就抽一批旧记录，作为这次更新的输入
19 │     if not replay.ready(): continue                      # 经验数量不足时继续收集
20 │
21 │     S, A, R, S_next, D = replay.sample_batch()            # 抽出一批状态、历史动作、奖励、后继状态、终止标记
22 │     # TD target · 用真实奖励和下一步的估值，为每条经验构造训练标签 Y
23 │     with no_grad:                                        # 只算固定标签 Y，这段运算不进入反传图
24 │         ε = clip(sample_target_noise(), -noise_clip, noise_clip)  # 限制加到下一动作上的噪声
25 │         A_next = clip(Actor_target(S_next) + ε, action_bounds)    # target Actor 给每条后继状态提议动作
26 │         Q_next = min(Q1_target(S_next, A_next), Q2_target(S_next, A_next))  # 两个 target Q 逐行取较小值
27 │         Y = R + γ * (1 - D) * Q_next                      # 即时奖励加折扣后的未来估值，终止时只保留奖励
28 │
29 │     # Critic update · 让两个 online Critic 对历史动作的预测拟合同一个 Y
30 │     loss_c = mean((Q1(S, A) - Y)^2 + (Q2(S, A) - Y)^2)    # 两个 online Critic 用历史动作拟合相同的 Y
31 │     critic_optimizer.zero_grad()                        # 清掉上次留下的 Critic 梯度
32 │     loss_c.backward()                                   # 求两个 online Critic 的参数梯度
33 │     critic_optimizer.step()                             # 按梯度更新两个 online Critic
34 │     critic_updates += 1                                  # 记下这次已经执行的 Critic 更新
35 │
36 │     # Delay gate · 按 Critic 更新次数决定何时更新 Actor 和 target networks
37 │     if critic_updates % policy_delay == 0:              # 每隔几次 Critic 更新，再进入 Actor 分支
38 │         # Actor update · 用 Q1 对当前 Actor 提议的动作打分，把梯度传回 Actor
39 │         freeze(Q1.parameters())                         # 冻结 Q1 权重，梯度仍可经过动作输入回到 Actor
40 │         A_policy = Actor(S)                             # 当前 Actor 对这批旧状态重新提议动作
41 │         Q_policy = Q1(S, A_policy)                       # 用刚更新的 Q1 给这些新动作打分
42 │         loss_a = -Q_policy.mean()                        # 用负均值把最大化 Q 转成最小化 loss
43 │         actor_optimizer.zero_grad()                      # 清掉上次留下的 Actor 梯度
44 │         loss_a.backward()                                # 梯度从 Q1 的动作输入继续传回 Actor
45 │
46 │         # Actor step · 用刚算出的梯度写回 Actor 参数
47 │         actor_optimizer.step()                          # 写回新的 Actor 参数
48 │         unfreeze(Q1.parameters())                       # 让 Q1 在下次 Critic 更新时继续学习
49 │
50 │         # Target sync · 把新 online 参数少量混入副本，让下次算 Y 的参数缓慢跟进
51 │         with no_grad:                                    # 不记录这次参数写入的梯度图
52 │             for online, target in ((Actor, Actor_target), (Q1, Q1_target), (Q2, Q2_target)):  # 每个 online 网络对应自己的副本
53 │                 paired_parameters = zip(online.parameters(), target.parameters(), strict=True)  # 按顺序配对这两个网络的参数
54 │                 for online_param, target_param in paired_parameters:  # 逐对读取对应的参数张量
55 │                     target_param.copy_((1 - τ) * target_param + τ * online_param)  # 直接写回 target 权重，无需 optimizer
```

| 读代码时会遇到的差异 | 本课与参考实现的写法 |
| --- | --- |
| 何时探索、何时开始训练 | 这里用 `warmup` 控制随机探索，用 `replay.ready()` 控制何时能开始更新。原作者实现用同一个环境步数阈值，先执行随机动作，再切换到 Actor 加探索噪声并开始训练。 [作者的采集循环](https://github.com/sfujim/TD3/blob/master/main.py#L103-L126) |
| 收集与更新的节奏 | 本课在 replay 就绪后每采一条经验做一次 Critic update。Spinning Up 默认每收集 50 步再集中做 50 次更新。两种写法都可以按 Critic 更新次数做 delayed policy updates。 [Spinning Up TD3](https://spinningup.openai.com/en/latest/algorithms/td3.html) |
| 工程接口 | 这里用 `D` 表示真正终止，和 Gymnasium 的 `terminated` 对应。外部时限截断时，从 reset 前的观察继续 bootstrap。Actor 更新中显式冻结 Q1 的参数以省掉这些参数的梯度。原作者实现的 Actor optimizer 也只写回 Actor。 [Gymnasium 的终止处理](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/) · [作者的更新代码](https://github.com/sfujim/TD3/blob/master/TD3.py#L98-L142) |

**TD3+BC** 是算法变种的一个例子。它直接从固定数据集抽样，省去在线 Collect，并给 Actor 的目标加上 behavior cloning 项，让 Actor 在追求高 Q 的同时贴近数据中的动作。论文还对输入状态做了归一化。按上面的 block 对照，变化最明显的是 Collect 的数据来源和 Actor update 的 loss。 [TD3+BC 论文 §5](https://arxiv.org/html/2106.06860#S5)

## 3 · 逐个 block 走读

每块先解释它解决哪一个训练问题，再贴相关代码。滑块数据只在需要区分两个量或解释一个结果时出现。行号沿用完整训练流程，Debugger 的阶段名用于核对执行（新增细分阶段后，不依赖旧步号）。

### Initialize

把学习用的网络与出标签用的网络分开，才能控制两者各自变化的速度。初始化时，先把每个 online 网络复制一份，让这一对从相同参数起步（Debugger 01）。

```text
 1 │ # Initialize · 建立可训练网络、用于算标签的副本和经验库
 2 │ Actor, Q1, Q2 = init_online_networks()                 # 创建由 optimizer 更新的三个网络
 3 │ Actor_target, Q1_target, Q2_target = copy_networks(Actor, Q1, Q2)  # 各复制一份结构和权重
 4 │ replay = empty_replay()                                 # 准备存放环境交互记录
 5 │ critic_updates = 0                                      # 按 Critic optimizer.step 的次数计数
 6 │ s = env.reset()                                         # 获得当前 episode 的初始观察
```

第 2 行建立 online Actor 和两个独立的 online Critic。第 3 行把每个网络的初始参数复制一份。比如 `Q1_target` 从 `Q1` 复制，对同一个输入，两者初始打分相同。对这一对网络，第 33 行只改 online 的 `Q1`，第 55 行才会改 `Q1_target`。第一次构造一步 TD 目标用的就是这些初始副本。

第 4 行的 replay 此时是空的，第 5 行的计数也是零。第 6 行向环境取到 `s = [0.3, 0]`，把它交给 Collect 选第一下推力。

### Collect

这一步把环境给出的 reward 和到达状态固定下来。哪怕下一步要 reset，训练也得保留这次真正到达的状态（Debugger 02/03）。

```text
 8 │ repeat:                                                 # 每次先采一条经验，再考虑更新
 9 │     # Collect · 在环境里执行一次动作，把真实反馈存进 replay
10 │     if warmup:                                          # 初期先收集足够的交互
11 │         a = random_action()                             # 从合法动作范围抽一个实际推力
12 │     else:                                               # warmup 结束后由 online Actor 提议
13 │         a = clip(Actor(s) + exploration_noise, action_bounds)  # 适度探索，并限制实际推力
14 │     s_next, r, terminated, truncated = env.step(a)       # 执行 a，记录奖励、后继观察与结束原因
15 │     replay.add(s, a, r, s_next, terminated)               # 存实际执行的 a 和 reset 前的 s_next
16 │     s = env.reset() if terminated or truncated else s_next  # episode 结束就重置，否则沿后继状态继续
```

第 10–13 行决定**实际要执行的推力**。Warmup 用随机动作收集经验，之后用 online Actor 的动作加探索噪声。第 14 行是这次实际交互。环境执行推力，按滑块的运动和 reward 规则返回结果。

第 15 行先把这次交互完整存下来，第 16 行再决定下一次从哪开始。第二次推到终点时，replay 保存的是 `s_next = [1.0, 0.4]` 和 `terminated = 1`。训练循环随后 reset，下一次用于选动作的 `s` 才回到 `[0.3, 0]`。若是外部截断，也要 reset，但 `terminated` 仍为 0，所以第 27 行仍能从截断前的到达状态估计未来。

### Sample

一次训练更新可以复用过去的交互。抽到的一条记录同时提供 Critic 的输入和构造训练目标所需的环境反馈，两边必须按样本对齐（Debugger 04）。

```text
18 │     # Sample · 经验够了就抽一批旧记录，作为这次更新的输入
19 │     if not replay.ready(): continue                      # 经验数量不足时继续收集
20 │
21 │     S, A, R, S_next, D = replay.sample_batch()            # 抽出一批状态、历史动作、奖励、后继状态、终止标记
```

第 19 行先检查 replay 有多少经验。就绪后，第 21 行按字段把多条记录拼成 batch。同一条经验在五个变量中的行号一致。

| 经验 | `S` 原状态 | `A` 实际推力 | `R` 即时奖励 | `S_next` 到达状态 | `D` 终止 |
| --- | --- | ---: | ---: | --- | ---: |
| 1 | `[0.3, 0]` | 0.6 | −0.1 | `[0.6, 0.3]` | 0 |
| 2 | `[0.6, 0.3]` | 0.5 | 1 | `[1.0, 0.4]` | 1 |

第 30 行拿 `S, A` 预测原动作的回报。第 24–27 行拿 `R, S_next, D` 构造这些预测的训练目标。Debugger 为了对照两轮更新，重复抽了这两条经验。

### TD target

要为原来那次推力估计回报，当步 reward 已经知道，剩下的回报要从到达状态开始算。两个时点各有一个动作，先分清它们，才知道每个 Q 分数在评价哪一步（Debugger Target 1 / Target 2）。

```text
22 │     # TD target · 用真实奖励和下一步的估值，为每条经验构造训练标签 Y
23 │     with no_grad:                                        # 只算固定标签 Y，这段运算不进入反传图
24 │         ε = clip(sample_target_noise(), -noise_clip, noise_clip)  # 限制加到下一动作上的噪声
25 │         A_next = clip(Actor_target(S_next) + ε, action_bounds)    # target Actor 给每条后继状态提议动作
26 │         Q_next = min(Q1_target(S_next, A_next), Q2_target(S_next, A_next))  # 两个 target Q 逐行取较小值
27 │         Y = R + γ * (1 - D) * Q_next                      # 即时奖励加折扣后的未来估值，终止时只保留奖励
```

这一块为 replay 中实际执行过的 `(S, A)` 构造一步 TD 目标。读代码时先追两路数据，最后在第 27 行汇合。

| 行号 | 产生的量 | 从哪来、交给谁 |
| --- | --- | --- |
| 24 | `ε` | 采样并限制 target smoothing noise，交给第 25 行。 |
| 25 | `A_next` | Target Actor 在到达状态 `S_next` 提议动作，加 `ε` 并限制到合法范围。把这个提议交给两个 target Critic 打分。 |
| 26 | `Q_next` | 两个 target Critic 对同一个 `(S_next, A_next)` 打分，每条经验取较小值。 |
| 27 | `Y` | 从 replay 取 `R` 和 `D`，合成 `R + γ × (1 − D) × Q_next`。第 30 行把它当回归标签。 |

第一条经验已经实际执行过 `a = 0.6`。从它到达的 `s_next = [0.6, 0.3]` 出发，target Actor 原始提议约 `0.187746`。第 24 行把采到的噪声 `0.3` 限制到 `0.2`，第 25 行相加得到 `A_next ≈ 0.387746`，仍在合法动作范围内。这个新推力是为估计**后续回报**而提议的，Critic 在第 30 行仍然用实际执行过的 `0.6` 做回归。

两个 target Critic 给 `0.387746` 分别打出约 `0.317509` 和 `0.176477`。第 26 行取后者作为第一条经验的 `Q_next`。它是在同一条经验的两个分数之间选，batch 里有两条经验就得到两个 `Q_next`。第 27 行再接上当步实得的 `−0.1`，第一条的一步 TD 目标是 `−0.1 + 0.9 × 0.176477 ≈ 0.058829`。第二条已经终止，`1 − D = 0`，目标只保留即时奖励 `1`。

### Critic update

两个 Critic 用同一批一步 TD 目标做回归。每个 Critic 的参数会同时影响多条样本，所以一条样本希望预测往下，整批更新仍可能让它往上（Debugger Critic forward / backward / step）。

```text
29 │     # Critic update · 让两个 online Critic 对历史动作的预测拟合同一个 Y
30 │     loss_c = mean((Q1(S, A) - Y)^2 + (Q2(S, A) - Y)^2)    # 两个 online Critic 用历史动作拟合相同的 Y
31 │     critic_optimizer.zero_grad()                        # 清掉上次留下的 Critic 梯度
32 │     loss_c.backward()                                   # 求两个 online Critic 的参数梯度
33 │     critic_optimizer.step()                             # 按梯度更新两个 online Critic
34 │     critic_updates += 1                                  # 记下这次已经执行的 Critic 更新
```

把第 30 行当成监督学习回归。输入是第 21 行的原状态和实际推力 `(S, A)`，标签是第 27 行的一步 TD 目标。第一轮里，`Q1` 给两条经验的预测分别是 `0.346` 和 `0.345`，标签分别约是 `0.058829` 和 `1`。第一条的平方误差倾向把预测往下拉，第二条倾向把预测往上拉。两条经验共享一套 Critic 权重，所以实际更新要合并整批贡献。在这个 batch 里，更新后第一条的 `Q1` 反而升到了约 `0.403474`。

第 32 行只沿两个 online Critic 的预测求梯度，标签来自第 23 行的 `no_grad` 区域。第 33 行写回两个 online Critic 的参数，第 34 行计数加一。下一块用这个计数决定 Actor 是否也要更新。

### Delay gate

Actor 依赖 Critic 给方向。让 Critic 先多做几次拟合，Actor 就不会跟着每一次评分变化立即调整（Debugger Delay gate）。

```text
36 │     # Delay gate · 按 Critic 更新次数决定何时更新 Actor 和 target networks
37 │     if critic_updates % policy_delay == 0:              # 每隔几次 Critic 更新，再进入 Actor 分支
```

第 37 行读取的是第 34 行刚加过一的 Critic 更新次数。Debugger 里 `policy_delay = 2`，所以第一次更新后 `1 % 2 = 1`，跳过第 38–55 行，Actor 和三个 target 副本都继续用旧权重。第二次更新后 `2 % 2 = 0`，依次算 Actor 梯度、写回 Actor、同步三个副本。把这个条件放在 Critic 更新之后，Actor 调整时就能用刚完成拟合的 `Q1`。

### Actor update

Critic 回归问的是「原来的动作值多少」，Actor 更新问的是「现在提议的动作该怎么改」。用同一状态重新提议动作，才能在那个动作附近求评分的斜率（Debugger Actor forward / backward）。

```text
38 │         # Actor update · 用 Q1 对当前 Actor 提议的动作打分，把梯度传回 Actor
39 │         freeze(Q1.parameters())                         # 冻结 Q1 权重，梯度仍可经过动作输入回到 Actor
40 │         A_policy = Actor(S)                             # 当前 Actor 对这批旧状态重新提议动作
41 │         Q_policy = Q1(S, A_policy)                       # 用刚更新的 Q1 给这些新动作打分
42 │         loss_a = -Q_policy.mean()                        # 用负均值把最大化 Q 转成最小化 loss
43 │         actor_optimizer.zero_grad()                      # 清掉上次留下的 Actor 梯度
44 │         loss_a.backward()                                # 梯度从 Q1 的动作输入继续传回 Actor
```

第 40 行复用 replay 的状态 `S`，重新问当前 online Actor「你现在会怎么推？」。比如第一条经验原来执行了 `0.6`，Actor 此时为同一个出发状态给出的 `A_policy` 约是 `0.158649`。第 41 行由刚更新的 online `Q1` 评估这次提议，给出的分数约是 `0.296558`。这里用当前状态 `S`，和第 25 行在到达状态 `S_next` 估计未来的动作是两个问题。

第 42 行用负号把提高平均 Q 转成最小化 loss。第 44 行反传时，先由 `−mean` 把 `−1/B` 分给每个 Q 分数，再通过 `Q1` 对动作的斜率传给 Actor。这个例子里第一条经验的动作斜率约为 `+0.499771`，表示在这个提议附近把推力稍微调大，`Q1` 的评分会上升。第 39 行冻结的是 `Q1` 自己的参数，这个对输入动作的斜率仍然可以用。常见算子的反传规则另见 [A1 Autograd 讲义](../../a1-transformer/autograd-primer.html)。

### Actor step

Backward 给出参数该往哪个方向变，optimizer 才实际写回参数。用同一个状态重新算推力，可以看出这次参数更新改变了什么（Debugger Actor step）。

```text
46 │         # Actor step · 用刚算出的梯度写回 Actor 参数
47 │         actor_optimizer.step()                          # 写回新的 Actor 参数
48 │         unfreeze(Q1.parameters())                       # 让 Q1 在下次 Critic 更新时继续学习
```

第 44 行算出的 Actor 梯度，直到第 47 行才被 optimizer 用来修改权重。拿第一条经验的状态重新 forward，Actor 的提议从约 `0.158649` 变为 `0.220747`。这两个数是同一状态在更新前后的输出。第 48 行解除 `Q1` 的冻结，让下一轮 Critic update 能继续训练它。接下来的 target sync 读取的是刚写回的 Actor 以及第 33 行已写回的两个 online Critic。

### Target sync

到这里，Critic 和 Actor 都已经更新过。下一轮还要靠网络估计「做完这次推力之后，还能拿到多少回报」。**这一块负责把本轮学到的参数，少量交给下一轮负责估值的副本**（Debugger Target sync）。

```text
50 │         # Target sync · 把新 online 参数少量混入副本，让下次算 Y 的参数缓慢跟进
51 │         with no_grad:                                    # 不记录这次参数写入的梯度图
52 │             for online, target in ((Actor, Actor_target), (Q1, Q1_target), (Q2, Q2_target)):  # 每个 online 网络对应自己的副本
53 │                 paired_parameters = zip(online.parameters(), target.parameters(), strict=True)  # 按顺序配对这两个网络的参数
54 │                 for online_param, target_param in paired_parameters:  # 逐对读取对应的参数张量
55 │                     target_param.copy_((1 - τ) * target_param + τ * online_param)  # 直接写回 target 权重，无需 optimizer
```

这里确实有两种**会改变网络权重**的操作。

- 第 33、47 行更新 **online 网络**。方向来自 loss 的梯度，optimizer 的学习率控制步长。以 SGD 为例，更新量是 `−lr × 梯度`。
- 第 55 行更新 **target 副本**。方向来自当前 online 参数与旧 target 参数的差，更新量是 `τ × (online_param − target_param)`。`τ` 是 **soft-update coefficient（软更新系数）**，表示每次同步走完差距的多少。这里直接用 `copy_` 写回副本，`no_grad` 让这次写入不进入反传图。原作者的实现也是逐个参数做这个加权平均。 [原作者的 Target sync 实现](https://github.com/sfujim/TD3/blob/master/TD3.py#L127-L142)

第 52 行列出三对网络，顺序都是 `(online, target)`。target 是第 3 行从对应的 online 网络复制来的，结构和参数顺序相同。第 53 行用 `zip` 把这一对网络的参数按顺序配起来，结果就是 `paired_parameters`。第 54 行取出一对参数，第 55 行只写回 target 那一侧。

例如 `Q2_target` 的 bias 原来是 `0.02`，当前 `Q2` 的 bias 约为 `0.138879`。Debugger 用 `τ = 0.1`，所以副本走完差距的 `10%`，得到 `0.02 + 0.1 × (0.138879 − 0.02) ≈ 0.031888`。原作者代码的默认值是 `τ = 0.005`。 [原作者的超参数](https://github.com/sfujim/TD3/blob/master/TD3.py#L64-L85)

为什么值得把副本更新得慢一些？online Critic 正在用一步 TD 目标做回归，目标里又有 Critic 自己这一类模型对下一步的估值。用一套更新较慢的参数生成这部分估值，可以减弱一次 online 更新对下一轮回归目标的即时影响。线性插值发生在**权重空间**。`10%` 指权重差距的比例。下一轮的动作和 Q 分数要用新权重重新 forward 才能得到。 [TD3 原论文第 5.2 节](https://proceedings.mlr.press/v80/fujimoto18a/fujimoto18a.pdf#page=5)

下一轮回到第 25–27 行才用上新副本。下面固定同一条 replay 经验和同一份 smoothing noise，只换同步前后的副本。

| 下一轮读到的量 | 同步前 | 同步后 |
| --- | ---: | ---: |
| 第 25 行 · target Actor 提议的估值动作 `A_next` | 0.387746 | 0.394926 |
| 第 26 行 · 两个 target Critic 给出的较小估值 `Q_next` | 0.176477 | 0.198093 |
| 第 27 行 · 加上即时奖励 `−0.1` 后的一步 TD 目标 | 0.058829 | 0.078284 |

第 12 步的表格用两套副本各做一次前向来展示这个差异。实际训练循环会在下一批经验上继续更新 Critic，Debugger 的第 13 步则直接演示部署。

## 4 · Debugger · 验证原理在执行时怎样表现

Debugger 在同一批两条滑块经验上做两轮更新。每一步会高亮当前代码，表格显示这些行读到和算出的 tensor。 [打开 Training Debugger](training-debugger.html)

1. **05 Target 1** 切换两条经验，观察终止标记为什么让第二条只保留即时奖励。
2. **Actor forward → backward → Actor step · 写回** 对同一个状态比较推力输出，核对算梯度和写回参数分别发生在哪一步。
3. **Target sync · 同步 targets** 先看一个 bias 怎样向 online 参数靠近，再固定经验和噪声，只换同步前后的副本，对比一步 TD 目标。

### 对照 tensor 时用到的模型

- 一批 `B` 个滑块观察的形状是 `[B, 2]`，每行为 `[position, velocity]`。Actor 用一层 affine 和 `tanh` 输出 `[B, 1]` 的推力。
- 每个 Critic 用 `[position, velocity, action, action²]` 四个特征，经过一层 affine 得到一个 Q。加上二次项后，可以直接从权重和输入动作算出动作斜率。
- 两轮更新使用固定数据和 SGD，便于核对各步骤计算。部署时使用更新后的 online Actor。

## 5 · Inference：训练好以后怎么用

```text
 1 │ load trained online Actor weights                       # 加载训练好的 online Actor 权重
 2 │ Actor.eval()                                            # 切到推理模式
 3 │ s = env.reset()                                         # 获取初始观察
 4 │ while episode continues:                                # 每次控制一步
 5 │     with no_grad:                                       # 推理只做前向计算
 6 │         a = Actor(observation_tensor(s))                 # [1, obs_dim] → [1, act_dim]
 7 │     s, r, terminated, truncated = env.step(to_controller_action(a))  # 转换动作尺度并获取下一观察
```

部署时保留训练时的 observation 预处理、动作尺度和控制频率。TD3 的 deterministic policy 直接执行 Actor 输出；通常关闭 exploration noise。两个 Critics、target networks、replay 和 optimizer 都留在训练端。Debugger 最后一帧会在同一个状态上并排显示初始 Actor 和训练两步后的 Actor 输出。

[导师问答与练习路线](tutor.md)含答案。第一轮试读可以停在 debugger 的第二次 Critic update，先说出第二行的 reward 为什么不接未来值。再走到 Actor 和 inference，核对自己的判断。

参考实现的更新顺序可对照 [Spinning Up TD3](https://spinningup.openai.com/en/latest/algorithms/td3.html) 和 [原作者实现](https://github.com/sfujim/TD3/blob/master/TD3.py)。

## 6 · 算法演进

### DPG、DDPG 和 TD3 对照

还是用滑块来想。Actor 从当前位置和速度直接给出一个推力，Critic 对推力打分。三种算法训练 Actor 时都问同一个问题：**当前推力稍微增大一点，Critic 预测的回报会升还是会降？** Critic 对动作的导数给出方向，再用链式法则把这个方向传回 Actor 参数。这就是 **deterministic policy gradient**。

| 对照点 | DPG（这里取 OPDAC） | DDPG | TD3 |
| --- | --- | --- | --- |
| Critic 用什么数据 | 刚采到的一条经验 | replay 抽一批 | replay 抽一批 |
| 下一动作从哪来 | 当前 Actor | target Actor | target Actor 加有界扰动 |
| 谁估计下一步回报 | 当前的一个 Critic | 一个 target Critic | 两个 target Critic 的较小值 |
| Actor 听谁的梯度 | 当前 Critic | online Critic | online `Q1` |
| Actor 和副本多久更新 | 本例每步更新 Actor，无参数副本 | 每次 Critic 更新后都更新 Actor 和副本 | 每隔 `policy_delay` 次 Critic 更新，才更新 Actor 和副本 |

下面是按论文改写的对照伪代码。`(s, a, r, s_next, d)` 是单条经验，`d` 表示真正终止。`S, A, R, S_next, D` 是把多条经验按行拼成的 batch，顺序与小写变量一一对应。这三段沿用滑块的设定，简写环境返回值为 `s_next, r, d`。第 2 节的完整代码另外处理外部截断。

`stop_grad` 把算出的一步 TD 目标当作这次更新的固定标签。`step_minimize` 表示用给定 loss 做一次梯度更新，写回括号中指定的模型参数。每段行号独立。

#### DPG（2014）· 用 Critic 的动作斜率更新 Actor

DPG 是策略梯度公式。这里用原论文第 4.2 节的 **OPDAC**，也就是它的一种 off-policy actor-critic 实现。探索策略 `β` 负责采集，确定性 Actor 负责提出要学的动作。每采到一条经验，用当前 Actor 和当前 Critic 构造一步 TD 目标，再分别算 Critic 和 Actor 的更新方向。

`w` 和 `θ` 分别是 Critic 和 Actor 的参数，`α_Q` 和 `α_actor` 是各自的学习率。第 10 行的导数只对 Actor 参数 `θ` 求，沿 `Q → 动作 → Actor 参数` 反传。论文中的两条更新式都用该步更新前的参数，所以先算两条梯度，再写回。 [Silver 等，第 4.2 节，式 (16)–(18)](https://proceedings.mlr.press/v32/silver14.pdf#page=5)

```text
 1 │ Actor_θ, Q_w = init_differentiable_models()
 2 │ s = env.reset()
 3 │ repeat:
 4 │     a = sample(β(. | s))                          # 探索策略选实际推力
 5 │     s_next, r, d = env.step(a)
 6 │     a_next = Actor_θ(s_next)                       # 用当前 Actor 提议下一动作
 7 │     y = stop_grad(r + γ * (1 - d) * Q_w(s_next, a_next))
 8 │     δ = y - Q_w(s, a)                              # 同一条经验的一步 TD error
 9 │     g_Q = δ * grad_w(Q_w(s, a))
10 │     g_actor = grad_θ(Q_w(s, Actor_θ(s)))            # 通过 Q 对动作求导
11 │     w = w + α_Q * g_Q
12 │     θ = θ + α_actor * g_actor
13 │     s = env.reset() if d else s_next
```

#### DDPG（2015 预印本）· 用 replay 和 target networks 训练深度网络

DDPG 保留确定性策略梯度，把 Actor 和 Critic 实现为深度网络。它把刚采到的经验存入 **replay buffer**，随机抽一批旧经验训练。相邻交互往往相关，随机抽样有助于降低一批样本的时间相关性，也能重复使用旧数据。另备一对 **target networks** 专门计算下一步估值，让在线 Critic 每次更新后，产生标签的参数只跟进一小步。第 15–16 行的 `soft_update` 指 `target ← (1 − τ) × target + τ × online`。 [Lillicrap 等，第 3 节和 Algorithm 1](https://arxiv.org/pdf/1509.02971#page=5)

```text
 1 │ Actor, Q = init_deep_networks()
 2 │ Actor_target, Q_target = copy_networks(Actor, Q)
 3 │ replay, s = empty_replay(), env.reset()
 4 │ repeat:
 5 │     a = clip(Actor(s) + exploration_noise, action_bounds)
 6 │     s_next, r, d = env.step(a)
 7 │     replay.add(s, a, r, s_next, d)
 8 │     s = env.reset() if d else s_next
 9 │     if not replay.ready(): continue
10 │     S, A, R, S_next, D = replay.sample_batch()
11 │     A_next = Actor_target(S_next)                  # 下一动作由慢更新的副本提议
12 │     Y = stop_grad(R + γ * (1 - D) * Q_target(S_next, A_next))
13 │     step_minimize(Q, mean((Q(S, A) - Y)^2))
14 │     step_minimize(Actor, -mean(Q(S, Actor(S))))      # 仍是确定性策略梯度
15 │     soft_update(Actor_target, Actor, τ)
16 │     soft_update(Q_target, Q, τ)
```

#### TD3（2018）· 改 Critic 的目标，也改 Actor 更新频率

TD3 保留 DDPG 的 replay、深度网络和参数副本。第 12–13 行在 target Actor 的下一动作附近加有界扰动，降低 Critic 对某个狭窄高分点的依赖。这份噪声只用于估值，第 6 行的探索噪声才会影响实际推力。第 14 行让两个 target Critic 评估**同一个**下一动作，并对每条经验分别取两个分数中的较小值，以抑制高估。第 18–20 行让 Actor 和副本隔几次 Critic 更新才动一次。第 19 行的 Actor 梯度仍来自一个 online Critic，这里是 `Q1`。 [Fujimoto 等，第 4.2、5.2–5.3 节和 Algorithm 1](https://proceedings.mlr.press/v80/fujimoto18a/fujimoto18a.pdf#page=6)

```text
 1 │ Actor, Q1, Q2 = init_deep_networks()
 2 │ Actor_target, Q1_target, Q2_target = copy_networks(Actor, Q1, Q2)
 3 │ replay, s = empty_replay(), env.reset()
 4 │ critic_updates = 0
 5 │ repeat:
 6 │     a = clip(Actor(s) + exploration_noise, action_bounds)
 7 │     s_next, r, d = env.step(a)
 8 │     replay.add(s, a, r, s_next, d)
 9 │     s = env.reset() if d else s_next
10 │     if not replay.ready(): continue
11 │     S, A, R, S_next, D = replay.sample_batch()
12 │     ε = clip(sample_target_noise(), -noise_clip, noise_clip)
13 │     A_next = clip(Actor_target(S_next) + ε, action_bounds)
14 │     Q_next = elementwise_min(Q1_target(S_next, A_next), Q2_target(S_next, A_next))
15 │     Y = stop_grad(R + γ * (1 - D) * Q_next)
16 │     step_minimize([Q1, Q2], mean((Q1(S, A) - Y)^2 + (Q2(S, A) - Y)^2))
17 │     critic_updates += 1
18 │     if critic_updates % policy_delay == 0:
19 │         step_minimize(Actor, -mean(Q1(S, Actor(S))))
20 │         soft_update_pairs((Actor_target, Actor), (Q1_target, Q1), (Q2_target, Q2), τ)
```

## 7 · Executable review reference

The [PR1 consensus delta](PR1-consensus-review.md) maps the agreed design to code and explicitly lists remaining UI work. The [PyTorch reference](reference/debugger.py) implements real autograd, no-momentum SGD, phase stepping, deep snapshots, restore and comparison forks. The browser remains a standalone analytic trace; it is not a PyTorch frontend.

The debugger now separates forward, backward and optimizer step, showing a fixed-state Q1 curve and shared-weight gradient contributions. The documented 0.058829 and 0.031888 fixtures have been checked against both implementations with the full configuration recorded in the review note. These are teaching fixtures, not training or learning-effect evidence.
