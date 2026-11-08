# 机器人 RL 温习：PPO、TD3、FB 与 DexLab

> T2 / T3 共用路线 · 2026-10-17 · [离线 HTML 阅读版](rl-control-review-primer.html) · [分段实践](rl-control-practice.md)
> 先通读，再由陪练按已有能力选择短 session。主线是 RL 基础知识及其在连续控制、sim2real 中的含义；DexLab 只作具体案例，源码导读为可选深入。现行顺序与证据门槛见 [课表](../../curriculum.md#rl-基础讲义与课次的对应t2t3-共用)。

## 0 · 三条路线，两个分类维度

**On-policy / off-policy** 问的是：训练目标中的 policy 与数据采集的 behavior policy 是什么关系？**Zero-shot** 问的是：遇到新任务时，还需要哪些数据、交互和参数更新？这两个维度可以交叉。FB 可以用 off-policy transitions 预训练，再对新 reward 做 zero-shot task inference。

| 路线 | 采什么数据 | 学什么 | 测试时怎样接新目标 | 本课程中的落点 |
|---|---|---|---|---|
| PPO | 当前 behavior policy 的新 rollout；一个 batch 内重复优化有限轮 | 随机 policy 与 state value | 通常需要训练或预先定义的 goal conditioning；PPO 本身不提供任意新 reward 接口 | T2·S02–S06、T3·S01；DexLab Stage 1 |
| TD3 | 带 exploration 的 behavior transitions，跨轮存入 replay | 双 Q、deterministic actor、target networks | 通常对指定 reward 优化；也可作为多任务学习组件 | T3·S02；接回 RLT / EXPO 的价值学习 |
| FB | 覆盖环境的 transitions；可在线采集，也可固定离线数据 | reward-independent successor representation 与 task-conditioned policy | 冻结网络，用新 reward 推断 task vector，再调用 policy | T3·S09；原理见 SR / SF / FB 专题 |

“Reward-free”不等于无损失、无训练、无采样成本；“off-policy”不等于“offline”。后者还规定训练期间不能继续收集环境交互。能消费 replay 的算法也不自动适合固定窄数据集：actor 仍可能追逐数据未覆盖动作上的虚高 Q。

当前研究对象还需要第四层知识：**simulation / control contract**。环境的 action、observation、time step、reset 和 reward 决定 RL 实际学的 MDP。更换 controller 或删掉 privileged inputs，可能改变问题本身，而不仅是换一个网络。

先读本讲义 §1–2 的共同接口，再接 §4 的 TD3 与[基础讲义 §4 的 SAC](rl-foundations-primer.html#4--连续动作的-off-policy-actor-critic)。经典更新对拍后，按 [P7/P8](rl-control-practice.md) 进入生成策略后训练或可微动力学。DexLab 和 FB 各有专题入口；已掌握的知识点只做一次上下文判读。

**执行过程阅读法（2026-10-18 补充）。** §2 在公式后补 REINFORCE、GAE 和 PPO 伪代码；TD(0)、SARSA/Q-learning、DQN、TD3/SAC 与 offline policy extraction 的流程见 [RL 基础讲义](rl-foundations-primer.html)。先找“数据采集 → 固定目标 → 更新谁 → 何时重新采集”，再把公式放回对应的一行。这里的伪代码用于理解执行顺序，不代表已经完成实现或训练。

## 1 · 所有方法共享的控制闭环

```text
物理状态 s_t → sensor / noise / history → observation o_t
    ↑                                      ↓
PhysX / dynamics ← PD / actuator / delay ← policy action a_t
    └──────── next observation、reward、termination ────────┘
```

### 1.1 State、observation 与 history

**State** 需要包含决定未来的变量；**observation** 是 policy 实际拿到的信息。物体质量、摩擦、接触状态、执行器延迟和 target accumulator 可能影响未来，却不都直接可测。只给单帧 joint position 的 policy 面对的是 partial observability。History encoder 用过去的 observation / action 估计有用的信息；有限 history 不能保证恢复完整 Markov state。

**Privileged information** 是训练可见、部署不可见的信息。两种常见接线要分开：

1. Asymmetric actor-critic：actor 只看可部署输入，critic 多看 simulator state；部署直接去掉 critic。
2. Privileged teacher → student：teacher actor 也使用 privileged inputs；必须训练替代输入的 student，并单独评估它。

DexLab 的 HORA / AnyRotate Stage 1 属于第二种。不要仅根据类名中的 `asymmetric` 就判断只有 critic 看到了特权状态；以 forward 中的 tensor 拼接为准。具体证据见 [DexLab 代码导读](../../references/dexlab-sim-rl-digest.md)。

### 1.2 Action 不一定是 torque

Policy 可以输出 torque、position target 或 target delta。若输出 delta，则 actuator 看到的是经过缩放、滤波、累积和限位后的 target；相同 action 在不同历史下会产生不同力矩。控制器状态必须在 reset 时恢复，policy / teacher / evaluator 的 action mapping 必须一致。

以 H13 配置为例：policy 为 10 Hz，physics 为 120 Hz，13 个主动控制量表示 position target 的增量，经滤波、累积和 position PD 执行；mimic joints 与 delay 也参与最终动作。理解 action 的含义，要从 policy 输出一路连到 actuator。

**时间单位影响算法含义。** Control period 为 Δt 时，gamma 对应的有效折扣时长约为 `Δt / (1 - gamma)`，指数时间常数为 `-Δt / log(gamma)`。保持 gamma 不变而把 control rate 翻倍，会缩短真实时间尺度上的视野。Velocity reward、按时间累积的成本、episode timeout、history duration 都要一起核对。这些量共同定义了训练问题的时间尺度。

**三个时间长度要分开。** AnyRotate Stage 1 的 H13 示例使用 `gamma=0.99`、`Δt=0.1 s`，有效折扣视野约为 `0.1 / 0.01 = 10 s`；训练 episode 上限为 `60 s`。10 秒表示远期 reward 的权重衰减尺度，超过这个时间仍有贡献；60 秒决定一条 episode 最长能持续多久。还有一个长度是每次收集的 rollout：这里为 8 步，即 0.8 秒，见 §2.2。

### 1.3 Termination、truncation 与 reset

Task terminal（例如定义为失败的掉落）一般不 bootstrap。外部 time limit 若只是截断一个 continuing task，应从 **reset 前的 final observation** bootstrap；如果 horizon 本来就是任务定义的一部分，则按任务 terminal 处理，必要时把剩余时间纳入 state。

Vector environment 的 autoreset 容易把新 episode 的初始 observation 当作上一条 transition 的 next observation。无论 PPO 还是 TD3，都要先验证 wrapper 返回值与 final observation 的存放位置；字段名字不能替代语义核查。

## 2 · PPO：新 rollout 如何变成策略更新

### 2.1 从 REINFORCE 到 actor-critic

Policy gradient 用 `grad log pi(a|o) × advantage` 提高好动作的概率。Baseline 降低方差；在 actor loss 中，采集后计算好的 advantage 是固定训练目标，不让其梯度顺路更新 value 或改变权重的含义。

Critic 拟合 return，actor 使用相对 baseline 的 advantage。Reward 更高不一定意味着 advantage 为正，因为 baseline 也可能更高。T2·S02/S03 分别练 score-function estimator 和 stop-gradient；这里把它们接到 rollout 更新流程。

#### 执行过程 · REINFORCE 加 learned baseline

**输入：** 当前 policy 采到的完整 terminal episodes；外部 truncation 不能直接当作零尾部。**更新：** stochastic actor 与独立 V baseline。下面明确采用起始状态折扣目标 `J = E[sum gamma^t r_t]`，所以 actor 项保留 `gamma^t`；`r_t` 表示执行本步动作后得到的 reward。

```text
repeat:
    用当前 pi 收集一批完整 episodes；期间不更新参数
    对每条 episode 从后往前算 G_t = r_t + gamma * G_(t+1)
        # 真正 terminal 后的 G_T = 0；t 在每条 episode 内从 0 计
    advantage_t = stopgrad(G_t - V(o_t))     # 在拟合 V 前固定
    actor_loss = -mean_over_episodes(
        sum_t gamma^t * advantage_t * log pi(a_t | o_t))
    minimize actor_loss over pi             # a_t 是采集时保存的动作
    minimize mean((V(o_t) - G_t)^2) over V
    丢弃这一批 episodes，重新用更新后的 policy 采集
```

若不学 baseline，令 advantage 为 `G_t` 就得到 REINFORCE。A2C 将完整 MC return 换成短 rollout 的 bootstrapped return；下一节 GAE 进一步组织多步 TD residual。常见实现会采用省略显式 `gamma^t` 的 surrogate，须说明目标/采样权重约定；它与这里严格起始状态目标的写法有区别。

### 2.2 GAE：两个 mask，各管一件事

设 `b_t` 表示该 transition 允许 bootstrap，`c_t` 表示下一条 advantage 仍属于同一条可继续的轨迹：

$$\delta_t = r_t + \gamma b_t V_{old}(o_{t+1}^{final}) - V_{old}(o_t)$$

$$\hat A_t = \delta_t + \gamma\lambda c_t\hat A_{t+1},\qquad \hat R_t=\hat A_t+V_{old}(o_t).$$

Continuing task 的 time-limit truncation 可以有 `b_t=1, c_t=0`：需要 final-state value，但不能把 reset 后新 episode 的 advantage 接过来。真正 terminal 两者为 0。普通 batch 末尾可以 bootstrap，递推在已有 buffer 边界结束。若实现用 reward correction 或别的等价写法，仍应对拍这两种语义。

Lambda 控制多步 TD 误差的累积：小 lambda 更依赖 critic，大 lambda 使用更长的 sampled credit assignment。比较时固定 rollout、value prediction 和其余参数，先区分 estimator 变化，再谈学习曲线。

**用 8 步 rollout 理解 bootstrap。** 这个示例的 `horizon_length=8`：每个 env 每批收集 0.8 秒数据，再构造 GAE。若中途没有 terminal，buffer 最前面的样本最多用到 8 步实测 reward，最后一个样本只用到 1 步；更远的影响由 critic value 补上。`lambda=1` 时可直接看成相应的 n-step return；此配置 `tau=0.95` 对应 GAE lambda，实际是这些多步估计的加权组合。最长 8 步项的尾部系数仍有 `gamma^8≈0.923`，所以短 rollout 下 critic 的准确性尤其重要；这个系数不是 critic 占整个 target 的百分比。增加并行 env 数会增加样本数，而每条轨迹的实测时间长度仍是 0.8 秒。

#### 执行过程 · GAE 的倒序递推

**输入：** 长度 T 的 rollout，以及采样时冻结的 `old_value[t]`、`next_value[t]`。后者评价该 transition **reset 前**的下一 observation；terminal 的 value 可直接置 0。**输出：** 固定的 advantage 与 value target，本阶段不更新网络。

```text
next_advantage = 0
for t = T-1, ..., 0:
    delta = r[t] + gamma * b[t] * next_value[t] - old_value[t]
    advantage[t] = delta + gamma * lambda * c[t] * next_advantage
    value_target[t] = advantage[t] + old_value[t]
    next_advantage = advantage[t]
advantage, value_target = stopgrad(advantage, value_target)
```

真正 terminal 用 `b=0, c=0`；continuing-task timeout 用 `b=1, c=0`；普通相邻 transition 用 `b=1, c=1`。Buffer 末尾的 `next_advantage=0` 结束递推，但该步仍可通过 `next_value` bootstrap。倒序是为了读取已算好的下一项 advantage，环境交互依然正序发生。

### 2.3 Ratio 与 clipping

$$r_t(\theta)=\exp\big(\log\pi_\theta(a_t|o_t)-\log\pi_{old}(a_t|o_t)\big)$$

$$L^{clip}=\mathbb E\left[\min\left(r_t\hat A_t,\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)\hat A_t\right)\right].$$

Actor 最大化这个 surrogate，代码通常最小化其负数，再加 value loss 与 entropy 项。`old_log_prob`、old values 和 advantage 必须固定在采样/目标构建时；每个 minibatch 重算 old log probability 会让 ratio 失去意义。

| 固定 advantage | 超过哪边后，surrogate 停止继续奖励有利变化 | 另一边为什么仍需有梯度 |
|---|---|---|
| A > 0 | ratio > 1 + epsilon | 好动作概率被过度压低时，需要纠正 |
| A < 0 | ratio < 1 - epsilon | 坏动作概率被过度抬高时，需要纠正 |

Clipping 不强制 policy 落在 KL ball 内，也不是把所有样本的 ratio 数值硬截断。多个 epochs 仍会产生 policy drift；因此监测 approximate KL、clip fraction、entropy 和实际 return。一个 rollout 多次 minibatch 更新仍属于 PPO 的常见 on-policy 工作方式，不意味着可以无限混入旧 replay。

### 2.4 连续动作与可执行 policy

Diagonal Gaussian 的 log probability 要沿 action dimensions 求和。使用 tanh squashing 时，通过变量变换的 Jacobian 计算变换后动作的 density。使用 raw Gaussian sample 后由环境 clamp 时，保存 raw sample 及其 log probability，并把 clamp 视为执行映射；executed action 另行记录。这样 rollout 中的采样事件、更新时的 likelihood 和实际执行动作都有明确对应。

Observation normalization、action scale、network weights、history/reset 逻辑共同构成部署 policy。把这些内容一起保存，并在 evaluation 恢复，才能复现采集时的行为。动作采样、映射和记录方式是连续控制实现的基础接口。

**一次完整 PPO 更新：** 冻结 behavior snapshot → 收集 rollout → 正确构建 bootstrap / GAE / return → 有限次 minibatch 更新 → 检查 drift 与独立 rollout → 丢弃该批数据并重新采集。[PPO 原论文](https://arxiv.org/abs/1707.06347)与 [Spinning Up 实现说明](https://spinningup.openai.com/en/latest/algorithms/ppo.html)给出目标与流程。

#### 执行过程 · PPO 的完整外循环

**初始化：** policy `pi`、独立 value network V；学习率、clip epsilon、每批 rollout 大小与最大 epochs 事先固定。**动作约定：** buffer 保存实际采样的 `a_sample` 与相应 density；通过 `action_map` 执行，raw Gaussian clipping / tanh density 按 §2.4 选择并保持一致。

```text
repeat:
    pi_old, V_old = 冻结当前网络的快照
    buffer = 用 pi_old 收集一批新的 rollout
        # 每步存 o, a_sample, old_logp, r, old_value, next_value, b, c
        # old_logp = log pi_old(a_sample | o)；next_value 不读 reset observation
    buffer.advantage, buffer.value_target = GAE(buffer)
        # 仅计算一次，固定后随原 transition 一起抽 minibatch

    for epoch = 1, ..., max_epochs:
        for batch in shuffled_minibatches(buffer):
            logp = log pi(batch.a_sample | batch.o)
            ratio = exp(logp - batch.old_logp)
            unclipped = ratio * batch.advantage
            clipped = clip(ratio, 1-epsilon, 1+epsilon) * batch.advantage
            minimize -mean(min(unclipped, clipped)) over pi
            minimize mean((V(batch.o) - batch.value_target)^2) over V
        if 测得的 policy KL 超过预设停止阈值:
            break                         # 提前结束本批的多轮优化
    独立评测、记录 KL / clip fraction / return
    丢弃 buffer，用新的 policy 重新采集
```

这里保留 actor clip 与 value regression 两个核心目标；entropy bonus、value clipping、advantage normalization 等实现选择需单列。固定的是采集时的 action、old log probability、advantage 与 value target；每个 minibatch 都重算的是当前 policy 的 `logp`。流程与 [Spinning Up PPO 伪代码](https://spinningup.openai.com/en/latest/algorithms/ppo.html#pseudocode)对照，timeout 与动作记录额外按本讲义的接口约定展开。

## 3 · DexLab：PPO teacher 后面还有什么

### 3.1 Framework 复用的边界

| 层 | 承担职责 | 阅读时的重点 |
|---|---|---|
| Isaac Sim / PhysX | 刚体、接触、仿真 stepping | timestep、接触材料、solver 与真实执行器误差 |
| IsaacLab | DirectRLEnv lifecycle、scene、vector env 接口 | step / reset / observation / reward 的调用与返回约定 |
| rl_games | Stage 1 PPO rollout / optimization 框架 | 配置如何进入 PPO，tensor 与 normalization 如何接入 |
| DexLab | task、controller、sensor、DR、curriculum、teacher/student、bank 与 evaluation | 被学习的任务是什么，训练信号能否部署 |

框架负责通用训练与仿真循环，研究实现定义具体任务及其部署条件。Reward 和 contact 语义、控制标定、history、随机化、蒸馏分布与独立评估，都能显著改变训练结果。温习时先掌握这几层职责，再按需要阅读对应代码。

### 3.2 HORA 与 AnyRotate 的两阶段接线

| 阶段 | HORA 路线（所读版本） | AnyRotate / H13 路线（所读配置） |
|---|---|---|
| Stage 1 | privileged encoder + proprioception 进入 actor/value；PPO 优化旋转行为 | 增加 tactile、command axis、物体状态等接口；PPO teacher |
| Stage 2 | history encoder 拟合 teacher latent，冻结其他网络；student action 采集后续状态 | history student；latent/action 蒸馏，配置允许训练 actor；rollout 有 teacher mixing |
| 部署检查 | history 能否补足 teacher 使用的隐藏信息 | 纯 student 是否在新的接触、形状和轴向下稳定 |

Stage 2 的默认训练信号是 supervised distillation；其中 optimizer 使用的监督目标决定了更新性质。AnyRotate 的 replay 与 finetune 是另外的可选路径，具体实验由所选配置决定。

**Teacher mixing** 改变 student 训练时访问的 state distribution。训练 return 包含 teacher 纠正的贡献，纯 student 能力通过独立 rollout 衡量。Latent MSE 描述表示误差，而 closed-loop performance 还取决于误差方向、actor sensitivity 与 history distribution。

H13 这个 student rollout 执行均值动作 `student_mu`，没有额外从 action distribution 采样。动作层的随机性来自 teacher/student 来源选择；reset、domain randomization、扰动与观测噪声仍可使访问到的状态多样化。**Deterministic action selection 与 deterministic trajectory 是两件事。**

读指标时同时看样本数：这个配置持续保留 teacher mixing，训练日志中用于纯 student 统计的 episode 集合始终为空，因此 `student_episode_reward=0.0` 且 `student_episode_count=0`。这里的 0 是“尚无该类评测样本”的占位值；实际能力需要另跑纯 student evaluation。具体过滤规则留在可选导读中。

### 3.3 Sim2real 的三个接口

1. **Dynamics / sensing**：system identification 校准执行器、延迟与触觉；DR 覆盖不确定性。范围更宽未必更好，错误的随机化关系可能制造不真实任务。
2. **Training distribution**：grasp bank 决定起点，object sampling 决定物体权重，curriculum 逐步改变重力或成功要求。这些都会改变训练难度与日志的含义。
3. **Deployment / evaluation**：固定 normalization、history、controller 和时间尺度；按物体、旋转轴、失败类型报告纯 student 表现，再讨论真实硬件迁移。

Bank 恢复包括 joint targets / preload 等 controller state，这样相同 pose 才有可比较的后续响应。Nominal bank replay 验证给定条件下的初始化；DR robustness 另用多种参数条件评估。Teacher 的仿真成绩、student 的仿真成绩和真机成绩分别记录。

需要追踪实现时再读 [DexLab digest](../../references/dexlab-sim-rl-digest.md)。完成本章基础温习只需解释信息流、训练信号和评测对象；源码导读补充固定版本的实现证据。

## 4 · TD3：用旧数据学习可改进策略的 Q

### 4.1 一个 batch 的计算图

Replay 保存 `(s, a_executed, r, s_next, bootstrap_mask)` 及所需 history。Actor 输出的 proposal 与环境执行动作若不同，要先明确 Q 的 action space；不允许把一个动作的后果随意标成另一个动作的 transition。

TD3 对下一个动作加 clipped smoothing noise，并限制到合法 action range：

$$\tilde a'=\operatorname{clip}\left(\mu_{\bar\theta}(s')+\operatorname{clip}(\epsilon,-c,c),a_{min},a_{max}\right),\quad \epsilon\sim\mathcal N(0,\sigma^2).$$

$$y=\operatorname{stopgrad}\left[r+\gamma b\min_{i=1,2}Q_{\bar\phi_i}(s',\tilde a')\right],\qquad L_{Q_i}=\mathbb E[(Q_{\phi_i}(s,a)-y)^2].$$

Actor 每隔若干 critic updates 更新一次，最小化 `-mean Q1(s, mu(s))`。Actor step 时固定 Q 的参数，但保留 **Q 对 action 的导数**，否则 actor 接收不到改进方向。Target networks 使用 Polyak averaging；target 不反向传播。[TD3 原论文](https://arxiv.org/abs/1802.09477)与 [Spinning Up](https://spinningup.openai.com/en/latest/algorithms/td3.html)是实现对照。

### 4.2 三项机制为何组合

| 机制 | 针对的风险 | 不保证什么 |
|---|---|---|
| Twin critics + target min | actor / bootstrap 放大乐观估计 | 两个相关的 critic 仍可能一起高估，也可能过于保守 |
| Delayed actor update | critic 尚不可靠时 actor 快速追逐错误 | 不保证 critic 有足够数据支持 |
| Target policy smoothing | 对狭窄 Q 峰的利用 | 不替代 behavior exploration noise；不是 SAC entropy objective |

**Update-to-data ratio (UTD)** 必须声明分母：每条新 transition 对应多少 optimizer updates，还是每次 vector step 对应多少次。Replay reuse 改善采样效率，也可能反复强化偏差。两条曲线应同时画 `environment transitions` 与 `optimizer updates / wall time`，否则不能判断“更快”。

SAC 是对照路线：stochastic actor 与 entropy-regularized objective；不能把 TD3 的 exploration noise 当作 SAC 的温度 alpha。REDQ 与 EXPO 的 critic ensemble 也不等同于 TD3 两个 critics 的全部设计。基础细节接 [原讲义知识点 13–17](rl-foundations-primer.html)。

### 4.3 为什么接到 RLT / EXPO 后更难

Q 的输入 action 可能是 residual、token、candidate selection 或 edit，而物理环境执行的是完整动作序列。需要追踪 `learned action → executor → executed action → transition` 的身份。若一步执行 K 个 primitive steps，则 semi-MDP target 通常使用相应累计 reward 和 `gamma^K`；episode 中途结束时按实际执行长度处理。

这条线只解释接口，具体 EXPO / RLT 机制及真实问题继续使用 [S06](s06-primer.html)、[actor-critic 基础衔接](actor-critic-foundations.md)和 [EXPO-FT 案例集](expo-ft-casebook.html)，不另造一份实验结论。

## 5 · FB：把训练阶段的表示用于新 reward

### 5.1 从标量 value 到 successor measure

SR 记录固定 policy 的未来访问，SF 记录未来 feature 累积。换 reward 可以重新评价这些行为；固定 policy 的 evaluation 本身不会自动产生最优控制。GPI 在已有 policy 的 action values 之间改进，FB 再把表示与 task-conditioned policy 联合学习。

沿用 S09 的 **next-state reward** 约定，令 `Q_r(s,a)=E[sum_{t>=0} gamma^t r(s_{t+1})]`。FB 用相对于 reference distribution rho 的 density 分解 successor measure：

$$M^{\pi_z}(s,a,ds^+)\approx F(s,a,z)^\top B(s^+)\,\rho(ds^+).$$

$$z_r=\mathbb E_{s\sim\rho}[B(s)r(s)],\qquad Q_r(s,a)\approx F(s,a,z_r)^\top z_r.$$

F 表达给定 task-conditioned policy 下的未来占用；B 给访问目标 state 的坐标。它们不是 forward dynamics / backward physics。要得到控制能力，还需要 policy 对 `F(s,a,z)^T z` 的改进与表示学习保持自洽。连续 action 的 actor 近似 greedy，不等于求到了精确 argmax。

### 5.2 哪些东西训练，哪些东西测试时冻结

| 阶段 | 允许的操作 | 应登记的成本与边界 |
|---|---|---|
| Pretraining | 采集/读取 transitions，采样 z，更新 F/B/policy/targets | 数据来源、coverage、表示维数、更新预算 |
| Task inference | 在同一 rho 的 reference samples 上评价新 reward，估计 z_r | reward evaluations 数量、采样分布、是否额外访问环境 |
| Execution | 冻结网络，以 z_r 调用 policy | test tasks、rollout 数、task return；不拿结果继续调参 |

沿用 S09 与 2023 论文的定义，按任务怎样给出区分：

- **Zero-shot**：显式给出 reward function，用已有 reference states 评价它并计算 z；随后直接执行，无额外 planning / learning。用有限 reference samples 近似积分，仍可属于这种任务指定方式。
- **Few-shot task specification / zero-planning**：任务仅由少量 `(state, reward)` 样本给出，通过这些样本估计 z，再直接执行。
- **Adaptation**：额外用环境 rollout 搜索 z，或更新网络参数；单列交互、搜索与优化成本。

共同点是利用预训练知识；区分点是 reward 的提供方式与测试时允许的操作。仅仅冻结网络，还不足以区分这三者。[定义：2023 §2](https://arxiv.org/html/2209.14935v2#S2)

Rho 不是无关符号：训练 density 的 reference measure 与推断 z 的积分必须一致。`C_B=E_rho[B B^T]` 是**未中心化二阶矩**；在给定 B 的最小二乘 reward projection 中可出现 `C_B^-1`。它不应被随意插入标准 FB task encoding `z_r=E[B r]`，否则改变坐标约定。完整推导与数值例见 [Zero-Shot 讲义 §3](zero-shot-rl-primer.html)。

### 5.3 FB 在灵巧操作上的问题是什么

Contact-rich control 的可达状态可能狭窄且不连通；训练集未覆盖的重抓取或掉落恢复，不会仅靠换 z 出现。Finite-rank representation 未必区分新 reward 所需的接触细节。History 与 hidden dynamics 还影响 successor measure 对应的是哪个状态定义。

因此要分开问：**同一 dynamics 的新 reward 能否零样本调用？新的物体 / 摩擦 / 硬件下能否迁移？** 后者需要额外的 coverage、conditioning、DR 或 adaptation 证据。FB 不自动替代 HORA 的 dynamics adaptation，也不直接替代触觉与 controller 校准。核心来源：[FB 2021](https://arxiv.org/abs/2103.07945)、[Zero-Shot RL 2023](https://arxiv.org/abs/2209.14935)。

## 6 · 相似词汇背后的不同接口

| 名字 | 编码什么 / 怎样获得 | 直接影响什么 | 如何验证 |
|---|---|---|---|
| HORA adaptation latent | history 推断 teacher 的 privileged latent | policy 对隐藏环境因素的响应 | student-only rollout、latent error 与动作/失败的关系 |
| AnyRotate command axis | 明确输入的期望旋转方向 | goal-conditioned 行为 | 各轴 holdout、真实旋转进展 |
| FB task vector z | reference states 上的 reward-weighted B 平均 | 用既有表示调用新 reward 的 policy | 冻结权重、未见 reward、task return |
| RLT / EXPO learned interface | token、residual、selection 或 edit，依方法定义 | frozen executor 实际执行的动作 | action identity、Q ranking 与真实闭环结果 |

“用了 latent”无法说明它们等价。先写明 latent 的 producer、consumer、训练信号、是否跨部署边界，再讨论迁移。也不要混淆 Successor Representation 与 success rate 的缩写 SR；本路线写全 Successor Representation，指标写 `success_rate`。

## 7 · 用什么证据判断实现与局限

| 看到的现象 | 至少两种解释 | 区分它们的检查 |
|---|---|---|
| PPO reward 上升，旋转能力没上升 | shaping 被利用；curriculum / object mix 改变 | 固定评测条件，分别报告真实 turns、掉落与各 reward 项 |
| Student latent loss 下降，纯 student 变差 | supervision 分布过窄；actor 对误差敏感 | 相同状态上 latent/action 误差，再做 teacher-only / student-only 对照 |
| TD3 Q 上升，return 下降 | Q 外推偏差；action / reward 接线错误 | 固定动作序列短程对拍、held-out rollout return 与 Q 校准；别只加 critics |
| FB TD loss 下降，新任务差 | coverage / rank 不足；task inference 或 policy 近似错误 | 固定网络、任务与初始状态分布，对比直接推断 z_r 和有限预算搜索 z 的独立评测表现 |
| 仿真成功，真机掉落 | dynamics / tactile mismatch；reset 与 controller state 不一致 | 先校验同一控制命令与感知接口，再定位学习侧问题 |

**FB 检查的解释边界。** 小型有限 MDP 可以用 exact oracle 对拍。真实控制任务可以冻结表示与 policy，用 Cross-Entropy Method（CEM）等方法搜索 z，再在未参与搜索的初始状态 / seeds 上评测。若搜索得到的 z 稳定更好，说明现有 conditional policy 已包含更好的行为，直接 task inference 尚有改进空间；原因还可来自表示近似、reward 积分误差等。若搜索未找到改进，表示、policy、coverage 以及搜索预算或算法本身都仍是候选原因。搜索后的成绩属于 adaptation，和直接推断的成绩分别报告；成本包括新增 rollout。相关扩展见 [S09 §6–8](zero-shot-rl-primer.html)。

这些是用于区分原因的检查。实现定义计算过程，独立评估提供能力证据；先明确各项证据能回答的问题，再选择下一步干预。

## 8 · 温习与继续阅读

| 本次想补的缺口 | 主材料 | 实践入口 |
|---|---|---|
| PPO 与连续动作训练闭环 | 本讲义 §1–2；T2 原路线 | [练习 P1–P2](rl-control-practice.md) |
| HORA / AnyRotate 的工程与部署边界 | §3 + [DexLab digest](../../references/dexlab-sim-rl-digest.md)；[HORA](https://haozhi.io/hora/)、[AnyRotate](https://maxyang27896.github.io/anyrotate/)原项目 | P3、P6 |
| TD3 / SAC 更新与价值学习诊断 | §4 + [基础讲义](rl-foundations-primer.html) | P4 的 replay 闭环与 SAC 固定 batch 对拍；可用已有正式复现证据替代同目标的 toy |
| FB 基础和泛化边界 | §5 + [Zero-Shot 专题](zero-shot-rl-primer.html) | P5；继续 [S09 八单元](s09-zero-shot-rl.md) |
| 形成当前项目的判断 | §6–7 + 已有正式 reports | P6 的一页 comparison / failure card |

本路线是跨课的复习入口，不增加 Lesson。练习单给完整设计、learner-owned core 与 verifier；读完材料或看过参考演示都不自动获得 Pass。
