# 机器人 RL：从经典更新到生成策略与 BPTT 的分段实践

> 先读对应讲义，再选一个 session。课次与顺序以 [curriculum](../../curriculum.md#rl-基础讲义与课次的对应t2t3-共用) 为准。P 编号标练习入口；学员进度单独记录。
> 本文件提供题目、实验设计和验收条件。开课时 AI 提供 harness、独立 oracle 和图表，学员拥有核心实现、预测与干预。设计预算与参考预期不算 learner evidence。

## 0 · 开课顺序与共同合同

推荐顺序：**P0 恢复 → P1/P2 PPO → P4 TD3、SAC → P7 生成策略 → P8 BPTT（选修）**。P8 可在经典算法对拍完成后独立进入。P3 DexLab、P5 FB 和 P6 综合诊断仍在原课次按需进入。每次到一个自然停止点；已学过的知识用一批情境题校准，按错题和疑点补讲。

三个 Pass 的边界：Understand 能解释一个带具体输入、输出和反例的机制；Implement 有自己完成的核心代码及独立 verifier；Improve 还要有 failure hypothesis、单变量 control 与 regression evidence。下列局部验收不自动替代 T2/T3 的课程级 Exit Gate。

每次运行前，在自己的 session record 中固定：

- environment / code / wrapper 版本与 seed、reward 和 action 单位、termination/truncation 语义；
- 自变量、保持不变的项目、机制预测、推翻预测的观察；
- transition / optimizer update / wall-time 上限、停止条件、评价与 checkpoint 选择规则；
- 数据身份与保存位置。训练结果不进 Git，正式项目证据链接 registry / report。

下面 CPU toy 数量是**开课时预算建议**，须在 session 中定稿；当前任务不启动训练。若正式 Isaac / GPU / 真机训练或产生可引用 checkpoint，按根仓库授权执行。现有正式实验可替代相同学习目标的 toy，但不能事后伪造预测或把同一证据计成一次新干预。

## P0 · 恢复价值学习

**对应** T2·S01；**先读** [基础讲义 §2](rl-foundations-primer.html#2--从数据学价值) 的 MC/TD、backup 和 Double Q。**停止点** 在新场景中能说明样本来自哪里、target 接哪段未来、如何判断 bootstrap；用一句自己的话记住最容易混淆的区别。MDP/Bellman 已清楚时直接从本组开始。

共用场景：机械臂把夹住的物块放入盒中。一次尝试是一个 episode；`step(a)` 执行动作并取得 reward 和实际到达的 observation；`reset()` 摆回初始物块和机器人姿态、开始下一次尝试。下面各题独立，可一次回复 `1. …；2. …`，不确定项标 `?`。每题一句判断或一行伪代码即可。

1. **记录与 reset。** 采样器规定每局最多采 20 步，第 20 步得到 `r=0`，到达盒口附近的 observation `f`，任务本身还可继续。`reset()` 返回物块在桌面的 `z`。这条 transition 的 `next_obs` 存什么？构造一步 target 时读取哪个 observation 的 V？
2. **时限变成任务定义。** 现在改成“必须在 20 步内放入”；policy 的状态包括剩余时间。仍在第 20 步到达 `f`，尚未完成。相比第 1 题，bootstrap 和 reset 哪一个决策会变？为什么？
3. **从数据到 target。** 一条完整尝试的 rewards 是 `[0, 0, 10]`，第三步后真正结束。为第一步的状态估值，用符号写出一步 TD、两步 TD、MC 各读哪些 reward 和哪个 V。若只改变第二步到达状态的 V、轨迹不变，哪些目标会变？
4. **评价谁。** replay 的第一步是人类实际执行的动作，得到 `r, s_next`；其后记录的人类动作是 `a_h`。现在要评价自己的 policy。TD target 的下一动作由谁给？当前 Q prediction 的动作是否也要换成自己的？
5. **多步的 off-policy 偏差。** 人类先对准、再稳妥放下成功；你的 policy 对准后会立刻松手而失败。若直接用人类的两步回报评价自己的“先对准”，偏差从哪一步带进来？若两步本来是已执行的一整个 macro-action，先要澄清哪一个 action 接口？
6. **Double Q 的挑和评。** 下一状态的候选是 `左、右`。`Q_A=[0.8,0.4]`，`Q_B=[0.3,0.7]`。这次更新 A，用 A 挑、B 评。target 的尾部应该读哪个动作的哪个分数？若改成各自取最大再取 min，还在评价同一个动作吗？
7. **Claim audit。** “只要 Q-learning 是 off-policy，就能从任何 replay 学出从未被执行过的动作有多好。”判断它的问题，并指出还需要什么证据。

陪练先按“episode 边界 / target 数据来源 / backup policy / 选择偏差”归类，再只补错题和疑点。未通过的项用新变体再检核；本组诊断不替代 T2·S01 的 tabular 实践，也不提前登记 Pass 或 ledger。

## P1 · PPO 的一次更新：先确认算的是什么

**对应** T2·S03/S04/S05；**先读** 统一讲义 §1.3、§2；**停止点** 数值和 gradient 语义能对拍，不训练 policy。

先对照[统一讲义 §2](rl-control-review-primer.html#2--ppo新-rollout-如何变成策略更新)，合上材料写出 `采样 → return / baseline → GAE → ratio/clip → actor/critic step → 下一轮采样`。指出 REINFORCE 的完整 return 到 GAE 的变化，以及更新 policy 时哪些数据保持不变。AI 提供固定小 tensor fixture 与独立标量 oracle；learner 完成 `compute_gae` 和 `ppo_surrogate` 的核心计算，并口述哪些输入 detach。

可直接用的首轮输入（教学 fixture，不是实测数据）：`gamma=0.9, lambda=0.8`；普通 buffer 的 `r=[1,0,2]`、`V_old=[0.4,0.5,0.6,0.7]`（末项为 bootstrap value）、`b=[1,1,1]`、`c=[1,1,0]`。另用单步 `r=1, V_old=2, V_final=4, V_reset=100` 分别标为 terminal 与 time-limit truncation，观察是否错误使用 reset value。Clip fixture 固定 `epsilon=0.2`、`A∈{+1,-1}`、`ratio∈{0.6,0.9,1.0,1.1,1.4}`；先写预测，再让 oracle 给答案。

| Fixture / intervention | 固定项 | Verifier |
|---|---|---|
| 一条普通 trajectory、一条真实 terminal、一条 continuing-task timeout | reward、old value、final observation；先给完整输入再预测 | 手工 TD residual / return 与向量代码一致；timeout bootstrap 使用 final value，新 episode 不串入 GAE |
| A 取正/负，ratio 横跨 1±epsilon；改变是否 clipping | 同一 old log probability、advantage 与 epsilon | 在非拐点处用 finite differences 验证 surrogate 斜率；负 A 分支不能照搬正 A 的平顶方向 |
| Actor loss 中 advantage 是否 detach | 同一计算图与输入 | 对 baseline 参数检查 gradient；learner 说明错误梯度改变了哪个目标 |

每次只选择一行作正式 intervention，其余是先决 correctness checks。验收需同时给出实现值与独立 oracle 值，不只 assert 两段相似代码相等。预算建议：CPU、单次检查不超过 60 秒。Claim audit：**“PPO clip 把所有 ratio 强制限制在区间内。”**

## P2 · 连续 PPO：接上真实闭环

**对应** T2·S06、T3·S01；**先读** §1–2；**环境** Gymnasium [`Pendulum-v1`](https://gymnasium.farama.org/environments/classic_control/pendulum/)，锁定 wrapper 与版本。它提供 3-D observation、1-D torque action（范围 -2 到 2），默认 200 steps truncation；是练习连续 action 的小环境，不是灵巧操作替代品。

AI 提供 env loop、记录、evaluation 与 plotting；learner 完成 diagonal Gaussian log probability、固定 rollout 的 PPO update，以及一个明确 action mapping。可以先用 raw Gaussian + 环境 clipping，但必须保存 raw sample 的 log probability，明确 evaluation 的 mean action；若改用 tanh policy，把 Jacobian 与 scale 纳入测试。

**开课前先验证参考 baseline。** 本课尚无已实测可行的 PPO 配置；原先的 `100k transitions / 10 分钟 CPU` 只保留为参考实现的首轮可行性探查上限，任一先到即停，达标与否如实记录。它不是学员必须学会 Pendulum 的预算保证。

AI 先用选定的参考实现检查可行性，再安排 learner training prediction：

1. 固定 reference revision、完整超参、action distribution、observation / reward normalization、硬件和线程数。特别说明采用普通 Gaussian 还是 gSDE（state-dependent exploration），这些选择会改变练习内容；选用新机制时先补相应讲解。
2. 预先固定 calibration seeds `{10,11,12}` 与每个 seed 的 20 个 evaluation episodes，评价原始、未归一化的 episode return。候选开课门槛为每个 seed 的 mean return ≥ −250；这是验收目标，当前尚无达标结果。先做 seed 10 的首轮探查，未达标即结束本轮准备并说明结果，不要求学员找 bug。
3. 参考方法达标后，将完整配置、全部 calibration 结果和实际 transition / wall-time 成本写入 session card，冻结可行的学员预算。若配置或预算仍需调整，作为课程准备单独记录并重新验证，避免把调参过程算成学员实验。
4. 学员用独立 training seeds `{0,1,2}` 与事先固定的 evaluation seeds 运行；对照各 arm 使用相同 seed 集合，评测数据仅用于预定报告，禁止因成绩差追加 seed 或延长预算。

本次只修订开课条件，未运行上述参考训练。环境/资源不足时，先完成 P1 与 policy / value / action 接口验证，训练练习待参考可行性明确后再开展。

**Intervention 二选一**：固定总 transitions 与 rollout size，改变 minibatch epochs；或保持其他项不变，改变 GAE lambda。先声明为何可能影响 sample efficiency / policy drift，别同时改 learning rate。

**Verifier**：报告每个 seed 的 return、episode length、approximate KL、clip fraction、entropy、value explained variance、环境样本数与更新数；画固定 evaluation return，不用 training shaped loss 代替效果。Baseline 未学会时，先与同条件的 reference 比较：两者均未达标，优先检查预算与配置可行性；reference 达标而自有实现明显偏离，再结合 P1 对拍与动作/归一化检查定位实现差异。低 return 本身不能确定是代码错误。

与锁定 revision 的 [CleanRL PPO continuous-action 实现](https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy)核对目标和接口；Pendulum 局部练习通过不自动满足 T2·S06 的完整参考成绩 gate。Claim audit：**“把 simulator 的 num_envs 增大，等同于把时间 horizon 拉长。”**

## P3 · Teacher/student：先讲机制，再按需读代码

**对应** T3·S03/S04，补 T3·S01；**先读** 统一讲义 §3。本段以 DexLab 作为案例；基础温习无需安装 simulator 或逐条追踪源码。

陪练先用一张 information / gradient flow 图讲解，再逐项讨论：

1. **两个训练信号**：Stage 1 的 PPO reward 与 Stage 2 的 supervised targets 各更新谁？
2. **部署输入**：teacher 的 privileged inputs 由 student 的 history 替代后，哪些不确定性可能仍然存在？
3. **采样分布**：teacher mixing、均值动作、reset 与 DR 分别怎样改变访问到的状态？
4. **指标与评测**：给出 `student_episode_reward=0, student_episode_count=0`，先解释分母，再说明如何测纯 student。该配置持续 mixing，训练统计按“任何 mixing 步均标 assisted”过滤，因此没有合格的纯 student episode；这与“有样本但 return 为零”含义不同。

**Learner-owned 产出**：一张简洁图，标明 teacher / student 可见信息、loss、更新参数与 evaluation 对象；附一个关于分布偏移或指标解释的判断。**Verifier**：能沿图解释一个行为例子，区分 supervised loss、混合训练 return 和纯 student 能力。

**可选源码深入**：沿 [固定版本导读](../../references/dexlab-sim-rl-digest.md)检查 dispatcher → observation / forward → optimizer → reset / evaluation。第 4 步可追踪 `episode_assisted` 从设置、过滤到计数 / reward / best selection 的路径。只为关键判断保留少量证据链接，无需逐字段整理整仓源码。

**Improve 设计**：提出 teacher mixing 与纯 student 性能的区分性对照，冻结 checkpoint / bank / object-axis / noise seeds。有实际条件时再执行；只完成设计时记录为设计产出。Claim audit：**“Stage 2 复用了 PPO player，所以 reward 会通过 PPO 更新 student。”**

## P4 · TD3：旧数据怎样给 actor 一个方向

**直接入门试读**：[TD3 第一篇](td3-first-lesson.md)先用短讲义看懂一次 update，无需先读完整 foundations 或通过 P0；导师另见[问答稿](td3-first-lesson-tutor.md)。读完再按下面内容分次做实现与工程练习，不将本篇阅读算作完成 P4。

**对应** T3·S02；**先读** §4 + 基础讲义 13–17。先固定 batch 做机制验证，再选择小训练或既有真实复现证据。

合上材料写出 `action + exploration → transition → replay → TD target → critic step → 按计数更新 actor / target`。Learner 写 `td3_target`、双 critic loss、`actor_loss`、delayed update 条件与 Polyak update；AI 管 replay、评测与画图。这里练一套可复用的 replay 闭环，后面的 SAC 只替换有算法差异的更新。

**先决 verifier**：

- 指定 Q1/Q2 常数与 noise fixture，逐项核对 target min、noise clipping、action clipping、bootstrap mask；
- 用可解析 `Q(a)=-(a-a*)^2` 验证 actor gradient 指向 a*，确保固定 critic 参数时保留 action gradient；
- 按固定 update counter 检查 actor / target 的更新时机与 Polyak 更新数值。

首轮 target fixture：`r=1, gamma=0.9, b∈{0,1}`，target actor 输出 `0.95`，预先给定 noise `0.30`、noise clip `0.20`、action bounds `[-1,1]`，两 target critics 在最终 action 上输出 `3` 和 `5`。Polyak fixture 约定 `target_new=(1-tau)*target_old+tau*online`，取 `tau=0.1, target_old=2, online=10`；注意有的代码把互补权重命名为 polyak。先手算，再与代码对拍，避免随机采样掩盖错误。

**小训练建议**：沿用 P2 的 `Pendulum-v1` task、action bounds 与评测方式，TD3 的参考配置与可行预算单独验证。固定 random warmup、replay capacity、batch size、exploration noise 和 UTD 的定义。选择 **target min → 只用 Q1 target** 或 **actor delay 2 → 1** 中一个 intervention；其余保持不变。参考准备、learner seeds 与停止规则沿用 P2 的流程，不预设两种算法需要相同样本数才学会。

**Verifier**：除 return，还画 Q1/Q2 差异、固定 evaluation states 上 Q 与折扣 rollout estimate 的偏差、action saturation、replay age / updates。短 rollout estimate 要注明 tail 截断；不能把它当精确无限时域 Q。单 seed 的 Q gap 不直接证明系统性 overestimation。

**怎样比较 PPO 与 TD3。** TD3 会反复利用 replay，在 Pendulum 这类任务上可能以更少的环境样本学会；PPO 需要持续采集新 rollout，因此样本效率出现明显差距是合理的观察，并不自动表示 PPO 实现有 bug。先分别验证可行 baseline，再画相同 transition 横轴的 learning curves，同时报告达到同一 return 门槛的样本数和 compute cost。探索方式、归一化与超参也是差异来源；若 gamma 不同，还需注明优化目标的时间视野不同。由此讨论样本效率，不将单个 toy 的结果外推为手内操作的方法排名。若使用 RLT / EXPO 正式证据，必须回答同一 target/action/behavior identity 问题，保持原 evaluation contract。Claim audit：**“增加 UTD 就等于拿到了更多环境经验。”**

### P4 内 SAC 对照：同一 replay，两种更新

**先读** [基础讲义知识点 15](rl-foundations-primer.html#知识点-15--sacmaximum-entropy-rl-与自动温度)。先在纸面重建整个 SAC 循环，并对照 TD3 标出：policy 本身如何探索、next action 由谁采样、target 是否含 `log_prob`、actor 优化什么、谁更新 `alpha`。注明当前采用常见的 twin-Q SAC 实现；target update 频率、固定或自动 temperature 是实现配置。

**Learner 写的部分**：squashed Gaussian 的重参数化采样与 log-prob（含 tanh Jacobian 和 affine action scale）；soft target；actor loss；`log_alpha` 更新。复用环境、replay、评测与 target critic 同步工具。Q 的参数在 actor step 冻结，但要保留 Q 对动作的梯度；temperature step 使用 detached 的 entropy 统计。

| 固定 batch 对照 | 一次只变什么 | 独立 verifier |
|---|---|---|
| Soft target | 固定 `next_action`、两份 target Q、`r`、`b` 与 `log_prob`，只打开/关闭 entropy 项 | 标量 oracle 核对 `r + gamma * b * (min_Q - alpha * log_prob)`；再用 `b=0` 检查整段尾部被屏蔽 |
| Action density | 固定 Gaussian、noise 和 action range，只切换是否补 Jacobian | 用一维变换密度对拍；验证采样动作与保存/重算的 log-prob 属于同一变量 |
| Temperature | 固定 actor 和 critic，给定分别高于、低于 target entropy 的 detached log-prob | 独立判断 `alpha` 的更新方向，核对该步不会更新 actor/critic；不同 log-alpha loss 可能有相同方向但不同尺度 |

首轮 soft target fixture：`r=1, gamma=0.9, min_Q=3, log_prob=-0.7, alpha=0.2, b∈{0,1}`；entropy 对照先用 `H_target=-1`，分别固定 `mean(log_prob)=0.5, 1.5`。每个 session 先锁预测，再选择一行作单变量干预；其余是先决正确性检查。Claim audit：**“SAC 只是在 TD3 的探索动作上多加一点随机噪声。”**

**停止点**：能在同一批 transition 上重建、对拍两种更新，并指出哪些代码可复用。可运行的学员更新加一次独立对照，验证的是 SAC 更新机制；评估是否训出有效控制还要单独跑参考配置与闭环。继续小训练时沿用 P2 的校准、预算与独立评测规则；将完整 PPO、TD3、SAC 的曲线放在一起是算法比较，不能归因成某一项 loss 的单变量效应。

## P5 · FB：换 reward，与重新训练分开计时

**对应** T3·S09；**先读** §5 与 Zero-Shot 讲义 §1–4；继续复用 [S09 的 SR / SF / FB 单元](s09-zero-shot-rl.md)，不另写一套已求解的走廊演示。

**第一站：exact oracle。** 在既有有限 MDP 中，冻结 dynamics、gamma、reference distribution rho、reward 的 next-state convention；从讲义参考模型得到 exact values 作 verifier。Learner 写 `infer_task_vector(B, rewards, rho_weights)`，验证积分权重与 shape；用小非正交 B 判断 task encoding 与 reward projection 的区别。

**第二站：new task。** 提前锁定 3 个不参与训练/选择的 reward functions，例如原走廊新的到达偏好与避让成本。在 frozen representation / policy 上推断 z，记录 reward evaluations、推断耗时、return 和 exact optimality gap。按任务来源标记结果：显式给出 reward function、在 reference states 上评价并推断 z，称 zero-shot；仅给少量 reward-labelled samples，称 few-shot task specification / zero-planning。两种方式都冻结网络，test episodes 仅用于评价。若额外用 rollout 搜索 z，则作为 adaptation 另列数据与预算。

**第三站：coverage intervention。** 在相同 environment 和表示容量下，仅改变预训练 transition 数据是否覆盖某段通路；训练预算与 reference/task-inference 规则保持一致。为避免“删数据导致 rho 也变了”的混淆，明确把 transition sampling 与独立 reference-state sampling 分开。固定 reference states 不能补回缺失 dynamics；若所用实现强制二者共用 replay，改用 dropout/reweight protocol 时需先说明新的对照含义。

完整 learned FB 的 F/B/policy/target training 继续使用 S09 练习，不把 exact-SR 计算包装成 learned FB。已有 neural toy 是讲义参考证据，不是本轮 learner run；本轮只在先写 prediction 后选择一个未完成的 control。预算建议：exact fixture CPU 60 秒内；neural control 先在 session 中确定且不超过已有参考训练规模，禁止为补图默认重跑。

**Verifier**：权重更新计数为零的 test 阶段、rho 一致性、task reward 未泄漏、同一 MDP oracle、各任务 return 与 coverage。改变 friction / latency 属于另一项 dynamics generalization 测试，不能与本实验的新 reward 同时改变。Claim audit：**“FB 的 z 与 HORA 的 adaptation latent 都是低维向量，因此可直接互换。”**

## P6 · 把方法放回灵巧操作与当前项目

**对应** T3·S03/S04/S08；**先读** §6–7；**停止点** 一页可审查 comparison / failure card。

选一个具体现象，例如“teacher 在仿真转得好，student 在某类 box 上频繁掉落”。先分层列出可检验解释：history/input gap、supervision distribution、controller / reset mismatch、object sampling / reward confound。每个解释至少给一个能否证它的检查，不以“调 DR”作为全部答案。

| 必填字段 | 最小内容 |
|---|---|
| 已知事实 | 固定 source/config/日志/评测身份；缺的证据明确留空 |
| 方法选择 | PPO、TD3 或 FB 将改变哪个接口；为何当前数据/预算允许它 |
| 核心 comparison | 固定一组物体/轴向/初始状态，teacher-only 与 student-only；再选择一个主要干预 |
| Score | fixed-horizon turns、掉落率、存活时间、各物体/轴表现；提前定义短失败 episode 与 timeout 的计分 |
| 限制 | 是否只有 sim；是否看过 test 结果后调参；是否存在 teacher assistance 或 privileged leakage |
| 下一步 | 最小需要的 evidence、预算和停止条件；关联正式计划入口 |

材料范围内的代码阅读与 proposal 可以完成 Understand；没有独立干预结果，不补记 Improve。要达成 T3·S08，还需满足原 capstone rubric 与研究目标，不能用这张练习表替代正式 proposal。

## P7 · 生成策略后训练：比较改进发生在哪里

**对应** T3·S05–S07；**前置** 三种经典更新已完成伪代码与固定 batch 对拍；先补 [基础讲义](rl-foundations-primer.html) 18–29 的 coverage、IQL/weighted extraction、expressive policy。讨论 denoising/flow 内部梯度前，先读 [A2](../../curriculum.md) 的生成机制。**停止点** 一份可复核的代码差异表，加一项机制证据。

先用 [S06 讲义](s06-primer.html) 的同一条 transition 问三件事：buffer 存的是哪个实际动作；backup 接哪个下一动作；actor 或生成器更新的 loss 从哪来。再把 `生成过程中写入改进` 和 `保留生成先验、另学决策` 分开。训练信号进入哪条路径，与 base 有没有被其它 loss 更新，分别记录。

| 机制 | 原文 / 现有材料 | 读代码时的辨认点 |
|---|---|---|
| likelihood policy gradient | [DPPO](https://arxiv.org/abs/2409.00588) | 概率比作用在哪个 denoising step；环境 step 和生成 step 是不同时间轴 |
| advantage-conditioned generation | [RECAP](https://arxiv.org/abs/2511.14759)、[AG](https://arxiv.org/html/2610.19783v1) | Advantage 怎样变成监督条件；AG 的高低条件引导发生在 flow field，`action imagination` 在同一个状态上采动作，不展开动力学 |
| reference-conditioned actor / latent steering | [RLT](s07-rl-token-online-adaptation.md)、[DSRL](https://diffusion-steering.github.io/) | RLT actor 输出动作 chunk；DSRL actor 决定送进冻结生成器的初始 noise |
| select/edit、backup 与 base 写回 | EXPO / EXPO-FT 代码导读（内部资料）、[DICE-RL](https://arxiv.org/html/2603.10263v2) | EXPO rollout 和 backup 都选 OTF；DICE-RL 的 best-of-N 执行与多样本平均 backup 分开读。EXPO-FT 的 RL 梯度与可训练 VLA 的监督更新是两条路径 |
| behavior support | [GSBR §4](https://arxiv.org/html/2611.05295v1) | Base 只供参考样本；actor 读参考集合。损失约束离估计 support 的距离，不要求复制各 mode 的频率 |

**实际只做一个首轮对照。** 优先复用 [EXPO-lite 练习](s06-expressive-policy-value-maximization.md) 或 [EXPO-FT 案例集](expo-ft-casebook.html) 的已有证据；另一选项是下表的 support toy。需要完整 T3·S06 的 Pass 时继续按原课的 evidence gate。

| 对照 | Learner 写什么 | 固定条件与验收 |
|---|---|---|
| 同一 backup 换下一动作 | `otf_select` 与 target；其余走现有 harness | 固定 transition、候选、critic 与数据，单次只把 next-action 规则从一个预指定 base 样本切到 OTF。独立枚举候选，核对 target 变化归因于哪一条；然后按原练习设计验证闭环 |
| 同一先验换 regularizer | `paired_penalty`、`support_penalty` 与统一目标的评分 | 固定一维候选网格 `[-2,2]`（步长 `0.01`）、base 样本 `[-1.1,-0.9,0.9,1.1]`、配对 reference `-1.1`、support 半径 `0.2`、`Q(a)=-(a-1.45)^2` 与权重 `1`。只把 `distance(a, reference)^2` 换成 `distance(a, union_of_intervals)^2`，用区间距离 oracle 验证 loss，记录最优动作相对各 mode 的位置、离 support 的距离与 score。此 toy 固定半径、固定 Q；它只验证约束几何 |

**源码产出**：在一个固定 revision 和配置上，把传统更新的 `data → next_action → target → loss → optimizer → execution` 对到一篇工作（首选仓库已有 pin 的 RLT 或 EXPO-FT）。标出三处有意义的改动、哪些参数只对某个 loss 冻结，并预测关掉一个改动的结果和反证。GSBR/AG 可先用原文算法和公式对照，不逐篇重写 trainer。`Understand` 要能在新案例里说清改进发生的位置；`Implement` 要有自己写的机制和对拍；`Improve` 需要已执行的失效对照及独立评测，或符合相同目标的正式证据。Claim audit：**“没有 RL 梯度穿过 VLA，就等于整个 EXPO-FT 训练都冻结 VLA。”**

## P8 · 可微动力学与 BPTT

**对应** T3·S10（选修）；**先读** [SuperMimic §3.2–3.3、§4.3](https://arxiv.org/pdf/2610.28480)。**前置** 三种经典更新已对拍，理解短 rollout 的 tail bootstrap 与 actor gradient。**停止点** 能解释同一前向轨迹为什么会有不同的反传路径，再把 toy 的变量对到论文。

SuperMimic 用 simulator transition 拟合可微 world model。在 actor update 内固定 world-model 参数，从 replay state 展开短 imagined rollout，以想象 reward 和末端 V 组成目标；BPTT（backpropagation through time）把后面状态和回报的梯度传回早期动作。外层仍会用新 simulator 数据拟合模型；真实 simulator 不需要提供梯度。末端 V 补的是截短 rollout 以后的 return；是否把这个 V 的输入梯度传回早期动作，是另一项选择。论文在同一个 world model 上还有 PPO-WM 对照：用同一个预测模型采样，不等于用模型的 Jacobian 更新 policy。

**最小实践，不搭人形仿真栈。** 用一个标量位置控制器，固定 `x0=1, theta=-0.1, gamma=0.95, H=8`。Policy `a=theta*x`，可微 dynamics `raw=0.9*x+0.5*a`，该步 reward `-(raw^2 + 0.1*a^2)`，`x_next=raw`；累加折扣 reward，末端加 `gamma^H * V(x_H)`，固定 `V(x)=-x^2`。Learner 写 `unroll` 和 loss，先口述梯度应经过哪些边；AI 提供固定输入、独立有限差分 oracle 与记录。

| 验收项 | 控制和证据 |
|---|---|
| 完整 BPTT 的正确性 | 固定参数、状态和 horizon；对同一个 rollout 目标做中心差分，对拍 `dJ/dtheta`。短步长检查数值稳定性；保持 world-model 参数固定且保留其输入梯度。 |
| 单变量 gradient 对照 | 保持相同前向值与 reward 表达；只在每步算完 `raw` 和该步 reward 后，将送往下一步及末端 V 的 `x_next` 改成 `detach(raw)`。比较相同 return 下的梯度；detach 版仍有即时 reward 对当前动作的路径，完整前向目标的数值导数只用来验证完整 BPTT。此对照不等价于 PPO。 |
| 进阶 model-error 诊断 | 另开一次实验，保持 policy、初始状态、horizon、reward 与训练预算不变，只改 imagined dynamics 的一个系数；在真实 toy dynamics 上独立评估同样的更新。报告预测误差、梯度方向/尺度和真实 return，区分模型偏差与优化失败。 |

`Understand`：能画 `sim 数据 → model fit → imagined rollout → actor / critic`，解释为什么 “同样用 world model” 可以对应 PPO-WM 和 BPTT 两种更新。`Implement`：自己写 rollout，完成完整梯度的独立对拍及 detach 干预。`Improve`：用 model-error 对照诊断改进为何在 imagined 与真实 toy 中不一致，并检查 held-out 初始状态。读论文时标注 model fit、imagined reward、V bootstrap、actor update 的位置；toy 证据仅覆盖梯度机制。Claim audit：**“SuperMimic 必须对原始物理模拟器求导。”**

## 陪练恢复提示

“读取我的 baseline、progress 和本练习单。恢复时用一批情境题集中校准；看过的内容按证据跳过。经典算法先让我重建伪代码和做固定 batch 对拍，之后再比较研究源码。先给完整实验设计，再由我做 prediction、核心实现和解释。每次推进到一个自然停止点，把实际证据登记到对应课次。”
