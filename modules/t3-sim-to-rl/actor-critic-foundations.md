# Actor–critic foundations · RLT 原理与诊断课表

## 0. 地图与范围

为“有 RL 理论档案、但 actor / critic / Q / TD3 的现役理解不稳”的学员设置。它是 T3·S07 从基础衔接到应用诊断的教学路线，不新增课次数，不替代 T2/T3 的完整实现 gate。最终目标：能沿自己 RLT 的数据与更新路径解释模块职责，针对 Q / loss 与真实行为的矛盾提出并验证一个区分性对照。

| 单元 | 必讲范围 | 学员任务 / 自然停止点 | 材料入口 |
|---|---|---|---|
| 1 · 角色与接口 | actor、critic、Q-function；reward 与 return；动作输出与评分 | 用自己的话标明系统图中谁输出什么；不要求求导 | 本文 §1 |
| 2 · 一次学习循环 | replay；真实 reward + 后续估计；critic/actor 更新；target networks；旧动作与新动作 | 沿一条 transition 解释信息来源、谁更新谁冻结、错误反馈怎样回到学习 | 本文 §2 |
| 3 · TD3 稳定机制 | twin-min、policy delay、target smoothing；各自的作用位置与局限 | 对一个不稳定场景选择对应机制并预测影响；不把三个机制都变成独立长实验 | 本文 §3 |
| 4 · 接回 RLT 实现 | actor/critic 网络结构选择与取舍；frozen VLA/token；flow reference 与 small chunk actor；BC anchoring；实际 chunk 执行时长；Oracle correction；online/test-time/real-world 三维区分 | 对照代码说明结构与输入/输出的关系，标出 proposal / executed action / BC target，解释与标准 TD3 的差异 | 本文 §2 接线校准与 [T3·S07 §9](s07-rl-token-online-adaptation.md#9--从真实实现问题切入的-session-routing) |
| 5 · 故障诊断与对照 | Data → Value → Policy → Behavior；估值误差、reward 对齐、覆盖、reference 与辅助成功的区别 | 对一个失败现象提出竞争解释、固定项、干预项与结果判据 | 本文 §4 与 T3·S07 §9 的短 session B/C |
| 6 · 最小实践与收口 | 一次本地 CPU toy 的完整 Predict → Verify → Intervene → Explain；联系 PR evidence | 学员选择一个干预并口述关键逻辑，AI 搭 harness；比较真实回报与估值，留下实验记录和诊断卡 | 下方实践约定 |

这是先前四段路线的展开：原“角色、学习信号、TD3”保留为1–3；原“RLT诊断”明确拆成实现接线、故障对照和验证收口。不是临时增加三个新主题。个人完成状态只维护在 `learners/<name>/progress.md`，不把“已讲解”标成“已掌握”。

**节奏与答题方式**：每次恢复先报“当前单元 / 本次目标 / 自然停止点”；每次跨单元更新个人进度。讲解后只给一个带场景的机制解释、代码路径判读或对照设计任务，通常2–3句话。已出现的有效回答直接作为证据，不为补表重考。公式只按需解释；不默认手算、求导或偏微分。提前讨论后续单元时明确标为预览，再回到主线。

**实践约定**：只选一个主要问题做小实验，例如“新增已执行失败动作的经验，能否纠正 critic 对这些动作的偏好”。运行前再完整给出 toy、reward、数据范围、固定预算、评测起点、单一干预和判据；学员先预测后运行。AI 负责环境、优化器与可视化，学员负责口述关键 update/data-flow 意图、选干预并解释结果。不能把人工固定评分曲线或AI单方面验证教具算作学习训练闭环。真实机器人/PR训练不在该 CPU 实践范围内。

**收口**：原理阶段以一张完整系统图和一个有证据边界的诊断解释结束；实践阶段以一个可运行结果与其局限结束。Understand / Implement / Improve 按原有 gates 分别记录；没有改善与 regression evidence 时不声称 Improve 完成。Flow matching 推导、从零实现完整 TD3、重做 π0.5 训练均不是本路线的额外必做题。

默认 Improve 是最终深度，每段有自然停止点。首段互动是概念校准，不冒充训练实验；完整 P-T-I-E 从第二段清楚展示实验设计后开始。学员负责机制解释、预测和判断；AI 提供图示、harness 和机械计算与验证。手算或求导不是本衔接课的默认 gate。

## 1. 四个词处在不同层级

- Actor：执行 policy 的模型。确定性写法为 `a = μθ(s)`；随机性 policy 则输出动作分布或其参数。`θ` 是 actor 参数。
- Reward `r`：环境根据实际 transition 按任务规则返回的即时信号。
- Return：从当前开始累积的 reward，常用折扣 `G = r0 + γ r1 + γ² r2 + …`；`γ` 决定未来 reward 权重。
- Q-function：`Qπ(s,a)` 是在 state `s` 先执行 action `a`、以后遵循 policy `π` 时的 expected return。`Q*` 是采用最优后续策略时的对应值；不要把不同后续策略的 Q 混为同一对象。
- Critic：学习 value estimate 的模型。在 TD3/RLT 的本课语境中是 `Qφ(s,a)`，输入 state 与 action、输出一个标量。`φ` 是其参数。更广义的 actor–critic 也可以采用 V-based critic。
- TD3：Twin Delayed Deep Deterministic Policy Gradient，一种训练 actor 与 twin Q critics 的算法；不是第三个网络，也不是 Q-function 的别名。

Qπ 是期望这个数学对象，critic 是对它或控制目标 Q 的参数化近似。Q estimate 不是环境真值，不一般等于成功概率；奖励尺度、折扣和后续行为都会影响它。学习到的网络也可能输出不合理的数值。

执行时 actor 可以直接给动作，不必每次调用 critic 枚举所有候选。训练 actor 时则利用 critic 对动作的评价/梯度来调整 policy。

演示：[一步对齐：动作、估值与反馈](demos/actor-critic-roles.html)。每次从 x=0 开始，目标 x=+0.6mm，action 是一次位移，范围 [-1,+1]mm；执行一次后结束。进入目标±0.1mm时 reward=1，否则0。这里没有后续 reward，所以 true Q 恰好等于 reward。多步课程会解除这一简化。

演示的 critic 输出是人为构造的错误估计，不是训练结果；滑块手动指定假设 actor 输出，不运行 actor 网络或梯度更新。通过先比较估值、再点击执行，观察“critic 看好”与“环境成功”可以不一致。

第一段自然停点：学员说明三个输出各是什么，并指出尚不清晰的关系。新概念未确认前，不跳到 TD target 算题。

## 2. 学习信号（后续 session）

### RLT 接线校准与本次短 session

依据 PR #70 已发布 snapshot `74c2d1c54245c960ef51fdeae396ac74b9e853f4` 的 networks.py（内部资料）：`ChunkActor.actor_mean` 从 `z_rl`、proprio 与 reference chunk 直接输出 `[batch, chunk_len, action_dim]` 的动作均值；`sample_action` 可以加入 fixed-std Gaussian noise。它不是输出 flow vector field 的 action expert。`QNetwork.apply` 接收 state features 和 action chunk，输出一个标量 Q estimate；可选 duration conditioning 要另看启用配置。

VLA action expert 的 flow vector field 经积分生成 reference action chunk；online small actor 利用 reference 与 RL token 生成自己的 action chunk。Flow integration time 不等于机器人控制时间。Critic 估计执行动作及后续行为的 discounted return，不是输出 reward function，更不是给 vector field 直接打 reward。即使在其他 flow-policy RL 方法中 actor 内部使用向量场，也必须先查 critic 的输入接口，不能据此推定 critic 评价向量场。

本次路线：接线与 gradient → 一次 critic/actor 交替更新 → TD3 三机制的作用位置 → RLT failure control。每段一个有上下文的主任务；学员先解释信息来源、预测变化方向或选择区分性干预，AI 再运行 CPU harness；正确则直接推进。公式是机制说明工具，不默认要求求导或手算；本次 Wayne session 明确采用原理与诊断导向。

可选数学演示（不作默认题目）：固定同一 state，单维控制、两步 chunk `A=(a0,a1)`，toy actor 直接用两个参数 `A=theta`，初值 `(0.2,0.8)`；人工 critic `Q(A)=1-(a0-0.6)^2-(a1-0.4)^2`。学习率0.1，梯度上升，只更新 actor；无 noise、BC、环境执行或 critic update。工具可展示梯度与更新结果，学员关注“评分变好是否意味着真实行为变好”。0.6/0.4仅为人工 critic 的峰值，不是已验证的环境正确动作；两步均为 toy 评分输入，不代表实际项目 chunk execution contract。

先展示完整 transition `(s,a,r,s',terminated)`，区分 actual executed action 与 BC correction target，再讲：

`y = r + γ (1−terminated) Qtarget(s', μtarget(s'))`

TD（temporal difference）用“已观察的 reward + 对后续的估计”构造 target，critic 向 target 回归；target 是估计，不是完整真实 return。终止时后续项为0。一般 time-limit truncation 不自动等于任务 terminal，须按任务语义处理。

`Lcritic = mean((Qφ(s,a)−stop_gradient(y))²)`

`Lactor = −mean(Qφ(s, μθ(s)))`

Actor update 时固定 critic 参数，但保留 critic 对 action 的梯度，经 action 传回 actor；不能把整个 Q 输出 detach。低 TD error 只说明拟合当前 target，不能单独证明 Q 准确。

先做 terminal case，再加一段 bootstrap；最后才将 γ 扩展为实际执行时长 γ^k。合成数据与 actor/critic 初值在出题前明确，学员锁预测后才运行。

## 3. 标准 TD3（后续 session）

TD3 针对估值误差被 policy 放大的问题，采用三项核心机制：

1. Twin critics：构造 bootstrap target 时取两个 target Q 的较小值；不是两网络永不同时错。
2. Delayed policy updates：critic 更新更频繁，actor 与 target networks 较少更新；不是提高采样延迟。
3. Target policy smoothing：计算 TD target 的动作加入截断噪声并限制在合法范围；与采集动作的 exploration noise 作用位置不同。

目标网络是主网络的慢更新副本，帮助稳定 target。标准 actor objective 使用 Q1，而非要求 actor 也总取 twin-min。

实际实现只看到 twin critics，不能直接判为标准 TD3；必须检查 target、更新频率、噪声和 regularization。项目的 fixed-std sampling、reference BC anchoring、chunk duration 等适配分别标注。

Policy delay提高的是每次actor更新之间的critic更新次数，不自动新增环境数据或扩大action coverage。更多更新只能重用已有经验；可能改善暂时拟合不足，也可能继续拟合错误target。不要将“延迟更新→估值必然更准→动作分布自动更广”作为因果链。

### 单元4的网络结构入口

Actor与critic是职责，TD3是更新算法，MLP/Transformer等才是网络结构。接线讲解时必须补充结构取舍：先检查输入信息是否足够，再看表达能力与归纳偏置，最后看更新成本。当前snapshot的ChunkActor用投影与MLP输出chunk均值、配fixed-std Gaussian；TwinCritic包含两套分别初始化的QNetwork。MLP适合这里已压缩的向量接口且更新便宜，不代表是唯一或已验证最优选择；更大网络不能补回缺失的接触/历史信息，单峰策略不能凭扩大网络自动变成多峰分布。具体替代结构的效果需控制输入、数据和预算比较，不能直接归因为参数量。

Reference只是输入，不保证actor具备精细chunk生成能力。该ChunkActor直接输出完整chunk，没有硬编码为reference加residual；需要足够容量表达相关时序关系。动作精度不等于网络深度，较小MLP同样输出连续精细数值。容量假设需固定输入/数据并控制训练与评测口径，对比表示与拟合能力；不能仅凭任务需要精度就断言网络必须扩大。

### 单元4的reference、executed action与BC target

Reference作为actor输入与作为BC监督目标是不同用途。当前snapshot的`Learner._step`中actor接收`reference`，critic拟合使用`action`，BC使用`bc_target`（缺省退回`reference`），并受eligibility mask约束。

教学时间线：s0下actor预测10步，仅执行前2步到s1；随后Oracle在s1生成并执行纠正动作B，达到s2并成功。Critic的两条transition分别记录s0下实际执行的actor前缀与该段reward、s1下实际执行的B与该段reward；bootstrap跨越实际执行时长，并以对应执行后state为起点。不能把Oracle那段即时reward/动作回填为actor段的实际执行事实；未来价值按明确的后续策略另行估计。

BC监督必须与目标动作所对应的起始state匹配：B作为s1的纠正标签，不能仅因整个episode成功就作为s0的正确动作标签。若Oracle独立针对s0另行生成建议，该建议可以在符合训练eligibility约定时作为s0的BC target，但未经执行不能替换critic的executed action。Oracle-assisted成功不等于actor独立成功。

## 4. 诊断地图（后续 session）

只记四问，具体仪表按需展开：

- Data：replay 记录的是实际 state/action/reward 吗，是否有关键成功与纠正覆盖？
- Value：critic 是在什么 actions 上估值，target 是否合理，actor-Q 是否偏离经验 return？
- Policy：actor 在追求什么，Q pressure 与 reference/BC 约束如何竞争？
- Behavior：动作落到环境后发生什么，reward、success、Oracle assistance 与几何 proxy 是否被混用？

这些是排查入口，不是从单一曲线唯一归因的规则。一次只选一个可区分的 hypothesis/control。

## 6 · 最小实践：模仿与critic指导的配对实验

Wayne的CPU脚手架：[s07-critic-guidance.py](../../learners/wayne/code/t3/s07-critic-guidance.py)。默认运行只展示合同；actor objective先留TODO与执行guard，在学员预测记录后按学习意图填入。现在可用`--run`复现固定比较。没有checkpoint输出，不运行GPU或真实机器人。

完整设计：一维横向纠偏，初始误差x在[-0.4,0.4]mm，一次动作a在[-1,1]mm；最终误差x+a，执行一次即terminal。实际reward=1-(x+a)^2，成功为绝对误差≤0.1mm。Reference=-x+0.25mm，故意保留0.25mm偏差；它是人工有偏prior，不是Oracle成功示范。

Critic使用17个训练状态（间隔0.05mm）×41个动作（[-1,1]间隔0.05mm）的697条环境执行记录，训练1000次后冻结。Actor输入[当前误差,reference]，结构2→16→1、tanh输出；critic输入[误差,动作]，结构2→32→32→1，隐藏层tanh。Actor两组完全相同初始化、17个训练状态和reference，Adam lr0.002、各600次full-batch更新。Arm A只模仿reference；Arm B同样模仿并接受冻结critic的指导；模仿权重1、critic指导权重4，固定不扫参。随机seed20260909、CPU单线程、训练墙钟上限60秒；失败/超时保留为结果，不为预期效果静默改参。

评价：16个训练网格间的固定新起点，[-0.375,0.375]mm间隔0.05mm。主指标为实际最终绝对误差均值；次指标为成功率、实际reward、critic评分与reference偏差，全部使用固定最后一次更新，不挑最好时刻。保存每个起点的配对结果以便解释差异；critic另有未参与训练的state/action网格误差检查。环境函数只用于生成replay与独立评测，不作为actor优化目标直接求导。记录数据identity和运行配置。

范围：真实优化小型actor/critic网络，但这是一段terminal的固定数据实验，刻意不含VLA、chunk、target bootstrap或完整TD3；不能外推成RLT/插孔收益。冻结训练后的critic可把“评分改变”与“actor跟随评分”拆开观察。学员先预测相对效果并说明所依赖的critic条件，再以口述意图补actor objective。对照切换是否启用critic指导是本轮唯一干预。

已运行教例：[16个配对起点的交互结果](demos/critic-guidance-results.html)；本次运行配置与汇总结论见[Wayne记录](../../learners/wayne/records/t3-s07.md#单元6固定比较结果)。图示只回放一次真实CPU运行的结果，不在浏览器里训练模型。

## 卡点问答：确定的 Q 数值为什么仍能给 actor 梯度？

单次输入的输出是数值，整个网络是函数。固定critic参数与state后，不同action仍可得到不同Q；gradient描述Q随连续action的局部变化，不要求critic输出概率分布。多维action的gradient是各分量的局部变化率组成的向量，也不是概率分布。

Critic update对critic参数求导，让replay动作的预测贴近TD target；actor update固定critic参数，但经Q对action的梯度更新actor参数。可微critic从初始化时通常就能对action求导；训练目标是让估值更有经验依据，不保证局部动作梯度正确。仅从冻结critic查询更多候选动作不会新增真实监督；把候选动作实际执行并记录后果才可能增加学习证据。

TD3机制选择题需自足地列明候选机制范围，不要求从远处回忆名称；同一失稳表象可能有多个根因，不把题目预期机制当作唯一正确修复。

## 卡点问答：纯BC与critic指导到底差在哪里？

术语约定：在本课，critic网络就是参数化的Q网络，不是critic之外另有Q模块；单次输出称“Q评分（预期累计回报估计）”。讨论训练时明确说“用critic指导actor”，不交替用裸词Q指网络、输出或训练项。

Behavior cloning (BC)是监督模仿：给定状态与示范动作，让actor输出接近该示范。纯BC的actor不使用critic评分来学习，也不需要训练critic。Oracle是示范的来源，不是与BC并列的一种loss；新增Oracle数据可能改善BC，也可能因覆盖、标签或执行问题未带来改善。

此前“BC+Q”指“模仿训练 + critic指导”：actor既学习接近示范，又利用critic对其动作的Q评分与梯度争取更高回报。区别是critic是否实际影响actor更新，而不是代码中是否存在critic对象。两组保留相同critic更新、仅切换actor的Q项是隔离实现变量的一种可选做法，不是纯BC的要求；初次教学不要用这个工程细节遮蔽BC定义。

检验额外收益的对照：使用完全相同的原数据与新增Oracle示范，比较纯模仿训练和模仿+critic指导，匹配actor初始化、更新预算和独立评测条件。不能把新增示范与启用critic指导同时发生后的改善直接归因RL。

## 来源链接

- [TD3 原论文](https://proceedings.mlr.press/v80/fujimoto18a.html)：估值误差与 TD3 动机。
- [Spinning Up TD3](https://spinningup.openai.com/en/latest/algorithms/td3.html)：定义、目标、三项机制与伪代码。
- [PI RLT](https://www.pi.website/research/rlt)：compact token、small actor/critic 与 reference action。
