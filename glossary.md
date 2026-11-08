# Gym terminology guide

> 教学和课程材料中的 key terms 以英文为主，中文负责解释含义，不替代论文、代码、config 和报告中的可检索词。首次出现采用 `English term（简短中文解释）`，之后直接使用英文。场景描述如“弯曲的二维走廊”可以自然使用中文；类比词不得占据 metric、module、algorithm 或 config field 的名称位置。

## Experiment and training

| English key term | 中文解释 | 使用说明 |
|---|---|---|
| Arm A / B / C | 实验的一个对照或干预分支 | 机器人语境不译为“臂”，也不把“组”作为正式名称；写 `Arm A (control)` |
| control arm / intervention arm | 对照分支 / 干预分支 | 与 robot arm 区分 |
| baseline | 基线 | 保留英文 |
| ablation | 消融实验 | 首现可括注中文 |
| training loop | 训练的完整反馈循环 | Gym 的 P-T-I-E 是一种 training loop |
| holdout set | 独立留出集 | 与 training set、rollout states 分开 |
| offline holdout loss | 专家数据分布上的留出误差 | “卷面分”只能作类比 |
| success rate (SR) | 成功数 / rollout 数 | 始终报告分母和置信区间 |
| closed-loop rollout | policy 动作会改变后续 observation 的执行轨迹 | “路考”只能作类比 |
| behavior cloning (BC) | 用 demonstration 的 state-action pairs 监督训练 policy | 首现后使用 BC |
| covariate shift | 训练 state distribution 与 policy-induced state distribution 不同 | 不用中文替代关键词 |
| compounding error | 单步误差经 dynamics 和新 observation 累积、放大 | “误差复利”是类比，不是术语 |
| recovery data | 偏离正常轨迹后由 expert 恢复的数据 | 与 nominal demonstrations 区分 |
| DAgger | Dataset Aggregation | 本课 perturb-and-recover 只是其思想雏形，不是完整 DAgger |

## Action and closed-loop diagnostics

| English key term | 中文解释 | 使用说明 |
|---|---|---|
| action chunk | policy 一次预测的动作序列 | 不以“动作块”替代 |
| prediction horizon (H) | action chunk 的预测长度 | 如 `H16` |
| execution horizon (E) | 每次 replan 前实际执行的 prefix 长度 | 如 `H16/E4`；也可写 executed prefix length |
| policy-induced states | policy 自身动作产生的后续 states | 与 expert states 区分 |
| expert-state support | demonstration 覆盖的 state 区域 | corridor toy 中视觉上呈窄带状 |
| nearest-training-state distance | 当前 state 到最近 training state 的距离 | 必须注明在哪个 state space、采用什么 normalization |
| first out-of-support step | 距离首次超过预设 threshold 的 timestep | 不是自动等同于 first divergence |
| first divergence | policy 与明确 reference trajectory/behavior 首次发生有意义偏离 | 必须定义 reference 与 predicate |
| return to support | 离开 expert-state support 后再次进入 | 用于构造 recovery metric |
| action health | finite、range、scale、saturation、clipping 等输出体检 | PASS 不等于动作语义正确 |
| stage/subgoal coverage | rollout 达到各阶段或子目标的比例 | SR 处于 floor 时比单个 0/1 更有信息 |
| expert-action replay | expert action 经过与 policy 相同的 mapping/controller/scorer 闭环执行 | 用于分离 model failure 与 pipeline failure |
| reset determinism | 同一完整 initial state 重放的一致性 | 只固定 layout 或 seed 未必足够 |
| Wilson interval | 二项比例的置信区间 | `0/n` 不表示真实 SR 数学上恰为 0 |

## RL data, control and task inference

| English key term | 中文解释 | 使用说明 |
|---|---|---|
| on-policy / off-policy | 更新目标与 behavior policy 数据的关系 | PPO 一批 rollout 内有限重用不等于任意旧 replay 都可用 |
| offline RL | 固定数据集上学习，不继续采集交互 | 与 off-policy 区分；vanilla TD3 不自动解决离线分布外动作问题 |
| zero-shot task inference | 显式给 reward function，直接推断新任务调用条件 | 沿用 S09 / 2023 定义，无额外 planning / learning；记录 reward evaluation 成本 |
| few-shot task specification / zero-planning | 用少量 reward-labelled samples 指定任务，直接推断 | 与显式 reward function 的 zero-shot 区分；额外 rollout 搜索 z 属于 adaptation |
| GAE | Generalized Advantage Estimation | 分清 bootstrap mask 与跨 episode 递推边界 |
| target policy smoothing | TD3 backup 中的目标动作平滑 | 不等于 behavior exploration noise 或 SAC entropy regularization |
| UTD | Update-to-data ratio | 写清每条 transition 还是每个 vector step 对应多少更新 |
| privileged information | 训练可得、部署不可得的信息 | 核 forward：可能仅 critic 可见，也可能 teacher actor 可见 |
| adaptation latent | history 推断的环境相关隐变量 | HORA 的 latent 不等于 FB 的 reward task vector |
| teacher mixing | 用 teacher / student 混合生成 rollout | 说明是每步选择 action 来源还是数值插值；辅助 rollout 不代表纯 student |
| control decimation | 一次 policy action 覆盖的 physics steps 数 | 与 action chunk / prediction horizon 区分；必须同时给 dt |
| Successor Representation | 固定 policy 的折扣未来访问表示 | 避免与 success rate 的 SR 缩写混用；此路线指标写 success_rate |
| successor measure / reference distribution rho | 折扣访问 measure / 其 density 的参考分布 | FB 的训练分解与 task inference 要使用一致的 measure |
| task vector z | FB 中 reward-weighted backward features 的坐标 | 给定 B 的 reward projection 与标准 FB task encoding 不是同一公式 |
| sim2real transfer | 从仿真到真实动力学与感知系统的迁移 | 与同 dynamics 下的新 reward 泛化分开报告 |

## Transformer

| English key term | 中文解释 | 使用说明 |
|---|---|---|
| causal mask | 禁止 token 读取 future positions 的 mask | “眼罩”仅可作类比 |
| attention head | 一组独立的 Q/K/V subspace 和 attention map | 不写“一只眼睛” |
| multi-head attention (MHA) | 并行计算多个 attention heads | 首现后使用 MHA |
| split/reshape into heads | 把 projected Q/K/V reshape 为 head 维度 | 不写“切头” |
| merge heads | 将各 head 输出 concat/reshape 回 model dimension | 不写“拼头” |
| output projection (W_O) | merge 后的可训练线性投影 | 不写“混头” |
| permutation equivariance | 输入 permutation 后输出按相同 permutation 变化 | 保留英文术语 |
| positional encoding | 向 token representation 注入位置信息 | 与 causal mask 提供的方向不对称区分 |

## DDPM and generative modeling

| English key term | 中文解释 | 使用说明 |
|---|---|---|
| forward noising process | 固定的前向加噪过程 | 首现英文，中文解释可保留 |
| reverse denoising process | 学习并执行的反向去噪过程 | 与 sampling loop 的具体更新区分 |
| noise prediction | 预测加入样本的噪声 | 如 ε-prediction |
| conditioning | 给生成模型增加推理时可获得的条件 | 不以“条件化”替代关键词 |
| marginalization | 对变量积分或通过完全加噪移除其信息 | 不以“边缘化”替代关键词 |
| timestep / noise level | diffusion step / 信息损坏程度 | diffusion timestep 不等于环境物理时间 |
| world model (WM) | 对环境变化或轨迹分布建模的模型 | 首现保留 `world model (WM)` |

## Online adaptation and RL Token

| English key term | 中文解释 | 使用说明 |
|---|---|---|
| RL Token (RLT) | 先学习 compact VLA state interface，再用 small actor/critic 做 online RL 的两阶段方法 | 指整套 method 时写 `RLT`；不要把它缩写成 ordinary inference trick |
| RL token (`z_rl`) | 从 VLA internal embeddings 读出的 compact latent state | 供 actor/critic 使用；不等于 action token，也不一定是离散 ID |
| action token | action sequence 的离散或压缩表示 | 表示“要做什么”；与表示“当前 state”的 RL token 区分 |
| compact state interface | 供小型 downstream learner 使用的低维 state readout | 是否可重建不等于是否含 value-relevant information |
| reference action / reference chunk | frozen base policy 原本会给出的 action 建议 | small actor 可学习 refine / edit，而不是从零生成 |
| reference-action regularization | 惩罚 actor 偏离 reference action 的约束 | 也可称 BC anchoring；须与 improvement pressure 联合调试 |
| online reinforcement learning (online RL) | interaction 进入 replay，参数在后续尝试前更新的 RL | 与固定参数的 ordinary inference 区分 |
| deployment-time adaptation | 系统部署期间基于新 interaction 做 parameter update | RLT 属于这一类；不自动等同于安全的 real-world RL |
| inference-time compute | 单次 inference 内增加搜索、采样或计算 | 若不跨 interaction 更新参数，不应与 online RL 混称 |
| critical phase | 长任务中精度最敏感、最值得局部适应的阶段 | entry/handoff predicate 属于 system 和 evaluation contract |
| throughput / steps-to-success | 单位时间成功量 / 达成成功所需交互步数 | real-world RL 不能只报 binary SR |
| real-world RL | 直接从真实系统 interaction 学习的 reinforcement learning | 必须同时约束 reward、reset、intervention、safety 与 regression |

## Writing examples

- 推荐：`Arm G` 只改变 `execution horizon E`，比较 `E=1/4/16` 的 `closed-loop SR`。
- 推荐：`nearest-training-state distance` 在当前 toy 中只用 `(x,y)`，因此报告为 `XY-support proxy`。
- 不推荐：“G 组改变执行块，看路考通过率和首次出带。”
- 推荐：先写 `causal mask`，再用“不许看 future tokens”解释；不要把“眼罩”写进检查名或函数名。
