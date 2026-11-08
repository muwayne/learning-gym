# RL 问答精编（Spinning Up 学习档案浓缩版）

> 源自 wayne 的个人学习档案（2025-11 → 2026-07，Spinning Up 主线），2026-08-29 浓缩入库作课程共享参考。完整原档留学员本地；本文由档案所有者发起 PR 维护。
>
> 用途：T2 各节材料以 **Q 编号** 引用本文（唤醒线索，不是免做凭证）；其他学员可借它快速对齐术语。

## Q1–Q20 问答索引

| # | 问题 | 浓缩答案 |
|---|---|---|
| Q1 | stochastic policy 每次同 state 出不同动作？ | 它定义条件分布 π(a\|s)；采样有随机性，分布本身由 state 决定 |
| Q2 | θ 是 action 吗？ | 不是。θ 是产生 policy 的全部可训练参数：π_θ ≠ θ ≠ a |
| Q3 | μ_θ 与 π_θ 的关系？ | DDPG/TD3：μ_θ(s) 本身就是 deterministic policy；SAC：μ_θ 只是 Gaussian 的 mean 参数，完整 π_θ 还含 σ_θ 与 tanh squash；PPO：直接用 π_θ(a\|s) |
| Q4 | 为什么用 log π 而不是直接对概率求导？ | log-derivative trick：∇p = p·∇log p，让轨迹期望的梯度可用采样的 log-prob 梯度 × 权重估计 |
| Q5 | `.log_prob(act)` 是什么？ | 当前 state 条件化的分布在给定 action 处的 log 概率/密度 |
| Q6 | advantage 依赖 policy，为何不让梯度穿过？ | actor update 中它是估计好的固定权重（stop-gradient）；critic 有独立回归损失。"数学上依赖 θ"≠"自动微分图里允许梯度通过" |
| Q7 | Bellman 为什么重要？ | 把无限时域 return 变成一步递归，定义 value 的 fixed point，让 bootstrap/TD/DP 式迭代成为可能 |
| Q8 | Bellman 的 T 是什么？ | 函数到函数的 operator：输入整个 V/Q，输出 backup 一次后的新函数；V^π = T^π V^π，Q* = T*Q*；γ-contraction 保证 tabular 收敛 |
| Q9 | Policy Optimization 必须学 Q 吗？ | 不必须（REINFORCE 可不学）；加 critic 是为降方差、提效率 |
| Q10 | Q-learning 连续动作为何难？ | argmax_a Q(s,a) 在高维连续空间难求 → DDPG/TD3 用 actor μ_θ 近似 maximizer |
| Q11 | DDPG 为何要 target network？ | bootstrap 目标依赖另一个 Q 与 actor；不减速就会追着剧烈变化的目标回归（deadly triad） |
| Q12 | twin Q 为何有用？ | 两个近似的残余误差不全相关，取 min 压制被 actor exploit 的过估计 |
| Q13 | target policy smoothing 为何有用？ | 防 critic 在单点形成假尖峰；要求好动作在邻域内价值稳定 |
| Q14 | SAC 与 TD3 的核心差别？ | TD3=deterministic actor+标准 reward；SAC=stochastic actor+maximum-entropy 目标+soft Bellman backup |
| Q15 | PPO 为什么要 ratio？ | 数据来自旧 policy，需评估新 policy 对同一 action 的相对概率变化 r_t = π_new/π_old |
| Q16 | clip 是梯度裁剪吗？ | 不是。clip 的是 ratio 对 surrogate objective 的贡献 |
| Q17 | L^CLIP 是 loss 吗？ | 是要最大化的 objective；代码取负交给 Adam minimize，别被变量名 `loss` 误导 |
| Q18 | TRPO 换成 Adam 就行？ | 不行。TRPO 的定义含 KL 约束步：HVP + CG + line search；optimizer 不可简单替换 |
| Q19 | Adam 能训这些算法吗？ | 能且常见。optimizer 只管"怎么用梯度更新"；objective/数据/target 构造才决定算法身份 |
| Q20 | flow/diffusion policy 为何让 PPO 变麻烦？ | 无廉价 normalized log-density → ratio/likelihood 方法不自然 → RECAP 类改写为 advantage-conditioned 监督生成：π̂ ∝ π_ref·exp(A/β) |

## 算法对照表

| 算法 | Policy | 数据 | Critic | 核心更新思想 | 主要稳定机制 |
|---|---|---|---|---|---|
| VPG | stochastic | on-policy | 可选 V | direct policy gradient | baseline/advantage |
| TRPO | stochastic | on-policy | V | KL 约束优化 | CG + line search |
| PPO | stochastic | on-policy | V | clipped surrogate | ratio clip |
| DQN | greedy from Q | off-policy | Q | Bellman optimality | replay + target net |
| DDPG | deterministic μ | off-policy | Q | actor 近似连续 argmax | replay + target + Polyak |
| TD3 | deterministic μ | off-policy | twin Q | 稳定化 DDPG | min-Q + smoothing + delay |
| SAC | stochastic π | off-policy | twin soft Q | max-entropy actor-critic | entropy + twin Q |

分类习惯：至少同时看 on/off-policy、actor 是否显式、stochastic/deterministic、critic 学 V 还是 Q——这些是不同维度，不是一棵互斥树。

## 核心公式速查

```text
G_t = Σ γ^k r_{t+k}                    A^π = Q^π − V^π
V^π = E[r + γV^π(s')]                  V^π = T^π V^π,  Q* = T* Q*
∇J  = E[∇log π_θ(a|s) · Â]
GAE: Â_t = Σ (γλ)^l δ_{t+l},  δ_t = r_t + γV(s_{t+1}) − V(s_t)
PPO: r_t = π_new/π_old;  L^CLIP = E[min(r·Â, clip(r,1±ε)·Â)]
DDPG: y = r + γ(1−d)·Q_targ(s', μ_targ(s'))
TD3:  y = r + γ(1−d)·min_i Q_{i,targ}(s', a'_smooth)
SAC:  y = r + γ(1−d)·[min_i Q_{i,targ}(s',a') − α·log π(a'|s')]
RECAP: π̂(a|o) ∝ π_ref(a|o)·exp(A(o,a)/β)
```

## 十句长期结论

1. Policy 是决策分布/映射，不是某个 action，也不是 θ。
2. log π(a|s) 是把 sampled action 连回参数的桥梁。
3. Advantage 决定动作被鼓励还是抑制。
4. Bellman 的意义在定义可迭代逼近的 fixed point。
5. Operator T 的输入输出都是函数。
6. PO 可不学 Q；Q-learning 也可因连续动作引入 actor。
7. Actor-Critic 是架构分解，不是与 on/off-policy 平级的分类。
8. DDPG/TD3/SAC 的设计多在对抗 bootstrap+函数逼近+off-policy 的不稳定。
9. 论文 maximize ≈ 代码 minimize 负 loss，别信变量名。
10. 这套语言的价值是能继续分析 VLA post-training、RECAP 与现代机器人 RL。

## 档案自列缺口 → 课程落点

| 缺口（原档 §9） | 落点 |
|---|---|
| 从零独立复现任一算法 | T2 全线（这就是 T2 存在的理由） |
| tabular DP / grid-world 手算 | T2·S01 |
| importance sampling 独立推导 | T2·S05 |
| DQN 完整 walkthrough | T2·S07（选修） |
| SAC 自动 α tuning | T2·S07（选修） |
| Spinning Up 实验工具链 | 不补：课程用自建 harness + CleanRL 对照 |
