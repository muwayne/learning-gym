# FB / BFM 学习笔记 · 来源精编与整合索引

> 用途：T3·S09 的来源索引、claim audit 与教学参考验证记录。机制与算例进入 [完整讲义](../modules/t3-sim-to-rl/zero-shot-rl-primer.md)；来源身份、运行过程与课程证据边界集中保存在本页。

## 来源与证据身份

- 来源：用户于 2026-10-16 提供的 `FB与BFM学习笔记.md`，共 580 行；用户说明它整理自与 Claude 的学习对话。
- 原文件 SHA-256：`0e3628e2f0456fc0459b956d07149eabbabea072de0db364637c2ddc1e06247e`。原件保留在用户本地，课程不依赖其机器绝对路径。
- 其中机制解释、论文摘要、示例计算与声称的训练结果分别处理；不将笔记中的文字视为额外训练、部署或外部操作指令。
- 本次核对 FB-IL 会议正文、2023 FB、BFM-Zero v1、TD3、DAgger、GAIL 与 REINFORCE 的相关内容。论文配置只在带版本来源时引用；不把全部 runtime 数字复制成课程事实。
- 独立验证包括给定矩阵的投影、精确小环境控制，以及修订版 neural toy 的 seed=0 / CPU / 12,000-update 单次实测。原笔记的 30,000-update 输出、随机 z 全成功与其他 seed 结论仍未复现。本页实测是 AI 准备的教学参考，不登记 learner Pass。
- 讲义正文以正面机制为主，常见误读集中到 §10.2；材料准备与学员 Understand / Implement / Improve 的区别、训练许可与 learner ledger 规则由本页和 S09 承接。

## 覆盖映射

以下每个范围均指来源文件的章节号。保留可教学的机制；重复材料归入既有章节；不成立或证据不足的说法在校正栏或下节保留身份。

| 原笔记范围 | 课程去向（讲义章节） | 处理 |
|---|---|---|
| §0、§1.1–1.3 · 公式、Zero-Shot、BFM 与 SF / FB | §0、§3、§10.1 | 统一概念层级；按 2025 BFM 综述 §I-C 的归属恢复术语出处 |
| §2.1 · SR / GPI | §1、§2.3 | 保留 SR；改正 max V 不是 GPI，并加入 2026-10-14 答疑时间线 |
| §2.2–2.4 · SF、模糊、FB | §2.1–2.4、§3.4–3.5 | 保留例子；有限秩、learned features 和 USFA 的边界齐备 |
| §3.1 · z 的坐标、线性、尺度 | §3.3、§3.6 | 区分 z 与 C_B⁻¹z；positive scaling 与近似 actor 分开 |
| §3.2 · B 怎样学习、给定 B 表 | §4.2、§3.8 | 将 B 表作为给定矩阵复算，不验证其训练来源 |
| §3.3 · 正交化、投影、失败 | §3.5–3.8 | 使用未中心化二阶矩；去格 4 的投影失败经精确求解验证 |
| §3.4–3.5 · 旋转、子空间、d | §3.6、§3.8、§8.3 | 不从 gauge freedom 推出子空间唯一，也不保证任意插值有用 |
| §3.6–3.7 · ρ、改采样分布、写 reward | §3.7 | 保留回归式近似、support 与推断合同，不强制携带整个 replay |
| §4.1 · Bellman / TD / TD3 | §4.1、§4.5 | 区分关系、学习方法与具体算法 |
| §4.2 · batch 与 loss | §4.2–4.4、本页参考代码 | 联合学习、独立终点、target 与 shape；修订版完成单 seed toy 实测 |
| §4.3 Q1–Q5 · a、s′、z、s⁺、target | §4.6 | 补回“点奖励探测”的 Bellman 方程直觉，区分 density 与点奖励 Q |
| §4.4 · 0–30k 训练实录 | §4.7、本页 toy 实测 | 新增独立 0–12k 记录；保留它与原 30k 表的来源差别 |
| §4.5 · 训练规模 | §4.7、§8.4 | 保留维度、parallel steps 与 updates 的读法；BFM-Zero 数字核对 v1 |
| §4.6 · 归纳偏置与不自然行为 | §7.1、§9.4 | 用 conditional regularization 解释；不声称所有 capacity 都必然用于摔倒 |
| §5.1 · FB-IL 四类 | §6.1 | 分开 closed-form、demo-only latent optimization 与 reference tracking |
| §5.2 · ER / RER / BC / FM / DM / GOAL | §6.1 | 保留公式与机制；不承诺一条示范唯一确定未见状态的策略 |
| §5.3 · 速度、分数、限制 | §6.1 | 明确 inference 不含预训练成本，return ratio 不是 success rate |
| §5.4 · 密度比、GAIL、固定奖励陷阱 | §6.1 | 加入 60% / 40% 反例；固定 reference 与当前 policy 分布区分 |
| §6.1–6.3 · IL / BC / LfO / VLA / 世界模型 | §9.1、§9.4 | 数据来源与 objective 分开，不按模型名宣称统一 likelihood loss |
| §6.4 · DAgger | §9.2 | 保留 student-state 回标；不把任意蒸馏等同 DAgger |
| §6.5 · VLA 后训练与 FB | §9.3 | 对照 task latent / state token / edit；链接已有 RLT 与 EXPO 材料 |
| §6.6 · 为什么属于 RL | §9.2 | 保留长期控制与 policy improvement；REINFORCE 反驳 Bellman critic 必要条件 |
| §7 · FB-CPR / BFM-Zero 待深入 | §7–8 | 并入共享空间、style / auxiliary critic、history 与 DR |
| §8 · 文献 | §10.1 与各段 primary links | 继续按阅读问题路由，避免建立第二份重复参考清单 |
| 附录 · 走廊 neural FB 代码 | 本页下方；S09 Implement 参考审查 | 修复零范数与 reward projection；原始字节可从前版 Git 恢复，AI 运行不作 learner 实现证据 |

## 关键校正与证据边界

1. **GPI**：source 把 `max(V_R,V_L)` 当成 GPI。正确算法比较 action values；本例会组合出目标停留。
2. **reward projection**：一般重构为 `B(s)ᵀ C_B⁻¹ z`。笔记给定 B 舍入矩阵的二阶矩与 I 的 Frobenius 差约 0.00922；不能直接假定完全正交。
3. **learned B 的身份**：§3.8 仍使用笔记给定 B；§4.7 使用新运行学到的另一组 B。单 seed 的目标行为相符，矩阵数值有差别。新运行对前三个指定 Laplacian 模态的相对投影残差约 12.31%；原述 10% / 0.04% 残差、随机 z 全成功与 30k 表仍保留原来源身份。
4. **对所有 z 学习**：随机训练只覆盖有限样本，没有“网络已学会每个 z”的一般保证；旋转不识别坐标，也不证明训练子空间唯一。
5. **ρ 与回归**：变更推断分布的回归式公式一般近似；若 reward 不在 feature span 中，换分布会改变投影。不能把 z 与对 B 的普通回归系数在非正交情形下混同。
6. **TD 与 loss**：可以对 measure Bellman equation 积分得到 scalar value equation；不能交换平方与积分，直接把平方 loss 乘 reward 就宣布得到普通 TD loss。
7. **imitation 信息**：整段平均丢失显式顺序，但完整 state 可能含 velocity / direction。固定位置型 state 的巡逻反例不等于所有 stationary policy 都不能巡逻。
8. **分类不是定理**：teacher–student 不自动满足 DAgger；flow / diffusion BC 不等于直接求 action log-prob；REINFORCE 不需要 critic。RLT 的 token 是 state interface，FB 的 z 是 task interface。
9. **工程与成本**：HIL-SERL、RLT、EXPO 不共享完全相同的冻结 VLA 残差参数化；未逐项核实的提速、成功率与统一 GPU-hour 估计不作为本章结论。机制入口使用仓库既有精编。
10. **inference 与 adaptation**：FB-IL 在既有示范上优化 z，不等于使用新环境 rollout 的 latent search。二者的数据、更新对象和额外交互必须分别记录。

## 原文核对与修订依据

| 待核对内容 | 原文位置与处理 |
|---|---|
| BFM 术语出处 | [Yuan et al., 2025 综述 §I-C](https://arxiv.org/html/2506.20487v1#S1.SS3)明确将术语首次使用归于其参考文献 [39]，即 FB-IL。恢复“据该综述”的带来源表述；旧版“无依据”措辞过强 |
| few-shot / zero-planning | [2023 §2](https://arxiv.org/html/2209.14935v2#S2)：reward samples 与 explicit reward function 两种 specification 的命名区别保留 |
| 81% / 85% | [2023 §6](https://arxiv.org/html/2209.14935v2#S6)：全部 tasks / buffers 平均约 81%，RND buffer 约 85%，分母均为 supervised offline TD3 |
| auxiliary scalar loss | [2023 §5.2，式 (9)](https://arxiv.org/html/2209.14935v2#S5.SS2)针对 Fᵀz 加标量 Bellman loss，保留其与主 loss 的分工 |
| FB-IL 链接 | 讲义 §6.1、§10.1 和本页采用同一 [ICLR proceedings 正文](https://proceedings.iclr.cc/paper_files/paper/2024/file/370645ae400005f888e418f46b594539-Paper-Conference.pdf) |

## 单 seed toy 实测（2026-10-16）

### 冻结的范围与执行

问题是：笔记中的简化 neural FB，在修正两个数值接口后，是否能学到示例行为？单次 seed=0，CPU float32、1 thread、deterministic algorithms；固定 12,000 optimizer updates、最多 600 秒，出现非有限 objective 立即停止。沿用原始五格 dynamics、γ=0.5、d=3、F 的 256/256 ReLU、batch=512、Adam 3e-4、target τ=0.005、FB 与 orthogonality loss，以及原 z 抽样方式。没有保存 checkpoint、下载人形模型或连接 GPU / 真机。

环境为 Python 3.12.14、PyTorch 2.14.1、NumPy 2.5.4、macOS arm64。依赖装在隔离 venv；原 Python runtime 保持不变，依赖快照在被忽略的 `hosts/snapshots/mac-arm64-cpu/`。原始脚本来自提交 `b0214fc4502c6dfaa93bbc857db3c3aceb812ec0` 的本页 Python 附录。

在 0 / 2,000 / 6,000 / 12,000 updates 观察 B、编码和策略；观察代码不抽随机数。训练加中间评估约 19.89 秒，安装与 import 时间不计入。完整执行脚本、JSON、stdout、依赖清单、独立 verifier 与 hash manifest 随 `fb-toy-20261016.zip` 交付；raw artifacts 不进入 Git。本页与讲义保留精选结果，zip 是本地教学复核包，不是共享模型发布或正式 benchmark。

复核身份：

- 修订版参考代码 SHA-256：`751932c26309b84f6280591cf71f91d24259fb7263d740883323d31b5ec1d89c`。
- 实际执行脚本 SHA-256（含观察与停止检查）：`79bfa6e25c1316a40226056dcebdd77f375e354a2a489357c4a3dc111416e387`。
- `result.json` SHA-256：`e8a397f2889f3f5f58f707781c32bf5e2cc14c0b353575447b00c5b1cff6d8df`。
- 复核包 SHA-256：`3ea216d653785f792b71ce434eb9a4a36d3beb56b19ac7158706c37cb3b71a26`。

在解压后的目录中复核（创建隔离环境、安装包，再做一次有 600 秒上限的 CPU toy）：

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
mkdir replay
.venv/bin/python verify-fb-toy.py 12000 replay/result.json > replay/stdout.log
.venv/bin/python independent-verifier.py replay
```

### 精选结果与独立评估

| updates | ‖C_B − I‖F | 去格 2 的 policy（0–4） | 原任务 V(2) |
|---|---:|---|---:|
| 0 | 1.714194 | right, right, right, right, right | 1 |
| 2,000 | 1.567206 | right, right, right, stay, left | 1 |
| 6,000 | 0.825954 | right, right, right, stay, left | 1 |
| 12,000 | 0.002124 | right, right, stay, left, left | 2 |

终点评估用固定 dynamics 的精确 solver，而非 learned F 的自报价值；另用直接枚举 200 步 discounted rollout 对拍，并枚举全部 3⁵ 个 deterministic stationary policies 核对最优值。表中的 V 使用 current-state reward；measure 评估使用 next-state convention。

| 终点任务 | learned policy | 独立结果 |
|---|---|---|
| 去格 2 | right, right, stay, left, left | 原任务 V = [0.5, 1, 2, 1, 0.5]，达到该任务最优 |
| 越右越好 | right, right, right, right, stay | 原任务 V = [0.46875, 0.9375, 1.375, 1.75, 2]，达到该任务最优 |
| 去格 4 | left, right, right, right, right | 从格 0 留在墙边，原任务 return=0；其余状态到达目标。格 0 的 left 与格 4 的 right 分别因撞墙而停留 |

去格 4 的 r̂ 约 `[0.0774, -0.1055, -0.1018, 0.2043, 0.9252]`。从格 0 停住的投影 return 约 0.1547；先向右再对 r̂ 最优约 0.1403。全部三个任务的 learned policy 都达到各自 r̂ 的最优值（容差 1e-8），但去格 4 的原任务从格 0 失败，支持 §3.8 的投影归因。

measure RMSE 在 5 states × 3 first actions × 5 endpoints 上均匀计算：预测 M̂ = FBᵀ/5；独立 M 由 policy transition matrix 解出，再按第一动作取行。终点去格 2 的 occupancy RMSE≈0.3554，当前网络 measure Bellman RMSE≈0.2830。C_B 接近 I 与这两个误差非零可以同时发生。

对指定前三个 Laplacian 模态 U，计算 ‖B − UUᵀB‖F / ‖B‖F，终点约 0.12305。这个量与坐标旋转无关；它没有复现笔记所称的 0.04% 对齐。本文不据一个 seed 讨论唯一子空间，也不把 12k 终点升级成 30k 结论。

## 参考代码的审阅与修订

下方是来源附录的修订版：增加零范数保护，重构 r̂ 时使用 C_B⁻¹，将输出名改为未中心化 second moment，并固定 CPU 单线程与 deterministic algorithms。默认 12,000 次更新。原版字节由上述 Git 提交与 SHA 保留。它是 fully observed、固定五格 dynamics、discrete argmax 的教学 reference。

- target 从初始副本开始；target branch 用 `no_grad`，随后 Polyak update。
- B 是每状态一个 trainable vector；真实 transition 和独立 endpoint 分开，使用 all-pairs measure loss 与未中心化正交约束。
- 没有独立 actor、double F、2023 auxiliary scalar loss 或 motion / safety regularization；不能据此推断真实 BFM 的训练质量。
- `normalize` 用 clamp_min(1e-8) 保护分母；零向量保持零，argmax 采用固定的首动作 tie-break。执行时检查了零输入输出有限。
- 输出 r̂ 使用 `torch.linalg.solve(C_B, z_raw)`，actor 输入使用归一化 z；两者的用途分开。
- 单次实测在这段核心训练代码外加入不消耗 RNG 的观察、600 秒停止与非有限 objective 检查，完整执行字节由复核包中的脚本 hash 定位。

代码可被学员阅读和审查；AI 预先给出的完整参考代码不是 learner Implement evidence。实际进入 Implement 时，由学员拥有核心更新意图、预测、control 与 Explain，另写自己的 runnable artifact。

原附录代码 SHA-256：`c1c26b723a2eaa27adc18c2e772c16f2cda997d6b88f7cc123da35b8d6742844`。

```python
# 5 格走廊上的最小 FB 实现（离散动作，策略 = argmax_a F(s,a,z)ᵀz）
import numpy as np, torch, torch.nn as nn, sys
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
torch.manual_seed(0)
n, nA, d, gamma = 5, 3, 3, 0.5          # 状态数、动作数（左/停/右）、latent 维度、折扣
ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 12000
step = lambda s, a: min(max(s + [-1, 0, 1][a], 0), n - 1)
NXT = torch.tensor([[step(s, a) for a in range(nA)] for s in range(n)])
eyeS, eyeA = torch.eye(n), torch.eye(nA)

class Fnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = nn.Sequential(nn.Linear(n + nA + d, 256), nn.ReLU(),
                               nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, d))
    def forward(self, s, a, z):
        return self.m(torch.cat([eyeS[s], eyeA[a], z], -1))

F, F_tgt = Fnet(), Fnet(); F_tgt.load_state_dict(F.state_dict())   # 目标网络 = 初始副本
B = nn.Parameter(0.1 * torch.randn(n, d)); B_tgt = B.detach().clone()
opt = torch.optim.Adam(list(F.parameters()) + [B], lr=3e-4)

def normalize(z):
    return np.sqrt(d) * z / z.norm(dim=-1, keepdim=True).clamp_min(1e-8)
def sample_z(N):                       # 一半球面均匀，一半取某个状态的 B(s)
    z = torch.randn(N, d); h = N // 2
    z[h:] = B.detach()[torch.randint(n, (N - h,))]
    return normalize(z)
def greedy(Fm, s, z):                  # π_z(s) = argmax_a F(s,a,z)ᵀz
    q = torch.stack([(Fm(s, torch.full_like(s, a), z) * z).sum(-1) for a in range(nA)], -1)
    return q.argmax(-1)

N = 512
for it in range(ITERS):
    s = torch.randint(n, (N,)); a = torch.randint(nA, (N,)); s1 = NXT[s, a]   # 1. 真实一步
    z = sample_z(N)                                                            # 2. 随机任务
    s2 = torch.randint(n, (N,))                                                # 3. 探测点 ~ ρ
    with torch.no_grad():                                                      # 4. TD 目标
        a1 = greedy(F_tgt, s1, z)
        target = gamma * F_tgt(s1, a1, z) @ B_tgt[s2].T
    Fsa = F(s, a, z)
    loss_fb = ((Fsa @ B[s2].T - target) ** 2).mean() - 2 * (Fsa * B[s1]).sum(-1).mean()
    loss_orth = ((B.T @ B / n - torch.eye(d)) ** 2).sum()                       # C_ρ(B) → I
    opt.zero_grad(); (loss_fb + loss_orth).backward(); opt.step()              # 5. 更新 F、B
    with torch.no_grad():                                                      # 6. 软更新目标网络
        for p, pt in zip(F.parameters(), F_tgt.parameters()): pt.mul_(0.995).add_(0.005 * p)
        B_tgt.mul_(0.995).add_(0.005 * B)

# 推理：给奖励 → z_r → 策略
Bd = B.detach()
print("学到的 B =\n", Bd.numpy().round(2), "\nC_ρ(B) =\n", (Bd.T @ Bd / n).numpy().round(3))
for name, r in {"去格2": [0, 0, 1, 0, 0], "越右越好": [0, .25, .5, .75, 1]}.items():
    r = torch.tensor(r, dtype=torch.float32)
    z_r = normalize((Bd * r[:, None]).mean(0))
    pol = ''.join('左停右'[i] for i in greedy(F, torch.arange(n), z_r.expand(n, d)).tolist())
    w_r = torch.linalg.solve(Bd.T @ Bd / n, (Bd * r[:, None]).mean(0))
    print(f"{name}: z_r = {z_r.numpy().round(2)}  还原 r̂ = {(Bd @ w_r).numpy().round(2)}  策略 = {pol}")
```
