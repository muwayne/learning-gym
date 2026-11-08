# DexLab：HORA / AnyRotate 的 sim-to-RL 代码导读

> 2026-10-17 · 服务于 [统一温习讲义](../modules/t3-sim-to-rl/rl-control-review-primer.html)与 [P3 / P6 实践](../modules/t3-sim-to-rl/rl-control-practice.md)。本文件是可选的实现附录，基础温习以统一讲义为主。这里保留固定源码的阅读证据，不作为训练复现报告或对作者原创贡献的鉴定。

## 1 · 阅读身份与恢复

- Repository：Synapath/dexlab（内部资料），需要组织访问权限。
- 本次阅读 pin：`5c066f4c4691659b1ec504daea74af6a33db2633`（2026-10-14），来自 `dev/jeff/dex_rotate_h13_multi_objs`。
- 先前检查的默认分支 `dev/jeff/dex_rotate` 停在较早的 `69b4e15722d16de2dad0a7a20609624fc36e5d92`；本文不把它称为 main，也不以分支名替代固定 SHA。
- 参考 clone 位置：主仓 `ref/synapath-dexlab`，detached checkout；不要求安装 Isaac、下载 LFS assets 或获取 checkpoint。本任务更新的 main 是 `tpc_manip` 的 main。
- 阅读范围：HORA、AnyRotate teacher/student，尤其 H13 taxel 配置；其他手型、BPS / spatial / finetune 变体需单独解释配置。未执行训练或真机验证。

新机器只读恢复（下载源码并写本地 `ref/`，不安装依赖；若目录已存在先检查 HEAD / status，不覆盖已有工作）：

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none "$DEXLAB_REPOSITORY" ref/synapath-dexlab
GIT_LFS_SKIP_SMUDGE=1 git -C ref/synapath-dexlab checkout --detach 5c066f4c4691659b1ec504daea74af6a33db2633
git -C ref/synapath-dexlab status --short
git -C ref/synapath-dexlab rev-parse HEAD
```

下文的源码位置均以同一 SHA 为准。若已有私有项目权限，先把 `DEXLAB_REPOSITORY` 设为你获准访问的 URL。历史窗口与配置值描述实现身份，不证明某个训练预算已经跑完。

## 2 · 先沿调用链读，不按目录大小猜职责

| 顺序 | 固定源码入口 | 读什么 |
|---|---|---|
| 1 | training/train.py（内部资料） | Task integration 有 custom trainer 时调用它；否则创建 rl_games PPO algorithm。IsaacLab 负责环境接口，不应把 PPO optimizer 也归到 IsaacLab |
| 2 | hora_stage1/env.py（内部资料） | 薄的组件绑定层，委托共享 in-hand 环境 |
| 3 | inhand/env.py（内部资料） | DirectRLEnv 子类；pre-physics / apply-action / observation / reward / reset lifecycle |
| 4 | simulation/control/joint.py（内部资料） | Incremental action、target、滤波与 controller；action 不是直接 torque |
| 5 | asymmetric_actor_critic.py（内部资料） | Privileged encoder 的 latent 与 proprio 拼接，actor / value 都消费；按 forward 判断信息可见性 |
| 6 | hora_stage2.py（内部资料） | `adapt_tconv` optimizer、detached teacher target、latent MSE 与 student rollout |
| 7 | anyrotate_stage2.py（内部资料） | Supervised losses、可选 replay、teacher mixing、assisted episode 标记与纯 student 指标 |

最重要的结构判断是：**framework 复用很多；task / control / sensing / adaptation / evaluation 的实现仍然实质性地决定行为。** 单个 env 文件短，不能推出工程只改了几个函数；也不能反过来以共享组件代码行数声称全是 Jeff 新写的算法。

## 3 · 先用 HORA 理解 adaptation，再进入触觉

HORA 的 observation（内部资料）将 joint position 与 target history 接入 policy；privileged state（内部资料）含 position、scale、mass、friction、COM，不能把全部 latent 严格解释为可辨识的物理参数。

Stage 1 用 PPO 学 teacher；Stage 2 用 history encoder 拟合 teacher latent。网络冻结逻辑（内部资料）与 trainer 的 optimizer 一起证明可训练部分。Reward 在 Stage 2 仍可记录，却不因此进入 PPO loss。阅读时要问：history 是否足以消除隐藏信息的不确定性？latent error 在 actor 敏感方向上多大？

AnyRotate 在此基础上把 tactile 与 command axis 纳入输入，并加入相关 task / reward / curriculum。论文层面的 HORA / AnyRotate 方法分别见 [HORA 项目](https://haozhi.io/hora/)与 [AnyRotate 项目](https://maxyang27896.github.io/anyrotate/)；DexLab 这里是另一套 simulator / embodiment 下的实现，不能直接沿用论文的维度、控制频率或结果。

## 4 · H13 的具体接口：数值都随配置走

以下以 Stage 1 H13 taxel scenario（内部资料）与 Stage 2 H13 taxel scenario（内部资料）为准，须结合各自 `templates/rl_tasks/` 的 task defaults。

| 接口 | 所读配置 / 实现 | 学习点 |
|---|---|---|
| 控制 | 13 active actions、3 mimic joints；120 Hz physics / 10 Hz policy；incremental delta EMA、bounded target、position PD；25 ms target latency | State 不只关节位置，还含控制器历史；control dt 改变时间视野 |
| 单帧 actor observation | `13 q + 13 previous delta + 13 target + 4 contact + 4 force + 8 UV + 3 axis = 58` | 拼接代码（内部资料）的通用 docstring 写 67-D；不能照抄给 H13 |
| Privileged input | baseline 为 `22 + 2 geometry = 24-D`；其余项为 hand quat、object position / quat / angular velocity、mass、COM、goal quat | builder 调用 privileged_geometry_dim（内部资料）：未配置 geometry encoding 返回 24，BPS 变体返回 `25 + basis_count`；observation 拼接后另有 shape 检查 |
| Student | 30-frame history、Text2Touch encoder；latent + raw-action NLL，actor 可训练 | history 名义覆盖约 3 秒控制采样；不能从单帧 58 推断全部 network input |
| Behavior mixing | teacher probability 先 1.0，5M agent steps 后用 10M steps 衰减到 0.5；此配置 `replay: null` | 每个 env 选择 teacher / student action，非两动作线性平均；部署评估需纯 student |
| 物体与 dynamics | 默认引用 21-object capsule/box set；density × final collision volume 得 mass；hand / object friction 可独立随机化 | geometry、mass、contact 的耦合比单独说“随机质量”更具体 |
| Tactile | response model、UV、packet dropout / hold；thumb 配置标注 transferred prior、无真实 thumb calibration data | 可以提出未校准 sensor gap 的检验，不据此断言真机必然失败 |

该 Stage 2 文件头注释仍写 130M，实际 `max_agent_steps` 为 260M。阅读预算应核最终字段与 run snapshot；两者都不是“已经训练完成”的证据。类似地，通用 observation docstring 与当前 H13 shape 不一致，是应追踪实际 tensor 的具体例子。

**日志中的零与空集合。** 该 H13 配置的 teacher probability 始终大于 0。Stage 2 每次 step 都将所有 `episode_assisted` 标为 true，再过滤结束的 episodes；即使某个 env 恰好一直选到 student action，也按这一保守规则排除。因此 `student_reward_window` 始终为空，训练日志的 `student_episode_reward=0.0`、`student_episode_count=0`，基于该窗口的 student-only best selection 没有候选。这里的 reward 零是空集合占位值，能力由独立 pure-student evaluation 衡量；定期保存 checkpoint 的逻辑仍然存在。

**时间尺度示例的来源。** Stage 1 模板（内部资料）设置 `horizon_length=8, gamma=0.99, tau=0.95`，结合 H13 的 10 Hz policy 与 60 秒 training episode，分别得到 0.8 秒 rollout、约 10 秒折扣视野和 60 秒 episode 上限。三者分别决定实测 return 长度、远期权重衰减和 episode 边界。

## 5 · Reward、curriculum、bank、evaluation 是训练问题的一部分

- AnyRotate reward（内部资料）与 curriculum（内部资料）：把真实旋转进展、shaping、成功计数与推进 curriculum 的统计分别追踪。比较学习曲线时，冻结物体权重和评测难度。
- 共享 reset（内部资料）：bank 恢复不仅是 object pose；还要核 joint position、targets / preload、hand pose、history 及动态属性的一致性。初始化可用性不是全 DR 范围的 robustness 证据。
- H13 rollout suite（内部资料）：按固定 object / axis / episode protocol 评测；先区分 teacher、assisted student、pure student，再聚合指标。Stage 2 scenario 的 evaluation horizon 为 30 秒，与 training episode 的 60 秒分开。

这些内容对应 T3·S03 的 DR / teacher-student 与 T3·S04 的 sim2real。接到 TD3 时重点转为 replay/action/target contract；接到 FB 时重点转为 dynamics / coverage 是否支持新 reward。不能把方法替换理解成只改 optimizer 名字。

## 6 · 可以继续讨论的局限及所需证据

| 待验证问题 | 已有源码支持什么 | 还缺什么，才足以判断 |
|---|---|---|
| Mixed rollout 是否掩盖 student 退化 | 配置明确保留 teacher takeover；代码标记 assisted episodes | 同 checkpoint、同环境条件的 pure-student 分物体/轴评测 |
| Latent supervision 是否充分 | teacher privileged latent 与 history student 的监督关系 | state-matched latent/action error、失败前 history、闭环对照 |
| DR 是否覆盖真实误差 | 有独立 friction、joint offset、mimic transmission、tactile packet 等配置 | 真实参数测量与联合分布、holdout 扰动；不是 DR 范围越宽越好 |
| Shape generalization 是否成立 | 存在 21-object baseline 和更大集合 / BPS 变体 | 各 run 的真实 config、训练/测试 split、checkpoint 与独立分组结果 |
| 任务指标是否一致 | reward / curriculum / evaluation 有各自统计入口 | 单位与分母的逐项对拍；不能只凭日志名中的 rotation 判断 full turns |

本次材料只将这些问题变成可教、可检验的路线，不新增“复现成功”或“已发现算法缺陷”的结论。
