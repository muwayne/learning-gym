# 进度 — wayne

> 唯一个人进度源。状态词汇见 [../../progress.md](../../progress.md)。图例：✅ U/I/M 为最高完成 Pass · ▶ 材料就绪可开课 · ・ 材料待起草（AI 在你到达前写好）

## 下一节

- **当前入口（2026-11-06）**：[P0 恢复题](../../modules/t3-sim-to-rl/rl-control-practice.md#p0--恢复价值学习) → PPO → TD3 → SAC，再按[课表](../../curriculum.md#rl-基础讲义与课次的对应t2t3-共用)进入生成策略与 BPTT。先集中核查 backup policy、n-step、reset 前 final state 与 Double Q；之后学员重建经典更新伪代码、固定 batch 对拍。RLT / EXPO 旧证据用于校准，P0 新题待答；尚无本轮 learner experiments。

- **阅读材料**：[RL 基础讲义](../../modules/t3-sim-to-rl/rl-foundations-primer.html) §2 起恢复；PPO 读[统一温习讲义](../../modules/t3-sim-to-rl/rl-control-review-primer.html) §2，TD3/SAC 回到基础讲义 §4。MDP/Bellman 按已读情况快速校准。
- T3·S06 EXPO select + edit（已暂停，[记录](records/t3-s06.md)）：讲义已读并有问答证据；经典更新对拍后，从 [P7](../../modules/t3-sim-to-rl/rl-control-practice.md#p7--生成策略后训练比较改进发生在哪里) 选一个机制对照，再进入已有 [EXPO-lite 练习](../../modules/t3-sim-to-rl/s06-expressive-policy-value-maximization.md) 或真实复现证据。Learner harness 的核心函数仍是 TODO，未运行学员实验。
- T 线另一入口：T1·S01 CPU toy 起点（[课表](../../modules/t1-training-stages/s01-training-stage-map.md) · 实验计划（内部资料） · [记录](records/t1-s01.md)）。PR #75 已合并；按反馈改为简单 MLP toy → SmolVLA + RoboTwin → Cosmos。公共 toy base 已准备，下一步为学员 source 选择与预测；SmolVLA 未训练，U/I/M 未完成。
- T3·S07 基础衔接与小实验已结课，本次不继续其完整 RLT 实践。
- 或继续 A 线：A1·S04 KV cache 与 prefill/decode（・材料待起草；本次不开始）

## RL 基础讲义进度（Understand 层）

对应关系见[课表](../../curriculum.md#rl-基础讲义与课次的对应t2t3-共用)。讲义已通读不等于 `✅ U`；还需一个有上下文的判读与 revised mental model。

| 课次 | 讲义知识点 | 状态 |
|---|---|---|
| T2·S01 | 1–10 | §1 自述清楚；§2 已作答一批，MC/TD、SARSA 比较和 n-step 有正确答项；backup-policy 身份错项、reset 和 Double Q 已讲解，待 P0 新场景确认，未完成 U |
| T2·S07（选修） | 11–12 | 待读 |
| T3·S02 | 13–17 | 待读 |
| T3·S05 | 18–27 | 待读 |
| T3·S06 | 28–29 + S06 讲义 | S06 讲义已通读；问答 6 题（[记录](records/t3-s06.md)），Understand 未收口 |
| T3·S07 | 30–33 | 待读；此前已有 actor–critic 基础衔接证据 |

T3 这四课的 `Implement` / `Improve` 层计划用 X2/X3 等真实复现工作的证据完成（P-T-I-E 结构，链接正式 registry / reports）。X2–X3.10 的问题与讨论已整理为 [EXPO-FT 复现案例集](../../modules/t3-sim-to-rl/expo-ft-casebook.html)（30 个案例，2026-10-13），可作为课上讨论入口；整理本身不记为学员作答证据。

## 当前 RLT 课内进度

课表：[T3·S07 · RLT 原理与诊断六单元路线](../../modules/t3-sim-to-rl/actor-critic-foundations.md#0-地图与范围)。这是一个 Lesson 内的分段，不新增 Gym 课次数。默认最终深度仍为 Improve；当前以原理和诊断为主，不默认数学练习。

| 单元 | 当前状态 | 已有证据 / 下一步 |
|---|---|---|
| 1 · 角色与接口 | 已讲解，已有局部作答证据 | actor 梯度方向及 Q≠真实成功已作答；critic/Q关系与RLT接线已澄清，不重新布置手算 |
| 2 · 一次学习循环 | 数据归因判读已完成；梯度卡点已补讲 | reference/executed action错配判读正确；函数与单次标量输出已区分讲解，不据此前手算断言梯度原理掌握 |
| 3 · TD3稳定机制 | 机制及边界已讲解；效果待实验验证 | 已区分更多critic更新、更多真实经验与估值准确性；未将讲解或“继续”当作独立掌握证明 |
| 4 · 接回RLT实现 | 结构、BC与纠正路径已讲；状态匹配题待答 | 不把跳到真实案例当作该题已通过；需后续结合实际数据确认executed action与BC target的来源 |
| 5 · 故障诊断与对照 | **对照方向已答对；补清BC与critic术语** | Wayne提出使用新增Oracle示范的纯BC对照；已解释BC是模仿方法、Oracle是数据来源、critic是Q网络。此处“新增数据必改善”需改为可能改善；单元6已完成固定toy对照 |
| 6 · 最小实践与收口 | **本次已收口：预测、实验、Explain完成** | 预测命中：A平均实际偏差0.250798mm、0/16成功；B为0.060882mm、16/16成功；同结构/数据/actor预算，改变是否接受critic指导；Wayne正确归因训练信号而非容量；ledger+1 |

本次停止点：单元6的Explain已完成，基础衔接与小实验收口。Wayne能以同结构对照说明critic指导的额外作用；actor输入和容量不变，训练信号改变。固定实验只运行一次，无调参选seed，结果交互图已就绪。后续统一称critic网络和Q评分。恢复时再接实际RLT的数据与训练路径；单轨迹overfit案例暂不追查，单元4纠正state匹配题仍待验证、不补记作答。整个T3·S07的U/I/M尚不因这个terminal toy自动完成。

## 总览

- 接触主题：**6 / 45** · `Understand 6 · Implement 6 · Improve 2` · ledger：29 条（命中 19 · 部分 4 · 未命中 6）· 教学层抓获：17 次（全部课规化）

## A 线 · 机制（3 / 19）

| 节 | 主题 | 状态 |
|---|---|---|
| A1·S01 | Attention 重激活：手算 + 50 行实现 | ✅ I 2026-08-29 · [记录](records/a1-s01.md) |
| A1·S02 | Causal mask + multi-head attention + mini decoder | ✅ I 2026-09-01 · [记录](records/a1-s02.md) |
| A1·S03 | 训练闭环：tiny GPT 训到能生成 | ✅ M 2026-09-08 · [记录](records/a1-s03.md) |
| A1·S04 | KV cache 与 prefill/decode | ・ |
| A1·S05 | RoPE / RMSNorm / SwiGLU 换装 | ・ |
| A1·S06 | Induction head | ・ |
| A1·S07 | burstiness → ICL 复现 | ・ |
| A2·S01 | action token + NLL | ・ |
| A2·S02 | Gaussian+MSE 的 mode averaging | ・ |
| A2·S03 | Diffusion（DDPM） | ・ 概念层已由 [DDPM 精编](../../references/ddpm-uwm-digest.md) 预热（2026-09-03）；训练闭环待课 |
| A2·S04 | Flow matching | ・ |
| A2·S05 | DiT / ACT / π0.5 三方对照 | ・ |
| A3·S01 | 视觉基础组件：ViT/CLIP/DINOv2 | ・ |
| A3·S02 | Projection / adapter | ・ |
| A3·S03 | Latent query / resampler | ・ |
| A3·S04 | 双塔与对比对齐 | ・ |
| A3·S05 | MoT：π0.5 解剖 | ・ |
| A3·S06 | MoE 实现 | ・ |
| A3·S07 | Generator × Reasoner / WAM | ・ |

## T 线 · 训练（3 / 26）

| 节 | 主题 | 状态 |
|---|---|---|
| T0·S01 | Loss 作为接口：第一次闭环 | ✅ I 2026-08-29 · [记录](records/t0-s01.md) |
| T0·S02 | Optimization geometry / training distribution：five-arm intervention | ✅ I 2026-08-30 · [记录](records/t0-s02.md) |
| T0·S03 | Closed-loop gap：offline loss ↔ SR | ✅ M 2026-09-04 · [记录](records/t0-s03.md) |
| T1·S01 | 阶段地图：CPU toy → SmolVLA 实践 → Cosmos 3 对照 | ▶ 2026-09-10 公共 toy base 就绪；待学员 source 选择与预测；U/I/M 尚未完成 |
| T1·S02 | Continual pretraining toy | ・ |
| T1·S03 | π0.5 配方与自身 SFT 定位 | ・ |
| T1·S04 | CT vs SFT 判断演练 | ・ |
| T2·S01 | Tabular：Bellman T 落成代码 | 进行中 · §2 已做首批诊断；P0 恢复题待答，未登记 Pass |
| T2·S02 | REINFORCE 从零 | ・ |
| T2·S03 | A2C 与 stop-gradient | ・ |
| T2·S04 | GAE 实现 + λ 扫描 | ・ |
| T2·S05 | IS 推导 → PPO ratio/clip | ・ |
| T2·S06 | PPO 对齐 CleanRL | ・ |
| T2·S07 | 选修：DQN / Double DQN | ・ |
| T3·S01 | PPO 连续控制 | ・ |
| T3·S02 | TD3 → SAC，共享 replay 与更新对照 | ▶ 温习讲义与 P4 设计就绪，未新增实践证据 |
| T3·S03 | Domain randomization + teacher-student | ・ |
| T3·S04 | Sim2Real 判断框架 | ・ |
| T3·S05 | Offline RL / AWR / IQL | ・ |
| T3·S06 | Expressive policy value maximization：EXPO select + edit | 进行中 2026-10-05 · 讲义已通读、索引问答 6 题后按学员要求暂停；Understand 未收口 · [记录](records/t3-s06.md) |
| T3·S07 | Online adaptation：RL Token / critical-phase real-world RL | ▶ 2026-09-09 基础衔接与小实验已收口；完整 U/I/M 尚未完成 |
| T3·S08 | Capstone：selective-RL proposal | ・ |
| T3·S09 | 选修：Zero-Shot RL，SR / SF / FB 与 BFM | ▶ 完整专题与 P5 温习入口就绪；讲义参考计算不计 learner Pass |
| T3·S10 | 选修：可微动力学与 BPTT | ▶ 原文入口与 P8 设计就绪；经典更新对拍后选做，未开课 |
| T4·S01 | 现代训练循环工程 + JAX 对照 | ・ |
| T4·S02 | 分布式：DDP / FSDP | ・ |

## 更新记录

| 日期 | 课次 | 状态变化 | 备注 |
|---|---|---|---|
| 2026-08-29 | — | v2 课表落盘；A1·S01、T0·S01 材料就绪；档案精编入库 references/ | 学习进度从 0 起算 |
| 2026-08-29 | T0·S01 | 未开始 → 进行中：P1/P2 锁定、手算与转译完成、梯度范数演示教学 | 口述驾驶模式确立；三条新课规入课表 |
| 2026-08-29 | T0·S01 | 进行中 → **已完成**：预测 2/2 命中；三运行消融链；温习卡入档 | 课规累计 6 条；教学体验原则入课表 |
| 2026-08-29 | A1·S01 | 未开始 → **已完成**：预测 3/3 命中；转置与乘法 mask 两坑亲手修复 | 2/41 |
| 2026-08-30 | T0·S03 | 新课次落盘：材料 + 闭环鸿沟交互演示就绪 | 源自 wayne 的 G2.0 VLA 诊断需求；课表 41→42 节 |
| 2026-08-30 | T0·S02 | 未开始 → **进行中**：确认/手算/追问/B 组 P-T-I-E 完成，ledger +4（首现未命中） | 抓获 #8–#11；B 组 tanh 昏迷反转剧本；剂量相变；暂停于 B 组病历句 |
| 2026-08-30 | T0·S02 | 进行中 → **已完成**：C/D/E 三组 P-T-I-E、ledger +7、三种效应总结与 mental model 入档 | 3/42；抓获 #12（直方图标判据）；调音台演示固化；grad norm/cosine 补课；迁移问答 |
| 2026-09-01 | A1·S02 | 未开始 → **已完成**：口述驾驶 7 零件、检查 A–D 全过、预测 3/3、ledger +3 | 4/42；抓获 #13–#15；多头/block 演示固化；卡点问答附录首创；两个静默 bug 证据抓获（S 抄行、mask 漏对角线） |
| 2026-09-03 | — | 课外输入入库：UWM §II 精读的 DDPM 笔记浓缩为 references/ddpm-uwm-digest.md（Q1–Q9）；课表 A2·S03–S05、A3·S07 挂钩；baseline 生成式动作头概念层 ◐→● | 不改课次数与完成计数；A2·S03 ① ③ 段快进、② ④ 照做 |
| 2026-09-03 | T0·S03 | 完整 tutorial + 第一题 hand calculation 完成；预测尚未锁定 | 抓获 #16 English-first terminology、#17 debugging map 不能退化为长 checklist；两项进入共享课规/材料 |
| 2026-09-04 | T0·S03 | ①–③ 概念教学与自检完成；暂停于 ④ 实验入口 | 已掌握 policy-induced state distribution、support metric 边界、四类 diagnostic categories，以及“更多 rollouts 估计 SR、instrumented rollouts 定位 failure”的区别；Arm F/G/H 尚未开始 |
| 2026-09-04 | T0·S03 | ④–⑤ three-arm intervention、G2.0 claim audit 完成 → **已完成** | Arm G 暴露 E16 sharp degradation；Arm H 提供 recovery data 的 directional positive evidence；ledger +5，累计 5/42 |
| 2026-09-04 | — | 新增 T3·S07 RL Token / online adaptation；原 capstone 顺延至 T3·S08；课表 42→43 节 | 课程只安排 mechanism、toy 与 π0.5 transfer audit，不构成 RoboDojo、Isaac 或真机训练授权 |
| 2026-09-08 | A1·S03 | 材料就绪 → **进行中**：开始 tiny GPT training loop Improve Pass | 先完整讲清 model/objective/training/generation contract，再进入实现；允许跨多个短 session |
| 2026-09-08 | A1·S03 | 进行中 → **已完成**：single-batch overfit、proper baseline 与 `y=x` objective control 全部完成 | Baseline true val loss/accuracy `0.7361/0.836`；control own loss `0.0003` 但 true val `11.7204/0.013`；明确区分 scored generalization 与 qualitative generation |
| 2026-09-09 | T3·S07 | 本次actor–critic基础衔接与最小CPU实验收口 | 预测命中，Explain正确区分训练信号与容量；ledger +1；完整课程U/I/M保持未完成 |
| 2026-09-09 | T1·S01 | 恢复方案详细化，待PR审阅后迁移c222 | SmolVLA小样本与真实阶段对照、W&B/rollout、课末Cosmos；未执行训练，不改完成计数与ledger |
| 2026-10-04 | T3·S06 | ・ → ▶：按 RLT / EXPO-FT 工作需要选作 RL 两日路线入口；课表扩为 EXPO select + edit，材料、演示与 CPU harness 就绪 | 不改课次数与完成计数；ledger 未变 |
| 2026-10-04 | T3·S06 | 按讲义先行方式改造：新增 29 个知识点的通读讲义（Markdown + HTML 阅读版），原课表改为练习单 | 学习方式偏好写入 baseline；不改完成计数 |
| 2026-10-05 | T3·S06 | ▶ → 进行中：通读讲义，索引问答 6 题；Wayne 要求改为系统补 RL 基础 | 新增 RL 基础讲义；教学抓获 3 次（对话 LaTeX 乱码、追问超出讲义、对已读论文者连续复述题）；ledger 未变 |
| 2026-10-05 | T2/T3 | 课表按方案 B 整合：RL 基础讲义承担 T2·S01、T2·S07、T3·S02/S05/S06/S07 的 Understand 层；T3 四课的 Implement/Improve 可用真实复现证据 | 课次数不变；本表新增讲义进度 |
| 2026-10-13 | T3 | 新增 EXPO-FT 复现案例集（X2–X3.10，30 个案例对回 RL 基础讲义）；S06 与讲义中过时的 X2/X3 状态改为引用 X3 终点与诊断报告 | 不改完成计数；ledger 未变 |
| 2026-10-17 | T2/T3 | PPO / TD3 / FB + DexLab 统一温习入口与六段实践设计就绪；T3·S02 明确 TD3 主线 | 未执行新 learner experiments，U/I/M 与 ledger 不变；总课次数分母同步 44，T 线按既有三条完成记录校正为 3/25，不新增完成记录 |
| 2026-11-06 | T2/T3 | 根据上次答题与本轮安排恢复 T2·S01；经典更新对拍先于生成策略 / BPTT 代码研究 | 只登记既有答题与待确认项；P0 新题和 P4/P7/P8 设计就绪，尚无本轮 learner experiment；U/I/M 与 ledger 不变 |
