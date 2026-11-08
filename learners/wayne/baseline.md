# Learner Baseline — wayne — 2026-08-29

> 由三份个人学习档案与工作背景整理预填，本人可修订。

## 能力矩阵

| 主题 | 概念层 | 实现/闭环层 | 备注 |
|---|---|---|---|
| Transformer / Attention / QKV | ◐ | ◐ | 2025-03 实现过 Vanilla Transformer（模块化+单元测试），训练闭环未闭合；现自感"似是而非" |
| 现代组件（KV cache / RoPE / RMSNorm / SwiGLU） | ○ | ○ | 个人档案 §52 自列缺口 |
| 生成式动作头（diffusion / flow matching） | ● | ○ | 2026-09 精读 UWM §II 闭合 DDPM 概念层（前向/反向/ε-prediction/条件化，见 [references/ddpm-uwm-digest.md](../../references/ddpm-uwm-digest.md)）；flow matching 只到端点对照；未亲手写 |
| RL 基础（MDP / Bellman / V·Q·A） | ●● | ○ | 2025-11→2026-07 完整一轮，到 operator/fixed point 层 |
| Policy gradient 族（VPG / PPO / GAE） | ●● | ○ | 理论熟；从未从零复现（档案 §9 自认） |
| Off-policy 族（DQN / DDPG / TD3 / SAC） | ●● | ○ | DQN 仅对照级；auto-α 未推导 |
| 训练阶段架构（pre / mid / post / continual） | ◐ | — | 零散；Cosmos 3 是关注对象 |
| 融合机制（双塔 / latent query / MoT / MoE） | ◐ | ○ | DETR object query 追问很深（latent query 的直系）；MoT/MoE 未拆过 |
| Sim2Real / domain randomization | ◐ | ○ | 2025-11 legged PPO 应用级讨论 |
| 本仓库系统（π0.5 / RoboTwin / eval 栈） | ●●● | ●● | 工作日常；G3–G5 实验主导者 |
| CNN / LSTM | CNN ●●；LSTM ○ | CNN ●●；LSTM ○ | 决定不单补 LSTM，对照带过 |

## 既有学习资产（本地个人文件，不入库）

- `~/Downloads/transformer_learning_history.md` — 2025-03 学习弧；§60/61 的两套 15 问 checklist 已被课程采纳为 A1/A2 Exit Gate 工具；浓缩版入库 [references/transformer-digest.md](../../references/transformer-digest.md)
- `~/Downloads/spinningup_rl_learning_history.md` — RL 完整档案；T2 材料将引用其 Q1–Q20 编号；浓缩版入库 [references/rl-spinningup-digest.md](../../references/rl-spinningup-digest.md)
- `~/Downloads/ICL到机器人了吗_S1×GEN-1.5_调研档案.html` — A1·S07 与 A3·S07 的直接输入；浓缩版入库 [references/icl-s1-gen15-digest.md](../../references/icl-s1-gen15-digest.md)
- `~/Downloads/claude_DDPM基础_UWM第二章精读笔记.md` — 2026-09 精读 UWM（RSS 2025）§II 时写成的 DDPM 追问笔记（原文/评述/推断三级标注）；A2·S03–S05 与 A3·S07 的直接输入；浓缩版入库 [references/ddpm-uwm-digest.md](../../references/ddpm-uwm-digest.md)
- 2025-03 的 Vanilla Transformer 代码仓库 — **待查找**；若在，A1·S01 增加"跑通当年单元测试"环节

## 我最想从课程里得到的三样东西

1. 把生疏的概念层全部重新焊死到"亲手闭环"级（Transformer 与 RL 同等对待）；
2. 成为能用 latent space fuse 各种模块的熟练工（双塔 / MoT / MoE / Generator×Reasoner 级别的接线能力）；
3. 一份踩在真地基上的 Action Expert selective-RL proposal。

## 学习模式偏好

- 默认 Pass：`Improve`
- 询问策略：不必每课重复确认；Wayne 明确提出临时或长期切换时再改变
- Session 偏好：以自然 checkpoint 为单位，避免为了做完整套流程而拉长；`Improve` 可跨多个 sessions 完成
- 可视化：沿用[全体学员共同教学原则](../../curriculum.md#learning-activity-toolbox)；2026-09-09 明确应适用于所有学员，不只作为个人偏好。
- 上下文与节奏（2026-09-09 重申）：先完整讲清课程主线，再出自足题目；每次一个问题，给足场景、必要定义与期望回答长度。答对即推进，卡点按需补讲，避免重复确认、冗长复述及为填模板延长课时。
- 学习目标（2026-09-09）：以原理理解、代码中的机制对应与实际问题诊断为主；不默认安排数值代入、求导或偏微分计算题，不用计算熟练度代替原理掌握证据。公式按需解释含义，机械计算交给工具；动手主要由学员预测行为、检查数据/梯度路径、改变一个机制并解释结果。只有学员明确选择推导训练时再布置相关计算。
- 实践直观性（2026-09-09 结课反馈）：知识讲解与末尾看一次结果仍不足。优先在同一环境交替讲概念、改训练条件、真实运行、看固定样例和独立评测；可点击图与设计卡片不能替代实际 optimizer 干预。T1·S01 选择 c222 上的 SmolVLA 小样本/真实数据两种尺度，W&B 曲线配动作与 rollout，Cosmos 只在末尾对照；具体方案见课表及其训练计划。
- 课表导航（2026-09-09）：课前提供完整分段、任务和结束标准；每次恢复或切换单元时标出“当前位置 / 本次目标 / 自然停止点”。区分已讲、已作答与已验证；插入后续案例时标为预览并返回主线，避免连续问答掩盖全貌。
- Toy 的尺度（2026-09-10）：先用远小于真实模型、CPU 秒级可验证的 toy 理解机制，再迁移到 SmolVLA 与真实数据，最后做 Cosmos 对照；缩小 SmolVLA 数据量不满足独立 toy 的目的。课前只做必要检测，课程可随学习反馈调整，不提前铺开整套参考实验。
- 讲义先行（2026-10-04，Wayne 总结的有效方式）：每课先生成一份覆盖整课所需知识的讲义（HTML 等可视化阅读版，图与交互示例内置），Wayne 通读一遍后再进入习题练习。练习单只放问答、判读、实现与实验，不在练习中引入讲义之外的新概念；讲义不提前给出 toy 结果，以免替代练习中的 prediction。
- 论文学习方式（2026-10-05）：已读过原论文的方法（如 RLT、EXPO-FT），机制以真实训练与复现为主线学习，不再用 toy 实验或逐题复述论文细节；课程负责补这些论文所依赖的系统 RL 基础（讲义先行），细节只在会改变实际复现决定或诊断判断时深挖。
- RL 温习范围（2026-10-17）：以基础知识整理和实践理解为主，DexLab 用作说明 PPO、teacher/student、sim2real 的案例；源码引用与逐文件追踪按需深入，不作为默认学习负担。
- RL 温习节奏（2026-11-06）：已看过的知识按一批情境题集中诊断，允许一次答完、标记不确定；仅围绕错题和疑点补讲，取代本 RL 温习的逐题节奏。先亲手复原 PPO、TD3、SAC 的伪代码与核心更新，再比较论文代码的差异；整个 RL 流程的深度目标仍是 Improve。
- 对话格式（2026-10-05）：聊天界面不渲染 LaTeX，对话中的公式用代码格式纯文本书写。
- 术语稳定性（RLT课反馈）：明确区分网络、输出、训练方法和数据来源；统一用“critic网络（Q网络）”“Q评分”，不在同一讲解中用裸词Q交替代指不同对象。先定义BC再讨论联合训练，不使用未解释的“BC+Q”等压缩写法。

## 节奏承诺

- 预计每周 session 数：待定
- 单个 session 可用时长：不固定；优先短 session 与独立收获
