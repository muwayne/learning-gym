# T1·S01 · Training stage map

## 0. 开课地图：先做阶段实验，最后对照 Cosmos

本课回答：**已有模型为什么还要分阶段适配，每个阶段改变了什么，我们怎样知道它有用？** 先用 CPU 上的二维点机器人和小 MLP 做秒级 toy，再将相同的阶段对照问题带到 SmolVLA + RoboTwin 真实示范，最后用 Cosmos 3 对照。阶段地图是实验的解释工具，不是整节课唯一的动手产物。

共同主线：看起点与数据 → 快速 toy 干预 → 真实阶段训练与 W&B → Cosmos 对照 → 一张有证据的阶段地图。案例可以更换，但必须保留来源、输入/目标和独立评测。Wayne 的 c222 数据选择、实验配方、预算建议与准备项唯一维护在SmolVLA 教学实验计划（内部资料），本课不另维护运行合同。

- Understand：区分训练阶段、训练方法和参数更新范围；用一个实验解释阶段职责及证据边界。
- Implement：口述或实现阶段数据路由，完成一次预测、单变量干预、实际运行与解释；地图或 AI 预先跑完的结果不替代该证据。
- Improve：在 Implement 之后做有对照的阶段选择，比较目标任务收益和保留能力；短课中结果不充分时如实停在已验证层，不以增加题目补齐。
- 单次 session 有自然停止点，可以分两次完成；默认最终深度不意味着一次坐完。
- 学员拥有：预测、关键实现意图、干预选择、结果解释。AI 负责环境、数据核验、harness、图表和机械计算；不要求手算或求导。
- 2026-09-10：按 Wayne 反馈将 toy 改成独立 CPU 小模型；真实模型准备可以后置，不要求先跑完整参考对照才能开课。

### 本课怎样交替学习与操作

以下时段是备课预算，不是固定课时承诺；训练与评测等待另计，需在开课前实测。讲解约占主动学习时间的三分之一，不连续讲完所有概念再发实验。

| 段落 | 主动学习参考 | 必要概念 | 学员操作与可见反馈 | 自然停止点 |
|---|---|---|---|---|
| 1 · 起点与地图 | 5–8 分钟 | pretrained checkpoint 已继承什么；本次更新什么 | 真实示范作为后续目标；先看二维 toy 的 expert 与公共 base 轨迹，明确输入、动作、数据和评测 | 能说清今天的输入、输出和要解决的问题 |
| 2 · CPU 小模型 toy | 10–15 分钟 | 数据分布、阶段适配与预算公平性；同一 loss 可以服务不同阶段 | 选择 source 与 target 的关系，预测后运行 direct/staged；看实际 loss 与到达轨迹 | 一次真实 optimizer 干预和解释结束；可在这里结束第一次 session |
| 3 · 真实阶段训练 | 15–20 分钟 | domain adaptation 与 target adaptation；预算公平性 | 口述阶段切换时的数据、参数及 optimizer 行为；在 W&B 对照直接适配与分阶段路线，并回放配对 rollout | 能指出阶段训练是否带来额外作用、哪个比较支持什么 |
| 4 · Cosmos 对照 | 8–10 分钟 | 从具体实验抽象出 pre/mid/post 职责 | 对照本文 §3，用数据、objective、可训练模块、evaluation 四列修订自己的地图 | 找出一项相同职责和一项不能类比的差异 |
| 5 · 固化与结课 | 3–5 分钟 | 结论与证据强度 | 用两三句话说明保留哪个方案及理由，记录一次 claim audit | 本次明确结课；不临时加开 RL、flow 推导或其他 Lesson |

第二段使用 **远小于 SmolVLA 的独立 MLP**，CPU 上真实更新权重；不使用图像、语言或机器人仿真依赖。第三段才进入 SmolVLA 与真实数据。两层共享 direct/staged 对照问题，但 observation/action、objective 和模型规模不同：toy 结果是机制练习，不能替代 VLA 能力证据。

### 交互与问答约定

- 每个操作前同屏给出当前段落、完整场景、固定项、唯一改动、已有概念和期望回答形式；首次遇到概念先看一个 worked example，再预测下一次干预。
- 学员可以亲手改配置、口述驾驶或检查留空实现；至少亲自决定 source 数据条件，并说明阶段切换的数据路由。AI 不先替学员运行干预再要求猜结果。
- 交互必须触发可观察的比较：选择 episode、切换 checkpoint、叠加动作轨迹、比较配对视频、修改训练条件后触发新 run。拖动展示控件不等于训练模型。
- 固定样例应同时显示示范视频、真实标签、模型 action chunk 和实际执行的 prefix；只有具备已验证 forward kinematics 时才画末端轨迹，否则画带单位的关节/动作曲线。
- 参考运行可在开课前准备；课程中清楚区分 `reference` 与 `learner`。参考结果没有学员预测，不进入其预测账本。
- 问题聚焦主线，不逐个新词强制复述；答对即推进。AI 或环境排错不占用连续问答来填时间。

### 训练等待与备课验收

课堂现场至少保留一次学员控制的 optimizer 更新和一次独立评测。若短 run 超出课堂等待预算，先展示参考 run，再在后续 session 读取本人的结果；不得把参考结果冒充本人运行，或默认无限延长课程。

CPU toy 开始前只需公共起点、实际轨迹与有界计时，不以 SmolVLA 安装或 W&B 为门槛。进入第三段前再交付真实样例/动作、W&B、最小 policy rollout 与实测耗时；参考对照按需准备，不提前代跑学员干预。若真实模型只有离线链路可用，明确闭环缺口，不将 toy 的成功冒充真实机器人能力。

原有[阶段地图演示](demos/training-stage-map.html)保留为概念参考；它不会训练模型，也不生成实验指标。新的实验图示完成后按 Gym 约定固化到本课 demos；W&B 中的数值和媒体必须来自具名 run。

### 三阶段的解释边界

| 本课所说的阶段 | 真实实验里是什么 | 不能宣称什么 |
|---|---|---|
| Pre-training 起点 | 已有 SmolVLA checkpoint 继承的 VLM 与机器人训练能力；查模型来源并观察 base | 未在本课从随机初始化重做全部预训练 |
| 中间适配阶段 | 同一 embodiment 上的相关 source tasks 继续训练；检验是否帮助后面的 target task | 不因处于中间就自动等同 Cosmos 的 mid-training；不保证正迁移 |
| 目标适配阶段 | 固定 target demonstrations 上的 imitation/flow-matching 训练与独立评测 | post-training 不等于必须使用 RL；输出更像示范不等于成功率更高 |

本课不增加新的课次数。T1·S02 的系统性遗忘/replay 研究、T1·S03 的 π0.5 配方解剖仍保留独立范围，不为本课补齐这些主题。

## 1. 三个阶段：关注点逐步变化

以下是工程阅读框架，不是所有模型都必须遵循的统一三段标准；mid-training 的命名尤其依项目而变。

| 阶段 | 主要职责 | 教学例：领域问答助手 | 需要验证什么 |
|---|---|---|---|
| Pre-training | 建立可复用的广泛能力 | 从广泛文本学习语言与知识模式 | held-out prediction 与相关通用能力 |
| Mid-training | 在基础与目标用途之间补能力、领域或接口缺口 | 在领域文本上继续 next-token training | 领域指标改善、通用能力 regression |
| Post-training | 让输出行为适配具体用途 | 用问题与示范回答做 SFT，或进一步做 RL | 目标任务表现与原有能力保留 |

“学到”“改善”都必须由 evaluation 支持。训练阶段名称本身不保证知识正确、指令遵循、物理一致性或 closed-loop success。

## 2. 阶段与方法是不同维度

- Fine-tuning：从已有训练权重出发继续适配；不指定唯一学习信号，也不等于全参数更新。
- SFT：用目标输出或 demonstration 做 supervised fine-tuning；LLM 的 instruction SFT 常以目标回答 tokens 的 cross-entropy 为目标。
- Post-training 是阶段称谓，可包含 SFT、preference optimization、RL 等方法；post-training 不等于 RL。
- Continual pretraining：继续预训练式学习，例如继续 next-token prediction；可承担某个项目中 mid-training 的工作，但不与 mid-training 严格同义。
- Full fine-tuning / LoRA 描述更新参数的方式，与 SFT / RL 的学习信号维度不同。
- SFT 不以“人工标注”为必要条件；synthetic demonstrations 也可监督训练。工具类名不决定科学阶段，例如 SFTTrainer 也支持纯 language-modeling 数据。

公式参考（本 session 不要求手算）：

\[
\mathcal L_{\mathrm{NTP}}=-\sum_t\log p_\theta(x_t\mid x_{<t})
\]

根据前文预测下一个 token。领域 continuation 与 instruction SFT 可以沿用相同的 loss 家族；数据分布、条件、loss mask 与目标用途改变，同样会改变训练的含义。

## 3. Cosmos 3：只看 Generator 三段路线

来源为原报告 v1 §4，读取于 2026-09-09。Reasoner 有自己的 pre-training / SFT 路线，不能把 Generator 的三段套到整个系统上。

- Pre-training：学习 image / video / audio generation。
- Mid-training：保留已有生成模式，引入 action 和 transfer supervision；前两阶段使用 rectified flow objective。
- Robot policy post-training：从 mid-trained checkpoint 出发，加入 proprioception 并适配 executable actions 和 closed-loop control。

本课推论：生成合理未来与完成真实控制任务需要不同的验证；动作相关训练并不自动证明可以部署在另一台机器人上。这里不把 Cosmos 3 配方改写为 π0.5 的正式配方或训练授权。

## 4. Cosmos 作为课末对照

此处在完成实验判读后使用，不把整篇论文精读作为课前门槛。仅核对 Generator §4 的阶段职责，Reasoner 作为来源背景，不展开新的架构课程。

| 对照项 | 学员从 SmolVLA 实验带来的问题 | 查 Cosmos 的位置 |
|---|---|---|
| 初始化与继承 | 这次训练开始前已经会什么？ | §4 的初始化说明、§4.2.1 |
| 数据与训练目标 | 增加的监督解决什么缺口；loss 是否必须改变？ | §4.2.2 |
| 目标用途 | 学动作与生成合理视频怎样分别验收？ | §4.2.5 及对应评测 |
| 证据边界 | 我们的实验能支持多强的阶段收益结论？ | 区分原文结果、本课实测与迁移假设 |

最终任务为解释，不是计算：用本次真实数据与结果补齐一张阶段地图，再说明为什么保留或拒绝中间适配。自然出现的 claim audit 可用“有中间阶段就一定更好”或“loss 降低就证明完成了闭环适配”；不为凑数重复出题。

## 来源

- [Hugging Face — Fine-tuning](https://huggingface.co/docs/transformers/en/training)：fine-tuning 的通用定义。
- [TRL — SFT Trainer](https://huggingface.co/docs/trl/en/sft_trainer)：监督目标、数据格式与 adapter 支持。
- [SmolVLA 官方教程](https://huggingface.co/docs/lerobot/main/smolvla)：模型与微调入口；性能数字仅为官方配置参考。
- [W&B 媒体记录](https://docs.wandb.ai/models/track/log/media)：按 run/step 比较实际样例。
- [Cosmos 3 v1 §4](https://arxiv.org/html/2606.02800v1#S4)：Generator 三阶段；§4.2.1、§4.2.2、§4.2.5 是本课案例定位。

## 卡点问答

**Fine-tuning 和 SFT 有什么区别？** 前者描述继续适配已有模型，后者指定用示范目标提供监督。SFT 可以全参数更新，也可以使用 LoRA；监督目标不局限于文本。
