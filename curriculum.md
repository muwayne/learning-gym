# Gym 双线课表 v2

> 唯一执行课表。总览与协作方式见 [README.md](README.md)，接力上下文见 [handoff.md](handoff.md)，个人进度见 `learners/<name>/progress.md`。

## 使用方式

- 共 **45 个 Lessons**（A 线 19 + T 线 26，含 3 节选修）；Lesson 是知识主题，不等同于一次 session；
- 每个 Lesson 可按 [learning modes](learning-modes.md) 分为 `Understand / Implement / Improve` 多次进入；不固定时长，到自然 checkpoint 即可停止；
- 按实际 **Pass / session** 留痕，不按日历周：间隔几天或几周都不算落后，进度看最高 Pass、ledger 与 records；
- 建议**双线并行**（每周 A 线、T 线各一节），也可单线推进；两线在 T3·S08 汇合；
- **档案 ≠ 现役**：材料引用既有学习档案时，把它当 routing evidence；需要时做最小 reactivation，已有现役 evidence 时允许缩短或替代，不为流程机械重复；
- 学员档案精编（Q 编号与 checklist 引用的解析目标）在 [references/](references/)；原始完整档案留学员本地；
- toy 实验记录用 [templates/session-log.md](templates/session-log.md)；只有 capstone 级实验（T3·S08 等）才使用完整 [templates/experiment-card.md](templates/experiment-card.md)。

## Learning activity toolbox

下列活动按本课目标和 Pass 选用，不要求五项齐全，也不要求为了填模板而延长 session。固定约束只有：先给完整 context 再提问；涉及实验时先展示完整 design 再锁 prediction。具体 completion gate 以 [learning modes](learning-modes.md) 为准。

| Activity | 何时使用 |
|---|---|
| Map / tutorial | 每次进入新 Lesson 时，先建立完整上下文。可做成覆盖整课知识的通读讲义（Markdown 源稿 + 离线 HTML 阅读版，图与交互示例内置），学员通读后再进入练习单；先例见 [A1·S01–S03 讲义](modules/a1-transformer/s01-s03-primer.md)、[T3·S06 讲义](modules/t3-sim-to-rl/s06-primer.md) |
| Reactivation / self-check | 需要验证旧知识是否仍在手时；已有证据可缩短 |
| Guided questions | 用于暴露边界与误解；一次只问一个自足问题 |
| Build / intervene | `Implement` 或 `Improve` 需要 runnable evidence 时，使用 P-T-I-E |
| Solidify | 本次出现了值得保留的新 mental model、prediction 或 evidence 时 |

**教学体验原则（先行者综合评语，2026-08-29）**：出题思路与考察要点保留，体验按以下重做——
1. **English key terms first，白话负责解释**：论文、代码、config 与报告中的可检索词保留英文，首现采用 `English term（简短中文解释）`，之后直接使用英文；中文场景描述和类比可以帮助理解，但不得占据 metric、module、algorithm 或 config field 的名称位置。实验分支写 `Arm A/B/...`，避免与 robot arm 冲突；统一写法见 [Gym terminology guide](glossary.md)。（2026-08-30 抓获 `arm→臂` 冲突；2026-09-01 抓获“眼罩/切头/置换等变”式自造译名；2026-09-03 明确教学对话同样执行。）
2. **由浅入深预设计**：概念阶梯提前写进材料，一次只引入一个新概念，不靠课中补救；
3. **可视化默认，适用于所有学员**：经常用清晰、可交互的可视化解释抽象概念；结构与动态概念先有图或交互演示、后有简短文字。交互应让学员观察条件变化带来的机制差异，静态关系能用清晰图示讲透时无需强加交互。图示不是装饰，也不替代学员自己的判断、实现和实验；ASCII 字符画不算图（2026-09-01 抓获；2026-09-09 明确为全体学员共同教学原则）；
4. **贴近实际**：每个概念配一句"这在你的真实训练工作里对应什么"；
5. **愉快感是教学指标**：阅读困难、刻意设计、刻板流程都算缺陷；表格与流程能省则省——判断动作本身才是目的，包装不是。

**④"亲手"的三种合法模式**：亲手打字不是重点，决策归属才是。A 亲手实现（可随时查 API、问语法，不是闭卷考）；B 口述驾驶——学员逐行口述意图，AI 打字，打之前学员先说这行干什么，且 AI 每轮重贴当前题目原文（口述场景学员无法回翻）；C 填空抓错——AI 生成留空/埋错的版本，学员补齐并抓出。三种都满足"亲手"；不变量是**预测先于运行、每行能说出为什么、干预变量由学员定**。唯一不合法：AI 一次性生成整段，学员只扫一眼。

**预测的先验门槛**：预测只对学员已有先验的量提出——首次接触的概念不设赌局（蒙不是校准）；且预测只能针对**已完整呈现的实验设计**提出——任务书在前，下注在后。正确顺序：先演示（优先可视化/交互，文字为辅），预测挪到该概念的下一次干预上。口述驾驶模式中，AI 重贴题目时须同时展开所有代号指代（组名、模块名）。

**演示固化**：为课次生成的可视化/交互演示是教案的一部分——固化为 `modules/<模块>/demos/*.html`（自包含、浏览器直接打开），并在课次材料中链接，不得只留在聊天里。讲解性问答同样沉淀：课中聊出的卡点解释整理为课件"卡点问答"附录，供后来学员直接引用（2026-09-01 学员建议）。

**术语首现三件套与负载前确认**：新术语第一次出现必须给定义与直观图示（结构性概念必须画出来），并经学员复述确认；进一步地，任何概念或方法在被用作题目前提或教学工具之前——即便属于"应该已学过"——AI 须先确认学员当前是否熟悉。"档案 ≠ 现役"同样约束教学话术本身。负载前确认不得与依赖该概念的提问同发——确认得到学员回复之前，题目不得发布（2026-09-01 抓获：同一条消息里既求确认又发四道追问，等于不教而诛）。

**开课地图先行（2026-08-30 现场抓获）**：任何 session 的第一条教学内容必须是本课的白话地图——这节课在哪、要做什么、流程几段、产出什么；开课确认与一切提问都排在地图之后，且每问注明它服务于今天哪一段。没有 context 的提问——包括复习性确认——都是"开放问题开场"的变体。材料排版同受此约束：开课确认不得写在 §0 之前。复习性/确认性提问同样必须自带作答上下文与公式工具，不得让学员凭空回忆上节课的数值或矩阵（2026-09-01 抓获）。

**公式表开卷（2026-08-30 现场抓获）**：每课材料附「本课公式参考表」（公式 + 白话一句），手算段默认开卷——考的是方向感与量级判断，不是公式背诵；口述驾驶重贴题目时，涉及的公式同贴。参考表可多给公式，让学员练"挑出有用的那条"。

**任务书三要素（2026-08-30 现场抓获）**：向学员发布任何任务时须注明——(a) 任务类型（口述配置 / 手算 / 下注 / 判读 / 解释）；(b) 期望产出的格式与长度（一句话 / 一个数 / 三选一）；(c) 它在本课流程中的位置与目的。学员不应需要先猜"这是让我干什么"。任务书须自足：题目全文（函数规格、验收判据）直接呈现在对话里，不得以"打开文件看 TODO"的指针代替（2026-09-01 抓获）；发布动手任务前须确认所选模式的物理前提——学员无顺手编辑器环境时默认口述驾驶 B（2026-09-01 抓获）。

**Lesson / Pass 完成定义**：状态必须标注最高完成层 `✅ U / ✅ I / ✅ M`，不得用无深度信息的“学过”代替。`Understand` 完成概念地图、一个有上下文的判读和 revised mental model；`Implement` 另需 runnable artifact、完整 P-T-I-E 与 ledger 至少 1 行；`Improve` 另需 failure hypothesis、区分性 instrumentation、falsifier/control、改善与 regression evidence。每个 Lesson 累计至少 1 条现场 AI claim audit，不要求每个 Pass 为凑数重复审题。
**模块完成定义**：所有课次完成 + 通过该模块 Exit Gate + 固化笔记归入个人 records。

---

# A 线 · 机制（19 节）

## A1 · 现代 Decoder-only Transformer（7 节）

目标：把 attention 从"似是而非"重新焊死，亲手补上训练闭环，换装现代组件，最后在自己的模型里复现 in-context learning 的出现条件。

| 节 | 主题 | 动手产出 |
|---|---|---|
| A1·S01 | Attention 重激活：手算 + 50 行单头实现 | 手算全对 + 数值对拍的 numpy attention |
| A1·S02 | Causal mask + multi-head attention + mini decoder forward | 完整 decoder-only forward，shape 全注释 |
| A1·S03 | **训练闭环**：tiny GPT 训到能生成 | loss 曲线 + 采样输出 + 一次过拟合小实验 |
| A1·S04 | KV cache 与 prefill/decode | 带 cache 的推理，与全量前向 assert 等价 |
| A1·S05 | 现代组件换装：RoPE / RMSNorm / SwiGLU | 逐个替换 + 单变量对照 |
| A1·S06 | Induction head：ICL 的机制细胞 | 2 层 attention-only 模型复现 copy 回路 |
| A1·S07 | 复现 burstiness → ICL（Chan et al. 玩具版） | 成簇采样 vs 均匀采样的 ICL 出现对比 |

A1·S03 补充阅读覆盖计算图、常见运算的梯度和一次 SGD 更新。[Autograd 讲义](modules/a1-transformer/autograd-primer.md) · [HTML 互动版](modules/a1-transformer/autograd-primer.html)

S06–S07 机制背景：[ICL 机制精编](references/icl-s1-gen15-digest.md)。

**Exit Gate**：用[「Transformer 15 问 debug checklist」](references/transformer-digest.md)逐问过一遍 π0.5 的 attention 相关代码并入档（pi05-training.md（内部资料） 为入口）。

## A2 · 生成式建模与动作头（5 节）

目标：从离散 NLL 走到 diffusion 与 flow matching，亲手理解现代生成式动作头在优化什么、没保证什么。

| 节 | 主题 | 动手产出 |
|---|---|---|
| A2·S01 | 离散动作：action token + NLL（接 A1 的 LM） | 离散控制 toy，policy = LM |
| A2·S02 | 连续动作的坑：Gaussian + MSE 的 mode averaging | 双峰 toy 上演示无效平均 |
| A2·S03 | Diffusion：DDPM forward noising / reverse denoising / ε-prediction（概念层可由 [DDPM 精编](references/ddpm-uwm-digest.md) Q1–Q3 唤醒快进；重心放在 training loop 与“μ 只在 sampling 端出现”的两套循环） | 2D 双峰上训 toy denoiser，对照 S02；ε̂ vector field 随 timestep 的扫描图 |
| A2·S04 | Flow matching：与 diffusion 的统一视角，vector field 监督与采样；端点严格性对照（ᾱ_T≈0 近似 vs t=1 恰好纯噪声，精编 Q9） | 1D/2D toy flow policy |
| A2·S05 | 架构落地：DiT（AdaLN conditioning）、ACT（CVAE + chunking）、π0.5 action expert 三方对照；condition 进入网络的四种方式与“conditioning 只改一处”在真实代码里定位（精编 Q4/Q5） | 对照表 + "flow loss ≠ log π" 代码级确认 |

S03–S05 机制背景：[DDPM 精编（UWM 第二章精读）](references/ddpm-uwm-digest.md)——学员 2026-09 自读闭合了概念层（前向/反向/ε-prediction/条件化）；按"档案 ≠ 现役"，S03 的 ① ③ 段按其 Q 编号唤醒并快进，② 手算与 ④ 训练闭环照做，精编末节「笔记未覆盖」即课程要补的清单。

**Exit Gate**：用[「机器人 Transformer 15 问」](references/transformer-digest.md)过一遍 π0.5 的动作链路（chunk、H/K、normalization、executor）。

## A3 · Latent-Space 融合机制（7 节）

目标：成为能用 latent space 组合模块的熟练工——每种融合机制亲手写最小版，再解剖真实系统。

| 节 | 主题 | 动手产出 |
|---|---|---|
| A3·S01 | 视觉基础组件：ViT / CLIP / DINOv2 / SigLIP 特征性质与选用 | 同图探针对比（语义对齐 vs 密集几何）+ 一页选用判断表 |
| A3·S02 | Projection / adapter（LLaVA 式） | 把 S01 探过的冻结特征注入自有 GPT |
| A3·S03 | Latent query / resampler（Perceiver、Q-Former；DETR object query 的直系后代） | 可学习 query 压缩变长输入的 toy |
| A3·S04 | 双塔与对比对齐（CLIP 式 late fusion）——S01 用特征，本课造特征 | 玩具双塔 + 对比损失对齐 |
| A3·S05 | MoT：共享 attention、分参数塔 | 解剖 π0.5 的 VLM 塔 + action expert 结构 |
| A3·S06 | MoE 实现：router / top-k / 负载均衡 | 在自有 GPT 里插一层 toy MoE |
| A3·S07 | Generator × Reasoner 双系统（Cosmos Predict/Reason、GR00T 式；WAM 的机制定位；UWM 作"timestep 当遮蔽旋钮、策略与动力学同一网络"的案例，精编 Q8） | tiny world-model generator + 读其 latent 的小 reasoner |

S05–S07 案例背景：[ICL 机制精编](references/icl-s1-gen15-digest.md)；S07 另见 [DDPM 精编](references/ddpm-uwm-digest.md) Q8。

**Exit Gate**：任选一个课程未讲过的前沿模型，独立画出它的 latent-interface 图（什么 latent 过界、谁训谁冻、损失落在谁头上），交同事 Skeptic 审。

---

# T 线 · 训练（26 节）

## T0 · 训练闭环解剖（3 节）

目标：在最小系统（MLP 二分类）上完成第一次完整训练闭环，把 formulation、优化几何与数据分布拆成三个可单独干预的对象。源自 v1 的 W1D1（保留其 formulation 转译与 A/B/C 设计），按"教程先行"重制，并补上 v1 缺失的 freeze 验证与容量对照两组。S03 为 2026-08-30 新增：把"答题模型"扩展为"开车模型"，正面回答 offline loss 与闭环能力为何脱节——源自学员真实 VLA 后训练诊断暴露的覆盖缺口。

| 节 | 主题 | 动手产出 |
|---|---|---|
| T0·S01 | Loss 作为接口：formulation 转译 + 第一次闭环 | 抓稳规则的六要素转译 + baseline（平衡+标准化）闭环 |
| T0·S02 | Optimization geometry 与 training distribution：five-arm intervention | Arm B scale×1000 / C class imbalance / D freeze+Δ=0 / E capacity bottleneck，逐 Arm 先预测后运行 |
| T0·S03 | Closed-loop gap：offline loss 与 SR 为何脱节 | corridor BC toy：Arm F first out-of-support metrics / G execution horizon E1·E4·E16 / H recovery data，三 Arm P-T-I-E |

**Exit Gate**：能用实验证据区分 representation capacity、optimization geometry 与 training distribution 三种效应；freeze 组有参数 Δ=0 的实证记录；能用首次偏离与恢复证据解释 offline loss 与闭环成功率的脱节，并完成一份真实诊断报告的证据类型标注（S03 应用作业）。

## T1 · 训练阶段架构（4 节）

目标：回答"我的 SFT 在整条 pipeline 的哪一段、上下游各保证什么"，把 pre/mid/post/continual 变成可判断的工程决策。

| 节 | 主题 | 动手产出 |
|---|---|---|
| T1·S01 | [阶段地图：SmolVLA 实验 → pre / mid / post 职责 → Cosmos 3 对照](modules/t1-training-stages/s01-training-stage-map.md) | 小样本干预、阶段训练与 W&B/rollout 判读；有证据边界的阶段地图 |
| T1·S02 | Continual pretraining toy：遗忘与迁移 | 域 A→B 训练的遗忘曲线 + replay 干预 |
| T1·S03 | π0.5 配方解剖与自身 SFT 定位 | 数据混合/保留能力的机制解释 |
| T1·S04 | 判断演练：CT vs SFT vs post-training | 三个假想任务的阶段选择 + 说理 |

**Exit Gate**：为一个真实候选任务写一页训练阶段方案，说明每段的证据边界，自审后入档。

## RL 基础讲义与课次的对应（T2/T3 共用）

**TD3-first 试读入口：** [第一篇](modules/t3-sim-to-rl/td3/README.md) → 分次问答 → P4 核心实现与固定 batch 验证 → TD3 对应公开代码 → 控制任务诊断与单变量改进。无需先通读 foundations、通过整套 P0 或完成 PPO/SAC。此路径不新增 Lesson，不更改 Pass 或课程级 Exit Gate；下列完整复习路线仍可选。

**整体学习顺序：恢复价值学习 → PPO → TD3 → SAC → 生成策略后训练 → 可微动力学 BPTT。** 用 [RL 基础讲义](modules/t3-sim-to-rl/rl-foundations-primer.html) 和 [统一温习讲义](modules/t3-sim-to-rl/rl-control-review-primer.html) 读机制，按 [分段实践](modules/t3-sim-to-rl/rl-control-practice.md) 做伪代码、实现和对照。P 编号是练习入口，沿用既有编号；不按编号线性推进。BPTT 在经典算法基础之后可独立选修，不依赖做完生成策略后训练。

| 阶段 | 课次 / 练习 | 亲手做什么；达到什么程度再推进 |
|---|---|---|
| 恢复 value learning | T2·S01 / [P0](modules/t3-sim-to-rl/rl-control-practice.md#p0--恢复价值学习) | 一批情境题；能从 transition 说清 bootstrap、backup 在评价谁、哪个 observation 可用。只补错题与不确定项。 |
| PPO | T2·S02–S06、T3·S01 / P1 → P2 | 先复原 REINFORCE → baseline → GAE → PPO 的伪代码，亲手写 GAE、ratio/clip 与 rollout 更新，再完成一个最小连续控制闭环。能用固定数据验证 mask、梯度与 action/log-prob 对齐。 |
| TD3 | T3·S02 / P4 | 亲手写 TD target、twin Q 更新、actor 梯度、delay 和 target update，跑通一套共享 replay 闭环。能沿执行顺序定位哪个网络什么时候更新。DQN/Double DQN 是选修桥。 |
| SAC | T3·S02 / P4 内 SAC 对照 | 独立复原全流程；复用 TD3 的环境、replay 和评测，亲手写 stochastic actor 的采样/log-prob、soft target、actor 和 temperature 更新。固定 batch 的对照是基础验收；完整小训练用于验证行为及进一步诊断。 |
| 生成策略后训练 | T3·S05–S07 / P7 | 先补 offline RL 的 coverage 与 policy extraction，再对照直接训练生成策略、冻结 base 后修正/选动作，以及混合流程。亲手实现或验证一个小机制，随后选一条真实源码路径解释它相对经典算法的改动。 |
| 可微动力学 BPTT | T3·S10（选修）/ P8 | 亲手写短时可微 rollout；用解析/数值 oracle 和断开 dynamics 梯度的对照验证 credit path，再读 SuperMimic。能区分 learned dynamics、critic bootstrap 和真实 simulator。 |
| 应用与其它专题 | T3·S03/S04/S08/S09 / P3、P5、P6 | DexLab teacher/student、sim2real、capstone 和 Zero-Shot RL 继续按原课次进入；无需插在三种经典算法之间。 |

**完整跨算法研究代码对照的门槛：** 在不看参考实现时，为 PPO、TD3、SAC 各重建一次更新的伪代码，标出数据来源、target、gradient path、optimizer 与更新时序，并在固定 batch 上用独立 oracle 验证三者的核心更新。PPO 和 TD3 的闭环是默认实践；参考训练尚未验证可行时，可以先完成上述门槛进入代码判读，闭环继续单独验收。[Pass 规则](learning-modes.md)仍适用：伪代码和判读属于 `Understand` 证据；`Implement` 要有学员核心代码、可运行验证和 P-T-I-E；`Improve` 还需能区分原因的干预及回归证据。原有课程级 Exit Gate 不变。TD3-first 路径可在完成 TD3 自身的核心重建与固定 batch 验证后，先阅读对应 TD3 update；不把三算法门槛套在这一局部判读上。

[RL 基础讲义：VLA online RL 背后的强化学习](modules/t3-sim-to-rl/rl-foundations-primer.md)（[HTML 阅读版](modules/t3-sim-to-rl/rl-foundations-primer.html)）承担下列课次的 `Understand` 层：学员通读对应知识点，完成有上下文的判读并写下 revised mental model；各课的实践深度见上表。

| 课次 | 讲义知识点 | `Implement` / `Improve` 层 |
|---|---|---|
| T2·S01 | 1–10（MDP、Bellman、TD、SARSA / Q-learning、n-step、maximization bias） | 原课：grid-world 手算 + 代码对拍 |
| T2·S07（选修） | 11–12（deadly triad、DQN / Double DQN） | 同一 batch 对拍 online/target 网络如何挑和评 |
| T3·S02 | 13–17（DDPG、TD3、SAC、REDQ / UTD、primacy bias / LayerNorm） | P4 的共享 replay 闭环与 TD3/SAC 核心对照；可引用同目标的真实复现证据 |
| T3·S05 | 18–27（policy improvement 边界、offline RL 四族、offline-to-online、replay 数据身份） | 二选一：AWR toy，或真实复现证据 |
| T3·S06 | 28–29（expressive policy RL 路线）+ [S06 讲义](modules/t3-sim-to-rl/s06-primer.md) | 二选一：EXPO-lite toy，或真实复现证据 |
| T3·S07 | 30–33（residual RL、action chunk / semi-MDP、人在环、reward 来源） | 二选一：RLT toy，或真实复现证据 |

**真实复现证据**指学员在根仓库已授权的训练、复现或诊断工作中取得的结果（以对应 registry 与 `reports/` 为准）。它要满足与 toy 相同的 P-T-I-E 结构：运行前写下带机制的预测与推翻条件，一次只改一个主要变量，运行后对照预测解释；`Improve` 另需 failure hypothesis、区分性检查与 regression evidence。课程本身不授予任何训练许可，Gym 记录只链接正式证据，不复制指标。[EXPO-FT 复现案例集](modules/t3-sim-to-rl/expo-ft-casebook.md)把 pi-expo-sim X2–X3.10 中的 30 个问题按同一结构整理并对回讲义知识点，是 T3·S02/S05/S06 使用真实证据时的阅读入口。

RLT / EXPO-FT 基础讲义未展开 T2·S02–S06 的 on-policy 线；现在由统一温习讲义 §2 与 P1–P2 衔接原 REINFORCE → PPO 实现路线，并用 DexLab 展示连续控制。T2 的 Exit Gate 不变。

## T2 · RL 从零到 PPO（7 节，S07 选修）

目标：把已有的理论档案（见 [RL 问答精编](references/rl-spinningup-digest.md)）变成自己的代码。每节开场先默写当日核心公式（重激活），再实现。

| 节 | 主题 | 动手产出 |
|---|---|---|
| T2·S01 | Tabular：value/policy iteration，把 Bellman T 落成 20 行循环（U：讲义 1–10） | grid-world 手算 + 代码收敛对拍 |
| T2·S02 | REINFORCE 从零（CartPole 级） | 亲手写 log_prob 与更新循环 |
| T2·S03 | Baseline → A2C；stop-gradient 亲手验证（精编 Q6） | advantage 挡不挡梯度的对照实验 |
| T2·S04 | GAE：实现 + λ 扫描 | bias-variance 曲线 |
| T2·S05 | Importance sampling 推导 → PPO ratio/clip 从零（Q15–Q17） | 正负 advantage 下 clip 行为的数值验证 |
| T2·S06 | PPO 完整版：对齐 CleanRL 参考 | 达到参考曲线区间 + 逐项差异说明 |
| T2·S07（选修） | DQN / Double DQN walkthrough（U：讲义 11–12） | 固定 next-state Q 表，对照选动作与评动作的网络；SAC 进入 T3·S02 |

**Exit Gate**：自有 PPO 在基准环境进入参考实现的成绩区间，能逐行说明与 CleanRL 的实现差异及其影响。

## T3 · Sim-to-RL 与机器人 RL（10 节，S09/S10 选修专题）

目标：从基准环境走向仿真机器人，补上 offline / online RL 谱系，最终形成真实工作的可审查产物。

| 节 | 主题 | 动手产出 |
|---|---|---|
| T3·S01 | 自有 PPO 上连续控制（MuJoCo 级） | 连续动作 Gaussian policy 跑通 |
| T3·S02 | TD3 → SAC：连续控制的两种 off-policy 更新（U：基础讲义 13–17 + 统一温习 §4） | [P4](modules/t3-sim-to-rl/rl-control-practice.md) 共享 replay 实现、TD3 单变量干预、SAC soft target / temperature 对拍；或同目标的真实复现证据 |
| T3·S03 | Domain randomization + teacher-student / privileged info；DexLab HORA → AnyRotate | P3 读清 teacher / history student 与 gradient；P6 设计 DR / distillation 单变量干预，执行后验收 |
| T3·S04 | Sim2Real：dynamics gap、system ID、control / sensor contract、何时 BC 何时 RL | H13 action / timing / tactile / bank / pure-student evaluation 案例；与 FB 新 reward 泛化分开判断 |
| T3·S05 | Offline RL 与 advantage-weighted 族（AWR / IQL；RECAP 的直系谱系）（U：讲义 18–27） | AWR 在离线数据上的 toy 实现；或真实复现证据 |
| T3·S06 | [生成策略后训练：以 EXPO select + edit 为锚](modules/t3-sim-to-rl/s06-expressive-policy-value-maximization.md)；DPPO 的 likelihood、RECAP / AG 的条件生成、DSRL 的噪声接口、DICE-RL / GSBR 的行为约束（U：讲义 28–29 + [S06 讲义](modules/t3-sim-to-rl/s06-primer.html)，生成器内部梯度前置 A2） | P7 从一个受控机制实验进入源码差异表；保留 EXPO-lite CPU toy / 真实复现证据，以及 EXPO-FT 迁移审计 |
| T3·S07 | [Online adaptation：RL Token 与 critical-phase real-world RL](modules/t3-sim-to-rl/s07-rl-token-online-adaptation.md)（前置：T0·S03、T3·S02、T3·S06；推荐 A3·S05；U：讲义 30–33） | frozen-VLA + small actor/critic toy 或真实复现证据；π0.5 transfer audit |
| T3·S08 | **Capstone**：Action Expert selective-RL preregistered proposal | 完整 experiment card + rubric 自审 + 同事审 |
| T3·S09（选修） | [Zero-Shot RL：从 SR / SF 到 FB 与人形 BFM](modules/t3-sim-to-rl/s09-zero-shot-rl.md)（前置 T2·S01、T3·S05；[完整 HTML 讲义](modules/t3-sim-to-rl/zero-shot-rl-primer.html)先行；[笔记精编](references/fb-bfm-notes-digest.md)） | 八个课内单元；2021 核心与 z / ρ → 2023 精读 → FB-IL / offline 修补 → Meta Motivo / BFM-Zero → IL / VLA 接口判读；按 Pass 选择机制判读、最小接口验证或 failure control |
| T3·S10（选修） | [可微动力学与 BPTT：SuperMimic](modules/t3-sim-to-rl/rl-control-practice.md#p8--可微动力学与-bptt)（前置：PPO/TD3/SAC 更新对照，能区分 bootstrap 与 gradient path） | 短 rollout 的解析/数值梯度对拍；保留或切断 dynamics gradient 的单变量对照；对回原文的数据收集、模型拟合与 actor/critic 更新 |

S09、S10 可在补齐各自前置后按需进入，不要求先完成 S08，也不改变 S08 capstone gate。S09 的 QP 暂不进入。RL（T2+T3）为 17/45 ≈ 37.8%，必修 Lesson 数仍为 42。讲义参考计算、论文结果与学员自己的实践证据分别记录。

T3 grounding（只读不复制）：[DexLab 固定版本代码导读](references/dexlab-sim-rl-digest.md)、g3-robotwin-post-training.md（内部资料）、frontier.md（内部资料）、G4 closeout（内部资料）、控制器 spec（内部资料）、[RLT → π0.5 / RoboDojo 课程精编](references/rlt-pi05-robodojo-digest.md)、pi-expo-sim（EXPO-FT 复现）（内部资料）。

**Exit Gate**：proposal 达到"可提交根仓库审批、或有据拒绝"的标准；未获授权不产生任何可引用 checkpoint（授权边界见 ml-engineering.md（内部资料））。

## T4 · 训练工程实践（2 节，位置灵活）

目标：把训练从"能跑"升级为"专业地跑"。建议 S01 在进入 T2 前完成，S02 在首次多卡需求前完成。

| 节 | 主题 | 动手产出 |
|---|---|---|
| T4·S01 | 现代训练循环工程：AMP / 梯度累积与裁剪 / lr 调度 / metrics 与实验跟踪；JAX（jit/lax/pytree）对照，openpi 片段研读 | 把 A1·S03 的 tiny GPT 训练循环升级并接上实验面板 |
| T4·S02 | 分布式训练：DDP / FSDP 机制、数据并行 vs 模型并行、吞吐 profiling | 双进程 DDP toy 跑通 + 一次 profiling 记录 |

**Exit Gate**：能对一个给定训练脚本说出它的精度策略、并行策略、显存去向与首要瓶颈。

---

# 课程级完成标准

1. 42 个 Lessons（选修除外）至少完成 `Understand`，并在 progress 中保留最高 Pass；
2. 8 个模块 Exit Gate 全过；
3. ledger 累计 ≥ 60 条预测判定，并做过一次命中率复盘；
4. A 线终点：一张独立完成并通过 Skeptic 审的 latent-interface 解剖图；
5. T 线终点：一份进入（或有据不进入）正式审批流程的 selective-RL proposal；
6. 每模块固化笔记归入 `learners/<name>/records/`，学员可自选追加进个人 learning archive。
