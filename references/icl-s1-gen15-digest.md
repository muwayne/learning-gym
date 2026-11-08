# ICL × Robotics 机制精编（S1 × GEN-1.5 调研档案浓缩版）

> 源自 wayne 的内部调研档案（2026-08-28），2026-08-29 浓缩入库。完整原档（含证据审计全文、问答存档、ego 数据推论）留学员本地。
>
> 纪律沿用原档：厂商数字均为自报未验证；【声称】= 厂商原文，【评】= 分析。用途：A1·S06–S07 的机制背景、A3·S05/S07 的案例背景。

## 事件与三个 verdict

2026-08 两家 Physical AI 公司同周宣称机器人版"GPT-3 时刻"（GEN-1.5 于 08-19，Skild S1 于 08-25）。

- **方向为真**：demonstration conditioning 由 ≥4 组独立团队同季度收敛；
- **量级存疑**：无论文、无权重、无第三方评测，关键口径系统性偏乐观；
- **叙事金融化**："GPT-3 moment" 话术全部来自两家 cap table 上的投资人。

## 核心：任务指定接口换代

```text
π(a | o, ℓ)  →  π(a | o, D)
任务被"描述"  →  任务被"演示"
```

语言是有损压缩，覆盖分布头部（常见原子动作甚至 zero-shot 就够）；demo 是长尾接口（"翻煎饼"在动作空间里无词可 ground，示范自 grounding）。【声称】100k 小时数据下 unseen 任务：demo 条件化 66% vs 语言条件化 9%。【评】这条交叉曲线的含义是"demo conditioning 把规模变现到长尾"，不是"demo 一定比语言好"。

## 真正的分岔在训练时的 context 组织

| | training context | inference context | 一致性 |
|---|---|---|---|
| 常规 VLA | [语言 goal; 观测史] → action | 同左 | 一致 |
| S1 | [另一条 episode 的 demo; 执行观测史] → action；demo 槽位训练时就存在，且刻意跨场景/视角/embodiment | 同左 | 一致 by construction（工程成就） |
| GEN-1.5 | 连续 span（obs+action 混排）→ next action；无 demo 槽位、无任务条件化结构 | [拼接的 demo 片段; 滚动观测] | 错位（分布外用法；"居然 work"就是涌现声称的全部） |

直觉锚点：demo ≈ LLM prompt 顶部的 few-shot examples（收窄 task belief）；观测历史 ≈ 当前对话上文（收窄 state belief）。

## 涌现假说：burstiness + Zipfian（Chan et al. 2022）

- **成簇性**：真实劳动同类动作扎堆——前一颗螺栓就是后一颗的免费 in-context demo → attend to context 有直接 loss 收益，梯度奖励 induction 回路；
- **长尾**：头部行为写进权重（zero-shot），尾部太罕见写不进 → 只能走 in-context；
- 保留失误段不剪 = 免费的 recovery 训练信号（fail→adjust→succeed）；
- **Caveat**：Chan et al. 是合成分类任务上的结论，向 sensorimotor 域的迁移未经验证——A1·S07 复现的正是合成版本，结论不得外推。

## S1 五局限（评）

配对监督质量即天花板；retrieval-and-copy 捷径风险（L5 退化为证）；泛化被 task family 界定；每任务需多条 episode 与 diversity 目标打架；demo 占 context 的 prefill 成本。

## L1–L5 双轴扰动协议（值得移植，独立于厂商数字）

| 级 | 扰动 | 结果（自报） |
|---|---|---|
| L1 | 与训练/示范同分布 | ICL 与 VLA 均稳 |
| L2–L3 | 位移 15→30cm、旋转 30→45° | 递减 |
| L4 | 同 affordance 换物 | ICL 仍可用 |
| L5 | 迫使执行计划改变（换对侧手臂） | ICL 显著退化——demo 的计划先验变成错误先验 |

**口径教训**：S1 的 66% 是 per-step 成功率 + 人工干预恢复的口径；formal rollout 严禁把人工干预混入成功率统计。

## 数字纪律

66%、380:1、59% 等一律禁止当效应量先验进入实验设计；可引用的只有：交叉趋势的方向性、L1–L5 协议设计、"质量换数量"的 QC 立场。

## 与课程的接口

- A1·S04：demo 作为静态前缀 = prefix KV 常驻的直接应用；
- A1·S06：induction head 是 ICL 的机制细胞；
- A1·S07：复现 burstiness→ICL（玩具版，合成任务）；
- A3·S05/S07：MoT 与 Generator×Reasoner 案例——GEN-1.5 本质是连续流 next-action 模型（WAM 近亲）；
- 团队 ego 数据推论（原档 ACTION 01–06）属工作决策，不属课程，见原档。
