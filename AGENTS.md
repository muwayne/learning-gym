# Learning Gym · AI 协作边界

本文件适用于整个 Learning Gym 仓库。

## 目的

`gym/` 是团队研究训练工作区（双线课表 v2），保存：

- 各学员的课次记录、预测账本、练习代码与固化笔记；
- 手算自检、P-T-I-E 实验与 AI claim audit；
- A 线（机制）与 T 线（训练）的共享课程材料。

## 事实边界

`gym/` 不是项目状态、训练许可、运行事实或正式结论的第二来源：

- 正式训练许可只看所属项目的 active training spec 和工程规约；
- 真实运行进入对应 `devlog/.../run_registry.csv`；
- 正式结论进入 `reports/`；跨期代码进入 `src/`；
- checkpoint、dataset、训练日志和大型 rollout 不进入本目录。

## 工作方式

1. 预测先于实验；每个实验只改一个主要变量；
2. 每课材料标明哪些代码必须学员亲手写；AI 只搭 harness、assert 与图表；
3. AI 生成的脚手架本身是审查对象：每课至少 1 条 claim audit（判据见 `ai-review-rubric.md`）；
4. toy/smoke 结果不得写成真实 π0.5/RoboTwin 能力证据；
5. 产生可引用 checkpoint 的正式训练由所属项目流程授权；
6. 同一实验逻辑第二次需要时考虑晋升 `src/`，不在 Gym 内复制；
7. 教学对话与课程材料中的 key terms 使用英文；首现可加简短中文解释，中文类比不得替代论文、代码、config 与报告中的可检索词；统一写法见 [`glossary.md`](glossary.md)；
8. 不为目录对称创建空文件。

## 多学员规则

- 共享材料（`curriculum.md`、`modules/`、`templates/`、`ai-review-rubric.md`）改动走 PR；
- 个人内容只进 `learners/<name>/`；AI 不得写入其他学员目录；
- 学员总览在 [progress.md](progress.md)；唯一个人进度源是 `learners/<name>/progress.md`；
- 出题提示与讲解按该学员 `baseline.md` 适度调整；课表结构、材料与完成标准对所有学员一致。

## 当前入口

- 总览与协作方式：`README.md`
- 接力 context：`handoff.md`
- 唯一课表：`curriculum.md`
- 学员总览：`progress.md`
- 审查规则：`ai-review-rubric.md`
- 术语表：`glossary.md`
- 档案精编：`references/`（学员档案浓缩版，Q 编号与 checklist 引用的解析目标）
- 课次记录模板：`templates/session-log.md`；capstone 用 `templates/experiment-card.md`

新 AI 首次进入按 `handoff.md` 第 6 节的冷启动顺序阅读。
