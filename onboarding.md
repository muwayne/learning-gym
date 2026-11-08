# Gym learner onboarding

> 新学员从这里开始。本文件只负责建立个人学习入口；课程内容看 [curriculum.md](curriculum.md)，学习深度与节奏看 [learning-modes.md](learning-modes.md)。

## 0 · 建立自己的工作分支

先从最新 `main` 建立个人 Gym 分支。`<name>` 使用简短的 ASCII lowercase 名称，必要时用 `-` 分隔：

```bash
git switch main
git pull --ff-only
git switch -c gym/<name>/start
```

个人学习记录正常 commit 到自己的分支；push 与 PR 遵守仓库协作约定。

## 1 · 让 AI 先读规则

在创建文件前，让陪练 AI 按顺序完整阅读：

1. 根目录 `AGENTS.md`；
2. [README.md](README.md)：Gym 的目标与协作边界；
3. [handoff.md](handoff.md)：AI 的接力上下文；
4. [learning-modes.md](learning-modes.md)：`Understand / Implement / Improve`；
5. [curriculum.md](curriculum.md)：共同课程主干。

这一步是为了让 AI 使用现有体系，不从其他学员的记录里临时发明一套课程。

## 2 · 建立个人目录

每位学员只维护自己的空间：

```text
learners/<name>/
├── baseline.md
├── progress.md
├── ledger.csv
├── records/
└── code/
```

- `baseline.md`：当前基础、既有资产、目标和可选的默认 Pass；
- `progress.md`：唯一的个人进度源，记录每个 Lesson 的最高 `U / I / M`；
- `ledger.csv`：实验前 prediction 与实际结果的校准账本；
- `records/`：每次学习的 reasoning、evidence 和 revised mental model；
- `code/`：个人练习代码。checkpoint、dataset、大型日志和 rollout 不放进 Git。

初始化规则：

- 从 [templates/learner-baseline.md](templates/learner-baseline.md) 创建 `baseline.md`；
- `ledger.csv` 的第一行固定为：

  ```csv
  date,session,claim,predicted,actual,hit,note
  ```

- `progress.md` 从 `curriculum.md` 的当前课表生成，不复制其他学员的完成状态；
- `records/` 与 `code/` 可以为空，第一次产生内容时再添加文件；
- 可以参考 `learners/wayne/` 的结构，但不得复制其中的 prediction、答案、结论或进度。

## 3 · 填写 baseline

baseline 不是入学考试，也不要求一开始决定所有课程的深度。请如实区分：

- concept 是否理解；
- implementation / training loop 是否亲手做过；
- 有哪些可核验的旧代码、笔记或项目 evidence；
- 最想通过 Gym 获得什么实际能力；
- 是否已有稳定的默认 Pass。

新学员的默认设置是：**每个新 Lesson 先看完整地图，再选择本次 Pass**。如果暂时不知道选什么，就从 `Understand（看懂）` 开始。以后可以在同一 Lesson 上继续 `Implement（做出）` 或 `Improve（改好）`。

## 4 · 登记学员入口

在 [progress.md](progress.md) 的学员表新增一行，只记录路标：名字、A/T 两线位置、日期和个人目录。详细进度只维护在 `learners/<name>/progress.md`，不要在多个文件重复维护同一事实。

个人目录只由本人和自己的 AI 修改；不得写入其他学员目录。共享的 curriculum、modules、templates 与 rubric 改动走 PR。

## 5 · 开始第一课

完成 baseline 后，从 `T0·S01` 或 `A1·S01` 开始。可以双线并行，也可以先走一条；不按日历追进度。

把下面这段发给陪练 AI，其中 `<name>` 换成自己的目录名：

```text
请先完整阅读根 AGENTS.md，以及 README.md、onboarding.md、
handoff.md、learning-modes.md、curriculum.md，
再读取 learners/<name>/baseline.md 和 progress.md。

请从我的下一节 Lesson 开始。第一条教学消息先完整说明：
1. 本课在课程中的位置与共同核心；
2. Understand / Implement / Improve 在本课分别会做到哪里；
3. 本次自然停止点与产物。

如果我没有稳定默认 Pass，给完地图后再询问本次选择；不要用无上下文问题开场。
教学中保留 English key terms，中文负责解释；一次只问一个自足问题。
活动按学习需要选择，不要求固定时长或五段齐全。
```

## 6 · 学习过程中如何留痕

- `Understand`：至少留下核心 mechanism、一个有上下文的判读和 revised mental model；
- `Implement`：在前者基础上留下 runnable artifact、一次 P-T-I-E 和 ledger prediction；
- `Improve`：再留下 failure hypothesis、区分性 instrumentation、falsifier/control 与 improvement/regression evidence；
- 每个 Lesson 累计至少一条现场 AI claim audit，不为了填表制造审题；
- 同一 Lesson 再学一遍时追加 Pass evidence，不覆盖第一次的 prediction 和理解变化。

结束一次 session 不等于必须结束整个 Lesson。到自然 checkpoint 就可以停，下次从 progress 与 record 恢复。

## 开户完成检查

- [ ] 个人分支已建立；
- [ ] `baseline.md` 已填写；
- [ ] `progress.md` 与 `ledger.csv` 已初始化；
- [ ] `progress.md` 已登记；
- [ ] AI 已读 handoff 与 learning modes；
- [ ] 第一课先给地图，再选择 Pass。
