# AI Proposal Review Rubric

> 目的：把“听起来合理”转化为可核验的机制、实现契约和证据。

## 两种判断必须分开

每条 AI claim 同时记录：

1. **Conceptual Verdict**：Correct / Plausible but vague / Conceptually wrong；
2. **Evidence Status**：Untested / Supported / Refuted / Inconclusive。

概念上成立但尚未实验的方案，不等于已被证明；偶然获得高 reward，也不会自动修复概念断链。

## 强制机制链

```text
observation / time contract
→ representation
→ information path
→ conditional action distribution / chunk
→ actually executed action or prefix
→ reward / termination
→ return / advantage / score estimator
→ optimizer
→ named trainable parameters
→ held-out closed-loop evaluation
```

- 链路缺段但没有已知矛盾：Plausible but vague；
- 链路明确断裂：Conceptually wrong；
- 每段均有代码、日志或最小实验可验证：才可能判为 Correct。

## 审查矩阵

| 维度 | Correct | Plausible but vague | Conceptually wrong |
|---|---|---|---|
| Claim / Falsifier | 条件、机制、预测与反证明确 | 只说“应提升”“更稳定” | 证据不能支持 claim |
| Representation | shape、time、units、normalization 明确 | 只说“更好 embedding/fusion” | 语义矛盾、信息缺失或 leakage |
| Distribution | train、rollout、eval 分开定义 | 只说“更多 trajectories” | 混用分布或 train/eval leakage |
| Objective / Reward | 明确优化什么、没保证什么 | 只给 loss/reward 名称 | proxy 与目标错位或 shortcut 明显 |
| Information Flow | 来源、consumer 与 intervention 明确 | 只说 attention/fusion | 必要信息到不了输出 |
| Gradient / Credit | 追踪到 named parameter groups | 只说 freeze/train expert | detach、漏参、错模块或时间错位 |
| Policy Estimator | score/log-prob 或替代 estimator 合法 | 只说 PPO/GRPO/KL | estimator 与 policy/sampler 不兼容 |
| Controls | 单变量、held constants、负控明确 | 多项一起探索 | 多变量实验却作单一因果归因 |
| Evaluation | held-out、paired seeds、CI、失败与行为健康 | 单 seed 或只给均值 | 只看 training loss/reward |
| Reproducibility | code/config/checkpoint/task/budget 可追溯 | 只保存部分身份 | baseline 或分支起点不可复现 |

## 判定规则

### Correct

- 条件明确；
- mechanism chain 闭合；
- 实现与描述一致；
- 有可观察 prediction 和 falsifier；
- 当前证据没有矛盾。

Correct 只在声明的 task、distribution、版本和条件内成立。

### Plausible but vague

- 没有明确错误，但关键环节缺定义、代码证据或反证条件；
- 无法判断信息是否到达、参数是否更新、指标是否识别机制；
- 不批准昂贵训练，先补最小证据。

### Conceptually wrong

任一 fatal break 即可判定：

- system/action/time contract 自相矛盾；
- 必要 information path 不存在；
- gradient 或 reward credit 到不了目标参数；
- objective / estimator 与声称算法不一致；
- distribution assumption 明确不成立；
- evaluation 或 negative control 已推翻机制。

## 固定审查表

| Claim | 类型 | 机制与条件 | 代码/配置证据 | 可观察预测 | 最小 Falsification | Verdict | Evidence |
|---|---|---|---|---|---|---|---|
|  | Fact / Hypothesis / Design Choice |  |  |  |  |  |  |

摘要：

- Fatal issues：
- Missing evidence：
- Smallest falsification：
- Allow pilot：Yes / No；
- 最大允许预算：

## π0.5 / RoboTwin / RL Fatal Red Flags

1. 未定义 action/chunk 的 policy score 或合法替代 estimator，却直接套 PPO/GRPO。
2. 把 flow/denoising regression loss 直接当作 `log π`、KL 或 importance ratio。
3. 假设 reward 可直接穿过 simulator 反向传播，却没有说明 estimator。
4. chunk 只执行 prefix 或会 replanning，却给未执行动作相同 credit。
5. 忽略 coordinate、units、joint order、gripper、normalization、clipping 或 control rate。
6. 声称 freeze VLM，却没有 parameter inventory、optimizer groups、gradient、delta 和 checksum。
7. zero advantage 仍产生未声明更新，或 permuted reward 也能稳定提升。
8. 同时修改 reward、algorithm、chunk、trainable layers 和 data mix。
9. offline loss 下降就宣称 closed-loop behavior 改进。
10. reward 上升但 true success、安全或邻近任务退化被忽略。
11. 只报告 training reward、单 seed 或平均 success。
12. 未获得 active training spec 授权就准备产生可引用 checkpoint。

## 审查顺序

1. claim 和 system/action contract；
2. objective/reward 和 estimator；
3. information/gradient/credit path；
4. intervention 与 negative controls；
5. 最后才看 reward、success 和 benchmark。

结果看起来好，不会自动修复机制链中的断点。
