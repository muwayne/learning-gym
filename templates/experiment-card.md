# Experiment Card

> 简单实验可将不适用字段标为 N/A 并说明理由。AI claim 判定见 [AI Review Rubric](../ai-review-rubric.md)。

## 1. Identity

- Experiment ID：
- Owner / Team：
- Reviewer / Skeptic：
- Prediction locked at：
- Code commit / config：
- Checkpoint / data identity：
- Task / environment version：
- Seeds / paired evaluation set：
- Budget：
- Artifact path：

## 2. Question and Decision

- Question：
- 这个实验将决定什么：
- Classical formulation：
- Learned formulation：

## 3. System Contract

### Observation / Representation

- fields、shape、dtype：
- history / time：
- preprocessing / normalization：
- representation module / shape：
- claimed information：
- frozen / trainable：

### Information Flow

- source → consumer：
- mask / fusion / bottleneck：
- causal-use intervention：

### Action / Execution

- modeled random variable：
- conditioning context：
- coordinates / units / joint order：
- normalization / inverse transform：
- bounds / clipping / gripper：
- chunk H / executed prefix K：
- control rate / replanning：

### Data / Distribution

- training：
- rollout：
- held-out evaluation：
- expert / failure / corrective data：
- leakage risks：

## 4. Objective, Reward and Credit

- objective / estimator：
- directly optimizes：
- does not guarantee：
- reward / termination：
- return / advantage / baseline：
- policy score/log-prob or alternative estimator：
- reward-to-action/chunk assignment：

```text
loss or reward
→ estimator
→ module
→ named parameter group
→ expected behavior change
```

- trainable groups / optimizer：
- frozen groups / verification：
- expected gradient and parameter-delta signature：

## 5. Hypothesis and Prediction

- Mechanism hypothesis：
- Primary prediction：
- Guardrail prediction：
- Expected failure-distribution change：
- Result that falsifies the mechanism：
- Main alternative explanation：

## 6. Intervention and Controls

- only changed variable：
- everything held constant：
- baseline：
- matched control：
- negative controls：
  - [ ] no-step / lr=0；
  - [ ] zero-advantage；
  - [ ] permuted reward；
  - [ ] shuffled condition/history；
  - [ ] temporal target shift；
  - [ ] other：

## 7. AI Claim Audit

| Claim | Type | Mechanism / Conditions | Evidence | Prediction | Falsification | Verdict | Evidence Status |
|---|---|---|---|---|---|---|---|
|  | Fact / Hypothesis / Design Choice |  |  |  |  |  |  |

- Fatal issues：
- Missing evidence：
- Allow current budget：Yes / No；

## 8. Pre-run Check

- [ ] Prediction locked；
- [ ] identity/config/budget recorded；
- [ ] shape/time/action assertions pass；
- [ ] optimizer groups match description；
- [ ] intended gradients are finite/nonzero；
- [ ] frozen delta/checksum check defined；
- [ ] baseline/control/intervention differ on one main variable；
- [ ] evaluation protocol frozen before results。

## 9. Evaluation

- primary endpoint / minimum effect：
- paired seeds / episode count / uncertainty：
- secondary metrics：
- failure taxonomy：
- action health：
- behavior drift / reference tasks：

## 10. Result and Diagnosis

- primary/secondary results：
- gradient/parameter evidence：
- failure slices / representative trajectories：
- surprise：
- alternative explanations still alive：
- logs/config/plots/rollouts：

## 11. Verdict

- Supported / Refuted / Inconclusive：
- Revised mental model：
- Go / No-go：
- Why：
- Next single-variable falsification：
