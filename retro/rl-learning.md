# Learning robotics RL with an AI tutor

## The intended capability

The learner wanted to reconstruct the update flows of PPO, TD3, and SAC at least in pseudocode, then use those flows to recognize what newer robotics RL implementations change. The [foundations primer](../modules/t3-sim-to-rl/rl-foundations-primer.md) provides a reference, and the [practice plan](../modules/t3-sim-to-rl/rl-control-practice.md) provides candidate exercises. The first practical outcome was still open. Reading a real update, reproducing it, and diagnosing a bad run call for different checks.

## Where the teaching lost context

One-question turns felt slow. Moving to a batch increased the number of questions per turn, but the tutor switched the quantity being estimated, the data source, the horizon, and the role of the networks between questions. The robot setting stayed the same while the work required from the learner changed. For example, the tutor first used Q scores of 8 and 2 as current estimates in an epsilon-greedy example. It later treated those same numbers as action values of a specified policy without making that premise explicit. The learner asked for the missing rewards and eventually asked what distinction the batch was supposed to teach.

The intended comparison was narrower. Given a transition reaching the box and an exploratory next action, SARSA bootstraps from the chosen action. Q-learning bootstraps from the maximum next-state action value. That comparison can be walked through with one transition, one table, and one change to the backup rule. It belongs in a practical session when it resolves a concrete obstacle in the update the learner wants to understand.

## Cases that give the discussion traction

| Case | What the learner could observe or change | What the record supports |
| --- | --- | --- |
| [A1 attention and decoder](../learners/wayne/records/a1-s01.md) | A shape error exposed an extra transpose. A separate assertion exposed probability assigned to a forbidden position. The learner dictated the repairs. | Concrete implementation and feedback. The same track also recorded missing task context in tutoring prompts. |
| [A1 Tiny GPT](../learners/wayne/records/a1-s03.md) | One arm predicted the current token and the other predicted the next token. Both were scored on the same held-out next-token task. | Lower training loss on a different objective did not establish the desired behavior. |
| [EXPO reading](../learners/wayne/records/t3-s06.md) | The learner redirected paper-detail questions toward the RL foundations shared across methods. | Reading and partial answers. The learner's EXPO implementation was still unfinished. |
| [RLT critic-guidance toy](../learners/wayne/records/t3-s07.md) | The learner predicted and explained a fixed comparison between imitation and imitation plus critic guidance. Actor architecture and initial weights stayed fixed. | The mechanism was tested in a synthetic terminal task. Its outcome is bounded to that toy. |

## A next check to discuss

First pick a real capability and a finish condition. For example, if the goal is to read TD3 code, trace one replay row through the critic update. State where the executed action comes from, which network proposes the next action, where the target is detached, and which parameters update. Keep the worked trace and a follow-up variation structurally alike. The learner can answer orally, annotate code, or implement the update according to the chosen goal.

Record the learner-produced result, clarification turns, and perceived usefulness. A later comparable trace can check retention. These cases suggest that context and feedback matter, but differ in prior familiarity, project relevance, and task size. They do not establish a comparative tutoring effect.
