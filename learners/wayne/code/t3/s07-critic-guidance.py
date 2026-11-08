"""T3 S07 CPU teaching harness: inspect first, predict, then implement the loss.

This is a one-step synthetic environment, not PI05, full RLT, or TD3.
Default execution only describes the frozen experiment. No checkpoints are saved.
Wayne owns the actor-objective intent and the pre-run prediction, recorded in
records/t3-s07.md. The loss below translates that intent into PyTorch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time


CONFIG = {
    "seed": 20260909,
    "device": "cpu",
    "threads": 1,
    "train_states": 17,
    "evaluation_states": 16,
    "replay_actions_per_state": 41,
    "reference_bias_mm": 0.25,
    "success_tolerance_mm": 0.1,
    "critic_updates": 1000,
    "actor_updates_per_arm": 600,
    "learning_rate": 0.002,
    "bc_weight": 1.0,
    "critic_guidance_weight": 4.0,
    "wall_seconds_cap": 60,
    "actor_objective_implemented": True,
}


def training_states():
    return [round(-0.4 + 0.05 * i, 6) for i in range(17)]


def evaluation_states():
    return [round(-0.375 + 0.05 * i, 6) for i in range(16)]


def reference_action(x):
    return -x + CONFIG["reference_bias_mm"]


def environment_outcome(x, action):
    """Physical mm; one action ends the episode. Reward is not success rate."""
    final_error = x + action
    reward = 1.0 - final_error**2
    success = abs(final_error) <= CONFIG["success_tolerance_mm"]
    return final_error, reward, success


def replay_rows():
    return [
        (x, round(-1.0 + 0.05 * i, 6),
         environment_outcome(x, round(-1.0 + 0.05 * i, 6))[1])
        for x in training_states()
        for i in range(41)
    ]


def describe():
    rows = replay_rows()
    return {
        "config": CONFIG,
        "task": "one horizontal move: final_error_mm = initial_error_mm + action_mm",
        "reward": "1 - final_error_mm squared; terminal after one move",
        "reference": "synthetic biased prior, NOT a successful Oracle demonstration",
        "replay_source": "697 synthetic executed transitions: 17 states x 41 actions",
        "replay_sha256": hashlib.sha256(json.dumps(rows).encode()).hexdigest(),
        "training_states_mm": training_states(),
        "evaluation_states_mm": evaluation_states(),
        "actor": "[current error, reference] -> Linear(2,16) -> Tanh -> Linear(16,1) -> Tanh; action in [-1,1] mm",
        "critic": "[current error, action] -> Linear(2,32) -> Tanh -> Linear(32,32) -> Tanh -> Linear(32,1)",
        "schedule": "train critic once on replay, then freeze it; clone identical actor initialization for both arms",
        "arms": {"imitation": "only match reference", "critic_guided": "same imitation plus critic guidance"},
        "primary_metric": "paired mean absolute final error in mm on 16 evaluation states",
        "secondary_metrics": ["success rate", "actual mean reward", "critic score", "reference deviation"],
        "selection": "fixed final update, no checkpoint/seed selection or hyperparameter search",
        "results": "design only; use --run for the recorded experiment",
    }


def actor_training_loss(actor, critic, states, references, use_critic):
    """Translate the learner's imitation versus critic-guidance comparison.

    The frozen critic must remain differentiable with respect to actor actions.
    Environment reward is an external verifier, never actor's training loss.
    """
    import torch
    actions = actor(torch.cat([states, references], dim=1))
    loss = CONFIG["bc_weight"] * (actions - references).square().mean()
    if use_critic:
        scores = critic(torch.cat([states, actions], dim=1))
        loss = loss - CONFIG["critic_guidance_weight"] * scores.mean()
    return loss


def run():
    if not CONFIG["actor_objective_implemented"]:
        raise SystemExit("Inspect and record prediction first; actor objective is not implemented.")
    import copy
    import torch
    from torch import nn

    torch.set_num_threads(1)
    torch.manual_seed(CONFIG["seed"])
    started = time.monotonic()

    def budget():
        if time.monotonic() - started > CONFIG["wall_seconds_cap"]:
            raise RuntimeError("CPU budget exhausted; incomplete result, no retuning.")

    critic = nn.Sequential(nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 32),
                           nn.Tanh(), nn.Linear(32, 1))
    rows = torch.tensor(replay_rows(), dtype=torch.float32)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=CONFIG["learning_rate"])
    for _ in range(CONFIG["critic_updates"]):
        budget()
        loss = (critic(rows[:, :2]).squeeze(-1) - rows[:, 2]).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite critic loss.")
        critic_optimizer.zero_grad()
        loss.backward()
        critic_optimizer.step()
    for parameter in critic.parameters():
        parameter.requires_grad_(False)
    frozen_critic = copy.deepcopy(critic.state_dict())

    actor_initial = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 1), nn.Tanh())
    train_x = torch.tensor(training_states(), dtype=torch.float32).reshape(-1, 1)
    train_ref = -train_x + CONFIG["reference_bias_mm"]
    eval_x = torch.tensor(evaluation_states(), dtype=torch.float32).reshape(-1, 1)
    eval_ref = -eval_x + CONFIG["reference_bias_mm"]

    def evaluate(actor):
        with torch.no_grad():
            actions = actor(torch.cat([eval_x, eval_ref], dim=1))
            q_values = critic(torch.cat([eval_x, actions], dim=1)).squeeze(-1).tolist()
            commands = actions.squeeze(-1).tolist()
        # Independent environment function consumes detached commands, not Q predictions.
        outcomes = [environment_outcome(x, a) for x, a in zip(evaluation_states(), commands)]
        n = len(outcomes)
        return {
            "mean_abs_final_error_mm": sum(abs(o[0]) for o in outcomes) / n,
            "success_rate": sum(o[2] for o in outcomes) / n,
            "mean_actual_reward": sum(o[1] for o in outcomes) / n,
            "mean_critic_score": sum(q_values) / n,
            "reference_mae_mm": sum(abs(a - reference_action(x)) for x, a in zip(evaluation_states(), commands)) / n,
            "per_start": [dict(x_mm=x, action_mm=a, final_error_mm=o[0],
                               reward=o[1], success=o[2], critic_score=q)
                          for x, a, o, q in zip(evaluation_states(), commands, outcomes, q_values)],
        }

    output = {"design": describe(), "initial_actor": evaluate(actor_initial), "arms": {}}
    output["design"]["results"] = "running"
    for name, use_critic in [("imitation", False), ("critic_guided", True)]:
        actor = copy.deepcopy(actor_initial)
        optimizer = torch.optim.Adam(actor.parameters(), lr=CONFIG["learning_rate"])
        trace = []
        for step in range(CONFIG["actor_updates_per_arm"]):
            budget()
            actor_loss = actor_training_loss(actor, critic, train_x, train_ref, use_critic)
            if not torch.isfinite(actor_loss):
                raise RuntimeError("Nonfinite actor loss.")
            optimizer.zero_grad()
            actor_loss.backward()
            optimizer.step()
            if (step + 1) % 100 == 0:
                metrics = evaluate(actor)
                metrics.pop("per_start")
                trace.append({"update": step + 1, **metrics})
        output["arms"][name] = {"final": evaluate(actor), "trace": trace}
    assert all(torch.equal(v, frozen_critic[k]) for k, v in critic.state_dict().items())
    with torch.no_grad():
        heldout = torch.tensor([(x, -0.95 + 0.1 * j) for x in evaluation_states()
                                for j in range(20)], dtype=torch.float32)
        truth = torch.tensor([environment_outcome(float(x), float(a))[1] for x, a in heldout])
        output["critic_heldout_mse"] = float((critic(heldout).squeeze(-1) - truth).square().mean())
    output["elapsed_seconds"] = time.monotonic() - started
    output["design"]["results"] = "complete"
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="run only after learner-owned objective is filled")
    args = parser.parse_args()
    if args.run:
        run()
    else:
        print(json.dumps(describe(), ensure_ascii=False, indent=2))
