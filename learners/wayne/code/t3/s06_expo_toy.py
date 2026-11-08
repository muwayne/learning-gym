# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy>=1.26,<3", "torch>=2.4,<3"]
# ///
"""T3 S06 CPU teaching harness: EXPO-lite on a 1D precision-alignment toy.

Frozen multimodal "VLA" sampler + learned Q ensemble + bounded edit + on-the-fly
(OTF) argmax selection. This is not PI05, ManiSkill, RoboDojo, or official
EXPO-FT; deviations are listed in modules/t3-sim-to-rl/s06-primer.md (knowledge point 29).

Default execution only describes the frozen contract. Three learner-owned
functions (otf_select / backup_next_action / td_target) stay TODO until Wayne
dictates their intent; --run refuses to start before that. No checkpoints are saved.

    uv run learners/wayne/code/t3/s06_expo_toy.py                # contract only
    uv run learners/wayne/code/t3/s06_expo_toy.py --run --arm expo --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

# ---------------------------------------------------------------- contract

ENV = {
    "decisions_per_episode": 3,  # each decision = one executed chunk
    "initial_error_range": [-0.6, 0.6],
    "action_range": [-1.0, 1.0],
    "env_noise_std": 0.02,
    "success_tolerance": 0.05,  # |x'| <= tol -> reward 1, terminal
    "drop_threshold": 1.0,  # |x'| > 1 -> reward 0, terminal
    "timeout_is_terminal": True,  # EXPO-FT DROID convention: mask=0, time is in the state
    "gamma": 0.99,
}

BASE = {  # frozen "VLA": a = -x + mode_bias + noise
    "p_good_mode": 0.6,
    "good_mode_bias": 0.0,
    "bad_mode_bias": 0.30,  # systematic error mode, larger than any edit bound
    "mode_noise_std": 0.08,
}

ARMS = {
    # name: overrides on top of DEFAULT_AGENT / BASE
    "expo": {},
    "base-only": {"n_candidates": 1, "edit_scale": 0.0},
    "backup-sample": {"backup": "sample"},
    "n1": {"n_candidates": 1},
    "no-edit": {"edit_scale": 0.0},
    "single-q": {"num_qs": 1, "num_min_qs": 1},
    # Improve pass: executor strength x selection width
    "weak-n8": {"p_good_mode": 0.05},
    "weak-n32": {"p_good_mode": 0.05, "n_candidates": 32},
    "weak-n32-demo": {"p_good_mode": 0.05, "n_candidates": 32, "replay_init_demos": 10},
    "mid-n8": {"p_good_mode": 0.15},
}

DEFAULT_AGENT = {
    "n_candidates": 8,  # EXPO-FT default N=8 base samples (+ N edited)
    "edit_scale": 0.1,  # tanh(edit) * scale; 0 disables edit
    "num_qs": 10,  # EXPO-FT num_qs=10
    "num_min_qs": 2,  # EXPO-FT num_min_qs=2 (random subset min in TD target)
    "select_reducer": "mean",  # how OTF scores candidates at rollout; audit vs official code
    "backup": "otf",  # "otf" | "sample": which next action the TD target evaluates
    "episodes": 150,
    "warmup_episodes": 10,  # EXPO-FT: no update for the first 10 episodes
    "replay_init_demos": 0,  # successful demo episodes put into replay before step 0 (design §3.2)
    "critic_minibatches_per_update": 20,
    "batch_size": 64,
    "lr": 3e-4,
    "polyak": 0.005,
    "hidden": 64,
}

EVAL = {"probe_states": 23, "probe_range": [-0.55, 0.55], "rollouts_per_probe": 40, "seed": 12345}

LEARNER_FUNCTIONS_IMPLEMENTED = {"otf_select": False, "backup_next_action": False, "td_target": False}


# ---------------------------------------------------------------- environment + frozen base


def base_sample(x: np.ndarray, n: int, cfg: dict, rng: np.random.Generator) -> np.ndarray:
    """Frozen executor. x [B] -> candidate actions [B, n]."""
    good = rng.random((x.shape[0], n)) < cfg["p_good_mode"]
    bias = np.where(good, cfg["good_mode_bias"], cfg["bad_mode_bias"])
    a = -x[:, None] + bias + rng.normal(0.0, cfg["mode_noise_std"], (x.shape[0], n))
    return np.clip(a, *ENV["action_range"])


def env_step(x: np.ndarray, a: np.ndarray, rng: np.random.Generator):
    x2 = x + a + rng.normal(0.0, ENV["env_noise_std"], x.shape)
    success = np.abs(x2) <= ENV["success_tolerance"]
    dropped = np.abs(x2) > ENV["drop_threshold"]
    return x2, success.astype(np.float64), success | dropped


# ---------------------------------------------------------------- networks


class QEnsemble(torch.nn.Module):
    """num_qs independent MLP heads Q_k(x, t, a) evaluated in one batched pass."""

    def __init__(self, k: int, hidden: int, din: int = 3):
        super().__init__()

        def w(*shape, fan_in):
            return torch.nn.Parameter(torch.randn(*shape) * fan_in**-0.5)

        self.w1, self.b1 = w(k, din, hidden, fan_in=din), torch.nn.Parameter(torch.zeros(k, 1, hidden))
        self.w2, self.b2 = w(k, hidden, hidden, fan_in=hidden), torch.nn.Parameter(torch.zeros(k, 1, hidden))
        self.w3, self.b3 = w(k, hidden, 1, fan_in=hidden), torch.nn.Parameter(torch.zeros(k, 1, 1))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:  # [B, din] -> [k, B]
        h = torch.relu(torch.einsum("bd,kdh->kbh", inp, self.w1) + self.b1)
        h = torch.relu(torch.bmm(h, self.w2) + self.b2)
        return (torch.bmm(h, self.w3) + self.b3).squeeze(-1)


def features(x, t, a) -> torch.Tensor:
    T = ENV["decisions_per_episode"]
    return torch.tensor(np.stack([x, np.asarray(t, dtype=np.float64) / T, a], -1), dtype=torch.float32)


class Agent:
    def __init__(self, cfg: dict, seed: int):
        torch.manual_seed(seed)
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.q = QEnsemble(cfg["num_qs"], cfg["hidden"])
        self.q_target = QEnsemble(cfg["num_qs"], cfg["hidden"])
        self.q_target.load_state_dict(self.q.state_dict())
        h = cfg["hidden"]
        self.edit = torch.nn.Sequential(torch.nn.Linear(3, h), torch.nn.ReLU(), torch.nn.Linear(h, 1))
        self.opt_q = torch.optim.Adam(self.q.parameters(), cfg["lr"])
        self.opt_edit = torch.optim.Adam(self.edit.parameters(), cfg["lr"])

    # candidates = N base samples + N edited copies (edit bounded by tanh * scale)
    def candidates(self, x, t, rng):
        n = self.cfg["n_candidates"]
        base = base_sample(x, n, self.cfg, rng)
        if self.cfg["edit_scale"] <= 0:
            return base, np.zeros_like(base, dtype=bool)
        xs, ts = np.repeat(x, n), np.repeat(t, n)
        with torch.no_grad():
            delta = torch.tanh(self.edit(features(xs, ts, base.reshape(-1)))).squeeze(-1).numpy()
        edited = np.clip(base.reshape(-1) + delta * self.cfg["edit_scale"], *ENV["action_range"])
        cand = np.concatenate([base, edited.reshape(base.shape)], axis=1)
        is_edit = np.concatenate([np.zeros_like(base, bool), np.ones_like(base, bool)], axis=1)
        return cand, is_edit

    def q_values(self, net, x, t, cand) -> torch.Tensor:  # -> [num_qs, B, M]
        b, m = cand.shape
        with torch.no_grad():
            return net(features(np.repeat(x, m), np.repeat(t, m), cand.reshape(-1))).view(-1, b, m)

    def random_min(self, qs: torch.Tensor, rng) -> torch.Tensor:
        k = qs.shape[0]
        idx = rng.choice(k, size=min(self.cfg["num_min_qs"], k), replace=False)
        return qs[idx].min(0).values

    def act(self, x, t, rng, net=None):
        """Rollout policy: OTF selection over candidates scored by the ensemble."""
        cand, is_edit = self.candidates(x, t, rng)
        qs = self.q_values(net or self.q, x, t, cand)
        scores = qs.mean(0) if self.cfg["select_reducer"] == "mean" else self.random_min(qs, rng)
        action, idx = otf_select(cand, scores.numpy())
        rows = np.arange(len(x))
        return action, scores.numpy()[rows, idx], is_edit[rows, idx]

    def update(self, replay: np.ndarray):
        cfg, rng = self.cfg, self.rng
        for _ in range(cfg["critic_minibatches_per_update"]):
            b = replay[rng.integers(0, len(replay), cfg["batch_size"])]
            x, t, a, r, x2, done = b.T
            a2 = backup_next_action(self, x2, t + 1, rng)
            q_next = self.random_min(self.q_target(features(x2, t + 1, a2)).detach(), rng)
            y = td_target(
                torch.tensor(r, dtype=torch.float32),
                torch.tensor(done, dtype=torch.float32),
                q_next,
                ENV["gamma"],
            )
            loss = ((self.q(features(x, t, a)) - y.detach()) ** 2).mean()
            self.opt_q.zero_grad()
            loss.backward()
            self.opt_q.step()
            with torch.no_grad():
                for p_t, p in zip(self.q_target.parameters(), self.q.parameters()):
                    p_t.mul_(1 - cfg["polyak"]).add_(cfg["polyak"] * p)
        if cfg["edit_scale"] > 0:  # one edit step: maximise ensemble-mean Q(s, a_base + edit)
            b = replay[rng.integers(0, len(replay), cfg["batch_size"])]
            x, t = b[:, 0], b[:, 1]
            a_base = torch.tensor(base_sample(x, 1, cfg, rng)[:, 0], dtype=torch.float32)
            inp = features(x, t, a_base.numpy())
            edited = a_base + torch.tanh(self.edit(inp)).squeeze(-1) * cfg["edit_scale"]
            inp = torch.stack([inp[:, 0], inp[:, 1], edited], -1)
            loss_edit = -self.q(inp).mean()
            self.opt_edit.zero_grad()
            loss_edit.backward()
            self.opt_edit.step()


# ---------------------------------------------------------------- learner-owned (Wayne dictates)


def otf_select(cand: np.ndarray, scores: np.ndarray):
    """cand, scores: [B, M]. Return (chosen action [B], chosen index [B]).

    Intent to fill from Wayne's dictation: which candidate does the OTF policy execute?
    """
    raise NotImplementedError("otf_select: waiting for learner intent (lesson §3 task I-1)")


def backup_next_action(agent: Agent, x2: np.ndarray, t2: np.ndarray, rng) -> np.ndarray:
    """Return the next action a' [B] whose target-Q enters the TD target.

    agent.cfg["backup"] == "otf"    -> ?
    agent.cfg["backup"] == "sample" -> ?
    Tools: agent.act(x2, t2, rng, net=agent.q_target), base_sample(x2, 1, agent.cfg, rng).
    """
    raise NotImplementedError("backup_next_action: waiting for learner intent (lesson §3 task I-2)")


def td_target(r: torch.Tensor, done: torch.Tensor, q_next: torch.Tensor, gamma: float) -> torch.Tensor:
    """One-decision TD target. done=1 covers success, drop and timeout (mask=0)."""
    raise NotImplementedError("td_target: waiting for learner intent (lesson §3 task I-3)")


# ---------------------------------------------------------------- run + instrumentation


def run_episode(agent: Agent, use_critic: bool, rng, replay: list) -> float:
    x = rng.uniform(*ENV["initial_error_range"], 1)
    T = ENV["decisions_per_episode"]
    for t in range(T):
        tt = np.array([float(t)])
        a = agent.act(x, tt, rng)[0] if use_critic else base_sample(x, 1, agent.cfg, rng)[:, 0]
        x2, r, terminal = env_step(x, a, rng)
        done = bool(terminal[0] or (t == T - 1 and ENV["timeout_is_terminal"]))
        replay.append((x[0], float(t), a[0], r[0], x2[0], float(done)))
        if done:
            return float(r[0])
        x = x2
    return 0.0


def evaluate(agent: Agent) -> dict:
    """Frozen policy on fixed probes: true SR, Q at the chosen first action vs MC return."""
    rng = np.random.default_rng(EVAL["seed"])
    probes = np.linspace(*EVAL["probe_range"], EVAL["probe_states"])
    m, T, g = EVAL["rollouts_per_probe"], ENV["decisions_per_episode"], ENV["gamma"]
    rows = []
    for x0 in probes:
        x = np.full(m, x0)
        a, q_claim, picked_edit = agent.act(x, np.zeros(m), rng)
        ret, alive = np.zeros(m), np.ones(m, bool)
        for t in range(T):
            if t > 0:
                a = agent.act(x, np.full(m, float(t)), rng)[0]
            x, r, terminal = env_step(x, a, rng)
            ret += alive * r * g**t
            alive &= ~terminal
        rows.append(
            {
                "x0": round(float(x0), 3),
                "success_rate": float((ret > 0).mean()),
                "mc_return": float(ret.mean()),
                "q_claim": float(q_claim.mean()),
                "picked_edit_ratio": float(picked_edit.mean()),
            }
        )
    mean = lambda k: float(np.mean([r[k] for r in rows]))  # noqa: E731
    return {
        "eval_success_rate": mean("success_rate"),
        "q_claim_minus_mc_return": mean("q_claim") - mean("mc_return"),
        "picked_edit_ratio": mean("picked_edit_ratio"),
        "probes": rows,
    }


def base_success_rate(cfg: dict, episodes: int = 4000) -> float:
    rng = np.random.default_rng(EVAL["seed"] + 1)
    x = rng.uniform(*ENV["initial_error_range"], episodes)
    alive, ok = np.ones(episodes, bool), np.zeros(episodes, bool)
    for _ in range(ENV["decisions_per_episode"]):
        x, r, terminal = env_step(x, base_sample(x, 1, cfg, rng)[:, 0], rng)
        ok |= alive & (r > 0)
        alive &= ~terminal
    return float(ok.mean())


def selection_space(cfg: dict, continuations: int = 20) -> float:
    """Pre-training gate (design G2 analogue): E[max_j p_j] - E[mean_j p_j] over N base
    candidates, p_j = true success of 'candidate j, then the frozen base continues'. No critic."""
    rng = np.random.default_rng(EVAL["seed"] + 2)
    probes = np.linspace(*EVAL["probe_range"], EVAL["probe_states"])
    n, T, gaps = cfg["n_candidates"], ENV["decisions_per_episode"], []
    for x0 in probes:
        cand = base_sample(np.array([x0]), n, cfg, rng)[0]
        p = []
        for a0 in cand:
            x = np.full(continuations, x0)
            a = np.full(continuations, a0)
            alive, ok = np.ones(continuations, bool), np.zeros(continuations, bool)
            for t in range(T):
                if t > 0:
                    a = base_sample(x, 1, cfg, rng)[:, 0]
                x, r, terminal = env_step(x, a, rng)
                ok |= alive & (r > 0)
                alive &= ~terminal
            p.append(ok.mean())
        gaps.append(max(p) - float(np.mean(p)))
    return float(np.mean(gaps))


def demo_transitions(cfg: dict, rng) -> list:
    """Successful scripted demos (a = -x, i.e. the good mode without noise); stored like rollouts."""
    rows = []
    for _ in range(cfg["replay_init_demos"]):
        x = rng.uniform(*ENV["initial_error_range"], 1)
        x2, r, terminal = env_step(x, -x, rng)
        rows.append((x[0], 0.0, -x[0], r[0], x2[0], float(terminal[0])))
    return rows


def run_arm(arm: str, seed: int, overrides: dict) -> dict:
    cfg = {**DEFAULT_AGENT, **BASE, **ARMS[arm], **overrides}
    torch.set_num_threads(1)
    agent, rng, successes = Agent(cfg, seed), np.random.default_rng(seed + 1000), []
    replay = demo_transitions(cfg, np.random.default_rng(seed + 2000))
    start = time.time()
    for ep in range(cfg["episodes"]):
        learning = ep >= cfg["warmup_episodes"]
        successes.append(run_episode(agent, learning, rng, replay))
        if learning:
            agent.update(np.asarray(replay))
    result = evaluate(agent)
    return {
        "arm": arm,
        "seed": seed,
        "config": cfg,
        "seconds": round(time.time() - start, 1),
        "base_success_rate": base_success_rate(cfg),
        "train_success_by_episode": successes,
        "train_success_last50": float(np.mean(successes[-50:])),
        **result,
    }


def describe() -> None:
    print(__doc__)
    print(json.dumps({"env": ENV, "base": BASE, "agent": DEFAULT_AGENT, "arms": ARMS, "eval": EVAL,
                      "learner_functions_implemented": LEARNER_FUNCTIONS_IMPLEMENTED}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--arm", choices=sorted(ARMS), default="expo")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--gate", action="store_true", help="only measure base SR + selection space (no training)")
    parser.add_argument("--out", help="write full JSON (curves + probes) here")
    args = parser.parse_args()

    if not (args.run or args.gate):
        describe()
        return
    cfg = {**DEFAULT_AGENT, **BASE, **ARMS[args.arm]}
    if args.gate:
        print(json.dumps({"arm": args.arm, "base_success_rate": base_success_rate(cfg),
                          "selection_space": selection_space(cfg)}))
        return
    missing = [k for k, v in LEARNER_FUNCTIONS_IMPLEMENTED.items() if not v]
    if missing:
        raise SystemExit(f"Refusing to run: learner-owned functions still TODO: {missing}")
    results = [run_arm(args.arm, s, {}) for s in args.seeds]
    for r in results:
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
                          if k in ("arm", "seed", "seconds", "base_success_rate", "train_success_last50",
                                   "eval_success_rate", "q_claim_minus_mc_return", "picked_edit_ratio")}))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f)


if __name__ == "__main__":
    main()
