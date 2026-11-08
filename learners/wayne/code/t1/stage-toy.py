"""CPU reaching lesson harness. Prepare a shared base; compare only after prediction.

Scientific contract: plans/smolvla/t1-s01-stage-learning.md, CPU toy section.
No pretrained VLA, external service, GPU, or robot simulator is used here.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


CONFIG = {
    "version": "cpu-reaching-v1", "seed": 20260910, "threads": 1,
    "layers": [4, 32, 32, 2], "step_limit": 0.12, "tolerance": 0.04,
    "max_steps": 30, "base_updates": 80, "stage_updates": 80,
    "batch_size": 32, "lr": 0.003, "base_demos": 32,
    "source_demos": 16, "target_demos": 8, "eval_episodes": 32,
    "wall_cap_seconds": 30,
    "angles_degrees": {"base": [-20, 20, 160, 200],
                       "near": [35, 65], "far": [-110, -70], "target": [70, 110]},
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def save_json(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)


def policy():
    return nn.Sequential(nn.Linear(4, 32), nn.Tanh(), nn.Linear(32, 32),
                         nn.Tanh(), nn.Linear(32, 2), nn.Tanh())


def limit(action):
    norm = np.linalg.norm(action, axis=-1, keepdims=True)
    return action * np.minimum(1.0, CONFIG["step_limit"] / np.maximum(norm, 1e-12))


def starts(domain, count, seed):
    rng = np.random.default_rng(seed)
    p = rng.uniform(-0.6, 0.6, (count, 2)).astype(np.float32)
    bounds = CONFIG["angles_degrees"][domain]
    angle = rng.uniform(bounds[0], bounds[1], count)
    if domain == "base":
        angle += rng.integers(0, 2, count) * 180
    angle = np.deg2rad(angle)
    radius = rng.uniform(0.4, 0.8, count)
    g = p + np.stack([np.cos(angle), np.sin(angle)], axis=-1) * radius[:, None]
    return p, g.astype(np.float32)


def demonstrations(domain, count, seed):
    p, g = starts(domain, count, seed)
    x, y = [], []
    for pos, goal in zip(p, g):
        for _ in range(CONFIG["max_steps"]):
            action = limit(goal - pos)
            x.append(np.concatenate([pos, goal]))
            y.append(action / CONFIG["step_limit"])
            if np.linalg.norm(goal - pos) < 1e-6:
                break
            pos = pos + action
    return torch.tensor(np.array(x), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32)


def datasets():
    return {name: demonstrations(name, count, CONFIG["seed"] + index)
            for index, (name, count) in enumerate([
                ("base", CONFIG["base_demos"]), ("near", CONFIG["source_demos"]),
                ("far", CONFIG["source_demos"]), ("target", CONFIG["target_demos"])])}


def train(model, data, updates, seed, deadline):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    rng = torch.Generator().manual_seed(seed)
    x, y = data
    curve, timings = [], []
    for step in range(updates):
        if time.perf_counter() > deadline:
            raise TimeoutError("CPU toy wall cap exceeded")
        tick = time.perf_counter()
        idx = torch.randint(len(x), (CONFIG["batch_size"],), generator=rng)
        optimizer.zero_grad(set_to_none=True)
        loss = (model(x[idx]) - y[idx]).square().mean()
        assert torch.isfinite(loss), "nonfinite loss"
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
        assert all(torch.isfinite(p).all() for p in model.parameters())
        timings.append(time.perf_counter() - tick)
        curve.append({"update": step + 1, "mse_normalized_action": loss.item()})
    return {"curve": curve, "seconds": sum(timings),
            "median_step_ms": float(np.median(timings) * 1000)}


@torch.no_grad()
def evaluate(model):
    tick = time.perf_counter()
    model.eval()
    result = {}
    for index, domain in enumerate(("base", "near", "far", "target")):
        seed = CONFIG["seed"] + 100 + index
        p, g = starts(domain, CONFIG["eval_episodes"], seed)
        x, y = demonstrations(domain, CONFIG["eval_episodes"], seed)
        mse = ((model(x) - y) * CONFIG["step_limit"]).square().mean().item()
        positions = p.copy()
        paths = [positions.copy()]
        done = np.linalg.norm(g - positions, axis=1) <= CONFIG["tolerance"]
        for _ in range(CONFIG["max_steps"]):
            obs = torch.tensor(np.concatenate([positions, g], axis=1), dtype=torch.float32)
            action = model(obs).numpy() * CONFIG["step_limit"]
            assert np.isfinite(action).all()
            positions = positions + limit(action) * (~done[:, None])
            done |= np.linalg.norm(g - positions, axis=1) <= CONFIG["tolerance"]
            paths.append(positions.copy())
        distance = np.linalg.norm(g - positions, axis=1)
        traces = np.stack(paths, axis=1)
        result[domain] = {
            "success_count": int(done.sum()), "denominator": len(done),
            "action_mse_unit2": mse, "mean_final_distance_units": float(distance.mean()),
            "episodes": [{"id": i, "start": p[i].tolist(), "goal": g[i].tolist(),
                          "success": bool(done[i]), "final_distance": float(distance[i]),
                          "path": traces[i].tolist()} for i in range(len(done))],
        }
    return {"domains": result, "seconds": time.perf_counter() - tick}


def self_check(data):
    # Separate episode seeds and exact state/goal rows; same limits in expert and rollout.
    for i, domain in enumerate(("base", "near", "far", "target")):
        train_rows = {tuple(row) for row in data[domain][0].numpy()}
        val, _ = demonstrations(domain, CONFIG["eval_episodes"], CONFIG["seed"] + 100 + i)
        assert not train_rows.intersection(tuple(row) for row in val.numpy())
        p, g = starts(domain, 32, 1234 + i)
        for _ in range(CONFIG["max_steps"]):
            a = limit(g - p)
            assert np.max(np.linalg.norm(a, axis=1)) <= CONFIG["step_limit"] + 1e-6
            p = p + a
        assert np.all(np.linalg.norm(g - p, axis=1) <= CONFIG["tolerance"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["describe", "prepare", "compare"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--source", choices=["near", "far"])
    parser.add_argument("--prediction")
    args = parser.parse_args()
    if args.mode == "describe":
        print(json.dumps(CONFIG, indent=2))
        return
    if args.output is None:
        parser.error("--output is required")
    if args.mode == "compare" and not (args.base and args.source and args.prediction):
        parser.error("compare requires --base, --source and the learner's --prediction")
    torch.set_num_threads(1)
    torch.manual_seed(CONFIG["seed"])
    torch.use_deterministic_algorithms(True)
    tick = time.perf_counter()
    deadline = tick + CONFIG["wall_cap_seconds"]
    args.output.mkdir(parents=True, exist_ok=False)
    data = datasets()
    self_check(data)
    run = {"run_id": args.output.name, "config": CONFIG,
           "config_sha256": digest(CONFIG), "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "role": "reference" if args.mode == "prepare" else "learner",
           "torch": torch.__version__, "device": "cpu", "self_check": "PASS",
           "data_sha256": {k: hashlib.sha256(x.numpy().tobytes() + y.numpy().tobytes()).hexdigest()
                           for k, (x, y) in data.items()}}
    model = policy()
    run["parameters"] = sum(p.numel() for p in model.parameters())
    if args.mode == "prepare":
        run["training"] = train(model, data["base"], CONFIG["base_updates"], 40, deadline)
        torch.save(model.state_dict(), args.output / "base.pt")
        reloaded = policy()
        reloaded.load_state_dict(torch.load(args.output / "base.pt", weights_only=True))
        with torch.no_grad():
            assert torch.equal(model(data["target"][0]), reloaded(data["target"][0]))
        run["reload"] = "PASS"
        run["evaluation"] = evaluate(reloaded)
        run["optimizer_updates"] = CONFIG["base_updates"]
    else:
        identity = json.loads((args.base / "result.json").read_text())
        assert identity["config_sha256"] == digest(CONFIG)
        assert hashlib.sha256((args.base / "base.pt").read_bytes()).hexdigest() == identity["checkpoints"]["base.pt"]
        model.load_state_dict(torch.load(args.base / "base.pt", weights_only=True))
        run.update(source=args.source, prediction=args.prediction, base_run=str(args.base))
        save_json(args.output / "prediction.json", {"source": args.source, "prediction": args.prediction})
        run["arms"] = {}
        direct = copy.deepcopy(model)
        staged = copy.deepcopy(model)
        for name, candidate, domain, sample_seed in [
            ("A", direct, "target", 51), ("C", direct, "target", 52),
            ("S", staged, args.source, 53), ("B", staged, "target", 51)]:
            training = train(candidate, data[domain], CONFIG["stage_updates"], sample_seed, deadline)
            torch.save(candidate.state_dict(), args.output / (name + ".pt"))
            loaded = policy()
            loaded.load_state_dict(torch.load(args.output / (name + ".pt"), weights_only=True))
            with torch.no_grad():
                assert torch.equal(candidate(data["target"][0]), loaded(data["target"][0]))
            run["arms"][name] = {"training_domain": domain, "training": training, "evaluation": evaluate(loaded)}
        run["optimizer_updates"] = 4 * CONFIG["stage_updates"]
    run["checkpoints"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.glob("*.pt")}
    run["elapsed_seconds"] = time.perf_counter() - tick
    assert run["elapsed_seconds"] <= CONFIG["wall_cap_seconds"]
    save_json(args.output / "result.json", run)
    print(json.dumps({k: run[k] for k in ("run_id", "role", "parameters", "optimizer_updates", "elapsed_seconds", "self_check")}, indent=2))


if __name__ == "__main__":
    main()
