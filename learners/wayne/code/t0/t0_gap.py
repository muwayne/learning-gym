# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["torch", "numpy", "matplotlib"]
# ///
"""T0·S03 closed-loop gap runner（AI scaffold；Arm config 由学员口述）。

用法：
  uv run t0_gap.py --name baseline                 # expert data → BC → H16/E1 rollouts
  uv run t0_gap.py --name G --E 1,4,16              # same policy, three execution horizons
  uv run t0_gap.py --name H --kick-prob 0.05 --kick-y 1.0 --kick-theta 0.5 \
                   --E 1,4,16 --eval-holdout gap-baseline/holdout.npz   # 恢复数据重训，同预算

Environment：x 0→100 的弯曲 corridor，中心线 c(x)，半宽 WALL=2.2。
Expert：看前方 LOOKAHEAD 处的 centerline point 并修正 heading。
Policy：MLP 输入 (x/100, y, θ)，输出 H=16 action chunk；BC objective 为 MSE。
Execution horizon E：每次 observation 后执行 action chunk 的前 E 步再 replan。

每次运行输出到 gap-<name>/：
  loss.png        train/holdout loss 曲线（原始线压淡 + EMA 粗线）
  rollouts_E*.png corridor + expert-state samples + rollout trajectories
  dist_E*.png     XY nearest-training-state distance（✕=first XY-support exit）
  holdout.npz     holdout set（供其他 Arm 用 --eval-holdout 对拍）
  model.pt / summary.json
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Hiragino Sans GB", "PingFang SC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
from matplotlib.ticker import FuncFormatter, NullFormatter

# ---------------- environment and expert（AI scaffold，学员不改） ----------------
XMAX, DX, MAXST, WALL, LOOKAHEAD, AMAX, K_EXPERT, H = 100.0, 0.5, 260, 2.2, 3.0, 0.3, 0.6, 16


def center(x):
    return 1.15 * np.sin(x / 12) + 0.5 * np.sin(x / 31 + 1.3)


def step(state, a, rng, wind):
    """One environment step: clipped steering action plus stochastic wind."""
    x, y, th = state
    th = th + float(np.clip(a, -AMAX, AMAX)) + rng.normal(0, wind)
    return np.array([x + DX * math.cos(th), y + DX * math.sin(th), th])


def off_center(state):
    return abs(state[1] - center(state[0]))


def expert_action(state):
    x, y, th = state
    th_star = math.atan2(center(x + LOOKAHEAD) - y, LOOKAHEAD)
    return float(np.clip(K_EXPERT * (th_star - th), -AMAX, AMAX))


def obs(state):
    return np.array([state[0] / XMAX, state[1], state[2]], dtype=np.float32)


def collect_expert(n_traj, rng, wind, kick_prob=0.0, kick_y=0.0, kick_theta=0.0):
    """Collect expert trajectories; optional kicks add recovery states and actions."""
    trajs = []
    for _ in range(n_traj):
        s = np.array([0.0, rng.uniform(-0.3, 0.3), rng.uniform(-0.15, 0.15)])
        S, A = [], []
        for _t in range(MAXST):
            if kick_prob > 0 and rng.random() < kick_prob:
                s = s + np.array([0.0, rng.uniform(-kick_y, kick_y), rng.uniform(-kick_theta, kick_theta)])
            a = expert_action(s)
            S.append(s.copy()); A.append(a)
            s = step(s, a, rng, wind)
            if s[0] >= XMAX or off_center(s) >= WALL:
                break
        trajs.append((np.array(S), np.array(A)))
    return trajs


def to_chunks(trajs):
    """Pair each state with the next H expert actions; pad the tail with its last action."""
    O, Y, XY = [], [], []
    for S, A in trajs:
        T = len(A)
        for t in range(T):
            idx = np.minimum(np.arange(t, t + H), T - 1)
            O.append(obs(S[t])); Y.append(A[idx]); XY.append(S[t, :2])
    return np.array(O, dtype=np.float32), np.array(Y, dtype=np.float32), np.array(XY)


# ---------------- XY-support proxy and first exit ----------------
def dist_to_band(xy, band_xy):
    d = np.sqrt(((xy[:, None, :] - band_xy[None, :, :]) ** 2).sum(-1))
    return d.min(1)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


# ---------------- closed-loop rollout ----------------
@torch.no_grad()
def rollout(model, E, rollout_seed, band_xy, tau, wind):
    rng = np.random.default_rng(rollout_seed)
    s = np.array([0.0, rng.uniform(-0.3, 0.3), rng.uniform(-0.15, 0.15)])
    initial_state = s.copy()
    S, chunk = [s.copy()], None
    raw_actions, executed_actions = [], []
    ok = False
    for t in range(MAXST):
        if t % E == 0:
            chunk = model(torch.tensor(obs(s))[None]).squeeze(0).numpy()
        raw_a = float(chunk[t % E])
        executed_a = float(np.clip(raw_a, -AMAX, AMAX))
        raw_actions.append(raw_a); executed_actions.append(executed_a)
        s = step(s, raw_a, rng, wind)
        S.append(s.copy())
        if off_center(s) >= WALL:
            break
        if s[0] >= XMAX:
            ok = True
            break
    S = np.array(S)
    d = dist_to_band(S[:, :2], band_xy)
    out = np.where(d > tau)[0]
    first_exit = int(out[0]) if len(out) else None
    return_step = None
    if first_exit is not None:
        returned = np.where(d[first_exit + 1:] <= tau)[0]
        if len(returned):
            return_step = int(first_exit + 1 + returned[0])
    return dict(
        S=S,
        ok=ok,
        rollout_seed=int(rollout_seed),
        initial_state=initial_state,
        d=d,
        first_exit=first_exit,
        return_step=return_step,
        raw_actions=np.asarray(raw_actions),
        executed_actions=np.asarray(executed_actions),
    )


def summarize_action_health(rs):
    raw = np.concatenate([r["raw_actions"] for r in rs])
    executed = np.concatenate([r["executed_actions"] for r in rs])
    return {
        "finite": int(np.isfinite(raw).sum()),
        "n": int(raw.size),
        "raw_range": [float(raw.min()), float(raw.max())],
        "raw_mean_abs": float(np.abs(raw).mean()),
        "executed_mean_abs": float(np.abs(executed).mean()),
        "clip_fraction": float((np.abs(raw) > AMAX).mean()),
    }


def ema(v, a=0.05):
    """log 空间、偏差修正的 EMA（曲线画在 log 轴上，线性空间的平均会被早期大值拖偏）。"""
    out, m = [], 0.0
    for i, x in enumerate(v, 1):
        m = a * math.log(x) + (1 - a) * m; out.append(math.exp(m / (1 - (1 - a) ** i)))
    return out


def plot_loss(hist, path):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    st = [h[0] for h in hist]
    for key, c in (("train", "#2a78d6"), ("holdout", "#d64545")):
        v = [h[1][key] for h in hist]
        ax.plot(st, v, color=c, alpha=0.25); ax.plot(st, ema(v), color=c, lw=2, label=key)
    ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("MSE (action chunk)"); ax.legend(); ax.grid(alpha=.3)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0e}")); ax.yaxis.set_minor_formatter(NullFormatter())
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_rollouts(rs, band_xy, E, path, title):
    fig, ax = plt.subplots(figsize=(11, 3.6))
    xs = np.linspace(0, XMAX, 400)
    ax.fill_between(xs, center(xs) - WALL, center(xs) + WALL, color="#999", alpha=.15, label="corridor")
    ax.scatter(band_xy[:, 0], band_xy[:, 1], s=1, color="#2a78d6", alpha=.15, label="expert-state samples")
    for r in rs:
        S = r["S"]; c = "#149a6b" if r["ok"] else "#d64545"
        ax.plot(S[:, 0], S[:, 1], color=c, lw=1, alpha=.8)
        if r["first_exit"] is not None:
            ax.plot(S[r["first_exit"], 0], S[r["first_exit"], 1], "x", color="k", ms=6)
    ax.set_xlim(0, XMAX); ax.set_ylim(-4.5, 4.5); ax.set_title(title); ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_dist(rs, tau, E, path):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    # Prioritize failures, then successful trajectories with support exits.
    selected = sorted(rs, key=lambda r: (r["ok"], r["first_exit"] is None))[:5]
    for i, r in enumerate(selected):
        status = "success" if r["ok"] else "failure"
        color = "#149a6b" if r["ok"] else "#d64545"
        ax.plot(
            r["d"],
            lw=1.2,
            ls="--" if r["ok"] else "-",
            color=color,
            alpha=.75,
            label=f"{status} #{i+1} (first exit {r['first_exit']})",
        )
    ax.axhline(tau, color="k", ls=":", label=f"support-exit threshold τ={tau:.2f}")
    ax.set_xlabel("step"); ax.set_ylabel("XY nearest-training-state distance")
    ax.set_title(f"E={E} · XY-support proxy"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--n-traj", type=int, default=60)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--E", default="1", help="execution horizons，如 1,4,16")
    ap.add_argument("--rollouts", type=int, default=20)
    ap.add_argument("--kick-prob", type=float, default=0.0)
    ap.add_argument("--kick-y", type=float, default=0.0)
    ap.add_argument("--kick-theta", type=float, default=0.0)
    ap.add_argument("--eval-holdout", default=None, help="另一 Arm 的 holdout.npz，用于固定口径对拍")
    ap.add_argument("--wind", type=float, default=0.05, help="每步朝向风扰 std（采数与路考同在）")
    ap.add_argument("--tau", type=float, default=0.5, help="first XY-support exit threshold")
    ap.add_argument("--seed", type=int, default=20260901)
    args = ap.parse_args()
    out = Path(f"gap-{args.name}"); out.mkdir(exist_ok=True)
    rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed)

    # 1. 采数（80/20 按轨迹切 train/holdout）
    trajs = collect_expert(args.n_traj, rng, args.wind, args.kick_prob, args.kick_y, args.kick_theta)
    n_tr = int(0.8 * len(trajs))
    Otr, Ytr, band_xy = to_chunks(trajs[:n_tr])
    Oho, Yho, ho_xy = to_chunks(trajs[n_tr:])
    np.savez(out / "holdout.npz", O=Oho, Y=Yho)
    d_ho = dist_to_band(ho_xy, band_xy)
    tau = args.tau
    print(f"train states {len(Otr)}  holdout states {len(Oho)}  holdout XY-distance p99={np.percentile(d_ho, 99):.3f}  tau={tau}")

    # 2. Train BC with H-step action chunks.
    model = nn.Sequential(nn.Linear(3, args.hidden), nn.Tanh(), nn.Linear(args.hidden, args.hidden), nn.Tanh(),
                          nn.Linear(args.hidden, H))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    Ot, Yt = torch.tensor(Otr), torch.tensor(Ytr); Oh, Yh = torch.tensor(Oho), torch.tensor(Yho)
    g = torch.Generator().manual_seed(args.seed)
    hist = []
    for it in range(args.steps):
        idx = torch.randint(0, len(Ot), (256,), generator=g)
        loss = ((model(Ot[idx]) - Yt[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 10 == 0 or it == args.steps - 1:
            with torch.no_grad():
                hl = ((model(Oh) - Yh) ** 2).mean().item()
            hist.append((it, {"train": loss.item(), "holdout": hl}))
    plot_loss(hist, out / "loss.png")
    torch.save(model.state_dict(), out / "model.pt")
    with torch.no_grad():
        holdout_sqerr = (model(Oh) - Yh) ** 2
    per_horizon = holdout_sqerr.mean(0).numpy()
    summary = {
        "args": vars(args),
        "support_metric": "XY nearest-training-state distance; heading theta is not included",
        "pairing_contract": "case i uses rollout seed args.seed + 7 + i for every E",
        "tau": tau,
        "final_train_loss": hist[-1][1]["train"],
        "final_holdout_loss": hist[-1][1]["holdout"],
        "per_horizon_holdout_mse": per_horizon.tolist(),
        "early_H8_holdout_mse": float(per_horizon[: H // 2].mean()),
        "tail_H8_holdout_mse": float(per_horizon[H // 2:].mean()),
        "rollouts": {},
    }
    if args.eval_holdout:
        z = np.load(args.eval_holdout)
        with torch.no_grad():
            summary["loss_on_eval_holdout"] = ((model(torch.tensor(z["O"])) - torch.tensor(z["Y"])) ** 2).mean().item()
        print(f"loss on {args.eval_holdout}: {summary['loss_on_eval_holdout']:.2e}")

    # 3. Closed-loop rollouts: same policy and paired initial-state RNG for every E.
    execution_horizons = [int(e) for e in args.E.split(",")]
    if any(E < 1 or E > H for E in execution_horizons):
        raise ValueError(f"execution horizon E must be within [1, {H}]")
    rollout_seeds = [args.seed + 7 + i for i in range(args.rollouts)]
    for E in execution_horizons:
        rs = [rollout(model, E, case_seed, band_xy, tau, args.wind) for case_seed in rollout_seeds]
        k = sum(r["ok"] for r in rs)
        exits = [r["first_exit"] for r in rs if r["first_exit"] is not None]
        returns = [r["return_step"] for r in rs if r["return_step"] is not None]
        lo, hi = wilson(k, len(rs))
        summary["rollouts"][f"E{E}"] = {
            "success": k,
            "n": len(rs),
            "wilson95": [lo, hi],
            "n_xy_support_exits": len(exits),
            "mean_first_xy_support_exit": float(np.mean(exits)) if exits else None,
            "first_xy_support_exit_each": [r["first_exit"] for r in rs],
            "n_returned_to_support": len(returns),
            "return_to_support_rate_given_exit": len(returns) / len(exits) if exits else None,
            "return_step_each": [r["return_step"] for r in rs],
            "rollout_seed_each": [r["rollout_seed"] for r in rs],
            "initial_state_each": [r["initial_state"].tolist() for r in rs],
            "action_health": summarize_action_health(rs),
            "final_x_each": [float(r["S"][-1, 0]) for r in rs],
        }
        print(f"E={E:2d}  SR {k}/{len(rs)}  Wilson95 [{lo:.2f},{hi:.2f}]  "
              f"XY-support exits {len(exits)}  mean first exit "
              f"{np.mean(exits) if exits else float('nan'):.1f}  returned {len(returns)}/{len(exits)}")
        plot_rollouts(rs, band_xy, E, out / f"rollouts_E{E}.png", f"{args.name} · E={E} · SR {k}/{len(rs)}")
        plot_dist(rs, tau, E, out / f"dist_E{E}.png")
    print(f"final train {summary['final_train_loss']:.2e}  holdout {summary['final_holdout_loss']:.2e}")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
