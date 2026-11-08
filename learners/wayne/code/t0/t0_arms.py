# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["torch", "numpy", "matplotlib"]
# ///
"""T0·S02 五臂干预统一 runner（AI 脚手架）。

用法：uv run t0_arms.py --arm A     → 输出到 arm-A/（B/C/D/E 同理）

A 臂 = 上节课 v3 baseline 的完整复刻（同数据、同 seed、同 3-16-16-16-1、
同 lr=1e-3 / 5000 步 / batch 256），RNG 消耗顺序与 t0_train.py 逐一对齐，
结果应与 baseline/ 一致。B/C/D/E 臂的配置差异按课程规则由学员口述后
填入各自的「学员口述配置区」，当前留空并退出。

每次运行输出全套仪表（沿用 report.py 画图习惯）：
  loss.png（原始线压淡 + EMA 粗趋势线）、gradnorm.png（逐层）、
  prob_hist.png（σ(z) 直方图）、summary.json（全部数值指标，
  含 per-class recall/gap、中间质量、逐层 max|Δ| 参数检查）。
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from report import save_report

HERE = Path(__file__).resolve().parent
SEED = 20260829
ACTS = {"tanh": nn.Tanh, "relu": nn.ReLU, "sigmoid": nn.Sigmoid}


def base_config():
    """v3 baseline 配置（= t0_train.py），A 臂原样使用，其余臂在口述区改动。"""
    return dict(
        seed=SEED,
        in_dim=3,
        hidden=[16, 16, 16],        # 结构 3-16-16-16-1
        act="tanh",
        lr=1e-3,
        steps=5000,
        batch=256,
        # preprocess(X_raw, mean, std) -> 特征矩阵；默认 = baseline 标准化
        preprocess=lambda X, mean, std: (X - mean) / std,
        # resample_train(X_raw, y) -> (X_raw, y)：训练集重抽样钩子（C 组用），默认无
        resample_train=None,
        # post_build(model)：建模后、建 optimizer 前的钩子（冻结/重初始化等），默认无
        post_build=None,
        note="v3 baseline replication",
    )


# ---------------------------------------------------------------- 各臂配置 --

def configure_arm_A(cfg):
    # A 臂 = 对照组：完整复刻 v3 baseline，不做任何改动。
    return cfg


def configure_arm_B(cfg):
    # === 学员口述配置区 · B 臂 ===
    # 学员口述（wayne，2026-08-30）：在第②步标准化处动手——force 列（第 0 列）
    # 跳过标准化并把原始值 ×1000（停疫苗造病）；slip、pose 两列照常标准化；
    # 其余一切与 A 组（对照组）相同。
    def preprocess(X, mean, std):
        Xp = (X - mean) / std           # slip/pose 照常
        Xp[:, 0] = X[:, 0] * 1000.0     # force：raw ×1000，不标准化
        return Xp
    cfg["preprocess"] = preprocess
    cfg["note"] = "B 单位灾难: force raw x1000 & no standardization (col 0)"
    # === 学员口述配置区 · B 臂 结束 ===
    return cfg


def configure_arm_C(cfg):
    # === 学员口述配置区 · C 组 ===
    # 学员口述（wayne，2026-08-30 续课）：三选一里选 (c) 重新抽一份训练集——
    # 总样本量与 A 组相同，90% 从"稳"样本抽（可放回）、10% 从"不稳"样本抽；
    # 测试集与标准化统计不动。
    def resample(X, y):
        rng = np.random.default_rng(SEED)
        n = len(y)
        n_s = int(round(n * 0.9)); n_u = n - n_s
        idx_s = np.flatnonzero(y == 1); idx_u = np.flatnonzero(y == 0)
        pick_s = rng.choice(idx_s, size=n_s, replace=True)
        pick_u = rng.choice(idx_u, size=n_u, replace=len(idx_u) < n_u)
        idx = rng.permutation(np.concatenate([pick_s, pick_u]))
        return X[idx], y[idx]
    cfg["resample_train"] = resample
    cfg["note"] = "C 数据偏食: train 重抽为 90% stable / 10% unstable（选项 c）"
    # === 学员口述配置区 · C 组 结束 ===
    return cfg


def configure_arm_D(cfg):
    # === 学员口述配置区 · D 组 ===
    # 学员口述（wayne，续课）：第 1 层 requires_grad=False，并因此不进优化器
    # 清单（runner 建优化器只收 requires_grad 的参数，两层保护一次落实）；
    # 尺子=训前训后逐元素比对，判据经确认收紧为 Δ 恰等于 0（非"小于预设值"）。
    def freeze_first(model):
        first = next(m for m in model if isinstance(m, nn.Linear))
        for p in first.parameters():
            p.requires_grad_(False)
    cfg["post_build"] = freeze_first
    cfg["note"] = "D 冻结纪律: layer1 requires_grad=False + 不入优化器清单，Δ 须恰为 0"
    # === 学员口述配置区 · D 组 结束 ===
    return cfg


def configure_arm_E(cfg):
    # === 学员口述配置区 · E 组 ===
    # 学员口述（wayne，续课）：结构砍到 3-2-1——hidden 从 [16,16,16] 改为 [2]
    # （单隐层、2 个神经元）；其余一切与 A 组相同。
    cfg["hidden"] = [2]
    cfg["note"] = "E 容量瘦身: 3-2-1 单隐层宽 2"
    # === 学员口述配置区 · E 组 结束 ===
    return cfg


CONFIGURE = {"A": configure_arm_A, "B": configure_arm_B, "C": configure_arm_C,
             "D": configure_arm_D, "E": configure_arm_E}


# ------------------------------------------------------------------ runner --

def build_model(cfg):
    """按 cfg 建网。层创建顺序与 t0_train.py 相同 → A 臂 RNG 消耗逐一对齐。"""
    act = ACTS[cfg["act"]]
    dims = [cfg["in_dim"], *cfg["hidden"], 1]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


def run(arm):
    outdir = HERE / f"arm-{arm}"
    cfg = CONFIGURE[arm](base_config())

    d = np.load(HERE / "t0_data.npz", allow_pickle=True)
    mean, std = d["mean"], d["std"]
    Xraw_tr, ytr = d["X_train_raw"], d["y_train"]
    if cfg.get("resample_train") is not None:
        Xraw_tr, ytr = cfg["resample_train"](Xraw_tr, ytr)
    Xtr = cfg["preprocess"](Xraw_tr, mean, std).astype(np.float32)
    Xte = cfg["preprocess"](d["X_test_raw"], mean, std).astype(np.float32)
    yte = d["y_test"]

    torch.manual_seed(cfg["seed"])
    model = build_model(cfg)
    if cfg["post_build"] is not None:
        cfg["post_build"](model)
    linears = [(f"layer{k + 1}", m) for k, m in
               enumerate(m for m in model if isinstance(m, nn.Linear))]
    # 训前参数快照（D 臂用：训后逐元素相减报告每层 max|Δ|）
    snap = {name: torch.cat([p.detach().flatten().clone() for p in m.parameters()])
            for name, m in linears}
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=cfg["lr"])
    lossf = nn.BCEWithLogitsLoss()

    Xt = torch.tensor(Xtr); yt = torch.tensor(ytr, dtype=torch.float32)
    g = torch.Generator().manual_seed(cfg["seed"])
    losses, gn = [], {name: [] for name, _ in linears}
    w1 = linears[0][1].weight                       # 第一层：逐输入列梯度范数（B 组主战场）
    col_gn = {j: [] for j in range(w1.shape[1])}
    for step in range(cfg["steps"]):
        idx = torch.randint(0, len(yt), (cfg["batch"],), generator=g)
        z = model(Xt[idx]).squeeze(1)
        loss = lossf(z, yt[idx])
        opt.zero_grad(); loss.backward()
        for name, m in linears:
            gs = [p.grad.flatten() for p in m.parameters() if p.grad is not None]
            gn[name].append(torch.linalg.norm(torch.cat(gs)).item() if gs else 0.0)
        if w1.grad is not None:
            for j in col_gn:
                col_gn[j].append(torch.linalg.norm(w1.grad[:, j]).item())
        opt.step()
        losses.append(loss.item())

    with torch.no_grad():
        probs = torch.sigmoid(model(torch.tensor(Xte)).squeeze(1)).numpy()

    rec = save_report(losses, gn, probs, yte, outdir=str(outdir))
    np.save(outdir / "losses.npy", np.array(losses))         # 跨组对比用

    if any(v for v in col_gn.values()):                      # 第一层逐列梯度范数图
        import matplotlib
        matplotlib.use("Agg")
        matplotlib.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "PingFang SC", "Arial Unicode MS", "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
        import matplotlib.pyplot as plt
        names = ["force", "slip", "pose"]
        fig, ax = plt.subplots(figsize=(7, 3.2))
        for j, vals in col_gn.items():
            ax.plot(vals, lw=1.2, label=f"{names[j] if j < 3 else j} 列 ‖∂L/∂W₁[:,{j}]‖")
        ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("grad norm (log)")
        ax.set_title(f"arm-{arm} · 第一层逐输入列梯度范数"); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(outdir / "layer1_col_gradnorm.png", dpi=130)
        plt.close(fig)

    # 数值指标（口径与 t0_train.py 一致）
    final_loss = float(np.mean(losses[-100:]))
    gap_pp = abs(rec[0] - rec[1]) * 100
    mid = float(((probs > 0.2) & (probs < 0.8)).mean())          # 中间质量：0.2<σ(z)<0.8
    sat = float(((probs > 0.9) | (probs < 0.1)).mean())
    delta = {name: float((torch.cat([p.detach().flatten() for p in m.parameters()])
                          - snap[name]).abs().max()) for name, m in linears}

    print(f"最后 100 步平均 train loss = {final_loss:.4f}")
    print(f"P1 判据：recall gap = {gap_pp:.1f} pp")
    print(f"P2 判据：0.2<σ<0.8 的中间样本占比 = {mid*100:.1f}%  (σ>0.9 或 <0.1 的占比 = {sat*100:.1f}%)")
    print("参数 Δ 检查：" + "  ".join(f"{n} max|Δ|={v:.4f}" for n, v in delta.items()))

    # 按边界族分解的 recall（数据生成时的 family 标签：broad/force/slip/pose）
    fam = d["fam_test"]
    fam_names = ["broad", "force_edge", "slip_edge", "pose_edge"]
    pred_stable = probs >= 0.5
    fam_recall = {}
    for k, nm in enumerate(fam_names):
        for cls, cname in ((0, "unstable"), (1, "stable")):
            m = (fam == k) & (yte == cls)
            if m.sum() > 0:
                fam_recall[f"{nm}/{cname}"] = round(float((pred_stable[m] == (cls == 1)).mean()), 4)
    print("边界族 recall：" + "  ".join(f"{k}={v}" for k, v in fam_recall.items()))

    arch = "-".join(map(str, [cfg["in_dim"], *cfg["hidden"], 1]))
    summary = dict(
        arm=arm,
        config=dict(seed=cfg["seed"], arch=arch, act=cfg["act"], opt="adam",
                    lr=cfg["lr"], batch=cfg["batch"], steps=cfg["steps"], note=cfg["note"]),
        n_params=sum(p.numel() for p in model.parameters()),
        final_loss_last100=final_loss,
        recall_unstable=rec[0], recall_stable=rec[1], recall_gap_pp=gap_pp,
        mid_mass_0p2_0p8=mid, saturated_mass=sat, family_recall=fam_recall,
        grad_norm_last100_mean={n: float(np.mean(v[-100:])) for n, v in gn.items()},
        param_max_abs_delta=delta,
    )
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"summary 已存 {outdir}/summary.json；图在 {outdir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="T0·S02 五臂干预统一 runner")
    ap.add_argument("--arm", required=True, choices=list("ABCDE"))
    run(ap.parse_args().arm)
