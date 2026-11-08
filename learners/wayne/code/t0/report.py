# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy", "matplotlib"]
# ///
"""T0 记录/画图 helper（AI 脚手架）。学员训练脚本 import 使用：

    from report import per_class_recall, save_report
    save_report(losses, grad_norms, probs, y_test, outdir="baseline")

生成 <outdir>/loss.png、gradnorm.png、prob_hist.png，并打印 per-class recall。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def per_class_recall(y, prob, thresh=0.5):
    pred = (np.asarray(prob) >= thresh).astype(int)
    y = np.asarray(y)
    out = {}
    for c in (0, 1):
        m = y == c
        out[c] = float((pred[m] == c).mean()) if m.any() else float("nan")
    return out  # {0: unstable_recall, 1: stable_recall}


def _ema(a, beta=0.98):
    out, m = [], a[0]
    for v in a:
        m = beta * m + (1 - beta) * v
        out.append(m)
    return out


def save_report(losses, grad_norms, probs, y_test, outdir="baseline"):
    os.makedirs(outdir, exist_ok=True)
    plt.figure()
    plt.plot(losses, alpha=0.22, lw=0.7)
    plt.plot(_ema(losses), lw=2)
    plt.xlabel("step"); plt.ylabel("BCE"); plt.title("train loss (粗线=EMA 平滑)")
    plt.savefig(f"{outdir}/loss.png", dpi=120); plt.close()

    plt.figure()
    for name, g in grad_norms.items():
        ln, = plt.plot(g, alpha=0.22, lw=0.7)
        plt.plot(_ema(g), color=ln.get_color(), lw=2, label=name)
    plt.xlabel("step"); plt.ylabel("||grad||"); plt.legend(); plt.title("per-layer grad norm (粗线=EMA)")
    plt.savefig(f"{outdir}/gradnorm.png", dpi=120); plt.close()

    probs = np.asarray(probs); y_test = np.asarray(y_test)
    plt.figure()
    plt.hist(probs[y_test == 0], bins=30, range=(0, 1), alpha=0.6, label="true unstable")
    plt.hist(probs[y_test == 1], bins=30, range=(0, 1), alpha=0.6, label="true stable")
    # 判据区间直接标在图上（画图规矩：判据数字须可从图读出）
    lo = float((probs <= 0.2).mean()); mid = float(((probs > 0.2) & (probs < 0.8)).mean())
    hi = float((probs >= 0.8).mean())
    plt.axvspan(0.2, 0.8, color="gray", alpha=0.12)
    ax = plt.gca()
    ax.text(0.10, 0.97, f"左端 {lo*100:.1f}%", ha="center", va="top", transform=ax.get_xaxis_transform(), fontsize=9)
    ax.text(0.50, 0.97, f"中间质量 {mid*100:.1f}%", ha="center", va="top", transform=ax.get_xaxis_transform(), fontsize=9, fontweight="bold")
    ax.text(0.90, 0.97, f"右端 {hi*100:.1f}%", ha="center", va="top", transform=ax.get_xaxis_transform(), fontsize=9)
    plt.xlabel("p(stable)"); plt.legend(); plt.title("output probability（灰带=0.2–0.8 犹豫区）")
    plt.savefig(f"{outdir}/prob_hist.png", dpi=120); plt.close()

    rec = per_class_recall(y_test, probs)
    print(f"[{outdir}] recall  unstable={rec[0]:.3f}  stable={rec[1]:.3f}  gap={abs(rec[0]-rec[1])*100:.1f}pp")
    print(f"[{outdir}] 图已存：{outdir}/loss.png gradnorm.png prob_hist.png")
    return rec
