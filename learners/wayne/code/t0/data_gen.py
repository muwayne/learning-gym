# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy"]
# ///
"""T0 数据生成器（AI 脚手架，spec 与 seed 写死 — S02 五臂共用，不得改动本文件）。

规则（来自 v1 W1D1 原题）：
    stable = normal_force > F  and  |slip_velocity| < S  and  pose_error < P

生成分布：70% 宽采样 + 3 个边界族各 10%（临界力 / 瞬时滑动 / 位姿边缘），
5% 标签翻转噪声，训练/测试均类别平衡。输出 raw 特征 + 标准化统计量分开保存，
baseline 用 standardize()；S02 的 B 臂（force×1000、不标准化）直接用 raw。

用法：uv run data_gen.py   → 生成 t0_data.npz 并打印摘要
"""
import json
import numpy as np

SEED = 20260829
SPEC = dict(F=5.0, S=0.5, P=0.15, label_noise=0.05,
            frac_broad=0.70, frac_force_edge=0.10, frac_slip_edge=0.10, frac_pose_edge=0.10,
            n_train=2000, n_test=1000)


def _sample(rng, n):
    n_b = int(n * SPEC["frac_broad"]); n_f = int(n * SPEC["frac_force_edge"])
    n_s = int(n * SPEC["frac_slip_edge"]); n_p = n - n_b - n_f - n_s
    def broad(m):
        return np.column_stack([rng.uniform(0, 10, m),
                                rng.normal(0, 0.4, m),
                                np.abs(rng.normal(0, 0.12, m))])
    def force_edge(m):
        x = broad(m); x[:, 0] = rng.normal(SPEC["F"], 0.3, m); return x
    def slip_edge(m):
        x = broad(m); x[:, 1] = rng.choice([-1, 1], m) * rng.normal(SPEC["S"], 0.05, m); return x
    def pose_edge(m):
        x = broad(m); x[:, 2] = np.abs(rng.normal(SPEC["P"], 0.02, m)); return x
    X = np.vstack([broad(n_b), force_edge(n_f), slip_edge(n_s), pose_edge(n_p)])
    fam = np.repeat([0, 1, 2, 3], [n_b, n_f, n_s, n_p])
    y = ((X[:, 0] > SPEC["F"]) & (np.abs(X[:, 1]) < SPEC["S"]) & (X[:, 2] < SPEC["P"])).astype(np.int64)
    flip = rng.random(len(y)) < SPEC["label_noise"]
    y[flip] = 1 - y[flip]
    return X, y, fam, int(flip.sum())


def _balanced(rng, n):
    """rejection 到两类各 n//2。"""
    Xs, ys, fams, flips = [], [], [], 0
    need = {0: n // 2, 1: n - n // 2}
    while need[0] > 0 or need[1] > 0:
        X, y, fam, nf = _sample(rng, n)
        flips += nf
        for c in (0, 1):
            idx = np.where(y == c)[0][: need[c]]
            Xs.append(X[idx]); ys.append(y[idx]); fams.append(fam[idx]); need[c] -= len(idx)
    X = np.vstack(Xs); y = np.concatenate(ys); fam = np.concatenate(fams)
    p = rng.permutation(len(y))
    return X[p], y[p], fam[p], flips


def standardize(X, mean, std):
    return (X - mean) / std


def main():
    rng = np.random.default_rng(SEED)
    Xtr, ytr, ftr, _ = _balanced(rng, SPEC["n_train"])
    Xte, yte, fte, _ = _balanced(rng, SPEC["n_test"])
    mean, std = Xtr.mean(0), Xtr.std(0)
    np.savez("t0_data.npz", X_train_raw=Xtr, y_train=ytr, fam_train=ftr,
             X_test_raw=Xte, y_test=yte, fam_test=fte,
             mean=mean, std=std, spec=json.dumps(SPEC), seed=SEED)
    print(f"seed={SEED}  train={len(ytr)} (stable {ytr.sum()})  test={len(yte)} (stable {yte.sum()})")
    print(f"family counts train (broad/force/slip/pose): {np.bincount(ftr, minlength=4)}")
    print(f"feature mean={mean.round(3)}  std={std.round(3)}")
    print("首 3 行 raw：\n", Xtr[:3].round(3), "labels:", ytr[:3])


if __name__ == "__main__":
    main()
