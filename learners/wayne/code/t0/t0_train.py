# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["torch", "numpy", "matplotlib"]
# ///
"""T0·S01 baseline（模式 B 口述驾驶）。
决策（wayne 口述，2026-08-29）：隐层宽度 3 + tanh；minibatch 256；5000 步。
lr：v1=1e-4 欠训练（存 baseline-lr1e-4/），v2 lr×10 → 1e-3（单隐层宽3，存 baseline-arch3-3-1/）。
v3（口述决策 c）：结构改为真·三隐层 3→16→16→16→1，其余与 v2 完全相同。
AI 脚手架：数据装载、记录、画图；核心结构/损失/循环按口述决策实现。
"""
import json
import numpy as np
import torch
import torch.nn as nn
from report import save_report

SEED = 20260829
d = np.load("t0_data.npz", allow_pickle=True)
mean, std = d["mean"], d["std"]
Xtr = ((d["X_train_raw"] - mean) / std).astype(np.float32)
Xte = ((d["X_test_raw"] - mean) / std).astype(np.float32)
ytr, yte = d["y_train"], d["y_test"]

torch.manual_seed(SEED)
model = nn.Sequential(nn.Linear(3, 16), nn.Tanh(), nn.Linear(16, 16), nn.Tanh(),
                      nn.Linear(16, 16), nn.Tanh(), nn.Linear(16, 1))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
lossf = nn.BCEWithLogitsLoss()

Xt = torch.tensor(Xtr); yt = torch.tensor(ytr, dtype=torch.float32)
g = torch.Generator().manual_seed(SEED)
STEPS, B = 5000, 256
losses, gn = [], {"layer1": [], "layer2": [], "layer3": [], "layer4": []}
LIDX = {"layer1": 0, "layer2": 2, "layer3": 4, "layer4": 6}
for step in range(STEPS):
    idx = torch.randint(0, len(yt), (B,), generator=g)
    z = model(Xt[idx]).squeeze(1)
    loss = lossf(z, yt[idx])
    opt.zero_grad(); loss.backward()
    for name, li in LIDX.items():
        gn[name].append(torch.linalg.norm(torch.cat([p.grad.flatten() for p in model[li].parameters()])).item())
    opt.step()
    losses.append(loss.item())

with torch.no_grad():
    probs = torch.sigmoid(model(torch.tensor(Xte)).squeeze(1)).numpy()
rec = save_report(losses, gn, probs, yte, outdir="baseline")
mid = float(((probs > 0.2) & (probs < 0.8)).mean())
print(f"最后 100 步平均 train loss = {np.mean(losses[-100:]):.4f}")
print(f"P1 判据：recall gap = {abs(rec[0]-rec[1])*100:.1f} pp")
print(f"P2 判据：0.2<σ<0.8 的中间样本占比 = {mid*100:.1f}%  (σ>0.9 或 <0.1 的占比 = {float(((probs>0.9)|(probs<0.1)).mean())*100:.1f}%)")
nparam = sum(p.numel() for p in model.parameters())
print(f"结构 3-16-16-16-1，参数量 = {nparam}")
json.dump(dict(seed=SEED, arch="3-16-16-16-1", act="tanh", opt="adam", lr=1e-3, batch=B, steps=STEPS),
          open("baseline/config.json", "w"))
print("config 已存 baseline/config.json；图在 baseline/")
