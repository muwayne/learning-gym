# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy"]
# ///
"""A1·S01 单头 attention。模式 B 口述驾驶：函数体五步由 wayne 口述、AI 代打；
softmax 数值稳定化、assert 与实验脚本为 AI 脚手架。"""
import numpy as np


def softmax_rows(S):
    S = S - S.max(axis=1, keepdims=True)
    E = np.exp(S)
    return E / E.sum(axis=1, keepdims=True)


def attention(X, WQ, WK, WV, mask=None):
    Q = X @ WQ                    # 口述①：QKV 分别 = X 乘对应权重
    K = X @ WK
    V = X @ WV
    d = X.shape[1]
    S = Q @ K.T / np.sqrt(d)      # 口述②：S = Q×Kᵀ / sqrt(d)
    if mask is not None:
        S = np.where(mask == 0, -1e9, S)   # 口述③修正：被禁格子填极大负数（masked_fill 语义）
    P = softmax_rows(S)
    O = P @ V                     # 口述④修正：O = P 乘 V（无转置）
    return O, P


if __name__ == "__main__":
    X = np.array([[2., 0.], [0., 2.], [1., 1.]])
    I = np.eye(2)

    print("== 检查 A · 对拍手算 ==")
    O, P = attention(X, I, I, I)
    P_hand = np.array([[0.7679, 0.0454, 0.1867], [0.0454, 0.7679, 0.1867], [1/3, 1/3, 1/3]])
    O_hand = np.array([[1.72, 0.28], [0.28, 1.72], [1., 1.]])
    assert np.allclose(P, P_hand, atol=0.02), f"P 与手算不符:\n{P}"
    assert np.allclose(O, O_hand, atol=0.02), f"O 与手算不符:\n{O}"
    print("A 通过：P、O 与手算一致")

    print("== 检查 B · 去缩放，d=4/64/256 ==")
    rng = np.random.default_rng(0)
    for d in (4, 64, 256):
        Xr = rng.standard_normal((6, d))
        Praw = softmax_rows(Xr @ Xr.T)          # 故意不除 √d
        Pscl = softmax_rows(Xr @ Xr.T / np.sqrt(d))
        print(f"d={d:3d}  无缩放每行最大概率均值={Praw.max(1).mean():.4f}   有缩放={Pscl.max(1).mean():.4f}")

    print("== 检查 C · mask 掉 (0,2) ==")
    mask = np.ones((3, 3)); mask[0, 2] = 0.0
    O2, P2 = attention(X, I, I, I, mask)
    print("行和 =", np.round(P2.sum(axis=1), 6))
    print("被禁位置权重 P2[0,2] =", round(float(P2[0, 2]), 4))
    assert P2[0, 2] < 1e-6, f"mask 无效：token0 仍以 {P2[0,2]:.4f} 的权重看着 token2"
    print("开放观察项：无 mask 时 O 第0行 =", np.round(O[0], 3), "→ mask 后 =", np.round(O2[0], 3))
    print("C 通过")
