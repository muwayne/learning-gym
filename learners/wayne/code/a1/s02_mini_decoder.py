# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy"]
# ///
"""A1·S02 causal mask + 多头 + Pre-LN block。模式 A：TODO(wayne) 的函数体由 wayne 亲手实现；
初始化、assert 与四项检查为 AI 脚手架（判据勿改）。在本目录下运行：uv run s02_mini_decoder.py"""
import numpy as np

from s01_attention import attention

# ---------- 核心实现（TODO(wayne)：亲手补齐函数体，可随时查 API） ----------


def causal_mask(L):
    """下三角允许表：允许=1、禁止=0，shape (L, L)。"""
    # 口述①（wayne）：列 index 小于等于行位置 → 1，否则 0
    # （初版漏了"等于"，对角线全 0：全禁行被 -1e9 填满后 softmax 变均匀——禁令全灭成放行，证据抓获后修复）
    return (np.arange(L)[None, :] <= np.arange(L)[:, None]).astype(float)


def split_heads(X, H):
    """(L, d) → (H, L, d_h)。切的是投影产出的列。"""
    # 口述②（wayne）：先把 d 拆成 H×d_h → (L, H, d_h)，再把 H 和 L 换位置 → (H, L, d_h)
    L, d = X.shape
    return X.reshape(L, H, d // H).transpose(1, 0, 2)


def merge_heads(Xh):
    """(H, L, d_h) → (L, d)。split 的逆操作。"""
    # 口述③（wayne）：先转置回 (L, H, d_h)，再把末尾两维并成 (L, d)
    H, L, d_h = Xh.shape
    return Xh.transpose(1, 0, 2).reshape(L, H * d_h)


def softmax_lastdim(S):
    """S01 的 softmax_rows 写死 axis=1：2D 时恰好是行，3D (H,L,L) 时就归一化到错误的轴上。
    批量版必须对 axis=-1 操作——轴的语义跟着维度数变，写死数字轴是隐患。"""
    S = S - S.max(axis=-1, keepdims=True)
    E = np.exp(S)
    return E / E.sum(axis=-1, keepdims=True)


def mha(X, WQ, WK, WV, WO, H, causal=False):
    """多头 attention。返回 (O, P)，O shape (L, d)，P shape (H, L, L)。"""
    # 口述⑥（wayne）：①Q/K/V 全部 split ②每头 S=QKᵀ/√d_h ③causal 时在 softmax 之前
    # 按 mask 选择（禁格填 -1e9）④三维批量算，不逐头循环
    L, d = X.shape
    d_h = d // H
    Qh = split_heads(X @ WQ, H)                      # (H, L, d_h)
    Kh = split_heads(X @ WK, H)
    Vh = split_heads(X @ WV, H)
    S = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(d_h)    # (H, L, L)
    if causal:
        S = np.where(causal_mask(L)[None, :, :] == 0, -1e9, S)
    P = softmax_lastdim(S)                           # (H, L, L)
    O = merge_heads(P @ Vh) @ WO                     # (L, d)
    return O, P


def layer_norm(x, gamma, beta, eps=1e-5):
    """对最后一维：减均值、除 sqrt(方差+eps)、乘 gamma 加 beta。每个 token 自己算自己的。"""
    # 口述④（wayne）：最后一维求均值方差（keep 维度）→ x 减均值、除以 √(方差+eps) → 乘 gamma 加 beta
    # （原话"除以它"按公式取 √(σ²+eps)；若按字面除以方差，每 token 的 std 就不是 1，微测可抓）
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return gamma * (x - mu) / np.sqrt(var + eps) + beta


def ffn(x, W1, b1, W2, b2):
    """d → 4d → ReLU → d，逐 token 独立。"""
    # 口述⑤（wayne）：最后一维 ·W1+b1 → ReLU → ·W2+b2；batch 维与 L 维不参与计算——逐 token 独立
    return np.maximum(x @ W1 + b1, 0.0) @ W2 + b2


def block(X, p, H, causal=True):
    """Pre-LN block。返回 (X, P)。"""
    # 口述⑦（wayne）：X' = X + mha(LN1(X))；输出 = X' + ffn(LN2(X'))；
    # residual 加的是 LN 之前的裸 X（Pre-LN 的灵魂：identity path 上传的是裸 X）；P 由 mha 产出
    h, P = mha(layer_norm(X, p["g1"], p["b1"]),
               p["WQ"], p["WK"], p["WV"], p["WO"], H, causal)
    X = X + h
    X = X + ffn(layer_norm(X, p["g2"], p["b2"]),
                p["W1"], p["bf1"], p["W2"], p["bf2"])
    return X, P


# ---------- 脚手架：初始化与四项检查（AI 写好，判据勿改） ----------


def init_params(d, seed=0):
    rng = np.random.default_rng(seed)
    W = lambda a, b: rng.standard_normal((a, b)) / np.sqrt(a)
    return dict(WQ=W(d, d), WK=W(d, d), WV=W(d, d), WO=W(d, d),
                g1=np.ones(d), b1=np.zeros(d), g2=np.ones(d), b2=np.zeros(d),
                W1=W(d, 4 * d), bf1=np.zeros(4 * d), W2=W(4 * d, d), bf2=np.zeros(d))


if __name__ == "__main__":
    X3 = np.array([[2., 0.], [0., 2.], [1., 1.]])   # S01 的 L=3 toy
    I2 = np.eye(2)

    print("== 检查 A · 退化对拍：H=1、无 causal、WO=单位阵 ⇒ 应与 S01 输出完全一致 ==")
    O_ref, P_ref = attention(X3, I2, I2, I2)
    O_new, P_new = mha(X3, I2, I2, I2, I2, H=1, causal=False)
    assert np.allclose(O_new, O_ref), f"O 与 S01 不一致:\n{O_new}\nvs\n{O_ref}"
    assert np.allclose(P_new[0], P_ref), "P 与 S01 不一致"
    print("A 通过")

    print("== 检查 B · 眼罩生效：L=3 toy + causal ==")
    O_c, P_c = mha(X3, I2, I2, I2, I2, H=1, causal=True)
    Pc = P_c[0]
    assert np.allclose(Pc, np.tril(Pc)), f"P 不是下三角:\n{Pc}"
    assert np.allclose(Pc.sum(axis=1), 1.0), f"行和不为 1: {Pc.sum(axis=1)}"
    assert np.allclose(Pc[0], [1., 0., 0.]), f"第 0 行不是 [1,0,0]: {Pc[0]}"
    print("B 通过：P =\n", np.round(Pc, 4))

    print("== 检查 C · shape 链：L=5, d=8, H=2 ==")
    rng = np.random.default_rng(1)
    X5 = rng.standard_normal((5, 8))
    Xh = split_heads(X5, 2)
    assert Xh.shape == (2, 5, 4), f"split 后 shape 错：{Xh.shape}，应为 (2, 5, 4)"
    assert np.allclose(Xh[0], X5[:, :4]), "头 0 应拿前 4 列（前 d_h 列归头 0）"
    assert np.allclose(merge_heads(Xh), X5), "merge(split(X)) ≠ X"
    p = init_params(8)
    O5, P5 = mha(X5, p["WQ"], p["WK"], p["WV"], p["WO"], H=2, causal=True)
    assert O5.shape == (5, 8) and P5.shape == (2, 5, 5), f"mha 输出 shape 错：{O5.shape}, {P5.shape}"
    Y5, _ = block(X5, p, H=2, causal=True)
    assert Y5.shape == (5, 8), f"block 输出 shape 错：{Y5.shape}"
    print("C 通过：split/merge/mha/block 全链 shape ✓")

    print("== 检查 D · 顺序实验：打乱输入行，置换等变还在吗？ ==")
    perm = np.array([2, 0, 3, 4, 1])
    for causal in (False, True):
        O_a, _ = mha(X5, p["WQ"], p["WK"], p["WV"], p["WO"], H=2, causal=causal)
        O_b, _ = mha(X5[perm], p["WQ"], p["WK"], p["WV"], p["WO"], H=2, causal=causal)
        same = np.allclose(O_b, O_a[perm], atol=1e-8)
        print(f"causal={causal}: 输出{'仍跟着打乱（置换等变保持）' if same else '不再跟着打乱（置换等变破坏）'}")
