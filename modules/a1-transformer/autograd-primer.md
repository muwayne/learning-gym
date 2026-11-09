# Autograd · 从 loss 到参数更新

## 0 · 三件事各由谁做

训练程序先用参数算出一个标量 `loss`，autograd 再算「参数稍微动一点，loss 会往哪个方向变」，最后 optimizer 读取这些梯度并更新参数。本页用两个预测共享一个权重的回归例子，沿同一张图走一遍 `forward → backward → SGD step`。然后把 `mean`、`min` 和 `clip` 的反传规则写进一个可以运行的标量引擎。

| 阶段 | 输入和产出 |
| --- | --- |
| **Forward** | 用当前参数计算预测和 loss，并记录各步计算之间的依赖。 |
| **Backward** | 从 loss 反向应用 chain rule（链式法则），得到 loss 对每个需要梯度的参数的导数。 |
| **Optimizer step** | 用参数的梯度计算更新量。这里用最简单的 SGD，`新参数 = 旧参数 − 学习率 × 梯度`。 |

梯度是**当前这次 forward 的局部变化率**。PyTorch 的 `loss.backward()` 默认把梯度累加到参数的 `.grad`。典型训练循环每轮先调用 `optimizer.zero_grad()`，再算 loss、backward、step。下一轮重新 forward。如果要用 gradient clipping，一般在 backward 之后、step 之前限制梯度本身。下文的 `clip` 作用于 forward 的预测值。 [PyTorch · Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html)

## 1 · 两个预测共用一个参数

模型只有一个可训练的标量权重 `w`。第一条样本把 `2w` 限制到 `[-1, 1]`，得到预测 `p₁`。第二条样本取 `w` 与 `0.75` 中较小的值，得到预测 `p₂`。真实标签分别是 `0.8` 和 `1.0`。Loss 是两个平方误差的平均。这样一张小图就能同时观察两条路径如何贡献梯度。

```text
       ┌─ 2w → clip(−1, 1) → p₁ → (p₁ − 0.8)² ─┐
w ─────┤                                        ├─ mean → loss
       └─  w → min(0.75)   → p₂ → (p₂ − 1.0)² ─┘
```

```reader-demo
backward-paths
```

从 `w = 0.25` 开始，两个预测是 `0.50` 和 `0.25`，loss 是 `0.32625`。Backward 从 `∂loss/∂loss = 1` 开始。`mean` 先给每个平方误差 `0.5` 的梯度。第一条路径接着乘平方误差对预测值的局部导数 `2 × (0.50 − 0.8)`，乘 `clip` 的局部斜率 `1`，再乘 `2w` 的局部斜率 `2`。所以第一条路径给 `w` 的贡献是 `−0.60`。第二条路径给 `w` 的贡献是 `−0.75`。**同一个参数被用两次，两个贡献相加**，得到 `w.grad = −1.35`。

用学习率 `0.1` 做一次 SGD，`w` 从 `0.25` 变为 `0.385`。用新参数重新 forward，loss 变为 `0.1895625`。切换到 `w = 0.60`，第一条路径的预测仍然有误差，但 `2w = 1.2` 落在 `clip` 的平坦区域，该路径传回 `w` 的梯度为零。再切到 `w = 0.90`，第二条路径也选了常量上限，两条贡献都是零，SGD 这一步就停在原地。

## 2 · 常见运算怎样传梯度

下面的 `g` 表示从下游传到这个运算输出的梯度。反传给某个输入的是 `g × 该输入的局部导数`。有多条路径指向同一个输入时，把贡献累加。

| 运算 | 反传给输入的规则 |
| --- | --- |
| `x + y`、`x − y` | 加法给两边各 `g`，减法给 `x` 传 `g`、给 `y` 传 `−g`。广播复用的元素会累加各处贡献。 |
| `x * y`、`x / y` | 乘法给两边分别传 `gy`、`gx`。实数除法在 `y ≠ 0` 时分别传 `g/y`、`−gx/y²`。 |
| `x**2`、`exp(x)`、`log(x)` | 分别乘局部导数 `2x`、`exp(x)`、`1/x`。实数 `log` 要求 `x > 0`。 |
| `X @ W` | 若输入是实数二维矩阵，上游梯度为 `G`，则给 `X` 传 `G @ W.T`，给 `W` 传 `X.T @ G`。 |
| `sum`、`mean` | `sum` 给参与同一个求和的元素各传 `g`。`mean` 各传 `g/N`，`N` 是这一组参与平均的元素数。 |
| `minimum`、`maximum` | 较小值或较大值唯一时，把 `g` 传给被选中的值。相等时由具体算子决定怎样分配。 |
| `clip(x, lo, hi)` | 固定上下界时，区间内传 `g`，区间外传 `0`。正好在边界处，左右斜率不同，要看框架的约定。 |
| `relu`、`tanh`、`sigmoid` | ReLU 正半轴乘 `1`、负半轴乘 `0`。若输出为 `y`，tanh 乘 `1 − y²`，sigmoid 乘 `y(1 − y)`。 |
| `reshape`、`transpose`、`gather` | 梯度按 forward 时的元素对应关系传回。用固定整数索引选择值时，梯度传给被取出的输入值，重复使用的值会累加。 |
| `softmax`、`log_softmax`、`cross_entropy` | 都有对 logits 的反传规则。Softmax 的各输出互相耦合。分类训练通常直接把 logits 交给 `cross_entropy`。 |
| `argmax`、`round` | `argmax` 返回整数索引，索引对分数没有可用的梯度路径。`round` 在台阶内斜率为零，在跳变处没有普通导数。 |

例子里用的是 `clip` 的普通可导点。对于并列最小值，PyTorch 的 `amin` 会在并列元素间分配梯度，带 `dim` 的 `min` 则把梯度传到它返回的索引所指的元素。写训练代码时，遇到并列或边界就查具体 API，再用两三个标量做对拍。 [PyTorch · amin 与 min](https://docs.pytorch.org/docs/stable/generated/torch.amin.html) · [PyTorch · 不可导点的处理](https://docs.pytorch.org/docs/stable/notes/autograd.html#gradients-for-non-differentiable-functions)

还有一个独立的问题是**计算有没有被 autograd 记录**。`no_grad` 内的运算不进反传图，`detach()` 会切断当前值和它上游的连接。`.item()` 则取出 Python 数值，用于日志时很方便。若用这个数重新构造 tensor 当 loss，原来的梯度路径就丢了。对已经清空梯度的参数，梯度是零表示走通了某条路径但局部变化为零，`.grad is None` 则可以提示这次 backward 没有给该参数积累梯度。 [PyTorch · Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html)

## 3 · 用标量写一个最小 autograd

Karpathy 的 micrograd 用标量计算图讲解 reverse-mode autodiff（从输出向输入反传）。这里采用同样的教学路径。每个节点保存 forward 的值和通向各输入的**边**，边上记录局部导数。Backward 先把节点排成输入在前、输出在后的顺序，再倒着走，保证共享节点先收齐两条路径的梯度。下面的代码可直接用 Python 运行。 [micrograd 源码](https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py) · [Karpathy 原课](https://www.youtube.com/watch?v=VMj-3S1tku0)

```python
class Scalar:
    def __init__(self, value, *edges):
        self.value = value
        self.edges = edges          # 每条边是 (输入节点, 输出对该输入的局部导数)
        self.grad = 0.0


def add(a, b):
    return Scalar(a.value + b.value, (a, 1.0), (b, 1.0))


def multiply(a, b):
    return Scalar(a.value * b.value, (a, b.value), (b, a.value))


def mean(a, b):
    return Scalar((a.value + b.value) / 2, (a, 0.5), (b, 0.5))


def minimum(a, b):
    # 这个小引擎在并列时平分梯度。
    if a.value == b.value:
        return Scalar(a.value, (a, 0.5), (b, 0.5))
    selected = a if a.value < b.value else b
    return Scalar(selected.value, (selected, 1.0))


def clip(x, lo, hi):
    value = min(max(x.value, lo), hi)
    slope = float(lo < x.value < hi)  # 这个小引擎在边界选择 0
    return Scalar(value, (x, slope))


def backward(loss):
    order = []
    seen = set()

    def visit(node):
        if node in seen:
            return
        seen.add(node)
        for parent, _ in node.edges:
            visit(parent)
        order.append(node)            # 输入先入列表，输出后入列表

    visit(loss)
    for node in order:
        node.grad = 0.0               # 这次演示从空梯度开始
    loss.grad = 1.0
    for node in reversed(order):
        for parent, slope in node.edges:
            parent.grad += node.grad * slope


def regression_loss(w):
    p1 = clip(multiply(Scalar(2.0), w), -1.0, 1.0)
    p2 = minimum(w, Scalar(0.75))
    e1 = add(p1, Scalar(-0.8))
    e2 = add(p2, Scalar(-1.0))
    return mean(multiply(e1, e1), multiply(e2, e2))


w = Scalar(0.25)
loss = regression_loss(w)
backward(loss)
w_new = Scalar(w.value - 0.1 * w.grad)
print(loss.value, w.grad, w_new.value, regression_loss(w_new).value)
# 约 0.32625, -1.35, 0.385, 0.1895625
```

注意 `multiply(e1, e1)` 有两条边指向同一个节点。`seen` 确保节点只进一次访问顺序，`+=` 确保两条边都贡献梯度。对 `w` 也是同一个道理。对拍时，可以在离折点较远的 `w` 上试 `(L(w+ε) − L(w−ε))/(2ε)`。这个有限差分用来检查方向和数值，backward 本身用的是链式法则。

把同一张图写成 PyTorch 训练代码，`backward()` 和 `optimizer.step()` 的分工就能直接对上。安装了 PyTorch 后可运行这一段。

```python
import torch

w = torch.nn.Parameter(torch.tensor(0.25))
optimizer = torch.optim.SGD([w], lr=0.1)
optimizer.zero_grad()

p1 = torch.clamp(2 * w, -1.0, 1.0)
p2 = torch.minimum(w, w.new_tensor(0.75))
loss = torch.stack(((p1 - 0.8) ** 2, (p2 - 1.0) ** 2)).mean()
loss.backward()     # 填写 w.grad
optimizer.step()    # 用 w.grad 更新 w
```

## 4 · 回到 TD3

**Actor update** 的 `loss_a = −Q1(S, Actor(S)).mean()` 会经过 `mean`，再穿过 Critic 对动作的计算，回到 Actor 参数。固定 Critic 权重时，Critic 仍然可以对输入动作求导。

**TD target** 的 `min(Q1_target, Q2_target)` 和 action clip 在 `no_grad` 里构造训练标签 `Y`。Critic backward 沿 `Q_online(S, A) → (Q_online − Y)² → mean` 这条图反传，`Y` 在这次更新中作为固定标签。 [TD3 完整训练流程](../t3-sim-to-rl/td3/README.md#2--完整训练流程)

## 5 · 一次预测和一次实现

- **判读。** 看过 `w = 0.25` 的完整演示后，先预测 `w = 0.60` 时两条路径对 `w` 的梯度各是零还是非零。用一句话说是哪一个运算决定的，再切到 Backward 核对。
- **亲手实现。** 把 `mean(a, b)` 改成任意长度的平均。保留其余代码，先用同一个节点计算 `mean(x, x, x)`，再调用 `backward`。验收是能解释为什么 `x.grad` 等于 `1`，并能用有限差分核对。
- **AI claim audit。** 「用了 `clip`，optimizer 就无法更新这个参数。」用 `w = 0.60` 的两条路径给出一个反例，说明什么条件才会让本例的总梯度变成零。
