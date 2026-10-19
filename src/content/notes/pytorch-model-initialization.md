---
title: '模型初始化'
description: '理解 Xavier、Kaiming 与 gain 的公式，按模块初始化参数，并避免重复遍历和破坏梯度记录。'
pubDate: 2026-08-08
updatedDate: 2026-10-19
order: 6
category: 'PyTorch'
tags: ['模型初始化']
---

## 模型初始化

初始化影响信号和梯度在网络中的传播。合适的初始化有助于训练，但不保证模型一定更准确或更快收敛。PyTorch 的 Linear、Conv 等层本身已有默认初始化，只有需要不同策略时才覆盖。

### torch.nn.init内容

| API | 用途 |
|---|---|
| `uniform_` / `normal_` | 均匀 / 正态分布 |
| `constant_` / `ones_` / `zeros_` | 指定常量 / 全 1 / 全 0 |
| `eye_` | 二维单位矩阵 |
| `dirac_` | 卷积中的恒等映射形式，需满足通道与分组约束 |
| `xavier_uniform_` / `xavier_normal_` | 同时考虑 fan_in 与 fan_out |
| `kaiming_uniform_` / `kaiming_normal_` | 根据非线性与 fan 模式缩放 |
| `orthogonal_` / `sparse_` | 正交 / 稀疏初始化 |
| `calculate_gain` | 计算适合指定非线性的增益 |

带下划线的方法原地修改参数，并在不记录梯度的上下文中执行，直接传入 Parameter 即可，无需 `.data`。`calculate_gain` 返回数值，不修改参数。

常见的 gain 为：

| 非线性 | gain |
|---|---|
| Linear / Conv / Sigmoid | $1$ |
| Tanh | $5/3$ |
| ReLU | $\sqrt{2}$ |
| LeakyReLU，负斜率 $a$ | $\sqrt{2/(1+a^2)}$ |

LeakyReLU 的分母是整个 $1+a^2$，不能写成缺少括号的其他表达式。

### Xavier 与 Kaiming 的尺度

Xavier 正态初始化的标准差为：

$$
\mathrm{std}=g\sqrt{\frac{2}{fan_{in}+fan_{out}}}.
$$

Xavier 均匀初始化使用 $[-b,b]$，其中：

$$
b=g\sqrt{\frac{6}{fan_{in}+fan_{out}}}.
$$

Kaiming 正态初始化使用：

$$
\mathrm{std}=\frac{g}{\sqrt{fan_{mode}}}.
$$

`mode="fan_in"` 侧重保持前向传播尺度，`mode="fan_out"` 侧重反向传播尺度。正确的参数名是 `fan_in`，不是 `fan__in`。配合 ReLU 时可显式指定 `nonlinearity="relu"`；LeakyReLU 则还要传入匹配的负斜率 `a`。

对 `nn.Linear`，权重布局为 `[out_features,in_features]`，前向通常计算 `x @ weight.T`。如果自己保存的是用于 `x @ weight` 的 `[in,out]` 矩阵，计算 fan 时要考虑转置，不能直接照搬层权重的约定。

### torch.nn.init使用

```python
import torch
from torch import nn

torch.manual_seed(42)
conv = nn.Conv2d(1, 3, 3)
linear = nn.Linear(10, 1)
nn.init.kaiming_normal_(conv.weight, mode="fan_in", nonlinearity="relu")
nn.init.zeros_(conv.bias)
nn.init.normal_(linear.weight, mean=0.0, std=0.01)
nn.init.zeros_(linear.bias)
assert conv.weight.requires_grad
```

`normal_(tensor, 0.1)` 的第二个位置参数是 **mean**，不是标准差。建议使用关键字 `mean=` 和 `std=`，避免把想要的“小方差初始化”写成“均值为 0.1、标准差仍为 1”。查看权重可用 `weight.detach()`。

### 初始化函数的封装

`model.apply(fn)` 会递归访问每个子模块并调用 `fn`。因此下面的回调只处理**当前模块**，不要在回调内部再次遍历 `model.modules()`，否则可能重复初始化同一层。

```python
def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        if module.weight is not None:  # affine=False 时没有这两个参数
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


model = nn.Sequential(
    nn.Conv2d(1, 4, 3, padding=1),
    nn.BatchNorm2d(4),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(4, 2),
)
model.apply(initialize_weights)
assert model(torch.randn(2, 1, 8, 8)).shape == (2, 2)
assert torch.count_nonzero(model[0].weight) > 0
torch.testing.assert_close(model[1].weight, torch.ones(4))
```

这个示例中卷积接 ReLU，输出层产生未激活的 logits，因此分别选用 Kaiming 与 Xavier。它是一种明确的教学策略，不是所有模型都必须采用的默认规则。

### 哪些参数可以初始化为零

把同层隐藏单元的权重全部设为相同常数，常会使它们无法打破对称性；在某些网络中还会使梯度传播受阻。把权重统一设为 0.01 或 0.1 也不能普遍解决对称性问题。

不过，**偏置初始化为零通常是合理的**，BatchNorm 的仿射缩放常设为 1、偏置设为 0。不能笼统地说“任何参数初始化为零都会导致梯度消失”。训练中也不要每个 epoch 重复初始化，否则会覆盖已经学到的参数。

## 参考

- [PyTorch：torch.nn.init](https://docs.pytorch.org/docs/stable/nn.init.html)
