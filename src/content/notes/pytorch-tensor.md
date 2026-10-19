---
title: '张量'
description: '理解张量的形状、创建方式、索引、视图、复制和广播规则。'
pubDate: 2026-08-07
updatedDate: 2026-10-19
order: 2
category: 'PyTorch'
tags: ['pytorch']
---

## 张量

PyTorch 的 Tensor 是具有 shape、dtype 和 device 的多维数值容器，不支持把字符串作为张量元素。这里的“维数”指轴的数量，不等同于矩阵秩，也不能仅凭轴数确定数据的语义。

### 简介

| 数据 | 常见 shape | 说明 |
|---|---|---|
| 标量 | `[]` | 0 维，例如单个损失值 |
| 特征向量 | `[D]` | 1 维 |
| 一批特征 | `[N,D]` | 2 维 |
| 单张彩色图片 | `[C,H,W]` | PyTorch 卷积常用通道在前的布局 |
| 一批图片 | `[N,C,H,W]` | 4 维 |
| 一批视频 | `[N,C,T,H,W]` | 一种常见的 5 维约定 |

时间序列也可能使用 `[N,T,D]`，图像库则常使用 `[H,W,C]`。这些是布局约定，不是“3 维一定是时间序列、4 维一定是图像”的规则。

### 创建张量

```python
import numpy as np
import torch

torch.manual_seed(42)
x = torch.rand(4, 3)                 # [0,1) 均匀采样
normal = torch.randn(4, 3)           # 标准正态采样
zeros = torch.zeros(4, 3, dtype=torch.long)
data = torch.tensor([5.5, 3.0])
ones = data.new_ones((4, 3))         # 默认继承 data 的 dtype、device
double_ones = data.new_ones((4, 3), dtype=torch.float64)
assert x.shape == (4, 3) and zeros.dtype == torch.long
assert double_ones.dtype == torch.float64
```

`torch.Size` 是 tuple 的子类，可以索引各轴长度。`new_ones` 等方法也可以通过显式参数覆盖继承的 dtype 或 device。

#### 全0矩阵的构建

```python
source = torch.ones(2, 3)
new_zeros = torch.zeros_like(source)  # 新建，不修改 source
assert source.sum().item() == 6
source.zero_()                       # Tensor 方法：原地清零
assert source.sum().item() == 0
```

正确写法是 `source.zero_()`，不是 `torch.zero_()`。带下划线的方法通常会原地修改张量，使用自动求导时还需遵守计算图的修改约束。

#### 常见的构造Tensor的方法

| 函数 | 功能 |
|---|---|
| `torch.tensor(data)` | 根据数据创建张量，复制数据 |
| `torch.empty(shape)` | 分配未初始化的存储，读取前必须写入 |
| `torch.ones/zeros(shape)` | 全 1 / 全 0 |
| `torch.eye(n)` | 单位矩阵 |
| `torch.arange(start,end,step)` | 不包含右端点的等差序列 |
| `torch.linspace(start,end,steps)` | 包含两端点，共 `steps` 个点 |
| `torch.rand/randn(shape)` | 均匀 / 标准正态随机数 |
| `torch.randperm(n)` | `0` 到 `n-1` 的随机排列 |

优先使用语义明确的工厂函数，避免用 `torch.Tensor(...)` 同时表达“传入数据”和“分配指定形状”。

### 张量的操作

#### 加法操作

```python
x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
y = torch.ones_like(x)
torch.testing.assert_close(x + y, torch.add(x, y))
y.add_(x)                           # 修改 y，而不是创建独立的新结果
assert y[0, 0].item() == 1
```

`*` 通常是逐元素乘法，二维矩阵乘法使用 `@` 或 `torch.mm`，二者不能混淆。

#### 索引操作

基本索引通常返回共享存储的视图；张量索引、布尔索引等高级索引读取通常返回副本。不能把“所有索引结果都共享内存”当成规则。

```python
x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
row = x[0, :]                       # 基本索引：视图
row.add_(10)
assert x[0, 0].item() == 10

selected = x[torch.tensor([0])]     # 高级索引读取：副本
selected.zero_()
assert x[0, 0].item() == 10

independent = x[0, :].clone()
independent.zero_()
assert x[0, 0].item() == 10
```

通过高级索引进行赋值，例如 `x[mask] = 0`，仍然会修改 `x`；这与“读取时返回副本”并不矛盾。复制 Tensor 应使用 `clone()`，而不是 NumPy 风格的 `.copy()`。

#### 维度变换

`x.view(...)` 返回另一个共享存储的 Tensor，不会直接改写 `x` 的 shape，也不是“同一个 Python 对象”。它要求尺寸与步幅兼容。

```python
x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
flat = x.view(12)
flat[0] = -1
assert x.shape == (3, 4) and x[0, 0].item() == -1

transposed = x.transpose(0, 1)
reshaped = transposed.reshape(12)    # 本例需要复制，不能直接 view(12)
independent = transposed.reshape(12).clone()
assert reshaped.shape == (12,)
```

`reshape` 是正常、受支持的接口：能返回视图时返回视图，否则可能复制。不要依赖它一定共享存储或一定复制。需要独立存储时显式使用 `clone()`；需要终止梯度联系并复制时使用 `detach().clone()`。仅 `detach()` 仍共享存储。

`clone()` 通常保留梯度联系；它也可能保留非连续布局，因此不能笼统地保证任意 `x.clone().view(...)` 都成功。交换轴用 `transpose`（两个轴）或 `permute`（轴顺序），并不是单纯 reshape。

### 广播机制

广播从**尾部维度对齐**：对应长度必须相同，或其中一个为 1；缺失的前导维度视为 1。

```python
matrix = torch.arange(6, dtype=torch.float32).reshape(2, 3)
offset = torch.tensor([10., 20., 30.])
assert (matrix + offset).shape == (2, 3)

column = torch.ones(3, 1)
row = torch.arange(3, dtype=torch.float32)
assert (column + row).shape == (3, 3)
```

最后一个例子也说明了回归损失中的常见陷阱：预测为 `[N,1]`、标签为 `[N]` 时，二者可能广播成 `[N,N]`。应先核对 shape，再计算损失。

### 与 NumPy 交换数据

```python
array = np.array([1., 2., 3.], dtype=np.float32)
shared = torch.from_numpy(array)   # 支持的 CPU dtype 下共享存储
copied = torch.tensor(array)       # 复制
array[0] = 9
assert shared[0].item() == 9 and copied[0].item() == 1
```

常见的导出方式是 `tensor.detach().cpu().numpy()`。得到的数组可能仍共享 CPU 张量的存储；需要独立数组时再 `.copy()`。单元素张量可用 `.item()` 取 Python 数值，这一步不可微，读取 GPU 数值还可能触发同步。

## 参考

- [PyTorch：Tensor Views](https://docs.pytorch.org/docs/stable/tensor_view.html)
- [PyTorch：Broadcasting semantics](https://docs.pytorch.org/docs/stable/notes/broadcasting.html)
