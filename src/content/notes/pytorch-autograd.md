---
title: '自动求导'
description: '理解计算图、叶子张量、梯度累积与向量—雅可比积，正确使用 detach 和 no_grad。'
pubDate: 2026-08-07
updatedDate: 2026-10-19
order: 3
category: 'PyTorch'
tags: ['梯度']
---

## 先修知识

- [PyTorch Tensor](/notes/pytorch-tensor/)
- [反向传播](/notes/deep-learning-backpropagation/)

## 自动求导

autograd 根据实际执行的、可微的张量运算构建动态计算图。本文以实数浮点张量为例；整数张量不能直接设置 `requires_grad=True`，也并非所有操作都有可用的梯度。

### Autograd简介

```python
import torch

x = torch.ones(2, 2, requires_grad=True)
y = x.square()
z = 3 * y.square()
out = z.mean()
assert x.is_leaf and x.grad_fn is None
assert not y.is_leaf and y.grad_fn is not None
```

`requires_grad=True` 表示需要对该张量求导。在普通梯度模式下，涉及它的可微运算会被记录。叶子张量通常由用户直接创建；`grad_fn=None` 也可能出现在无需求导、已 detach 或在 no_grad 中得到的结果上，不能仅据此判断张量是否由用户手动创建。

默认情况下，`.backward()` 把梯度累积到需要梯度的叶子张量的 `.grad`。非叶子结果如 `y` 不默认保存 `.grad`；确实需要观察时，可在反向传播前调用 `y.retain_grad()`。

### 梯度

上述计算可以写为：

$$
L=\frac{3}{N}\sum_{i=1}^{N}x_i^4,
\qquad \frac{\partial L}{\partial x_i}=\frac{12}{N}x_i^3.
$$

这里 $N=4$，所有 $x_i=1$，所以每个梯度为 3。

```python
out.backward()
torch.testing.assert_close(x.grad, torch.full_like(x, 3.0))

# 新建一次前向计算，梯度会继续累积。
x.sum().backward()
torch.testing.assert_close(x.grad, torch.full_like(x, 4.0))

x.grad = None
x.sum().backward()
torch.testing.assert_close(x.grad, torch.ones_like(x))
```

训练时通常使用 `optimizer.zero_grad()`，在新的前向与反向传播之前清理梯度。计算图保存的中间量通常在一次反向传播后释放；不要把“梯度可累积”误解成“可无限次对同一张旧计算图调用 backward”。

### 向量输出与雅可比矩阵

对于 $y=f(x)\in\mathbb{R}^{m}$、$x\in\mathbb{R}^{n}$，雅可比矩阵为 $J_{ij}=\partial y_i/\partial x_j$。若传入上游梯度列向量 $v\in\mathbb{R}^{m}$，反向传播得到：

$$
\nabla_x(v^T y)=J^T v.
$$

这叫向量—雅可比积的反向模式计算；用行向量记号则写为 $v^T J$。它不是直接返回完整的雅可比矩阵，也不应与前向模式的 $Jv$ 混淆。

```python
x = torch.tensor([1., 2., 3.], requires_grad=True)
y = x.square()
v = torch.tensor([0.1, 1.0, 0.001])
y.backward(v)
torch.testing.assert_close(x.grad, 2 * x.detach() * v)

J = torch.autograd.functional.jacobian(lambda t: t.square(), x)
torch.testing.assert_close(J, torch.diag(2 * x.detach()))
```

标量实数输出通常无需传入 `gradient`；多元素输出需要与输出形状匹配的上游梯度。PyTorch **可以**通过 Jacobian 接口计算完整雅可比，只是对大模型这样做通常非常昂贵。

### 使用 no_grad 与 detach

```python
x = torch.tensor([2.], requires_grad=True)
with torch.no_grad():
    prediction = x.square()
assert not prediction.requires_grad

detached = x.detach()               # 不再连接原计算图，但共享存储
snapshot = x.detach().clone()       # 同时复制存储
with torch.no_grad():
    x.add_(1)
assert detached.item() == 3 and snapshot.item() == 2
```

`no_grad` 关闭上下文中大部分运算的梯度记录，并不等于 `model.eval()`；后者切换 Dropout、BatchNorm 等层的行为。`detach()` 切断计算图联系，也不等于数据副本。

不要通过 `.data` 绕过 autograd 修改数值。这可能绕过必要的正确性检查，导致反向传播使用被篡改的数据。即使用 `no_grad()`，也不应随意修改已经参与前向、且反向传播还需要的值。

下面仅演示在完成反向传播之后手动更新参数；实际训练一般使用优化器：

```python
parameter = torch.tensor([2.], requires_grad=True)
loss = parameter.square().sum()
loss.backward()
with torch.no_grad():
    parameter -= 0.1 * parameter.grad
parameter.grad = None
torch.testing.assert_close(parameter.detach(), torch.tensor([1.6]))
```

## 参考

- [PyTorch：Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html)
- [PyTorch：torch.autograd.functional.jacobian](https://docs.pytorch.org/docs/stable/generated/torch.autograd.functional.jacobian.html)
