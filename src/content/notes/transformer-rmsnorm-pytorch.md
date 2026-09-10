---
title: 'LLaMA（5）：从零实现 RMSNorm'
description: '以课程 RMSNorm.ipynb 为代码主线，从最小实现出发补齐 epsilon、混合精度、正确验证和 PyTorch 对照测试。'
pubDate: 2026-09-10
category: 'LLaMA'
tags: ['RMSNorm', 'PyTorch', '从零实现', '混合精度']
order: 5
---

## 学习目标

本文以课程 `RMSNorm.ipynb` 为主要代码来源。学完后应能独立写出 RMSNorm，解释为什么构造函数传入 `dim`，正确验证输出，并知道教学实现距离训练级实现还差哪些数值稳定细节。

## 先修知识

- [RMSNorm 原理及其与 LayerNorm 的区别](/notes/transformer-rmsnorm/)
- [Pre-Norm 与 Post-Norm](/notes/transformer-pre-norm-post-norm/)
- [PyTorch 模型构建](/notes/pytorch-model-construction/)
- [PyTorch 自动求导](/notes/pytorch-autograd/)

## 1. 公式到代码的对应关系

RMSNorm 公式为：

$$
\operatorname{RMS}(x)
=
\sqrt{\frac{1}{D}\sum_{i=1}^{D}x_i^2+\epsilon}
$$

$$
y=\frac{x}{\operatorname{RMS}(x)}\odot\gamma
$$

| 数学符号 | Notebook/代码 |
|---|---|
| $x$ | 输入张量 `x` |
| $D$ | `dim`，最后一个隐藏维度 |
| $\frac{1}{D}\sum x_i^2$ | `torch.mean(x ** 2, dim=-1, keepdim=True)` |
| $\operatorname{RMS}(x)$ | `torch.sqrt(...)` |
| $\gamma$ | `self.scale` 或常见实现中的 `self.weight` |
| $y$ | `forward` 的返回值 |

## 2. Notebook 中的最小实现

课程代码的核心非常清晰：

```python
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(
            torch.mean(x ** 2, dim=-1, keepdim=True)
        )
        return x / rms * self.scale
```

这段代码完整对应了 `平方 → 均值 → 开根 → 相除 → 缩放` 五个步骤，适合第一次理解 RMSNorm。

## 3. 为什么构造函数传入 `dim`

输入通常是：

```text
x: [batch_size, seq_len, dim]
```

RMS 值由输入动态计算，不需要为 batch 或序列位置保存参数；可学习缩放却要为最后一个特征各保存一个值：

```text
self.scale: [dim]
```

PyTorch 广播时：

```text
x / rms:    [B, T, D]
scale:            [D]
output:      [B, T, D]
```

因此 `RMSNorm(dim)` 传入的是需要归一化的特征维数，不是说 `nn.Module` 只能接收张量。`nn.Module` 子类的构造函数参数完全由模块设计者决定。

## 4. Notebook 测试中需要纠正的两个认识

Notebook 打印了：

```python
print(output.mean(dim=-1))
print(output.std(dim=-1))
```

这些值可以用于观察，但不应期待 `mean≈0`、`std≈1`，因为 RMSNorm 没有减均值。更合适的验证量是均方根：

```python
output_rms = torch.sqrt(
    output.pow(2).mean(dim=-1)
)
print(output_rms)
```

当初始 `scale=1` 且忽略 $\epsilon$ 时，`output_rms` 应接近 1。

另一个需要修正的说法是“不是 1 就不是标准分布”。RMSNorm 不要求输出服从标准正态分布，`scale` 也正是允许模型学习不同特征尺度的参数。

## 5. 为什么必须加 epsilon

如果输入全为 0：

```python
x = torch.zeros(2, 8, 128)
```

Notebook 最小实现会得到 `rms=0`，除法产生 `NaN`。因此应写成：

$$
\sqrt{\operatorname{mean}(x^2)+\epsilon}
$$

注意 $\epsilon$ 通常放在根号内部。

## 6. 推荐的训练级实现

下面保持 Notebook 的类结构，同时补充 $\epsilon$、float32 归一化和 dtype 恢复：

```python
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """LLaMA 风格 RMSNorm，沿最后一个维度归一化。"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(mean_square + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 低精度训练时，用 float32 计算平方均值更稳定。
        normalized = self._norm(x.float()).to(dtype=x.dtype)
        return normalized * self.weight
```

`torch.rsqrt(z)` 等价于 $1/\sqrt{z}$，可以直接与 `x` 相乘。

## 7. 为什么先转 float32

若输入为 float16 或 bfloat16，平方操作更容易发生上溢、下溢或精度损失。归一化统计量通常用 float32 计算：

```text
低精度输入
→ float32 计算 mean_square 与 rsqrt
→ 转回输入 dtype
→ 乘可学习 weight
```

这并不意味着整个模型都使用 float32；只是对数值敏感的小段计算提高精度。

## 8. 最小正确性测试

### 8.1 Shape 和均方根

```python
torch.manual_seed(42)

B, T, D = 2, 8, 128
x = torch.randn(B, T, D)
norm = RMSNorm(D)
output = norm(x)

assert output.shape == x.shape

# 初始 weight=1 时，均方根应接近 1。
output_rms = output.pow(2).mean(dim=-1).sqrt()
assert torch.allclose(
    output_rms,
    torch.ones_like(output_rms),
    atol=1e-4,
    rtol=1e-4,
)
```

由于加入 $\epsilon$，极小输入下不会严格等于 1；普通随机输入的误差通常很小。

### 8.2 零输入不能出现 NaN

```python
zero = torch.zeros(B, T, D)
zero_output = norm(zero)

assert torch.isfinite(zero_output).all()
assert torch.equal(zero_output, torch.zeros_like(zero_output))
```

### 8.3 梯度能够传播

```python
x = torch.randn(B, T, D, requires_grad=True)
loss = norm(x).square().mean()
loss.backward()

assert x.grad is not None
assert norm.weight.grad is not None
assert torch.isfinite(x.grad).all()
assert torch.isfinite(norm.weight.grad).all()
```

## 9. 与手工参考结果对照

```python
def rms_norm_reference(x, weight, eps):
    rms = torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return (x.float() / rms).to(x.dtype) * weight


x = torch.randn(2, 3, 16)
norm = RMSNorm(16, eps=1e-6)

actual = norm(x)
expected = rms_norm_reference(x, norm.weight, norm.eps)

assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
```

如果当前 PyTorch 版本包含 `nn.RMSNorm`，还可以在统一 $\epsilon$ 和 `weight` 后与官方模块交叉验证；官方模块用于核对，不替代对手写公式的理解。

## 10. 接入 LLaMA Block

```python
class LlamaBlock(nn.Module):
    def __init__(self, dim, attention, ffn, eps=1e-6):
        super().__init__()
        self.attention_norm = RMSNorm(dim, eps)
        self.ffn_norm = RMSNorm(dim, eps)
        self.attention = attention
        self.ffn = ffn

    def forward(self, x, mask=None):
        h = x + self.attention(self.attention_norm(x), mask=mask)
        out = h + self.ffn(self.ffn_norm(h))
        return out
```

Attention 与 FFN 前使用不同的 RMSNorm 参数，不能只创建一个实例到处共享。

## 11. 常见错误清单

### 错误一：把 `mean(x ** 2)` 写成 `mean(x) ** 2`

平方的均值不等于均值的平方，后者还可能因正负抵消而接近 0。

### 错误二：沿 batch 或序列维归一化

LLaMA RMSNorm 应沿 `dim=-1` 处理每个 token 的隐藏特征。

### 错误三：忘记 `keepdim=True`

会使 `[B,T]` 难以直接广播回 `[B,T,D]`。

### 错误四：期待输出均值为 0

RMSNorm 没有中心化；应该检查均方根。

### 错误五：没有 epsilon 或把 epsilon 放在除法外

正确目标是稳定分母 `sqrt(mean_square + eps)`。

### 错误六：用同一个 Norm 参数服务所有层

每个 Block 的 Attention 与 FFN 前通常各有独立 `weight`。

## 12. 实践建议

- 教学阶段先实现 Notebook 简版，理解五步计算；
- 训练阶段使用带 $\epsilon$ 与 float32 统计量的版本；
- 不要只测随机输入，还要测零输入、低精度和反向传播；
- 性能敏感时优先使用框架或推理引擎的融合 RMSNorm 内核；
- `eps` 必须与目标模型配置保持一致，加载权重时不要凭经验随意修改。

## 一句话总结

RMSNorm 的代码核心只有“平方均值—倒数平方根—缩放”，但一个可靠实现还必须正确处理最后一维、$\epsilon$、广播、低精度统计和验证指标。

## 复习卡片

- 主线代码：课程 `RMSNorm.ipynb`；
- 参数 shape：`weight [D]`；
- 统计量 shape：`[B,T,1]`；
- 核心：`x * rsqrt(mean(x²) + eps)`；
- 正确验证：均方根约为 1，不是均值 0；
- 低精度：统计量转 float32；
- 必测：shape、零输入、对照结果、反向传播。

## 自测问题

1. 为什么 `RMSNorm(dim)` 只需要隐藏维度？
2. `keepdim=True` 对广播有什么作用？
3. 为什么 RMSNorm 的输出标准差不一定等于 1？
4. 为什么混合精度下建议用 float32 计算平方均值？
5. 如何设计一个测试发现实现遗漏了 $\epsilon$？

## 参考与推荐阅读

- 本地课程代码：`E:/BrotherDingOfLargeModel/Daily Study Schedule/level_2/code/RMSNorm.ipynb`。
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)：公式、性质和原始实验。
- [PyTorch `nn.RMSNorm`](https://docs.pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html)：框架实现的接口说明。
- [LLaMA 1 论文](https://arxiv.org/abs/2302.13971)：RMSNorm 在 LLaMA 中的使用位置。
