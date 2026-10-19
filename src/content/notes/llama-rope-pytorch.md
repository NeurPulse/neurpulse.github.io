---
title: 'LLaMA（10）：用 PyTorch 从零实现 RoPE'
description: '以课程 ROPE.ipynb 为主线，完成频率预计算、复数旋转、广播与 shape 检查，并验证范数和相对位置性质。'
pubDate: 2026-09-11
updatedDate: 2026-10-19
category: 'LLaMA'
tags: ['LLaMA 1', 'RoPE', 'PyTorch', '从零实现', '复数']
order: 10
---

## 学习目标

本文以 `ROPE.ipynb` 为代码主线，把手写推导映射为可以复用的 PyTorch 实现。完成后应能解释 `view(..., 2)`、`torch.polar`、`view_as_complex` 和广播分别对应哪一步数学操作。

## 先修知识

- [LLaMA（7）：RoPE 如何捕获向量之间的相对位置](/notes/llama-rope-relative-position/)
- [PyTorch：张量基础](/notes/pytorch-tensor/)

## 1. Notebook 的最小实验

课程代码先创建二维张量：

```python
seq_len = 10
dim = 128
q = torch.randn(seq_len, dim)
k = torch.randn(seq_len, dim)

q_pairs = q.view(seq_len, dim // 2, 2)
k_pairs = k.view(seq_len, dim // 2, 2)
```

`view(..., 2)` 不是“把向量分成两拨分别做 sin、cos”，而是把每两个相邻实数解释为一个二维平面坐标：

```text
[x0, x1, x2, x3, ...]
  ↓
[(x0, x1), (x2, x3), ...]
```

这要求 `dim` 为偶数。

## 2. 预计算旋转因子

```python
import torch


def precompute_freqs_cis(
    seq_len: int,
    head_dim: int,
    base: float = 10000.0,
    device=None,
):
    if head_dim % 2 != 0:
        raise ValueError("head_dim 必须是偶数")

    pair_index = torch.arange(0, head_dim, 2, device=device).float()
    inv_freq = base ** (-pair_index / head_dim)
    positions = torch.arange(seq_len, device=device).float()
    angles = torch.outer(positions, inv_freq)

    # 模长为 1、相角为 angles 的复数：cos(angle) + i sin(angle)
    return torch.polar(torch.ones_like(angles), angles)
```

形状变化：

```text
pair_index: [head_dim / 2]
inv_freq:   [head_dim / 2]
positions:  [seq_len]
angles:     [seq_len, head_dim / 2]
freqs_cis:  [seq_len, head_dim / 2]  # complex
```

## 3. 对 Q、K 应用旋转

下面使用常见的 `[batch, heads, seq_len, head_dim]` 布局：

```python
def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """x: [batch, heads, seq_len, head_dim]"""
    if x.ndim != 4 or not x.is_floating_point():
        raise ValueError("x 必须是四维浮点张量")
    if freqs_cis.shape != (x.size(-2), x.size(-1) // 2) or freqs_cis.device != x.device:
        raise ValueError("旋转表长度、维度和设备必须与输入匹配")
    if x.shape[-1] % 2 != 0:
        raise ValueError("head_dim 必须是偶数")

    original_dtype = x.dtype

    # view_as_complex 需要最后一维长度为 2；为兼容混合精度先转 float32。
    work = x if x.dtype == torch.float64 else x.float()
    x_pairs = work.reshape(*x.shape[:-1], -1, 2).contiguous()
    x_complex = torch.view_as_complex(x_pairs)

    # [seq_len, head_dim / 2] -> [1, 1, seq_len, head_dim / 2]
    rotation = freqs_cis[None, None, :, :]
    x_rotated = x_complex * rotation

    x_real = torch.view_as_real(x_rotated)
    return x_real.flatten(-2).to(original_dtype)
```

调用方式：

```python
batch, heads, seq_len, head_dim = 2, 4, 10, 128
q = torch.randn(batch, heads, seq_len, head_dim)
k = torch.randn(batch, heads, seq_len, head_dim)

freqs_cis = precompute_freqs_cis(seq_len, head_dim, device=q.device)
q_rot = apply_rope(q, freqs_cis)
k_rot = apply_rope(k, freqs_cis)
```

## 4. 为什么复数乘法就是旋转

将二维对 $(x,y)$ 看成 $z=x+iy$，旋转因子为：

$$
e^{im\theta}=\cos(m\theta)+i\sin(m\theta)
$$

于是：

$$
z'=z e^{im\theta}
$$

实部与虚部展开后正好得到二维旋转矩阵。

## 5. 两个必须通过的测试

### 5.1 旋转保持范数

```python
torch.testing.assert_close(
    q.norm(dim=-1),
    q_rot.norm(dim=-1),
    rtol=1e-5,
    atol=1e-5,
)
```

### 5.2 共同平移不改变相对位置点积

```python
head_dim = 8
freqs = precompute_freqs_cis(seq_len=16, head_dim=head_dim)

q0 = torch.randn(1, 1, 1, head_dim)
k0 = torch.randn(1, 1, 1, head_dim)

def rotate_at(x, position):
    return apply_rope(x, freqs[position:position + 1])

score_2_5 = (rotate_at(q0, 2) * rotate_at(k0, 5)).sum()
score_7_10 = (rotate_at(q0, 7) * rotate_at(k0, 10)).sum()

torch.testing.assert_close(score_2_5, score_7_10)
```

两组位置的差都为 3，所以分数相同。

## 6. 与课程 Notebook 对照

课程实现的核心思路是正确的：

```text
Q/K → 二维配对 → 转为复数 → 乘单位复数旋转因子 → 转回实数
```

整理成工程函数时需要补充：

- 显式检查 `head_dim` 为偶数；
- 明确 batch、head 和 sequence 的广播位置；
- 混合精度输入先转 `float32` 做复数运算，再转回原 dtype；
- Q 与 K 必须分别从自己的张量构造，不能误把 K 写成 Q；
- 缓存频率表，避免每一层、每一步重复计算。

## 常见误区

### 误区一：`view_as_complex` 会自动复制数据

它返回视图，并要求最后一维长度为 2，同时对 dtype 和 stride 有要求。

### 误区二：对整个 `dim` 使用同一个频率

每一个二维对都有自己的频率。

### 误区三：旋转 Value

标准 LLaMA 风格实现通常只旋转 Q 和 K。

## 一句话总结

RoPE 的 PyTorch 实现本质是：把相邻两维看成复数，用位置和频率生成单位复数相位，乘法完成旋转，再恢复原张量形状。

## 复习卡片

- 二维配对：`[..., head_dim / 2, 2]`；
- 角度表：`outer(position, inv_freq)`；
- 旋转因子：`torch.polar(1, angle)`；
- 复数旋转：`x_complex * freqs_cis`；
- 必测性质：范数保持、共同平移不变。

## 自测问题

1. `view(..., 2)` 在数学上代表什么？
2. 为什么 `freqs_cis` 的模长必须为 1？
3. 广播时哪一维对应位置，哪一维对应频率？
4. 混合精度下为什么先转 `float32`？

## 下一篇

[LLaMA（11）：用代码验证 RoPE 的远程衰减](/notes/llama-rope-decay-experiment/)

## 参考与推荐阅读

- 本地课程文件 `ROPE.ipynb`：本文代码主线，不随博客发布。
- [PyTorch `view_as_complex`](https://docs.pytorch.org/docs/stable/generated/torch.view_as_complex.html)
- [RoFormer 原论文](https://arxiv.org/abs/2104.09864)
