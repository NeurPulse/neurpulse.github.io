---
title: 'LLaMA（11）：用代码验证 RoPE 的远程衰减'
description: '修正并扩展 ROPE.ipynb 的实验，用位置核曲线、平移不变测试和距离分桶统计理解 RoPE 的振荡与远程衰减倾向。'
pubDate: 2026-09-11
updatedDate: 2026-10-19
category: 'LLaMA'
tags: ['LLaMA 1', 'RoPE', '远程衰减', 'PyTorch', '实验验证']
order: 11
---

## 学习目标

本文延续 `ROPE.ipynb` 的最后一个实验。目标是知道曲线究竟测量了什么、修正代码中的易错点，并用更可靠的统计方式观察 RoPE 的周期振荡与远程衰减倾向。

## 先修知识

- [LLaMA（9）：RoPE 对长距离和短距离依赖有什么区别？](/notes/llama-rope-long-range-decay/)
- [LLaMA（10）：用 PyTorch 从零实现 RoPE](/notes/llama-rope-pytorch/)

## 1. 原实验在测量什么

Notebook 把所有位置的 Q、K 都设为全 1，然后旋转并计算：

$$
S=\frac{Q'K'^\top}{\sqrt d}
$$

固定一个 Query 位置后绘制一整行分数。因为所有位置内容完全相同，曲线只反映 RoPE 位置核，不包含真实语义差异。

这很适合教学，但要注意它不是：

- 真实模型的注意力权重；
- softmax 后的概率；
- 加入因果 Mask 后的可见范围；
- 训练后 Q、K 的真实分布。

## 2. 先修正两个实现细节

课程代码中出现：

```python
K_split = Q.view(Q.shape[0], -1, 2)
```

通用实现应该写成：

```python
K_split = K.view(K.shape[0], -1, 2)
```

当 Q、K 都是全 1 时，这个错误不会改变结果，因此很隐蔽；换成随机 K 后才会暴露。

另外，如果 Query 位于 32，而模型使用因果注意力，那么位置 33 之后的 Key 都应该被 Mask。为了观察完整位置核可以暂时不加 Mask，但必须在图注中说明。

## 3. 可复现实验

```python
import torch
import matplotlib.pyplot as plt


def rope_kernel(max_tokens=10000, query_index=32, dim=128, base=10000.0):
    if dim % 2 != 0:
        raise ValueError("dim 必须为偶数")

    q = torch.ones(max_tokens, dim)
    k = torch.ones(max_tokens, dim)

    pair_index = torch.arange(0, dim, 2).float()
    inv_freq = base ** (-pair_index / dim)
    positions = torch.arange(max_tokens).float()
    angles = torch.outer(positions, inv_freq)
    rotation = torch.polar(torch.ones_like(angles), angles)

    q_complex = torch.view_as_complex(q.reshape(max_tokens, -1, 2))
    k_complex = torch.view_as_complex(k.reshape(max_tokens, -1, 2))

    q_rot = torch.view_as_real(q_complex * rotation).flatten(-2)
    k_rot = torch.view_as_real(k_complex * rotation).flatten(-2)

    # 只需要固定 Query 对全部 Key 的一行分数，无需构造 [T, T] 矩阵。
    query = q_rot[query_index]
    return (k_rot * query).sum(dim=-1) / dim**0.5


scores = rope_kernel()
plt.plot(scores.numpy())
plt.axvline(32, color="tomato", linestyle="--", label="query position")
plt.xlabel("key position")
plt.ylabel("position-only score")
plt.legend()
plt.show()
```

## 4. 应该如何读曲线

曲线通常具有三个特点：

1. Query 附近的相位差较小；
2. 随距离增加出现明显振荡，而不是单调下降；
3. 在部分距离区间，多频率相位抵消后可能观察到分数尺度下降，也可能出现反弹。

如果只看某一个远处峰值，很容易错误地得出“完全没有衰减”。分桶均值、上界或滑动平均只能描述选定内容、频率和观察区间的趋势，不能证明任意向量或任意长距离都必然衰减。

## 5. 按距离分桶观察

```python
query_index = 4096
scores = rope_kernel(max_tokens=8192, query_index=query_index, dim=128)

distance = (torch.arange(scores.numel()) - query_index).abs()
bucket_size = 128
bucket_id = distance // bucket_size

bucket_mean_abs = torch.zeros(int(bucket_id.max()) + 1)
for bucket in range(bucket_mean_abs.numel()):
    selected = scores[bucket_id == bucket]
    bucket_mean_abs[bucket] = selected.abs().mean()

plt.plot(torch.arange(bucket_mean_abs.numel()) * bucket_size,
         bucket_mean_abs.numpy())
plt.xlabel("absolute relative distance")
plt.ylabel("mean absolute score in bucket")
plt.show()
```

分桶仍不能代表训练后模型的注意力，但比单条振荡曲线更适合观察总体尺度变化。

## 6. 验证相对位置而不是绝对位置

```python
scores_a = rope_kernel(max_tokens=256, query_index=32, dim=128)
scores_b = rope_kernel(max_tokens=256, query_index=64, dim=128)

# 两组位置对的相对距离都为 10
torch.testing.assert_close(scores_a[42], scores_b[74])
```

该测试直接验证：内容相同且没有边界、Mask 等其他因素时，RoPE 核只依赖相对距离。

## 7. 慢频率尺度的计算

Notebook 用四分之一周期估计慢频率开始显著转动的距离：

```python
import math

dim = 128
i = dim // 2 - 1
theta_min = 10000 ** (-2 * i / dim)
quarter_period = math.pi / (2 * theta_min)
print(quarter_period)
```

这个数值只能解释最慢二维通道的相位尺度，不能当作模型上下文长度或性能断崖点。

## 8. 实验结论的边界

这个实验可以证明或展示：

- 旋转核依赖相对位置；
- 不同频率造成振荡；
- 特定全 1 内容在所选距离范围内的多频率相位抵消现象。

它不能单独证明：

- 某个 LLaMA 模型的真实有效上下文长度；
- 远距离语义一定无法被关注；
- 增大 `rope_theta` 就必然提升长文本任务；
- 原始 RoPE 能在任意长度上稳定外推。

## 常见误区

### 误区一：把原始 score 当注意力概率

还需要 Mask 和 softmax 才能得到注意力权重。

### 误区二：用全 1 的 Q、K 代表训练后的模型

全 1 只用于隔离位置因素，是位置核实验。

### 误区三：看到曲线反弹就否定远程衰减

周期振荡与总体衰减倾向可以同时存在。

## 一句话总结

RoPE 远程衰减实验测量的是“内容固定时的位置核”：曲线逐点振荡，但多频率项在远距离更容易抵消，应通过相对距离对齐和分桶统计理解，而不能把它直接当作真实注意力。

## 复习卡片

- 全 1 Q/K：隔离位置因素；
- 原始矩阵：attention logits，不是概率；
- 因果模型：未来位置实际会被 Mask；
- 代码易错点：`K_split` 必须来自 K；
- 观察方法：逐点曲线 + 距离分桶；
- 四分之一周期只是频率尺度。

## 自测问题

1. 为什么全 1 Q、K 适合观察 RoPE 位置核？
2. 为什么固定位置 32 时不能把右侧曲线当成真实因果注意力？
3. 分桶统计比逐点曲线多说明了什么？
4. 这个实验为什么不能证明模型的有效上下文长度？

## 参考与推荐阅读

- 本地课程文件 `ROPE.ipynb`：原始实现和绘图实验，不随博客发布。
- [RoFormer 原论文](https://arxiv.org/abs/2104.09864)：相对位置与远程衰减性质。
- [PyTorch 复数张量文档](https://docs.pytorch.org/docs/stable/complex_numbers.html)
