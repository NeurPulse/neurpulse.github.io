---
title: 'Transformer（二）：缩放点积注意力、QKV 与 Mask'
description: '从一个查询如何检索上下文出发，推导缩放点积注意力的公式、张量形状、Mask 机制与正确 PyTorch 实现。'
pubDate: 2026-08-26
category: 'Transformer'
tags: ['Self-Attention', 'Scaled Dot-Product Attention', 'QKV', 'Mask']
order: 2
featured: true
---

## 学习目标

学完本文后，应该能够：

- 用自己的话解释 Query、Key、Value；
- 写出并推导缩放点积注意力公式；
- 检查 $QK^T$、注意力权重与输出的 shape；
- 区分因果 Mask 和 Padding Mask；
- 写出同时支持自注意力与交叉注意力的实现。

## 先修知识

- [Transformer 的动机与整体架构](/notes/transformer-overview/)
- [Token 编码与词嵌入](/notes/deep-learning-token-encoding/)
- [Softmax 多分类](/notes/deep-learning-multi-class-classification/)

## 代码主线说明

本文代码以本地 `Attention.ipynb` 的计算顺序为主线：先计算 $QK^T$，再缩放、Mask、softmax，最后与 $V$ 相乘。为使同一函数还能用于后续交叉注意力，代码补充了 $T\ne S$ 的 shape 约定、Padding Mask 和数值稳定处理；这些属于对 Notebook 教学实现的完善，不改变其核心流程。

## 1. 最小直觉：一次带权检索

把注意力看成一次“查询—匹配—取值”：

- **Query**：我现在想找什么；
- **Key**：每个候选项用什么特征接受匹配；
- **Value**：匹配后真正取回的信息。

例如查询是“水果”，候选项是“金毛、西瓜、某个人名”。Query 与各 Key 计算相关性后，西瓜得到更大权重，最后输出会更接近西瓜对应的 Value。

注意：$Q$、$K$、$V$ 是三个不同角色，不必具有相同的数值。即使它们都由同一个输入 $X$ 投影得到，也会使用不同的可学习矩阵。

## 2. 从单个 Query 到公式

设一个查询向量为 $q\in\mathbb{R}^{d_k}$，共有 $S$ 个 Key：

$$
K = [k_1;k_2;\ldots;k_S] \in \mathbb{R}^{S\times d_k}
$$

第一步，计算查询与所有 Key 的点积分数：

$$
s_j = q\cdot k_j
$$

第二步，除以 $\sqrt{d_k}$ 并做 softmax：

$$
\alpha_j = \frac{\exp(s_j / \sqrt{d_k})}{\sum_{m=1}^{S}\exp(s_m / \sqrt{d_k})}
$$

第三步，用权重加权 Value：

$$
o = \sum_{j=1}^{S}\alpha_j v_j
$$

因此，一个 Query 得到一个聚合后的输出向量。

## 3. 矩阵形式

当有 $T$ 个 Query 时，把它们堆叠成矩阵：

$$
Q\in\mathbb{R}^{T\times d_k},\quad
K\in\mathbb{R}^{S\times d_k},\quad
V\in\mathbb{R}^{S\times d_v}
$$

缩放点积注意力为：

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}+M\right)V
$$

$M$ 是可选的加性 Mask。允许注意的位置加 $0$，禁止注意的位置加一个极大负数或 $-\infty$。

复习时可沿着 $X\to Q,K,V\to A\to AV$ 检查每一步的矩阵形状；只要最后得到的输出仍是 $L\times d_v$，这条数据流就保持了序列长度。

## 4. 完整 Shape 检查

加入 batch 和 head 维度后，设：

- batch 大小为 $B$；
- 头数为 $H$；
- Query 长度为 $T$；
- Key/Value 长度为 $S$；
- 每头维度为 $D_h$。

```text
Q:       [B, H, T, Dh]
K:       [B, H, S, Dh]
V:       [B, H, S, Dv]
K^T:     [B, H, Dh, S]
QK^T:    [B, H, T, S]
Mask:    [T, S]、[B, 1, T, S] 或可广播形状
weights: [B, H, T, S]
output:  [B, H, T, Dv]
```

最重要的两点：

1. 注意力权重最后两维是 **Query 长度 × Key 长度**；
2. 输出长度由 Query 的长度 $T$ 决定。

这就是为什么交叉注意力可以接收不同长度的源序列和目标序列。

## 5. 为什么要除以 $\sqrt{d_k}$

假设 $q_i$ 和 $k_i$ 相互独立，均值为 0、方差为 1。点积为：

$$
q\cdot k = \sum_{i=1}^{d_k}q_i k_i
$$

其方差大约会随 $d_k$ 增长：

$$
\operatorname{Var}(q\cdot k)\approx d_k
$$

当维度很大时，logits 的绝对值容易变大，softmax 会变得非常尖锐，许多位置的梯度接近 0。除以 $\sqrt{d_k}$ 后，方差重新回到约 1 的量级：

$$
\operatorname{Var}\left(\frac{q\cdot k}{\sqrt{d_k}}\right)\approx 1
$$

因此，这个缩放主要是在控制 softmax 输入的数值尺度。

## 6. Self-Attention 与 Cross-Attention

### 6.1 Self-Attention

$Q$、$K$、$V$ 都由同一序列 $X$ 产生：

$$
Q=XW^Q,\quad K=XW^K,\quad V=XW^V
$$

此时通常 $T=S=L$，注意力矩阵为 $L\times L$。

### 6.2 Cross-Attention

Decoder 表示产生 Query，Encoder 输出产生 Key 和 Value：

$$
Q=H_{dec}W^Q,\quad K=H_{enc}W^K,\quad V=H_{enc}W^V
$$

```text
H_dec: [B, T, D]
H_enc: [B, S, D]
scores:[B, H, T, S]
output:[B, T, D]
```

因此，实现注意力时不能把 $K$、$V$ 强行 reshape 成 Query 的长度 $T$。

## 7. 两种 Mask

### 7.1 Padding Mask

同一 batch 中的句子长度不同，短句需要补 `<pad>`。这些填充位置没有语义，不应被其他 Query 读取。

典型 shape：

```text
key_padding_mask: [B, S]
```

它会在 batch 和 head 维度上广播到注意力分数。

### 7.2 Causal Mask

在自回归 Decoder 中，第 $t$ 个位置不能看到未来位置。禁止区域是注意力矩阵的严格上三角：

```text
允许关系（1 表示可见）

1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

加性 Mask 则写成：

```text
0   -inf -inf -inf
0    0   -inf -inf
0    0    0   -inf
0    0    0    0
```

### 7.3 Mask 应加在 softmax 之前

正确顺序：

```text
QK^T / sqrt(dk) → 加 Mask → softmax → dropout → 乘 V
```

如果在 softmax 之后直接把某些权重改成 0，剩余权重之和通常不再等于 1。

## 8. 最小 PyTorch 实现

下面先实现单头/已拆头后的核心函数。这里约定布尔 Mask 中 `True` 表示“禁止注意”。

```python
import math
import torch
import torch.nn.functional as F


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    blocked_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    training: bool = False,
):
    """
    query: [B, H, T, Dh]
    key:   [B, H, S, Dh]
    value: [B, H, S, Dv]
    blocked_mask: 可广播到 [B, H, T, S]，True 表示禁止
    """
    d_k = query.size(-1)
    scores = query @ key.transpose(-2, -1)
    scores = scores / math.sqrt(d_k)

    if blocked_mask is not None:
        scores = scores.masked_fill(blocked_mask, float("-inf"))

    weights = torch.softmax(scores.float(), dim=-1).to(query.dtype)
    weights = F.dropout(weights, p=dropout_p, training=training)
    output = weights @ value
    return output, weights
```

### 为什么 softmax 前临时转成 float32

混合精度训练中，分数可能是 float16 或 bfloat16。用 float32 计算 softmax 通常更加稳定，之后再转回原 dtype。

## 9. 一个最小 Shape 测试

```python
B, H, T, S, Dh = 2, 4, 3, 5, 8

q = torch.randn(B, H, T, Dh)
k = torch.randn(B, H, S, Dh)
v = torch.randn(B, H, S, Dh)

out, weights = scaled_dot_product_attention(q, k, v)

assert out.shape == (B, H, T, Dh)
assert weights.shape == (B, H, T, S)
assert torch.allclose(
    weights.sum(dim=-1),
    torch.ones(B, H, T),
    atol=1e-5,
)
```

这组测试故意令 $T\ne S$，可以及时发现“误把交叉注意力当成等长自注意力”的实现错误。

## 10. PyTorch API 的 Mask 语义陷阱

不同 API 的布尔 Mask 语义不完全一致：

- `nn.MultiheadAttention`：布尔 Mask 中 `True` 通常表示禁止；
- `F.scaled_dot_product_attention`：布尔 `attn_mask` 中 `True` 表示允许参与注意力。

从一个 API 迁移到另一个 API 时，必须重新核对文档，不能只看 shape 一样就直接复用。

## 11. 计算复杂度

构造 $QK^T$ 和 $AV$ 的主要复杂度为：

$$
O(TSD_h)
$$

自注意力中 $T=S=L$，因此为：

$$
O(L^2D_h)
$$

注意力权重矩阵的空间复杂度是 $O(TS)$。长序列的主要显存压力通常来自这里，而不是 $Q$、$K$、$V$ 本身。

## 12. 常见误区与代码错误

### 误区一：Value 也参与相似度计算

相似度由 $Q$ 和 $K$ 计算，$V$ 是被权重聚合的内容。

### 误区二：softmax 对错维度

应该沿 Key 维度，也就是分数张量的最后一维做 softmax，使每个 Query 对所有 Key 的权重和为 1。

### 误区三：所有注意力矩阵都是方阵

只有 $T=S$ 时才是方阵。交叉注意力通常是 $T\times S$。

### 误区四：因果 Mask 可以代替 Padding Mask

前者限制时间方向，后者忽略填充 token，解决的是两个不同问题，训练 Decoder 时常常需要同时使用。

### 误区五：注意力权重就是模型决策原因

注意力权重能展示一次信息聚合的分配，但经过多层、残差和 FFN 后，不能简单把它等同为完整的因果解释。

## 一句话总结

缩放点积注意力的本质是：用 $QK^T$ 计算每个 Query 对所有 Key 的相关性，经缩放、Mask 和 softmax 后对 Value 加权汇总；理解它的关键是始终跟踪 $[T,S]$ 两个序列维度。

## 复习卡片

- 核心公式：$\operatorname{softmax}(QK^T/\sqrt{d_k}+M)V$；
- 分数 shape：$[B,H,T,S]$；
- 输出长度：由 Query 长度 $T$ 决定；
- 缩放原因：控制点积方差与 softmax 饱和；
- Padding Mask：屏蔽填充 Key；
- Causal Mask：屏蔽未来 Key。

## 自测问题

1. Query、Key、Value 各自承担什么角色？
2. 为什么 $QK^T$ 的 shape 是 $T\times S$？
3. 除以 $\sqrt{d_k}$ 对 softmax 有什么影响？
4. 为什么交叉注意力实现必须分别读取 Query 长度和 Key 长度？
5. Padding Mask 和 Causal Mask 能否互相替代？

## 参考与推荐阅读

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)：3.2 节给出缩放点积注意力与多头注意力的原始定义。
- [注意力机制](https://www.rethink.fun/chapter15/%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6.html)：适合用例子建立 QKV 直觉。
- [PyTorch `nn.MultiheadAttention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.activation.MultiheadAttention.html)：核对输入输出与 Mask shape。
- [PyTorch `scaled_dot_product_attention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)：了解官方融合实现及布尔 Mask 语义。
