---
title: 'Transformer（二）：缩放点积注意力、QKV 与 Mask'
description: '从一个查询如何检索上下文出发，推导缩放点积注意力的公式、张量形状、Mask 机制与正确 PyTorch 实现。'
pubDate: 2026-08-26
updatedDate: 2026-10-19
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

$M$ 是可选的加性 Mask。允许注意的位置加 $0$，禁止注意的位置加 $-\infty$。用有限负数替代时，要结合 dtype 和分数范围选择数值；它并不严格等价于屏蔽。

复习时可沿着 $Q,K,V\to A\to AV$ 检查形状：权重为 $T\times S$，输出为 $T\times d_v$。自注意力中 $T=S=L$，此时才可将输出长度写为 $L$。

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

<span id="5-为什么要除以-dksqrtd_kdk"></span>

## 5. 为什么要按维度缩放

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

需要先增加维度，再广播到注意力分数。

具体写法是 `key_padding_mask[:, None, None, :]`，得到 `[B,1,1,S]`，沿 head 和 Query 维广播。不要把 `[B,S]` 直接传给下面的核心函数：广播从尾维对齐，可能报错，也可能恰好尺寸相同而错误广播。屏蔽 Key 也不会自动清零填充 Query 的输出，后者通常还需在损失计算等位置排除。

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

下面实现已拆头后的核心函数，单头时令 $H=1$。示例使用 Python 3.10+ 和 PyTorch 2.x，要求 Q、K、V 是同设备、同 dtype 的有限浮点张量，batch 和 head 数一致；不实现 GQA、KV Cache 或 QKV 投影。

这里约定布尔 Mask 中 `True` 表示“禁止注意”。如果某个 Query 的所有 Key 都被禁止，softmax 在数学上没有定义；本实现明确约定该行的权重和输出为 0，避免 NaN。

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
    if any(x.ndim != 4 for x in (query, key, value)):
        raise ValueError("Q、K、V 必须是四维张量")
    if not query.is_floating_point():
        raise TypeError("Q、K、V 必须为浮点张量")
    if any(x.dtype != query.dtype or x.device != query.device for x in (key, value)):
        raise ValueError("Q、K、V 必须使用相同的 dtype 和 device")
    b, h, t, d_k = query.shape
    if (key.shape[:2] != (b, h) or value.shape[:2] != (b, h)
            or key.size(-1) != d_k or key.size(-2) != value.size(-2)):
        raise ValueError("Q/K 特征维、K/V 长度和 batch/head 维度必须匹配")
    if min(*query.shape, *key.shape, *value.shape) <= 0:
        raise ValueError("各维度必须为正")
    if not 0.0 <= dropout_p <= 1.0:
        raise ValueError("dropout_p 必须位于 [0, 1]")
    if blocked_mask is not None:
        if blocked_mask.dtype != torch.bool or blocked_mask.device != query.device:
            raise ValueError("blocked_mask 必须是同设备的布尔张量")

    # 先提高精度，再做点积；禁用外层 autocast 对这里矩阵乘法的降精度。
    work_dtype = torch.float64 if query.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=query.device.type, enabled=False):
        q = query.to(work_dtype)
        k = key.to(work_dtype)
        v = value.to(work_dtype)
        scores = (q / math.sqrt(d_k)) @ k.transpose(-2, -1)

        if blocked_mask is None:
            weights = torch.softmax(scores, dim=-1)
        else:
            mask = torch.broadcast_to(blocked_mask, scores.shape)
            all_blocked = mask.all(dim=-1, keepdim=True)
            scores = scores.masked_fill(mask, float("-inf"))
            # 先消除整行 -inf，再做 softmax，反向传播也不会经过 NaN。
            scores = scores.masked_fill(all_blocked, 0.0)
            weights = torch.softmax(scores, dim=-1)
            weights = weights.masked_fill(all_blocked, 0.0)

        weights = F.dropout(weights, p=dropout_p, training=training)
        output = weights @ v
    return output.to(query.dtype), weights.to(query.dtype)
```

<span id="为什么-softmax-前临时转成-float32"></span>

### 为什么在点积之前提高精度

如果先在 float16 中计算点积，再调用 `scores.float()`，已经溢出为 `inf` 的值无法恢复。因此，这里将低精度输入转为 float32 后才计算点积和 softmax；float64 输入则保留 float64。返回时恢复输入 dtype。这是便于核对的教学实现，生产环境通常优先使用官方 SDPA 及适合硬件的后端。

提高计算精度可以降低溢出风险，但不能保证任意大幅值的输入都安全；如果数值仍超过工作 dtype 的表示范围，需要进一步检查输入和模型的数值尺度。

只在 softmax 后把 NaN 替换为 0 也不够稳妥：反向传播仍可能经过产生 NaN 的节点。这里在 softmax 之前处理全屏蔽行。返回的权重已经经过 dropout；启用训练态 dropout 后，单次计算的行和不保证为 1。

## 9. 一个最小 Shape 测试

```python
B, H, T, S, Dh, Dv = 2, 4, 3, 5, 8, 6

q = torch.randn(B, H, T, Dh)
k = torch.randn(B, H, S, Dh)
v = torch.randn(B, H, S, Dv)

out, weights = scaled_dot_product_attention(q, k, v)

assert out.shape == (B, H, T, Dv)
assert weights.shape == (B, H, T, S)
assert torch.allclose(
    weights.sum(dim=-1),
    torch.ones(B, H, T),
    atol=1e-5,
)
```

这组测试故意令 $T\ne S$，可以及时发现“误把交叉注意力当成等长自注意力”的实现错误。

再检查 Padding Mask、官方实现对照和全屏蔽行：

```python
# True 表示填充；第二个样本故意全部填充，以检查空注意力行。
key_padding_mask = torch.tensor([
    [False, False, False, True, True],
    [True, True, True, True, True],
])
blocked = key_padding_mask[:, None, None, :]
out, weights = scaled_dot_product_attention(q, k, v, blocked_mask=blocked)
assert torch.isfinite(out).all() and torch.isfinite(weights).all()
assert torch.count_nonzero(out[1]) == 0
assert torch.count_nonzero(weights[0, :, :, 3:]) == 0

# 对有效样本核对官方 API；其布尔 Mask 的 True 表示允许。
reference = F.scaled_dot_product_attention(
    q[:1], k[:1], v[:1], attn_mask=~blocked[:1], dropout_p=0.0
)
torch.testing.assert_close(out[:1], reference, rtol=1e-5, atol=1e-6)
```

对于等长自注意力，可用 `torch.ones(T, T, dtype=torch.bool, device=q.device).triu(1)` 构造因果屏蔽，再与展开后的 Padding Mask 做按位或。交叉注意力通常不直接使用这种上三角因果关系；带 KV Cache 的非方阵因果屏蔽也需要考虑位置偏移。

## 10. PyTorch API 的 Mask 语义陷阱

不同 API 的布尔 Mask 语义不完全一致：

- `nn.MultiheadAttention`：布尔 Mask 中 `True` 通常表示禁止；
- `F.scaled_dot_product_attention`：布尔 `attn_mask` 中 `True` 表示允许参与注意力。

从一个 API 迁移到另一个 API 时，必须重新核对文档，不能只看 shape 一样就直接复用。

## 11. 计算复杂度

不计 QKV 与输出投影，单个 batch、单个 head 构造 $QK^T$ 和 $AV$ 的主要复杂度分别为 $O(TSd_k)$ 与 $O(TSd_v)$，合计：

$$
O\bigl(TS(d_k+d_v)\bigr).
$$

当 $d_k=d_v=D_h$ 时可简写为 $O(TSD_h)$。自注意力中 $T=S=L$，因此为：

$$
O(L^2D_h)
$$

本实现显式保存分数与权重，单个 head 的这部分空间复杂度为 $O(TS)$；加入 batch 与 head 后是 $O(BHTS)$。当序列长度很大时，这会成为显存开销的重要来源。FlashAttention 等实现通过分块和融合避免完整存储注意力矩阵，不能把这里的显存结论套用于所有后端。

## 12. 常见误区与代码错误

### 误区一：Value 也参与相似度计算

相似度由 $Q$ 和 $K$ 计算，$V$ 是被权重聚合的内容。

### 误区二：softmax 对错维度

应该沿 Key 维度，也就是分数张量的最后一维做 softmax。有有效 Key 且未启用 dropout 时，每个 Query 的权重和为 1；本实现的全屏蔽行则约定为 0。

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
- [PyTorch：Transformer building blocks](https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html)：全屏蔽注意力行的处理及实现差异。
