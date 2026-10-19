---
title: 'Transformer（三）：多头注意力与张量变形'
description: '理解多头注意力为什么有效，逐步追踪拆头、并行计算、拼接与输出投影的张量形状，并给出可用于交叉注意力的实现。'
pubDate: 2026-08-26
updatedDate: 2026-10-19
category: 'Transformer'
tags: ['Multi-Head Attention', 'Shape', 'PyTorch', 'Attention']
order: 3
---

## 学习目标

学完本文后，应该能够：

- 解释为什么需要多个注意力头；
- 写出多头注意力的完整公式；
- 熟练完成 $[B,L,D]$ 与 $[B,H,L,D_h]$ 之间的变形；
- 理解输出投影 $W^O$ 的作用；
- 使用 `nn.MultiheadAttention` 并正确读取注意力权重 shape。

## 先修知识

- [缩放点积注意力、QKV 与 Mask](/notes/transformer-self-attention/)
- [Transformer 的整体架构](/notes/transformer-overview/)

## 代码主线说明

本文手写类以本地 `MultiHeadAttention.ipynb` 为主要参考，保留 `Wq/Wk/Wv/Wo → 拆头 → 注意力 → 拼头` 的实现顺序。为了能直接接入 `Decoder.ipynb`，修正版分别记录 Query 长度 $T$ 与 Key/Value 长度 $S$，并把 `is_casual` 统一更名为 `is_causal`。官方 API 只放在后文作为接口对照。

## 1. 为什么单头还不够

单个注意力头只产生一套相似度与聚合结果。多个头允许模型在不同的投影子空间中建立不同关系，例如：

- 一个头关注指代关系；
- 一个头关注局部搭配；
- 一个头关注主谓关系；
- 一个头关注较远位置的信息。

这只是帮助理解的直觉，不代表训练后每个头一定对应一个清晰的人类语言学概念。

多头注意力的关键不是把同一个注意力复制多次，而是每个头具有不同的投影参数。

## 2. 数学定义

设模型维度为 $D$，头数为 $H$，通常每头维度为：

$$
D_h = \frac{D}{H}
$$

第 $i$ 个头为：

$$
\operatorname{head}_i
=
\operatorname{Attention}(QW_i^Q,KW_i^K,VW_i^V)
$$

所有头拼接，再通过输出投影：

$$
\operatorname{MHA}(Q,K,V)
=
\operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_H)W^O
$$

当 $H D_h=D$ 时，拼接后的维度仍然是 $D$。

## 3. 从单矩阵到多头的 Shape

设 Query 长度为 $T$，Key/Value 长度为 $S$。

### 3.1 线性投影

```text
q: [B, T, D] --Wq--> Q: [B, T, D]
k: [B, S, D] --Wk--> K: [B, S, D]
v: [B, S, D] --Wv--> V: [B, S, D]
```

代码中常用一个大矩阵投影到 $D$ 维，再通过 reshape 拆成 $H$ 个头。这与分别保存 $H$ 组小矩阵在数学上等价，矩阵计算更高效。

### 3.2 拆头

```text
[B, T, D]
→ view [B, T, H, Dh]
→ transpose [B, H, T, Dh]
```

Key 和 Value 同理，但长度是 $S$：

```text
K, V: [B, H, S, Dh]
```

### 3.3 每个头并行计算注意力

```text
scores:  [B, H, T, S]
weights: [B, H, T, S]
heads:   [B, H, T, Dh]
```

### 3.4 合并头

```text
[B, H, T, Dh]
→ transpose [B, T, H, Dh]
→ contiguous + view [B, T, D]
→ Wo [B, T, D]
```

可以用“拆头—独立注意力—拼接—输出投影”这条主线检查多头注意力的总体流程。

## 4. 为什么拼接后还需要 $W^O$

不同头的输出只是沿特征维度机械拼接。$W^O$ 提供一次可学习的线性混合，使模型能够重新组合不同头的信息。

若没有 $W^O$，后续层仍能看到各个头的结果，但不同头之间缺少当前子层内的显式融合。

## 5. 参数量检查

在最常见的 $D_h=D/H$ 设置下：

- $W^Q$：$D\times D$；
- $W^K$：$D\times D$；
- $W^V$：$D\times D$；
- $W^O$：$D\times D$。

忽略 bias，多头注意力的投影参数量约为：

$$
4D^2
$$

增加头数 $H$ 并不会在这种设置下让投影参数量按 $H$ 倍增长，因为每个头的维度会相应缩小。

## 6. 正确的手写实现

下面的实现支持 $T\ne S$，既能用于自注意力，也能用于交叉注意力。约定布尔 Mask 中 `True` 表示禁止注意。`[B,S]` 的 Padding Mask 必须先扩为 `[B,1,1,S]`。全屏蔽行在本例中约定权重及聚合值为 0，点积前提高计算精度以减小低精度溢出风险。

```python
import math
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        x = x.view(batch, length, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        blocked_mask: torch.Tensor | None = None,
    ):
        batch, target_len, _ = query.shape
        source_len = key.size(1)

        if value.size(1) != source_len:
            raise ValueError("key 和 value 的序列长度必须一致")

        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))

        work_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=q.device.type, enabled=False):
            scores = (q.to(work_dtype) / math.sqrt(self.head_dim)) @ k.to(work_dtype).transpose(-2, -1)
            if blocked_mask is not None:
                if blocked_mask.dtype != torch.bool or blocked_mask.device != q.device:
                    raise ValueError("Mask 必须是同设备的布尔张量")
                mask = torch.broadcast_to(blocked_mask, scores.shape)
                all_blocked = mask.all(dim=-1, keepdim=True)
                scores = scores.masked_fill(mask, float("-inf")).masked_fill(all_blocked, 0.0)
            weights = torch.softmax(scores, dim=-1)
            if blocked_mask is not None:
                weights = weights.masked_fill(all_blocked, 0.0)
            weights = self.attn_dropout(weights)
            output = (weights @ v.to(work_dtype)).to(q.dtype)
        weights = weights.to(q.dtype)
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch, target_len, self.dim)
        output = self.out_dropout(self.out_proj(output))
        return output, weights
```

## 7. 为什么 `contiguous()` 常出现在这里

`transpose` 只改变张量的视图和步长，不一定重新排列内存。转置后的张量若步长不兼容，直接 `view` 会报错；`view` 不会悄悄按错误内存顺序重排数据。

典型写法是：

```python
output = output.transpose(1, 2).contiguous()
output = output.view(batch, target_len, dim)
```

也可以使用更能处理非连续张量的 `reshape`，但理解内存连续性仍然很重要。

## 8. 自注意力与交叉注意力调用方式

```python
# Encoder 自注意力
self_out, weights = mha(x, x, x, blocked_mask=padding_mask)

# Decoder 交叉注意力
cross_out, weights = mha(
    query=decoder_states,       # [B, T, D]
    key=encoder_states,         # [B, S, D]
    value=encoder_states,       # [B, S, D]
    blocked_mask=source_padding_mask,
)
```

虽然函数相同，交叉注意力必须分别保留 `target_len` 和 `source_len`。

## 9. `nn.MultiheadAttention` 的最小用法

```python
import torch
import torch.nn as nn

B, T, S, D, H = 2, 3, 5, 32, 4

mha = nn.MultiheadAttention(
    embed_dim=D,
    num_heads=H,
    dropout=0.1,
    batch_first=True,
)

query = torch.randn(B, T, D)
memory = torch.randn(B, S, D)
source_is_pad = torch.zeros(B, S, dtype=torch.bool)

output, weights = mha(
    query=query,
    key=memory,
    value=memory,
    key_padding_mask=source_is_pad,
    need_weights=True,
    average_attn_weights=False,
)

assert output.shape == (B, T, D)
assert weights.shape == (B, H, T, S)
```

常用参数：

- `batch_first=True`：输入使用 `[B,L,D]`；
- `key_padding_mask`：shape 为 `[B,S]`，`True` 表示忽略该 Key；
- `attn_mask`：控制位置之间是否允许注意；
- `average_attn_weights=False`：保留每个头的权重；
- `need_weights=False`：不需要权重时更容易走高性能路径。

## 10. Notebook 代码核对

`MultiHeadAttention.ipynb` 很适合观察拆头和注意力热力图，但整合到 Decoder 时要修正一个重要假设：

```python
batch_size, seq_len, dim = q.shape
K = K.view(batch_size, seq_len, n_heads, head_dim)
```

这里把 Key 长度也写成了 Query 的 `seq_len`。自注意力中没有问题，但交叉注意力的源长度 $S$ 与目标长度 $T$ 可能不同。正确做法是分别读取：

```python
target_len = q.size(1)
source_len = k.size(1)
```

然后按各自长度 reshape。

另一个命名问题是代码中的 `is_casual`，应写作 `is_causal`。前者表示“随意的”，后者才表示“因果的”。

## 11. 常见误区

### 误区一：头越多一定越好

当 $D$ 固定时，头数越多，每头维度越小。过小的 $D_h$ 可能限制单个头的表达能力，还会增加一些调度开销。

### 误区二：多头就是把输入切片

标准实现先用可学习矩阵投影，再拆头。不同头看到的是不同投影子空间。

### 误区三：注意力权重默认保留每个头

PyTorch 默认可能对头求平均。想分析单独的头，需要设置 `average_attn_weights=False`。

### 误区四：`batch_first=True` 会改变所有 Mask 规则

它主要改变输入输出张量的前两个维度，不会自动替你改变 `attn_mask` 和 `key_padding_mask` 的语义。

## 一句话总结

多头注意力的本质是：把同一表示投影到多个较小子空间中并行执行注意力，再拼接并用 $W^O$ 融合；实现时最关键的是正确维护 batch、head、Query 长度和 Key 长度四类维度。

## 复习卡片

- 每头维度：$D_h=D/H$；
- 拆头：`[B,L,D] → [B,H,L,Dh]`；
- 权重：`[B,H,T,S]`；
- 合并：`[B,H,T,Dh] → [B,T,D]`；
- 参数量：常见配置下约 $4D^2$；
- 关键陷阱：交叉注意力中 $T$ 不必等于 $S$。

## 自测问题

1. 增加头数时，为什么参数量通常不会按头数线性增加？
2. `transpose(1, 2)` 前后各维度分别代表什么？
3. 为什么合并头后还需要 $W^O$？
4. 如何让 PyTorch 返回 `[B,H,T,S]` 的逐头注意力权重？
5. Notebook 中用 Query 长度 reshape Key 会在哪种场景出错？

## 参考与推荐阅读

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)：3.2.2 节。
- [PyTorch `nn.MultiheadAttention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.activation.MultiheadAttention.html)：重点看输入输出 shape、Mask 和 `average_attn_weights`。
- [PyTorch 高性能 SDPA 教程](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html)：从手写实现过渡到融合算子。
- 本地学习代码：`MultiHeadAttention.ipynb`、`Attention.ipynb`。
