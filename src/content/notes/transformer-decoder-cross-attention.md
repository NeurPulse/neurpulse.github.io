---
title: 'Transformer（六）：Decoder、因果注意力与交叉注意力'
description: '拆解 Transformer Decoder 的三个子层，理解因果 Mask、交叉注意力和源/目标不等长时的张量形状。'
pubDate: 2026-08-26
category: 'Transformer'
tags: ['Transformer Decoder', 'Causal Mask', 'Cross-Attention', 'Seq2Seq']
order: 6
---

## 学习目标

学完本文后，应该能够：

- 解释 Decoder 为什么比 Encoder 多一个注意力子层；
- 说明因果 Mask 如何防止训练时泄漏未来答案；
- 明确交叉注意力中 $Q$、$K$、$V$ 的来源；
- 正确追踪源长度 $S$ 与目标长度 $T$；
- 写出 Pre-LN Decoder Layer 的核心实现。

## 先修知识

- [Encoder、LayerNorm、残差与 FFN](/notes/transformer-encoder-block/)
- [缩放点积注意力与 Mask](/notes/transformer-self-attention/)
- [多头注意力与张量变形](/notes/transformer-multi-head-attention/)

## 代码主线说明

本文代码以本地 `Decoder.ipynb` 为主，保留 `DecoderLayer`、`Decoder` 以及 Pre-LN 的组装方式。唯一影响正确性的结构性调整是：交叉注意力分别处理目标长度 $T$ 和源长度 $S$；另外补入真实批训练所需的源端、目标端 Padding Mask。

## 1. Decoder 在解决什么问题

在机器翻译中，Encoder 已经把源句子编码成：

```text
memory: [B, S, D]
```

Decoder 要根据：

1. 已经给出的目标端前缀；
2. Encoder 对源句子的表示；

预测目标序列的下一个 token。

因此一个 Decoder Layer 包含三个子层：

```text
因果自注意力 → 交叉注意力 → FFN
```

## 2. 因果自注意力

训练时，目标序列的多个位置可以并行输入 Decoder。如果没有 Mask，“I” 的位置可能直接读取后面的 “am a dog”，等于提前看到了答案。

因果 Mask 让第 $t$ 个 Query 只能读取位置 $0\ldots t$。


### 2.1 构造上三角 Mask

```python
def make_causal_blocked_mask(length: int, device=None):
    return torch.triu(
        torch.ones(length, length, dtype=torch.bool, device=device),
        diagonal=1,
    )
```

结果中 `True` 位于严格上三角，表示屏蔽未来位置。

### 2.2 训练并行不等于可看未来

训练时可以一次计算所有目标位置，是因为真实目标前缀已经提供；因果 Mask 保证每个位置使用的信息与推理时一致。

## 3. 交叉注意力

交叉注意力连接 Decoder 和 Encoder：

- Query 来自 Decoder 当前隐表示；
- Key、Value 来自 Encoder memory。

$$
Q=H_{dec}W^Q
$$

$$
K=H_{enc}W^K,\quad V=H_{enc}W^V
$$

$$
\operatorname{CrossAttention}
=
\operatorname{softmax}\left(\frac{QK^T}{\sqrt{D_h}}+M_{src}\right)V
$$

目标端 Query 会对源端 Key/Value 计算一个 $T\times S$ 的权重矩阵：每一行对应一个目标位置，每一列对应一个源位置。

## 4. 最关键的 Shape：$T$ 不必等于 $S$

设：

- 源序列长度为 $S$；
- 目标前缀长度为 $T$。

```text
Decoder states: [B, T, D]
Encoder memory: [B, S, D]

Q: [B, H, T, Dh]
K: [B, H, S, Dh]
V: [B, H, S, Dh]

scores:  [B, H, T, S]
weights: [B, H, T, S]
output:  [B, H, T, Dh]
merged:  [B, T, D]
```

输出长度跟随 Query，因此交叉注意力输出仍属于目标端。

## 5. Decoder 的两类 Mask

### 5.1 目标端组合 Mask

Decoder 自注意力通常同时需要：

- 目标端 Causal Mask：不能看未来；
- 目标端 Padding Mask：不能读取 `<pad>`。

二者都作用在目标端 Key 上，可以组合后广播到 `[B,H,T,T]`。

### 5.2 源端 Padding Mask

交叉注意力使用源端 Padding Mask，禁止 Decoder 读取 Encoder memory 中对应 `<pad>` 的位置：

```text
source_is_pad: [B, S]
广播后:       [B, 1, 1, S]
```

交叉注意力通常不需要因果 Mask，因为源句子整体已知。

## 6. Pre-LN Decoder Layer 实现

下面假设已经有前两篇中的 `MultiHeadAttention` 与 `FeedForward`。

```python
import torch
import torch.nn as nn


class DecoderLayer(nn.Module):
    def __init__(self, dim, num_heads, hidden_dim, dropout):
        super().__init__()
        self.self_norm = nn.LayerNorm(dim)
        self.cross_norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)

        self.self_attn = MultiHeadAttention(dim, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(dim, num_heads, dropout)
        self.ffn = FeedForward(dim, hidden_dim, dropout)

    def forward(
        self,
        x,
        memory,
        target_blocked_mask=None,
        source_blocked_mask=None,
    ):
        norm_x = self.self_norm(x)
        self_out, _ = self.self_attn(
            norm_x,
            norm_x,
            norm_x,
            blocked_mask=target_blocked_mask,
        )
        x = x + self_out

        norm_x = self.cross_norm(x)
        cross_out, cross_weights = self.cross_attn(
            query=norm_x,
            key=memory,
            value=memory,
            blocked_mask=source_blocked_mask,
        )
        x = x + cross_out

        x = x + self.ffn(self.ffn_norm(x))
        return x, cross_weights
```

## 7. 多层 Decoder

```python
class Decoder(nn.Module):
    def __init__(self, layers, dim):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x, memory, target_mask=None, source_mask=None):
        cross_weights = None
        for layer in self.layers:
            x, cross_weights = layer(
                x,
                memory,
                target_blocked_mask=target_mask,
                source_blocked_mask=source_mask,
            )
        return self.final_norm(x), cross_weights
```

每层 Decoder 都会重新读取相同的 Encoder memory，但使用自己的投影参数。

## 8. Decoder-only 模型有什么不同

GPT、LLaMA 等 Decoder-only 模型通常只保留：

```text
因果自注意力 → FFN
```

它们没有独立 Encoder，因此也没有标准的 Encoder-Decoder 交叉注意力。所有上下文都作为同一条因果序列输入。

“Decoder-only” 是结构家族名称，不代表必须照搬原始 Decoder 的三个子层。

## 9. Notebook 代码核对

`Decoder.ipynb` 的组装顺序采用 Pre-LN，整体思路正确，但其 `MultiHeadAttention.forward` 从 Query 读取单个 `seq_len`，再用它 reshape $Q$、$K$、$V$：

```python
batch_size, seq_len, dim = q.shape
K = K.view(batch_size, seq_len, n_heads, head_dim)
```

示例中 Encoder 和 Decoder 都设置成长度 512，所以问题被掩盖。当翻译数据的源长度和目标长度不一致时，交叉注意力会 reshape 失败或产生错误语义。

应改为：

```python
target_len = q.size(1)
source_len = k.size(1)
```

并分别 reshape $Q$ 与 $K,V$。

另一个实现层面的建议是：源端与目标端应分别传入 Padding Mask。原 Notebook 的简化实现只包含 Causal Mask，没有完整处理 `<pad>`。

## 10. 常见误区

### 误区一：Decoder 的第二个注意力也是自注意力

它是交叉注意力，Query 与 Key/Value 来自不同序列。

### 误区二：训练时一次输入完整目标，所以发生了信息泄漏

只要正确应用因果 Mask，每个位置仍无法访问未来 token。

### 误区三：交叉注意力权重一定是方阵

它的 shape 是 $T\times S$，翻译中通常源目标长度不同。

### 误区四：Decoder-only 模型也必须有交叉注意力

标准 Decoder-only 语言模型没有独立 Encoder memory。

## 一句话总结

Transformer Decoder 的本质是：先用因果自注意力整理已生成前缀，再以 Decoder 表示为 Query 从 Encoder memory 中检索源信息，最后通过 FFN 完成逐 token 变换。

## 复习卡片

- Decoder 三子层：因果自注意力、交叉注意力、FFN；
- Causal Mask：阻止读取未来目标 token；
- Cross Query：Decoder；Cross Key/Value：Encoder；
- 交叉权重：`[B,H,T,S]`；
- 源 Padding Mask：`[B,S]`，广播到 Key 维；
- Decoder-only：通常没有交叉注意力。

## 自测问题

1. 为什么训练时 Decoder 能并行计算所有目标位置？
2. 交叉注意力的输出长度为什么是 $T$？
3. 目标端 Causal Mask 与 Padding Mask 分别解决什么问题？
4. 为什么源端交叉注意力通常不需要因果 Mask？
5. 给出的 Notebook 在 $S\ne T$ 时会在哪里出错？

## 参考与推荐阅读

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)：3.1 节 Decoder 结构与 3.2.3 节注意力应用方式。
- [把这一切组装起来](https://www.rethink.fun/chapter15/%E6%8A%8A%E8%BF%99%E4%B8%80%E5%88%87%E7%BB%84%E8%A3%85%E8%B5%B7%E6%9D%A5.html)：适合对照完整 Encoder-Decoder 图。
- [PyTorch `nn.Transformer`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Transformer.html)：重点看 $S$、$T$ 与各种 Mask。
- 本地学习代码：`Decoder.ipynb`、`Loss.ipynb`。
