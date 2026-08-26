---
title: 'Transformer（八）：PyTorch 组装、代码核对与调试'
description: '用 PyTorch 组装可训练的 Encoder-Decoder Transformer，建立公式—模块—Notebook 对照，并整理最容易出错的 shape 与 Mask 检查。'
pubDate: 2026-08-26
category: 'Transformer'
tags: ['PyTorch', 'nn.Transformer', 'Implementation', 'Debugging']
order: 8
---

## 学习目标

学完本文后，应该能够：

- 把前七篇中的模块映射到 PyTorch API；
- 组装一个支持源目标不等长、Padding Mask 与 Causal Mask 的翻译模型；
- 正确构造 token 级交叉熵；
- 用 shape 断言和最小测试定位实现错误；
- 识别给定 Notebook 中应当修正的代码细节。

## 先修知识

- [Transformer 的整体架构](/notes/transformer-overview/)
- [Encoder、LayerNorm、残差与 FFN](/notes/transformer-encoder-block/)
- [Decoder、因果注意力与交叉注意力](/notes/transformer-decoder-cross-attention/)
- [训练、损失与自回归推理](/notes/transformer-training-inference/)
- [PyTorch 模型构建](/notes/pytorch-model-construction/)

## 1. 先建立代码地图

| 知识模块 | Notebook | 推荐核对点 |
|---|---|---|
| 缩放点积注意力 | `Attention.ipynb` | `QK^T`、softmax 维度、`AV` |
| 多头注意力 | `MultiHeadAttention.ipynb` | 拆头、拼头、Causal Mask |
| 位置编码 | `PositionEncoding.ipynb` | sin/cos、buffer、长度切片 |
| LayerNorm | `LayerNorm.ipynb` | 最后一维、方差、`gamma/beta` |
| FFN | `FeedForward.ipynb` | `D → Dff → D` |
| Encoder | `Encoder.ipynb` | Pre-LN、残差、层堆叠 |
| Decoder | `Decoder.ipynb` | 因果自注意力、交叉注意力 |
| 整体与损失 | `Loss.ipynb` | 目标右移、词表投影、交叉熵 |
| Tokenizer | `Tokenizer.ipynb` | BPE、特殊 token、encode/decode |
| 解码策略 | `Sampling.ipynb` | temperature、top-k、top-p |

这组 Notebook 的优点是模块拆分清楚；学习时应当把它们当成“教学骨架”，再用本篇的检查项补齐真实训练所需的边界条件。

## 2. 为什么先用官方模块组装

手写注意力适合理解公式，但一个可训练 Transformer 还需要处理：

- 源序列和目标序列长度不同；
- 多种 Mask 的 shape 和布尔语义；
- dropout 与训练/推理状态；
- LayerNorm 的精确实现；
- 参数初始化、dtype、device；
- 高性能注意力内核。

因此推荐学习顺序：

```text
手写单个公式
→ 手写 MHA/Encoder/Decoder 理解结构
→ 用 nn.Transformer 建立正确基线
→ 再逐个替换为自己的模块并做等价测试
```

## 3. 配置类

```python
from dataclasses import dataclass


@dataclass
class TransformerConfig:
    src_vocab_size: int
    tgt_vocab_size: int
    pad_id: int
    dim: int = 512
    num_heads: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    hidden_dim: int = 2048
    dropout: float = 0.1
    max_len: int = 512

    def __post_init__(self):
        if self.dim % self.num_heads != 0:
            raise ValueError("dim 必须能被 num_heads 整除")
```

## 4. 位置编码

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, dim, max_len, dropout):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("此实现要求 dim 为偶数")

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        frequency = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / dim)
        )

        pe = torch.zeros(1, max_len, dim)
        pe[0, :, 0::2] = torch.sin(position * frequency)
        pe[0, :, 1::2] = torch.cos(position * frequency)
        self.register_buffer("pe", pe, persistent=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        length = x.size(1)
        x = x + self.pe[:, :length].to(dtype=x.dtype)
        return self.dropout(x)
```

## 5. 一个完整的 Encoder-Decoder 基线

```python
class Seq2SeqTransformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.src_embedding = nn.Embedding(
            config.src_vocab_size,
            config.dim,
            padding_idx=config.pad_id,
        )
        self.tgt_embedding = nn.Embedding(
            config.tgt_vocab_size,
            config.dim,
            padding_idx=config.pad_id,
        )
        self.position = SinusoidalPositionEncoding(
            config.dim,
            config.max_len,
            config.dropout,
        )

        self.transformer = nn.Transformer(
            d_model=config.dim,
            nhead=config.num_heads,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dim_feedforward=config.hidden_dim,
            dropout=config.dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.lm_head = nn.Linear(
            config.dim,
            config.tgt_vocab_size,
            bias=False,
        )

    def embed_source(self, token_ids):
        x = self.src_embedding(token_ids) * math.sqrt(self.config.dim)
        return self.position(x)

    def embed_target(self, token_ids):
        x = self.tgt_embedding(token_ids) * math.sqrt(self.config.dim)
        return self.position(x)

    def forward(self, src_ids, tgt_input_ids):
        # src_ids:       [B, S]
        # tgt_input_ids: [B, T]
        source_is_pad = src_ids.eq(self.config.pad_id)
        target_is_pad = tgt_input_ids.eq(self.config.pad_id)

        src = self.embed_source(src_ids)
        tgt = self.embed_target(tgt_input_ids)

        target_len = tgt_input_ids.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            target_len,
            device=tgt_input_ids.device,
        )

        hidden = self.transformer(
            src=src,
            tgt=tgt,
            tgt_mask=causal_mask,
            src_key_padding_mask=source_is_pad,
            tgt_key_padding_mask=target_is_pad,
            memory_key_padding_mask=source_is_pad,
        )
        return self.lm_head(hidden)
```

### 5.1 输入输出 Shape

```text
src_ids:       [B, S]
tgt_input_ids: [B, T]
src embedding: [B, S, D]
tgt embedding: [B, T, D]
hidden:        [B, T, D]
logits:        [B, T, Vtgt]
```

输出长度为 $T$，因为 Decoder 的 Query 来自目标序列。

## 6. 训练步骤

```python
def training_step(model, src_ids, full_target_ids, pad_id):
    # full_target_ids: <bos> ... <eos> <pad> ...
    tgt_input_ids = full_target_ids[:, :-1]
    labels = full_target_ids[:, 1:]

    logits = model(src_ids, tgt_input_ids)
    assert logits.shape[:2] == labels.shape

    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=pad_id,
    )
    return loss
```

优化器的标准流程仍然是：

```python
optimizer.zero_grad(set_to_none=True)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

梯度裁剪并非原始 Transformer 公式的一部分，但在某些训练配置中有助于防止异常梯度破坏训练。

## 7. 最小 Shape 测试

测试时故意令 $S\ne T$，避免等长样例掩盖交叉注意力 bug。

```python
config = TransformerConfig(
    src_vocab_size=100,
    tgt_vocab_size=120,
    pad_id=0,
    dim=32,
    num_heads=4,
    num_encoder_layers=2,
    num_decoder_layers=2,
    hidden_dim=64,
    max_len=32,
)

model = Seq2SeqTransformer(config)
src_ids = torch.randint(1, 100, (2, 7))
tgt_ids = torch.randint(1, 120, (2, 5))

logits = model(src_ids, tgt_ids)
assert logits.shape == (2, 5, 120)
```

还应至少测试：

- 源端包含 `<pad>`；
- 目标端包含 `<pad>`；
- batch 大小为 1；
- 序列长度为 1；
- 输入长度接近 `max_len`；
- `model.train()` 与 `model.eval()` 输出 shape 一致。

## 8. Notebook 中需要修正的重点

### 8.1 LayerNorm

独立 Notebook 使用了 `- beta`，应为 `+ beta`。标准公式是：

```python
(x - mean) / torch.sqrt(var + eps) * gamma + beta
```

并使用 `unbiased=False` 的方差，或直接使用 `nn.LayerNorm`。

### 8.2 位置编码

应按真实输入长度切片：

```python
x = x + self.pe[:, :x.size(1)]
```

### 8.3 交叉注意力长度

不能从 Query 读取一个 `seq_len` 后同时 reshape $Q,K,V$。应分别读取目标长度 $T$ 和源长度 $S$。

### 8.4 Padding Mask

简化 Notebook 主要实现了 Causal Mask。真实 batch 还需要：

- Encoder 的源 Padding Mask；
- Decoder 自注意力的目标 Padding Mask；
- 交叉注意力的源 memory Padding Mask；
- 损失中的 `ignore_index=pad_id`。

### 8.5 命名

`is_casual` 应改为 `is_causal`。准确命名能减少 Mask 逻辑被误读。

### 8.6 推理并不只是取一次最后 logits

`Loss.ipynb` 在 `targets is None` 时只返回最后一个位置的 logits，这只是**单次解码步**。完整推理仍需外层循环，把预测 token 追加到输入并检查 `<eos>`。

## 9. PyTorch Mask 语义速查

在 `nn.Transformer` 与 `nn.MultiheadAttention` 中，布尔 Mask 的 `True` 通常表示禁止参与注意力；但 `F.scaled_dot_product_attention` 的布尔 `attn_mask=True` 表示允许。

| API | 布尔 `True` 的常见含义 |
|---|---|
| `nn.Transformer` 的各类布尔 mask | 禁止 |
| `nn.MultiheadAttention` 的 mask | 禁止 |
| `F.scaled_dot_product_attention` 的 `attn_mask` | 允许 |

这是最容易在“手写实现 → 官方高性能算子”迁移时引入的静默错误之一。

## 10. 调试顺序

遇到报错或训练不收敛时，按以下顺序检查：

1. **token id**：dtype 是否为 `long`，是否超出词表；
2. **shape**：明确 $B,S,T,D,H,D_h,V$；
3. **目标右移**：输入和标签是否错一位；
4. **Mask shape**：能否广播到正确分数矩阵；
5. **Mask 语义**：`True` 到底是允许还是禁止；
6. **softmax 维度**：是否沿 Key 维；
7. **Padding loss**：是否设置 `ignore_index`；
8. **训练状态**：`train/eval`、dropout、`no_grad`；
9. **数值稳定性**：是否出现 NaN/Inf；
10. **小数据过拟合**：模型能否先记住极小 batch。

## 11. 小数据过拟合测试

在完整训练前，先让模型反复训练 1～4 个样本。若损失无法明显下降，优先怀疑数据对齐、Mask、标签右移或代码逻辑，而不是盲目扩大数据和模型。

这个测试是深度学习工程里最划算的排错手段之一。

## 12. `nn.Transformer` 的定位

PyTorch 官方说明中，`nn.Transformer` 和 `nn.MultiheadAttention` 更偏向原始架构的参考实现，适合学习和建立基线。

现代大模型工程通常还会使用：

- `F.scaled_dot_product_attention`；
- Flash Attention 等融合内核；
- RoPE、RMSNorm、SwiGLU；
- MQA/GQA；
- KV Cache 与分页缓存；
- 张量并行、流水线并行和编译优化。

这些属于掌握标准 Transformer 之后的进阶路线。

## 13. 常见误区

### 误区一：代码能运行就代表注意力正确

当示例恰好令 $S=T$ 时，错误的 reshape 也可能运行。必须用不等长测试。

### 误区二：只检查最终 logits shape

许多错误 Mask 不改变 shape，却会改变语义。还要检查注意力权重中被屏蔽位置是否为 0。

### 误区三：手写代码一定比官方模块更适合训练

手写适合理解；官方算子通常有更完整的边界处理和更高性能实现。

### 误区四：Transformer 不需要调试数据

目标右移、特殊 token、padding 和词表 id 错误，往往比网络公式本身更常见。

## 一句话总结

实现 Transformer 的关键不只是把模块类写出来，而是让源长度、目标长度、Mask、目标右移、损失和推理循环在同一套 shape 约定下完全一致，并用不等长与 padding 测试主动暴露错误。

## 复习卡片

- 正确基线：`nn.Transformer(batch_first=True, norm_first=True)`；
- 源输入：`[B,S]`；目标输入：`[B,T]`；
- logits：`[B,T,V]`；
- 三个 Padding Mask 入口：src、tgt、memory；
- 目标标签：目标序列右移后的下一 token；
- 必做测试：$S\ne T$、padding、长度 1、小 batch 过拟合。

## 自测问题

1. 为什么应先建立官方模块基线，再替换为手写模块？
2. `src_key_padding_mask` 与 `memory_key_padding_mask` 为什么常使用同一个源端 Mask？
3. 如何构造 `tgt_input_ids` 和 labels？
4. 哪些错误不会改变输出 shape，却会破坏模型语义？
5. 为什么小数据过拟合测试能快速定位实现问题？

## 参考与推荐阅读

- [PyTorch `nn.Transformer`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Transformer.html)：完整参数、输入输出与 Mask 约定。
- [PyTorch `nn.MultiheadAttention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.activation.MultiheadAttention.html)：MHA 参考实现与权重 shape。
- [PyTorch 高性能 Transformer 构建模块](https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html)：了解 SDPA、Nested Tensor 与编译优化。
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)：用来核对标准架构和训练配置。
- 本地学习代码：`Attention.ipynb`、`MultiHeadAttention.ipynb`、`PositionEncoding.ipynb`、`LayerNorm.ipynb`、`FeedForward.ipynb`、`Encoder.ipynb`、`Decoder.ipynb`、`Loss.ipynb`、`Tokenizer.ipynb`、`Sampling.ipynb`。

