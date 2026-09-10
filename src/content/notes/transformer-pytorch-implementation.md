---
title: 'Transformer（八）：基于 Notebook 的手写组装与代码调试'
description: '以课程 Notebook 的 ModelArgs、MultiHeadAttention、Encoder、Decoder 与 Transformer 类为主线，完成代码整合、错误修正和最小测试。'
pubDate: 2026-08-26
category: 'Transformer'
tags: ['PyTorch', 'Notebook', 'Implementation', 'Debugging']
order: 8
---

## 学习目标

学完本文后，应该能够：

- 说清 10 个 Notebook 在完整 Transformer 中的位置；
- 按 Notebook 的类名和 Pre-LN 顺序组装 Encoder-Decoder Transformer；
- 理解教学代码中每一处必要修正，而不是机械替换；
- 构造目标右移、Padding Mask、Causal Mask 和 token 级损失；
- 用不等长序列与小数据过拟合测试检查实现。

## 先修知识

- [Transformer 的整体架构](/notes/transformer-overview/)
- [缩放点积注意力、QKV 与 Mask](/notes/transformer-self-attention/)
- [多头注意力与张量变形](/notes/transformer-multi-head-attention/)
- [Encoder、LayerNorm、残差与 FFN](/notes/transformer-encoder-block/)
- [Decoder、因果注意力与交叉注意力](/notes/transformer-decoder-cross-attention/)
- [训练、损失与自回归推理](/notes/transformer-training-inference/)

## 1. 本篇的代码来源原则

本篇以给出的 `.ipynb` 为**主要代码来源**，保留其中的类名、模块拆分和 Pre-LN 组装顺序。论文、PyTorch 文档和其他资料只承担两项工作：

1. 解释 Notebook 背后的原理；
2. 核对 shape、Mask、损失和边界条件。

因此，后文不会把 `nn.Transformer` 当成主实现。Notebook 中明显影响正确性的写法会被修正，并说明“原写法—问题—修正”之间的关系。

## 2. Notebook 与模块地图

| Notebook | 在模型中的职责 | 本篇处理方式 |
|---|---|---|
| `Attention.ipynb` | $QK^T/\sqrt{d_k}$、softmax、$AV$ | 保留计算顺序 |
| `MultiHeadAttention.ipynb` | QKV 投影、拆头、拼头 | 保留类结构，修正 $S\ne T$ |
| `PositionEncoding.ipynb` | 正弦位置编码 | 保留 buffer，补长度切片 |
| `LayerNorm.ipynb` | 最后一维归一化 | 修正偏移与方差计算 |
| `FeedForward.ipynb` | `D → Dff → D` | 保留 `FFN` 类 |
| `Encoder.ipynb` | Pre-LN Encoder 堆叠 | 保留 `EncoderLayer/Encoder` |
| `Decoder.ipynb` | 因果自注意力与交叉注意力 | 保留 `DecoderLayer/Decoder` |
| `Loss.ipynb` | 完整模型、右移目标、交叉熵 | 作为整体组装主线 |
| `Tokenizer.ipynb` | BPE 与 token id | 用于数据入口 |
| `Sampling.ipynb` | temperature、top-k、top-p | 用于生成出口 |

完整调用链是：

```text
Tokenizer
→ ModelArgs
→ Embedding + PositionEncoding
→ EncoderLayer × N
→ DecoderLayer × N
→ lm_head
→ Loss / Sampling
```

## 3. 配置类：沿用 `ModelArgs`

下面沿用 Notebook 的字段，只增加 `pad_id`，以便注意力和损失共同忽略填充位置。

```python
from dataclasses import dataclass


@dataclass
class ModelArgs:
    n_heads: int
    dim: int
    hidden_dim: int
    dropout: float
    max_seq_len: int
    n_layer: int
    vocab_size: int
    pad_id: int = 0

    def __post_init__(self):
        if self.dim % self.n_heads != 0:
            raise ValueError("dim 必须能被 n_heads 整除")
```

这样做的好处是：阅读任意一个 Notebook 时，参数命名仍然一致，不必在 `dim/d_model`、`n_heads/num_heads` 之间反复切换。

## 4. 三处基础修正

### 4.1 `LayerNorm.ipynb`

标准 LayerNorm 是：

$$
\operatorname{LN}(x)
=
\gamma\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta
$$

Notebook 的独立版本曾出现 `- beta`，应改成 `+ beta`。手写时还应使用总体方差：

```python
import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * normalized + self.beta
```

### 4.2 `PositionEncoding.ipynb`

位置表按 `max_seq_len` 预生成，但前向时必须只取当前输入长度：

```python
def forward(self, x):
    return x + self.pe[:, :x.size(1)].to(dtype=x.dtype)
```

`Loss.ipynb` 中已经使用了这一正确切片方式。

### 4.3 `MultiHeadAttention.ipynb`

Notebook 从 Query 读取单个 `seq_len`，再用它 reshape $Q,K,V$。这在自注意力中可运行，但交叉注意力中目标长度 $T$ 与源长度 $S$ 往往不同。

```python
batch_size, target_len, _ = q.shape
source_len = k.size(1)

Q = self.wq(q).view(batch_size, target_len, self.n_heads, self.head_dim)
K = self.wk(k).view(batch_size, source_len, self.n_heads, self.head_dim)
V = self.wv(v).view(batch_size, source_len, self.n_heads, self.head_dim)
```

同时把 `is_casual` 更名为 `is_causal`。`causal` 表示“因果的”，`casual` 表示“随意的”。

## 5. 沿用 Notebook 的模块组装顺序

Notebook 采用 Pre-LN，层内顺序如下。

为保证后面的整体代码能够直接拼接，本篇统一沿用 Notebook 风格的接口：

```python
MultiHeadAttention(args, is_causal=False)
attention(q, k, v, blocked_mask=None)  # 只返回 output
Encoder(args)
encoder(x, source_mask=None)
Decoder(args)
decoder(x, enc_out, target_mask=None, source_mask=None)
```

前几篇为了展示注意力权重，部分示例返回 `(output, weights)`；整合到这里时只取 `output`，或在 `MultiHeadAttention` 中增加 `need_weights` 开关。不要把两套返回值约定混在一起。

### 5.1 Encoder Layer

```python
norm_x = self.attention_norm(x)
h = x + self.attention(
    norm_x,
    norm_x,
    norm_x,
    blocked_mask=source_mask,
)
out = h + self.feed_forward(self.fnn_norm(h))
```

### 5.2 Decoder Layer

```python
norm_x = self.attention_mask_norm(x)
x = x + self.mask_attention(
    norm_x,
    norm_x,
    norm_x,
    blocked_mask=target_mask,
)

norm_x = self.attention_norm(x)
h = x + self.cross_attention(
    norm_x,
    enc_out,
    enc_out,
    blocked_mask=source_mask,
)

out = h + self.feed_forward(self.ffn_norm(h))
```

这里的两个 Mask 不能混用：

- `target_mask`：目标端 Causal Mask 与目标 Padding Mask；
- `source_mask`：源端 Padding Mask，供 Encoder 自注意力和 Decoder 交叉注意力使用。

## 6. 以 `Loss.ipynb` 为主线的整体模型

下面保留 Notebook 的 `ModuleDict`、共享 token embedding、`lm_head` 和 `targets=None` 分支。它依赖前文修正后的 `PositionEncoding`、`Encoder` 和 `Decoder`。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(
                args.vocab_size,
                args.dim,
                padding_idx=args.pad_id,
            ),
            "wpe": PositionEncoding(args),
            "drop": nn.Dropout(args.dropout),
            "encoder": Encoder(args),
            "decoder": Decoder(args),
        })
        self.lm_head = nn.Linear(args.dim, args.vocab_size, bias=False)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, src_input, tgt_input, targets=None):
        # src_input: [B, S]；tgt_input: [B, T]
        if src_input.size(1) > self.args.max_seq_len:
            raise ValueError("源序列超过 max_seq_len")
        if tgt_input.size(1) > self.args.max_seq_len:
            raise ValueError("目标序列超过 max_seq_len")

        src_is_pad = src_input.eq(self.args.pad_id)       # [B, S]
        tgt_is_pad = tgt_input.eq(self.args.pad_id)       # [B, T]

        # 手写 MHA 约定 True 表示禁止读取。
        src_mask = src_is_pad[:, None, None, :]           # [B, 1, 1, S]
        tgt_pad_mask = tgt_is_pad[:, None, None, :]       # [B, 1, 1, T]
        target_len = tgt_input.size(1)
        causal_mask = torch.triu(
            torch.ones(
                target_len,
                target_len,
                dtype=torch.bool,
                device=tgt_input.device,
            ),
            diagonal=1,
        )[None, None, :, :]                               # [1, 1, T, T]
        tgt_mask = tgt_pad_mask | causal_mask

        src_emb = self.transformer.wte(src_input)
        tgt_emb = self.transformer.wte(tgt_input)

        src_emb = self.transformer.drop(
            self.transformer.wpe(src_emb)
        )
        tgt_emb = self.transformer.drop(
            self.transformer.wpe(tgt_emb)
        )

        enc_out = self.transformer.encoder(src_emb, src_mask)
        dec_out = self.transformer.decoder(
            tgt_emb,
            enc_out,
            target_mask=tgt_mask,
            source_mask=src_mask,
        )

        logits = self.lm_head(dec_out)                    # [B, T, V]

        if targets is None:
            # 与 Loss.ipynb 一致：单次解码只返回最后位置。
            return logits[:, [-1], :], None

        if targets.shape != tgt_input.shape:
            raise ValueError("targets 与 tgt_input 必须具有相同 shape")

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=self.args.pad_id,
        )
        return logits, loss
```

> 说明：这里让源端与目标端共享 `wte`，是为了忠实保留 `Loss.ipynb` 的设计。如果翻译两端使用完全不同的词表，可以改成 `src_embedding` 和 `tgt_embedding` 两套 Embedding。

## 7. 目标右移：与 Notebook 的样本构造一致

```text
src_input:  <bos> A B C <eos>
tgt_input:  <bos> D E F
targets:          D E F <eos>
```

批数据中可以这样得到 Decoder 输入与监督标签：

```python
tgt_input = full_target[:, :-1]
targets = full_target[:, 1:]

logits, loss = model(src_input, tgt_input, targets)
```

不要在传给 `F.cross_entropy` 前手动 softmax；交叉熵直接接收 logits。

## 8. 推理分支还需要外层循环

`targets is None` 时返回最后位置 logits，只完成了“一步预测”。完整生成仍需不断把新 token 接回目标前缀：

```python
@torch.no_grad()
def greedy_decode(model, src_input, bos_id, eos_id, max_new_tokens):
    generated = torch.full(
        (src_input.size(0), 1),
        bos_id,
        dtype=torch.long,
        device=src_input.device,
    )

    for _ in range(max_new_tokens):
        last_logits, _ = model(src_input, generated)
        next_id = last_logits[:, -1].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_id], dim=1)

        if torch.all(next_id.squeeze(-1).eq(eos_id)):
            break

    return generated
```

若要使用 temperature、top-k 或 top-p，只需把 `argmax` 替换为 `Sampling.ipynb` 中对应的概率调整与采样步骤。

## 9. 必做的最小测试

### 9.1 故意令 $S\ne T$

```python
args = ModelArgs(
    n_heads=4,
    dim=32,
    hidden_dim=64,
    dropout=0.1,
    max_seq_len=32,
    n_layer=2,
    vocab_size=100,
    pad_id=0,
)

model = Transformer(args)
src_input = torch.randint(1, 100, (2, 7))
tgt_input = torch.randint(1, 100, (2, 5))
targets = torch.randint(1, 100, (2, 5))

logits, loss = model(src_input, tgt_input, targets)
assert logits.shape == (2, 5, 100)
assert loss.ndim == 0
```

这个测试会直接暴露“用 Query 长度 reshape Key/Value”的交叉注意力错误。

### 9.2 还应测试

- 源端和目标端都包含 `<pad>`；
- batch 大小为 1；
- 序列长度为 1；
- 长度接近 `max_seq_len`；
- Causal Mask 屏蔽位置的注意力权重接近 0；
- `model.train()` 与 `model.eval()` 的输出 shape 一致；
- 1～4 个样本能被模型快速过拟合。

## 10. Notebook 原写法与修正版对照

| 位置 | Notebook 教学写法 | 修正版 | 原因 |
|---|---|---|---|
| LayerNorm | 曾出现 `- beta` | `+ beta` | 标准仿射偏移 |
| LayerNorm | `std + eps` | `sqrt(var + eps)` | 更符合标准定义 |
| PositionEncoding | 可能加完整位置表 | 按输入长度切片 | 支持变长输入 |
| MHA | 一个 `seq_len` 处理 QKV | 分开 $T$ 与 $S$ | 支持交叉注意力 |
| 命名 | `is_casual` | `is_causal` | 避免语义误读 |
| Mask | 主要只有 Causal Mask | 再加 src/tgt Padding Mask | 处理真实 batch |
| Loss | 所有位置计算损失 | `ignore_index=pad_id` | 忽略填充标签 |
| 推理 | 返回最后 logits | 外层自回归循环 | 完成整段生成 |

这些调整不是另起一套代码，而是把 Notebook 的教学骨架补全到能够处理真实变长数据的程度。

## 11. 官方模块只用于交叉验证

手写模块完成后，可以用 `nn.MultiheadAttention` 或 `nn.Transformer` 核对以下内容：

- 输入输出 shape 是否一致；
- $S\ne T$ 时交叉注意力是否正常；
- Causal Mask 与 Padding Mask 的广播是否正确；
- 关闭 dropout 后，简化条件下的数值是否接近。

需要特别注意：不同 PyTorch API 对布尔 Mask 的定义并不总一致。本文手写代码约定 `True` 表示“禁止注意”，迁移到其他 API 前必须重新查接口说明。

## 12. 推荐调试顺序

1. 检查 token id 的 dtype、范围与特殊 token；
2. 写出每一步的 $B,S,T,D,H,D_h,V$；
3. 检查目标输入和标签是否右移一位；
4. 检查 Mask 的 shape、device 和布尔语义；
5. 检查 softmax 是否沿 Key 维；
6. 检查 `<pad>` 是否同时从注意力和损失中排除；
7. 检查是否出现 NaN/Inf；
8. 最后做小数据过拟合测试。

## 常见误区

### 误区一：Notebook 示例能运行，交叉注意力就一定正确

当示例恰好令 $S=T$ 时，错误 reshape 也可能通过。必须主动使用不等长输入。

### 误区二：修正版就不再是 Notebook 代码

类名、模块边界、Pre-LN 顺序和整体前向仍来自 Notebook；修正的是会影响变长数据、Mask 或损失的边界条件。

### 误区三：`targets=None` 已经完成句子生成

它只产生当前前缀的下一 token 分布，完整句子仍需要自回归循环。

### 误区四：只要最终 logits shape 正确，Mask 就正确

Mask 语义错误通常不会改变 shape。还应检查被屏蔽位置的注意力权重。

## 一句话总结

这套实现的主线是课程 Notebook 的手写 Transformer：按 `ModelArgs → MHA → Encoder/Decoder → Transformer → Loss/Sampling` 逐层组装，再通过长度切分、Padding Mask、正确 LayerNorm 和自回归循环补齐真实任务需要的边界条件。

## 复习卡片

- 主实现来源：给出的 10 个 Notebook；
- 结构顺序：Pre-LN；
- 整体骨架：`ModuleDict + Encoder + Decoder + lm_head`；
- 必要修正：LayerNorm、位置切片、$S/T$ 分离、Padding Mask；
- 训练：目标右移，损失忽略 `<pad>`；
- 推理：最后位置 logits + 外层生成循环；
- 必做测试：$S\ne T$、padding、小数据过拟合。

## 自测问题

1. `Loss.ipynb` 在这组代码中为什么适合作为整体组装主线？
2. 为什么交叉注意力必须分别保存 $T$ 和 $S$？
3. `targets is None` 分支完成了推理的哪一步？
4. Padding Mask 为什么既要进入注意力，也要进入损失？
5. 如何证明自己的修正没有改变 Notebook 的核心结构？

## 参考与推荐阅读

- 本地代码主线：`Attention.ipynb`、`MultiHeadAttention.ipynb`、`PositionEncoding.ipynb`、`LayerNorm.ipynb`、`FeedForward.ipynb`、`Encoder.ipynb`、`Decoder.ipynb`、`Loss.ipynb`、`Tokenizer.ipynb`、`Sampling.ipynb`。
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)：用于核对标准架构与公式。
- [PyTorch `nn.Transformer`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Transformer.html)：用于核对输入输出和 Mask 接口，不作为本文主实现。
- [PyTorch `nn.MultiheadAttention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.activation.MultiheadAttention.html)：用于交叉验证多头注意力 shape。
- [利用 Transformer 实现翻译模型](https://www.rethink.fun/chapter15/%E5%88%A9%E7%94%A8Transformer%E5%AE%9E%E7%8E%B0%E7%BF%BB%E8%AF%91%E6%A8%A1%E5%9E%8B.html)：用于对照数据准备与训练流程。
