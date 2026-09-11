---
title: 'Transformer（一）：动机、整体架构与学习路线'
description: '从 RNN 的瓶颈出发，建立 Transformer 编码器—解码器架构的完整地图，并理解它的优势、代价与主要变体。'
pubDate: 2026-08-26
category: 'Transformer'
tags: ['Transformer', 'Attention', 'Encoder-Decoder', '序列建模']
order: 1
---

## 学习目标

学完本文后，应该能够：

- 解释 Transformer 为什么要摆脱 RNN 的循环计算；
- 说清 Encoder、Decoder、三种注意力分别负责什么；
- 区分 Encoder-only、Decoder-only 和 Encoder-Decoder 模型；
- 客观看待 Transformer 的并行优势与二次复杂度限制。

## 先修知识

阅读本文前，建议先理解：

- [自然语言处理中的常见任务](/notes/deep-learning-nlp-common-tasks/)
- [Token 编码与词嵌入](/notes/deep-learning-token-encoding/)
- [RNN 的隐状态与串行计算](/notes/deep-learning-rnn/)
- [LSTM 的长期记忆机制](/notes/deep-learning-lstm/)

## 专题阅读路线

本专题按照“先看全局，再拆模块，最后组装”的顺序阅读：

1. **动机与整体架构**：本文；
2. [缩放点积注意力与 Mask](/notes/transformer-self-attention/)；
3. [多头注意力与张量变形](/notes/transformer-multi-head-attention/)；
4. [位置编码](/notes/transformer-positional-encoding/)；
5. [Encoder、LayerNorm 与 FFN](/notes/transformer-encoder-block/)；
6. [Decoder、因果注意力与交叉注意力](/notes/transformer-decoder-cross-attention/)；
7. [训练、损失与自回归推理](/notes/transformer-training-inference/)；
8. [PyTorch 组装、代码核对与调试](/notes/transformer-pytorch-implementation/)。

## 1. 为什么需要 Transformer

### 1.1 RNN 的两个核心瓶颈

RNN 在第 $t$ 个时间步必须等待第 $t-1$ 个时间步的隐状态：

$$
h_t = f(x_t, h_{t-1})
$$

这种递归结构带来两个问题：

1. **难以并行**：一个序列内部必须逐步计算，GPU 很难同时处理所有 token；
2. **长距离路径过长**：序列首尾的信息需要经过许多次状态传递，容易衰减，也增加优化难度。

LSTM 和 GRU 缓解了长距离依赖问题，却没有消除时间步之间的串行依赖。

### 1.2 CNN 也不是最终答案

一维 CNN 可以并行处理序列，但一个卷积核通常只看到局部窗口。想让第一个 token 与很远的 token 交互，需要堆叠多层、扩大卷积核或使用空洞卷积。

Transformer 的问题意识是：

> 能否让任意两个位置直接交换信息，并且一次计算整个序列？

自注意力给出的答案是：让每个 token 主动查询序列中的所有 token，再按相关程度聚合信息。

## 2. Transformer 的一句话定义

Transformer 是一种以注意力为核心、用位置编码补充顺序信息、通过残差连接、归一化和前馈网络构建深层表示的序列模型。

它最初在 2017 年的论文《Attention Is All You Need》中用于机器翻译。原论文的关键贡献并不是“发明了所有注意力”，而是证明了：**不依赖循环和卷积，仅靠注意力也能构建高性能的序列到序列模型。**

## 3. 整体架构

初学时不必一次理解所有数据流，只需先把编码器、解码器和输出层看成三块相互衔接的结构；后续文章会逐块展开。

标准 Transformer 分为两半：

- 左侧 **Encoder**：把源序列编码成带上下文的表示；
- 右侧 **Decoder**：根据已经生成的目标 token 和 Encoder 表示，预测下一个 token。

### 3.1 输入端

设源序列 token 数为 $S$，目标序列 token 数为 $T$，模型维度为 $D$。

```text
源 token id:       [B, S]
源 token embedding:[B, S, D]
源位置编码:        [1, S, D] 或 [B, S, D]

目标 token id:       [B, T]
目标 token embedding:[B, T, D]
目标位置编码:        [1, T, D] 或 [B, T, D]
```

Embedding 提供“这个 token 是什么”，位置编码提供“这个 token 在哪里”。两者相加后进入网络。

### 3.2 Encoder Layer

一个 Encoder Layer 主要包含：

```text
多头自注意力 → 残差连接与 LayerNorm → FFN → 残差连接与 LayerNorm
```

其中，自注意力负责 token 之间的信息交换；FFN 对每个 token 的特征做相同的非线性变换。

### 3.3 Decoder Layer

一个 Decoder Layer 主要包含：

```text
带因果 Mask 的自注意力
→ 交叉注意力
→ FFN
```

每个子层附近都有残差连接和 LayerNorm。

Decoder 里的三部分分别解决：

- **因果自注意力**：汇总已经生成的目标端上下文，同时不能偷看未来答案；
- **交叉注意力**：从 Encoder 输出中查找与当前生成位置有关的源端信息；
- **FFN**：逐 token 变换特征。

### 3.4 输出端

Decoder 输出的 shape 仍为 $[B,T,D]$。线性层把 $D$ 维表示投影到词表大小 $V$：

$$
Z = HW_{vocab} + b
$$

```text
Decoder 输出 H: [B, T, D]
词表投影 W:     [D, V]
logits Z:       [B, T, V]
```

对最后一维做 softmax，便得到每个位置对词表中所有 token 的概率分布。

## 4. 三种注意力不要混淆

| 模块 | Query 来自 | Key、Value 来自 | 是否需要因果 Mask |
|---|---|---|---|
| Encoder 自注意力 | Encoder 当前序列 | Encoder 当前序列 | 否 |
| Decoder 自注意力 | Decoder 当前序列 | Decoder 当前序列 | 是 |
| Encoder-Decoder 交叉注意力 | Decoder 隐表示 | Encoder 输出 | 通常不需要 |

自注意力与交叉注意力使用同一个核心公式，差别主要在 $Q$、$K$、$V$ 的来源和序列长度。

## 5. 为什么它能并行

RNN 的时间步之间有依赖：

```text
x1 → h1 → h2 → h3 → ... → hn
```

Transformer 训练时可以把整个序列一次组成矩阵：

```text
X [B, L, D]
  ├─ XWq → Q
  ├─ XWk → K
  └─ XWv → V

Attention(Q, K, V) 一次得到所有位置的输出
```

这里的“并行”主要指：**训练时，同一层内所有位置可以并行计算。**

不要把它误解为“生成时也完全并行”。自回归 Decoder 推理仍然要先生成第 1 个 token，再生成第 2 个 token。KV Cache 可以减少重复计算，但不能取消 token 之间的生成依赖。

## 6. RNN、CNN 与自注意力的对比

设序列长度为 $L$，表示维度为 $D$，卷积核宽度为 $K$。

| 结构 | 单层主要复杂度 | 序列内并行 | 任意两点最短信息路径 |
|---|---:|---:|---:|
| RNN | $O(LD^2)$ | 否 | $O(L)$ |
| 一维 CNN | $O(KLD^2)$ | 是 | 随层数缩短 |
| 自注意力 | $O(L^2D)$ | 是 | $O(1)$ |

自注意力的优势是全局连接和短路径；代价是需要构造 $L\times L$ 的注意力矩阵。因此：

- 当 $L$ 不是特别大、$D$ 较大时，自注意力通常很有吸引力；
- 当 $L$ 极长时，$O(L^2)$ 的计算量和显存会成为主要瓶颈。

## 7. Transformer 的三类常见形态

### 7.1 Encoder-only

代表：BERT 一类模型。

每个位置可以看到左右两侧上下文，适合理解型任务：分类、NER、检索表示等。

### 7.2 Decoder-only

代表：GPT、LLaMA 一类模型。

使用因果 Mask，只能看到当前位置及其左侧内容，适合自回归生成。现代大语言模型大多采用这种形态。

### 7.3 Encoder-Decoder

代表：原始 Transformer、T5 一类模型。

Encoder 理解输入，Decoder 生成输出，特别适合翻译、摘要和其他输入输出不等长的 Seq2Seq 任务。

## 8. 原始 Transformer 的标准配置

论文中的 Base 模型常用配置如下：

| 超参数 | 数值 | 含义 |
|---|---:|---|
| Encoder 层数 | 6 | 堆叠 6 个 Encoder Layer |
| Decoder 层数 | 6 | 堆叠 6 个 Decoder Layer |
| $D$ | 512 | token 表示维度 |
| 头数 $H$ | 8 | 8 个注意力头 |
| 每头维度 $D/H$ | 64 | 每个头的 $Q,K,V$ 维度 |
| FFN 隐层维度 | 2048 | 先扩展 4 倍再压回 512 |
| Dropout | 0.1 | 正则化 |

这些数字是理解原论文的参照，不是所有现代模型都必须遵守的固定规则。

## 9. 优势与局限

### 优势

- 序列位置可以在训练阶段高度并行；
- 任意两个 token 可以在一层内直接交互；
- 架构容易扩大到更深、更宽和更多数据；
- 可以扩展到文本、图像、语音和多模态任务。

### 局限

- 全局注意力的时间和显存通常随序列长度平方增长；
- 自注意力本身不认识顺序，必须额外注入位置信息；
- 自回归生成仍有串行瓶颈；
- 注意力权重并不天然等于可解释的“原因”；
- 训练大模型需要大量数据、算力与工程优化。

## 10. 一个资料中的数值纠正

原论文报告的是：

- WMT 2014 英译德：Transformer Base 达到 **27.3 BLEU**，Big 达到 **28.4 BLEU**；
- WMT 2014 英译法：Transformer Big 达到 **41.8 BLEU**。

因此，不能把 28.4 和 41.8 写成同一个英德任务上从 0.284 提升到 0.418；它们来自不同翻译方向。阅读二手资料时，应回到原论文核对实验表格。

## 11. 常见误区

### 误区一：Transformer 完全没有顺序

注意力计算本身对排列是等变的，但模型会通过位置编码或相对位置机制加入顺序信息。

### 误区二：Transformer 能无限处理长文本

理论公式可以接受变长输入，但显存、最大位置范围、训练分布和注意力复杂度都会限制实际上下文长度。

### 误区三：Transformer 就等于注意力

注意力是核心，但残差连接、LayerNorm、FFN、位置机制、Mask、训练目标同样不可缺少。

### 误区四：Encoder 和 Decoder 的注意力完全一样

公式相似，但可见范围、输入来源和序列 shape 不同。特别是交叉注意力中，源长度 $S$ 与目标长度 $T$ 可以不相等。

## 一句话总结

Transformer 的本质是：用全局注意力缩短序列信息路径并提高训练并行性，再通过位置、残差、归一化和 FFN 把它组装成可训练的深层网络，但全局注意力要付出 $O(L^2)$ 的代价。

## 复习卡片

- 核心问题：RNN 串行计算与长距离信息路径过长；
- 核心机制：注意力让任意位置直接交互；
- 主要模块：Embedding、位置编码、MHA、FFN、残差、LayerNorm；
- 三种注意力：Encoder 自注意力、因果自注意力、交叉注意力；
- 三类架构：Encoder-only、Decoder-only、Encoder-Decoder；
- 主要限制：长序列下的二次复杂度与自回归生成串行性。

## 自测问题

1. Transformer 解决了 RNN 的哪两个核心瓶颈？
2. 为什么 Transformer 训练能并行，而自回归推理仍然串行？
3. Encoder 自注意力、Decoder 自注意力和交叉注意力的输入分别来自哪里？
4. 为什么 Decoder 输出长度由目标输入长度 $T$ 决定，而不是源长度 $S$？
5. 自注意力的 $O(L^2)$ 来自哪个中间矩阵？

## 参考与推荐阅读

- [Vaswani 等：Attention Is All You Need](https://arxiv.org/abs/1706.03762)：原始论文，重点看图 1、3.1～3.5 节和表 1～3。
- [动机与影响](https://www.rethink.fun/chapter15/%E5%8A%A8%E6%9C%BA%E4%B8%8E%E5%BD%B1%E5%93%8D.html)：适合先建立直觉，实验数值请以原论文为准。
- [Dive into Deep Learning：Self-Attention and Positional Encoding](https://d2l.ai/chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.html)：适合补充复杂度与位置编码推导。
- 课程课件《transformer算法原理-S》：本文架构图和矩阵图的主要参考。
