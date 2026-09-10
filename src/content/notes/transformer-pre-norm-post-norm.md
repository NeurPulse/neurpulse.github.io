---
title: 'LLaMA（4）：Pre-Norm 与 Post-Norm 的结构、梯度和取舍'
description: '从残差路径出发比较 Pre-Norm 与 Post-Norm，理解两者的公式、梯度传播、训练稳定性和 LLaMA 的选择。'
pubDate: 2026-09-10
category: 'LLaMA'
tags: ['Pre-Norm', 'Post-Norm', '残差连接', '训练稳定性', 'LLaMA 1']
order: 4
---

## 学习目标

本文回答一个常见面试题：Pre-Norm 和 Post-Norm 有什么区别？学完后应能根据结构图写出公式，从梯度路径解释训练稳定性，并知道为什么 LLaMA 选择 Pre-Norm。

## 先修知识

- [RMSNorm 原理及其与 LayerNorm 的区别](/notes/transformer-rmsnorm/)
- [Encoder、LayerNorm、残差与 FFN](/notes/transformer-encoder-block/)
- [反向传播与链式法则](/notes/deep-learning-backpropagation/)

## 1. 先区分“类型”和“位置”

RMSNorm 与 LayerNorm 描述归一化的**计算方式**；Pre-Norm 与 Post-Norm 描述归一化相对残差子层的**放置位置**。

为了专注位置差异，下面统一用 $\operatorname{Norm}$ 表示任意归一化，用 $F$ 表示 Attention 或 FFN 子层。

## 2. Post-Norm

原始 Transformer 使用 Post-Norm：先计算子层并残差相加，再做归一化。

$$
x_{l+1}=\operatorname{Norm}\left(x_l+F_l(x_l)\right)
$$

流程为：

```text
x_l ───────────────┐
  └→ F_l(x_l) ─→ 加法 ─→ Norm ─→ x_{l+1}
```

“Post”指 Norm 位于残差加法之后，不是指模型输出之后。

## 3. Pre-Norm

Pre-Norm 先归一化，再把子层输出加入未归一化的残差主干：

$$
x_{l+1}=x_l+F_l\left(\operatorname{Norm}(x_l)\right)
$$

流程为：

```text
x_l ───────────────────────→ 加法 ─→ x_{l+1}
  └→ Norm ─→ F_l(·) ──────────┘
```

LLaMA 的每个 Decoder Block 对 Attention 和 FFN 都采用这种形式，并在全部层之后再加 Final RMSNorm。

## 4. 最直观的差异：残差高速公路

Pre-Norm 中存在从 $x_l$ 到 $x_{l+1}$ 的直接恒等路径：

$$
\frac{\partial x_{l+1}}{\partial x_l}
=I+\frac{\partial F_l(\operatorname{Norm}(x_l))}{\partial x_l}
$$

梯度至少拥有一条不必先穿过 Norm 的恒等通路。这通常使深层网络初始化阶段的梯度更稳定，对 warmup 和学习率更不敏感。

Post-Norm 的梯度必须经过最外层 Norm：

$$
\frac{\partial x_{l+1}}{\partial x_l}
=J_{\operatorname{Norm}}
\left(I+J_{F_l}\right)
$$

多层连乘后，Norm 的 Jacobian 会反复进入主梯度路径。研究表明，原始 Post-LN Transformer 在初始化时靠近输出层的梯度可能较大，因此通常更依赖精心设计的学习率 warmup。

## 5. 展开多层 Pre-Norm

令：

$$
x_{l+1}=x_l+F_l(\operatorname{Norm}(x_l))
$$

递推展开：

$$
x_L=x_0+\sum_{l=0}^{L-1}F_l(\operatorname{Norm}(x_l))
$$

这说明最终残差流保留 $x_0$ 的直接贡献，并累积每一层的增量。它有利于优化，但也意味着残差流尺度可能随深度增长，所以 LLaMA 在堆叠层末尾使用 Final RMSNorm，再送入输出投影。

## 6. 不要把方差推导过度简化

课堂手写推导常用“$x$ 与 $F(x)$ 独立、方差都为 1”建立直觉，但真实网络中二者通常相关：

$$
\operatorname{Var}[x+F(x)]
=\operatorname{Var}[x]+\operatorname{Var}[F(x)]
+2\operatorname{Cov}[x,F(x)]
$$

因此不能严格断言：

- 每经过一个残差块，方差一定恰好乘 2；
- Pre-Norm 的梯度永远等于 1；
- Post-Norm 一定无法训练。

这些简化只能帮助理解趋势，不能代替实际梯度、协方差和初始化分析。

## 7. 两者对比

| 对比维度 | Pre-Norm | Post-Norm |
|---|---|---|
| 公式 | $x+F(\operatorname{Norm}(x))$ | $\operatorname{Norm}(x+F(x))$ |
| Norm 位置 | 子层之前 | 残差加法之后 |
| 残差主路径 | 存在直接恒等路径 | 主路径经过 Norm |
| 深层训练 | 通常更稳定 | 通常更依赖 warmup 和初始化 |
| 最后一层 | 常增加 Final Norm | 每块输出已经被 Norm |
| 代表架构 | GPT-2/3、LLaMA 等 | 原始 Transformer |

“Pre-Norm 更稳定”不等于“任何任务上最终效果一定更好”。部分研究观察到 Post-Norm 在调优充分时可能有较好的表示效果，但训练更敏感；实际选择还取决于模型规模、初始化、优化器和训练配方。

## 8. LLaMA Block 中的实际位置

设 `attention_norm` 与 `ffn_norm` 均为 RMSNorm：

```python
h = x + attention(attention_norm(x))
out = h + ffn(ffn_norm(h))
```

堆叠完成后：

```python
logits = output_linear(final_norm(out))
```

这就是 **Pre-RMSNorm + Final RMSNorm**。同一层里有两个归一化模块，它们各自拥有独立的可学习缩放参数。

## 9. 为什么 Pre-Norm 训练更稳但仍需调参

Pre-Norm 改善梯度路径，但无法消除：

- 学习率过大；
- 低精度溢出；
- 数据异常；
- Attention logits 过大；
- 梯度累积和分布式通信错误；
- 残差流随深度增长。

所以真实大模型仍需要合理初始化、优化器、学习率计划、梯度裁剪或监控，以及稳定的归一化内核。

## 10. 面试回答模板

可以用四句话回答：

1. Post-Norm 是 $\operatorname{Norm}(x+F(x))$，Pre-Norm 是 $x+F(\operatorname{Norm}(x))$；
2. Pre-Norm 保留不经过归一化的恒等残差路径，梯度传播通常更稳定；
3. Post-Norm 是原始 Transformer 的结构，通常更依赖 warmup 和精心初始化；
4. LLaMA 使用 Pre-RMSNorm，并在所有 Block 之后再做一次 Final RMSNorm。

## 11. 常见误区

### 误区一：Post-Norm 就是在模型最后做一次 Norm

Post-Norm 指每个残差子层内部的 Norm 位于加法之后。

### 误区二：Pre-Norm 不需要 Final Norm

许多 Pre-Norm 模型在堆叠末尾仍需要 Final Norm，以稳定送入输出头的残差流尺度。

### 误区三：Pre-Norm 的每一层都共享同一个 Norm

不同层以及 Attention/FFN 前通常都有独立归一化参数。

### 误区四：选择 RMSNorm 就自动变成 Pre-Norm

归一化公式与放置位置是正交设计选择。

## 一句话总结

Pre-Norm 的本质是把归一化移入残差分支前，让主干保留恒等梯度通路，从而更易训练深层模型；Post-Norm 输出更直接受归一化约束，但优化通常更敏感。

## 复习卡片

- Post：`Norm(x + F(x))`；
- Pre：`x + F(Norm(x))`；
- 原始 Transformer：Post-LN；
- LLaMA：Pre-RMSNorm + Final RMSNorm；
- Pre 优势：直接残差/梯度通路；
- 注意：稳定不等于无需 warmup 或一定效果更好；
- 方差相加要考虑协方差。

## 自测问题

1. 如何只看公式判断是 Pre-Norm 还是 Post-Norm？
2. Pre-Norm 的恒等路径为什么有利于梯度传播？
3. 为什么多层 Pre-Norm 末尾常有 Final Norm？
4. 为什么不能假设残差相加后方差一定等于 2？
5. 如何在 30 秒内完整回答 Pre-Norm 与 Post-Norm 的面试题？

## 下一篇

[LLaMA（5）：从零实现 RMSNorm](/notes/transformer-rmsnorm-pytorch/)

## 参考与推荐阅读

- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)：Pre-LN 与 Post-LN 的梯度分析。
- [LLaMA 1 论文](https://arxiv.org/abs/2302.13971)：LLaMA 的 Pre-RMSNorm 设计。
- [深层网络中的残差连接](/notes/deep-learning-resnet/)：复习恒等映射和残差梯度路径。
