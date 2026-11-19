---
title: '稀疏门控 MoE 精读｜为什么能有 4096 个专家、每次只用 4 个'
description: '从一个能手算的门控例子出发，读 Shazeer 等人的 ICLR 2017 论文，区分专家数、每步计算量、负载和困惑度。'
pubDate: 2026-11-19
kind: record
category: '论文精读'
tags: ['论文笔记', 'MoE', '条件计算', '负载均衡']
---

## 论文信息

- **题名与作者**　*Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*。作者为 Noam Shazeer、Azalia Mirhoseini、Krzysztof Maziarz、Andy Davis、Quoc Le、Geoffrey Hinton、Jeff Dean，来自 Google Brain 和雅盖隆大学。
- **发表**　ICLR 2017。下文图号和页码对应作者公开的 arXiv v1，会议版也列有文中比较的表 7。[会议页面](https://openreview.net/forum?id=BkFEbNwIg) · [会议版 PDF](https://openreview.net/pdf?id=BkFEbNwIg) · [配图版本 PDF](https://arxiv.org/pdf/1701.06538v1) · [Tensor2Tensor 专家分发实现](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/expert_utils.py)

## 一句话总结

在 1 Billion Word 语言建模实验中，作者用门控网络让每个输入始终只用 4 个专家，把备选专家从 4 个扩到 4096 个分层专家时，每步前向乘加从 840 万次增至 890 万次，10 轮训练后的测试困惑度从 45.0 降到 34.1。[第 5.1 节](https://arxiv.org/pdf/1701.06538v1#page=6) · [表 7](https://arxiv.org/pdf/1701.06538v1#page=15)

## 研究动机

混合专家（Mixture of Experts，MoE）里的“专家”各是一套有独立参数的小网络，“门控”则是决定当前输入由哪些专家处理的另一个网络。如果整个模型对每个词都使用同一套参数，扩大网络一般会增加每个词的计算成本。作者想把模型能保存多少知识和处理一个输入要做多少乘法拆开设计。难点在于把一个大批次拆给很多专家之后，每个专家可能只接到少数输入，GPU 难以高效工作。有的专家长期接不到样本，也很难学到有用的权重。[引言与第 3 节](https://arxiv.org/pdf/1701.06538v1#page=2)

## 核心方法

给定输入向量 $x$，门控网络为每个专家打分，只保留分数最高的 $k$ 个，在保留下来的分数上做 softmax。输出 $y=\sum_{i\in\mathrm{TopK}}g_i(x)E_i(x)$。其中 $g_i$ 是一个标量，$E_i(x)$ 是专家计算的向量。训练时，作者向分数加入幅度可学习的高斯噪声，让更多专家有机会被选中。用于语言建模时，作者把这一层放在两层 LSTM 中间。[第 2 节与公式 1–5](https://arxiv.org/pdf/1701.06538v1#page=3)

<figure class="paper-figure" id="fig-1">
  <a class="paper-panel" href="/images/research/moe-papers/shazeer-fig-1.webp"><img src="/images/research/moe-papers/shazeer-fig-1.webp" width="1448" height="734" loading="lazy" decoding="async" alt="循环语言模型中，门控网络为一个输入选中两个前馈专家，将它们的结果加权后传给上层。"></a>
  <figcaption><strong>图 1</strong>　先沿一个词从下层 LSTM 走到门控网络，再沿两条有权重的连线走到上层 LSTM。示意图画了两个被选专家，后面表 7 的实验每个输入选 4 个。 <a href="https://arxiv.org/pdf/1701.06538v1#page=2">Shazeer 等，论文原图</a></figcaption>
</figure>

**门控权重怎样作用在向量上？** 为便于手算，取 $k=2$，设专家 A、C 入选，分数分别是 $\ln 3$ 和 0。指数化后是 3 和 1，归一化得到 0.75 和 0.25。假设两位专家的输出为 $(4,0)$ 和 $(0,8)$，整层的输出就是 $(3,2)$。这个例子只解释一次前向，真实分数和专家权重都是训练学到的。

作者还区分两个容易混淆的量。某专家的 **importance** 是一批输入分给它的门控权重之和，**load** 关注送给它多少个输入，并使用可求导的期望负载来训练。一个专家接两个权重为 0.5 的输入，另一个接四个权重为 0.25 的输入，权重和都是 1，但后者要处理的输入数是前者的两倍。论文分别对二者施加均衡损失。[第 4 节](https://arxiv.org/pdf/1701.06538v1#page=5) · [附录 A](https://arxiv.org/pdf/1701.06538v1#page=13)

## 相对前作的创新点

- **门控已有前例。** Jacobs 等在 1991 年便用输入相关的门控组合多个专家，Eigen 等随后研究了多层混合专家。2017 年这篇论文把稀疏 top-k、带噪门控及均衡训练组合起来，并在大规模语言任务中验证了这种设计。[Jacobs 等原文](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf) · [Eigen 等原文](https://arxiv.org/abs/1312.4314)
- **把条件计算变成可跑的多卡训练。** 普通层可用数据并行，各设备上的输入按路由汇集给分布在不同设备的专家。论文还测试两级门控，先选专家组，再在组内选专家，以便扩展专家总数。[第 3 节](https://arxiv.org/pdf/1701.06538v1#page=4) · [附录 B、C](https://arxiv.org/pdf/1701.06538v1#page=14)

## 实验设置

以下只看 1 Billion Word 数据集上的低计算量实验。模型用两层 LSTM 夹一个 MoE 层，在 16 张 Tesla K40 上同步训练，批次约 30 万词，10 轮约 2.7 万步。4、32、256 专家用单层门控，表中 4096 专家用两级门控，每个输入最终都激活 4 个专家。每步乘加统计为一个输入在一个时间步的前向计算，排除最后的 softmax 输出层。表中参数列还排除了嵌入层。[第 5.1 节](https://arxiv.org/pdf/1701.06538v1#page=6) · [附录 C 与表 7](https://arxiv.org/pdf/1701.06538v1#page=15)

## 主要结果

| 备选专家数 | 每个输入用几个专家 | 每步前向乘加（百万） | 除嵌入和输出层的参数（百万） | 测试困惑度 ↓ |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 4 | 8.4 | 8.4 | 45.0 |
| 32 | 4 | 8.4 | 37.8 | 39.7 |
| 256 | 4 | 8.6 | 272.9 | 35.7 |
| 4096（分层） | 4 | 8.9 | 4303.4 | 34.1 |

困惑度越低，表示模型分给真实下一个词的概率在该测试集上的几何平均值越高。看前两行，乘加量相同，专家从 4 增至 32，困惑度下降 5.3。看最后一行，43.034 亿是排除嵌入和输出层后的参数量。比较第 1、4 行时还应说明门控结构同时变了。[原论文表 7](https://arxiv.org/pdf/1701.06538v1#page=15)

<figure class="paper-figure" id="fig-2">
  <a class="paper-panel" href="/images/research/moe-papers/shazeer-fig-2.webp"><img src="/images/research/moe-papers/shazeer-fig-2.webp" width="1479" height="530" loading="lazy" decoding="async" alt="图 2 左侧展示近似相同计算量下模型参数与测试困惑度的关系，右侧横轴改为每个时间步的计算量。"></a>
  <figcaption><strong>图 2</strong>　两幅图纵轴都是困惑度。汇报上表时先讲左图，它把每步预算控制在约 800 万次乘加。右图改变了计算预算，回答的是另一个问题。 <a href="https://arxiv.org/pdf/1701.06538v1#page=6">论文原图</a></figcaption>
</figure>

## 优点与不足

作者同时报告参数、前向乘加、测试困惑度及每张 GPU 的实测计算效率。在表 7 中，256 个平铺专家对应 0.81 TFLOPS/GPU，4096 个分层专家是 0.74，提高模型容量仍会付出调度和通信代价。这套实验使用循环网络、大批次与多卡训练，把这些数字用于 Transformer 自回归推理时需要重新测量。[表 7](https://arxiv.org/pdf/1701.06538v1#page=15)

## 对我的启发

做一个只有 8 个 token 的演示便可讲清为什么要观察每个专家的批量。每个 token 选 2 个专家，一共 16 条路由，若平均分给 4 个专家，每个专家收到 4 条。把专家数量扩到 16 而总路由条数不变，平均每个专家只剩 1 条。这样既能解释稀疏计算如何省乘法，也能引出为什么后来的系统论文要研究有效批量和数据搬运。[Switch 的单专家路由](/research/switch-transformers-paper-reading/) · [ScatterMoE 的 GPU 实现](/research/scattermoe-paper-reading/)
