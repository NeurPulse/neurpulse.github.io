---
title: 'Mixtral 精读｜一个 token 怎样从八个专家里选两个'
description: '读懂 Mixtral 8×7B 的前馈层、参数口径和跨领域路由统计，用同一份评测表比较它与 Llama 2 70B。'
pubDate: 2026-11-19
kind: record
category: '论文精读'
tags: ['论文笔记', 'MoE', '专家路由', '大语言模型']
---

## 论文信息

- **题名与作者**　*Mixtral of Experts*，Albert Q. Jiang、Alexandre Sablayrolles 等 26 位作者，Mistral AI。
- **版本**　权重于 2023 年 12 月发布。本文按 2024 年 1 月的作者技术报告 arXiv:2401.04088v1 阅读。[论文全文](https://arxiv.org/html/2401.04088v1) · [论文 PDF](https://arxiv.org/pdf/2401.04088) · [官方推理实现](https://github.com/mistralai/mistral-inference)

## 一句话总结

Mixtral 8×7B 把解码器每层的前馈网络换成 8 个专家，每个 token 只运行其中 2 个，用约 12.9B 激活参数在作者复测的 MMLU 上取得 70.6%，同期稠密 Llama 2 70B 为 69.9%。[论文表 2](https://arxiv.org/pdf/2401.04088#page=4) · [作者的参数口径](https://mistral.ai/news/mixtral-of-experts/)

## 研究动机

稠密 Transformer 的每个 token 都要通过同一套前馈权重。想给模型更多参数时，直接加宽网络会让每个 token 做更多乘法。Mixtral 把可供选择的权重分成多套，让路由器决定当前 token 使用哪两套。模型要为全部专家保存权重，而一次前向只需读取被选专家的相应权重。总存储、每个 token 的计算量和实际服务速度由此成为三个可以分别测量的问题。[第 2.1 节与第 3 节](https://arxiv.org/html/2401.04088v1#S2.SS1)

## 核心方法

每个专家是一套独立的前馈网络。对当前层的 token 向量 $x$，路由器先计算 8 个分数，保留最高的两个，再在这两个分数上做 softmax。若被选中的专家是 $a,b$，这一层的专家输出就是 $p_aE_a(x)+p_bE_b(x)$。下一层会对该 token 当时的隐藏状态重新路由。注意力及其他共享部分照常计算。[第 2.1 节](https://arxiv.org/html/2401.04088v1#S2.SS1)

<figure class="paper-figure" id="fig-1">
  <a class="paper-panel" href="/images/research/moe-papers/mixtral-fig-1.webp"><img src="/images/research/moe-papers/mixtral-fig-1.webp" width="925" height="424" loading="lazy" decoding="async" alt="路由器为一个输入选出两个专家，两份专家输出经过门控加权后汇合。"></a>
  <figcaption><strong>图 1</strong>　从左往右跟一个输入，分清路由器产生的权重和专家产生的向量。最后的加号把同一个 token 的两份结果合起来。 <a href="https://arxiv.org/pdf/2401.04088#page=2">Jiang 等，论文原图</a> · <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a></figcaption>
</figure>

**用两个数走一遍。** 假设选中专家 2 和 5，归一化后的权重为 0.75、0.25，把这两个专家的输出简化为标量 8 和 4。合并后得到 $0.75\times8+0.25\times4=7$。另一个 token 即使选了相同的两个专家，权重和专家输出都可以不同。真实模型在每个隐藏维度上做同样的加权。这个教学例子的系数是给定的，实际由路由器学习。

**8×7B 怎么数？** 每层有 8 套专家前馈参数，注意力、嵌入等参数仍由专家共享。作者给出的整模型总量是 46.7B，每个 token 用到约 12.9B，二者都包括相应的共享参数。讨论显存容量时先看总量，讨论主要乘加量时再看激活量。[作者发布说明](https://mistral.ai/news/mixtral-of-experts/)

## 相对前作的创新点

- **与 Mistral 7B 对比。** 两者同属解码器模型。Mixtral 保留分组查询注意力等设计，把前馈块换成稀疏专家，并在其配置中使用完整的 32K 注意力上下文。用结构表可以把前馈层的变化与注意力设计分开。[Mistral 7B 第 2 节](https://arxiv.org/pdf/2310.06825#page=2) · [Mixtral 表 1](https://arxiv.org/pdf/2401.04088#page=2)
- **与已有稀疏门控对比。** 按输入选少量专家、再加权相加，Shazeer 等人在 2017 年已明确写出。Mixtral 的贡献落在开放权重的 8×7B 解码器、与稠密模型的系统评测，以及对实际路由分布的观察。[Shazeer 等，第 2 节](https://openreview.net/pdf?id=BkFEbNwIg) · [Mixtral 第 5 节](https://arxiv.org/html/2401.04088v1#S5)

## 实验设置

表 1 给出 32 层、模型宽度 4096、专家隐藏宽度 14336、32 个查询头和 8 个键值头，词表大小 32000，上下文 32768。表 2 对比基础模型，包含 Mistral 7B、Mixtral 和若干 Llama 基线。这里读模型能力与激活参数量，指令微调版及聊天偏好评估在另外的章节。第 5 节在 The Pile 验证集的多个领域统计每层选中各专家的比例。[结构与评测](https://arxiv.org/pdf/2401.04088#page=2) · [路由统计](https://arxiv.org/pdf/2401.04088#page=7)

## 主要结果

| 表 2 的基础模型评测 | Mixtral 8×7B | Llama 2 70B | 相差多少 |
| --- | ---: | ---: | ---: |
| MMLU | 70.6% | 69.9% | +0.7 个百分点 |
| HumanEval | 40.2% | 29.3% | +10.9 个百分点 |
| GSM8K | 74.4% | 69.6% | +4.8 个百分点 |
| HellaSwag | 84.4% | 85.4% | −1.0 个百分点 |

这里每列是一整个模型的任务分数。以 HumanEval 为例，40.2 减去 29.3 得到的是 10.9 个百分点。作者没有用这张表报告毫秒延迟。要把较少的激活参数换算为服务吞吐，还需要批大小、设备、精度和路由实现。[表 2 与第 3 节 Size and Efficiency](https://arxiv.org/pdf/2401.04088#page=4)

<figure class="paper-figure" id="fig-7">
  <a class="paper-panel" href="/images/research/moe-papers/mixtral-fig-7.webp"><img src="/images/research/moe-papers/mixtral-fig-7.webp" width="1578" height="1068" loading="lazy" decoding="async" alt="在三个不同层，多个文本领域使用八个专家的比例大体相近，部分数据集和层有偏差。"></a>
  <figcaption><strong>图 7</strong>　三排分别是第 0、15、31 层，颜色表示语料领域，虚线是均匀分配的 1/8。作者将第一和第二选择都计入统计。 <a href="https://arxiv.org/pdf/2401.04088#page=7">论文原图</a></figcaption>
</figure>

如果有人问“是不是有一个数学专家、一个生物专家”，可以请他先在图 7 同一组柱里比较 ArXiv、PubMed 和 PhilPapers 的颜色。作者观察到这几个领域的专家分布相近，DM Mathematics 在部分层有所偏离。领域级频率回答的是专家被选了多少次，单个专家学到什么功能还需要更细的实验。

## 优点与不足

开放权重和推理代码让一条输入的 top-2 路由可以实际检查。横向评测覆盖推理、代码与常识，也列出了 Mixtral 得分较低的任务。这些是不同预训练模型之间的比较，无法把全部分差单独归因于 top-2。推理时保存全部专家权重、为不同专家取数，同样会占用显存和带宽。

## 对我的启发

讲这篇论文时，可以把“全部权重存在哪里”和“当前 token 做了哪些乘法”画成两列。一个最小验证是给 4 个 token 指定 top-2 结果，共有 8 条路由记录。先逐 token 计算，再按专家将记录分组、计算并写回，最后检查两种实现的输出是否一致。[查看 ScatterMoE 怎样融合分组和乘法](/research/scattermoe-paper-reading/#fig-1)
