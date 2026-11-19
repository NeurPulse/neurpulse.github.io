---
title: 'DeepSeekMoE 精读｜为什么把专家拆小，还要留一个共享专家'
description: '从固定专家计算量的手算例子出发，读懂细粒度路由、共享专家，以及关闭共享专家后损失为什么会上升。'
pubDate: 2026-11-19
kind: record
category: '论文精读'
tags: ['论文笔记', 'MoE', '专家路由', '大语言模型']
---

## 论文信息

- **题名**　*DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models*。
- **作者与机构**　Damai Dai、Chengqi Deng 等。作者来自 DeepSeek-AI、北京大学、清华大学及南京大学。
- **版本**　ACL 2024 长文，1280–1297 页。图表编号采用 ACL 正式版。[会议记录与 DOI](https://aclanthology.org/2024.acl-long.70/) · [论文 PDF](https://aclanthology.org/2024.acl-long.70.pdf) · [作者仓库与模型权重](https://github.com/deepseek-ai/DeepSeek-MoE)

## 一句话总结

DeepSeekMoE 把大专家切成可独立选择的小专家，并让每个 token 都经过少量共享专家，在作者的 2B 参数对照实验中，与 GShard 每次激活相同规模的参数，Pile 测试损失从 1.867 降至 1.808。[表 1](https://aclanthology.org/2024.acl-long.70.pdf#page=5)

## 研究动机

在普通 top-2 MoE 层里，路由器为当前 token 选两个前馈网络。如果某个大专家同时学到两种能力，只要用到其中一种，整个专家都要运行。作者还担心，若许多 token 都需要相似的变换，不同专家可能反复用各自的权重去学习它。把专家拆细，能否让路由选择更灵活？留出始终工作的专家，能否让其余专家承担更有区别的工作？[第 1、3 节](https://aclanthology.org/2024.acl-long.70.pdf#page=2)

## 核心方法

看图 1 的三栏。左边从 $N$ 个完整专家中选 2 个。中间把每个专家的前馈隐藏宽度减半，从 $2N$ 个小专家中选 4 个。右边把其中 1 个定为共享专家，每个 token 都经过它，路由器从剩余的小专家里再选 3 个。共享分支的输出直接相加，选中的路由分支按门控权重相加，最后还有残差连接。[式 (6)–(11)](https://aclanthology.org/2024.acl-long.70.pdf#page=3)

<figure class="paper-figure" id="fig-1">
  <a class="paper-panel" href="/images/research/moe-papers/deepseek-fig-1.webp"><img src="/images/research/moe-papers/deepseek-fig-1.webp" width="1819" height="892" loading="lazy" decoding="async" alt="从普通 top-2，到四个半宽专家，再到一个共享专家和三个按 token 选择的专家。"></a>
  <figcaption><strong>图 1</strong>　沿最右栏的绿色分支往上看，每个输入都会经过这个专家。蓝色分支的选择随输入而变。 <a href="https://aclanthology.org/2024.acl-long.70.pdf#page=3">Dai 等，ACL 2024 原图</a> · <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a></figcaption>
</figure>

**手算一个教学例子。** 设 4 个专家的隐藏宽度都是 8，原来每次选 2 个，合计经过 16 个隐藏通道。把每个专家拆成两个宽度为 4 的专家后，总宽度仍为 32，每次选 4 个，合计仍为 16。此时路由器可以分别选来自四个不同大专家的小块。再固定一个共享专家，设两个 token 的选择分别为“共享、1、3、5”和“共享、2、3、7”。两者共享一条始终工作的路径，专家 3 又可以被两个 token 同时选中。在这个例子里，每个 token 用到的隐藏通道数不变，专家线性层的主要乘加量相当。实际执行时，要合并的专家输出从 2 份变为 4 份，其中 1 份可以由所有 token 共用的分支计算。

## 相对前作的创新点

- **对照 GShard 的门控。** GShard 的算法 1 为 token 选择至多两个完整专家，并处理容量限制。DeepSeekMoE 沿用按 token 打分和加权的计算方式，把专家宽度和激活个数一起调整，使同等激活宽度有更多组合。[GShard 算法 1](https://arxiv.org/pdf/2006.16668#page=7) · [本文第 3.1 节](https://aclanthology.org/2024.acl-long.70.pdf#page=3)
- **共享路径有前例。** DeepSpeed-MoE 的 Residual-MoE 已让固定的稠密分支和动态专家共同处理输入。DeepSeekMoE 在第 3.2 节引用了这项前作，它把共享专家纳入细粒度专家预算，并用固定总参数和激活参数的实验分开比较两项改动。[DeepSpeed-MoE 第 4.1 节](https://arxiv.org/pdf/2201.05596#page=8) · [本文第 3.2、4.4 节](https://aclanthology.org/2024.acl-long.70.pdf#page=4)

## 实验设置

- **验证架构的 2B 实验。** 模型有 9 层，隐藏维度 1280，训练 100B 个 token。表 1 中 GShard 和 DeepSeekMoE 都约有 2.0B 总参数、0.3B 激活参数，每 2K 个 token 的计算量均为 4.3T FLOPs。语言建模看 Pile 交叉熵，越低越好。[第 4.1 节与表 1](https://aclanthology.org/2024.acl-long.70.pdf#page=4)
- **放大到 16B 的实验。** 共 28 层，第一层保留稠密前馈网络。其余各层有 2 个共享专家和 64 个路由专家，每个 token 激活两个共享专家以及 6 个路由专家。训练量为 2T 个 token。论文附录 B 记录了 A100、H800 集群。表 2 报的是计算量和模型质量。[第 5.1 节](https://aclanthology.org/2024.acl-long.70.pdf#page=7) · [附录 B](https://aclanthology.org/2024.acl-long.70.pdf#page=15)

## 主要结果

| 对照 | 作者报告的结果 | 怎么解读 |
| --- | --- | --- |
| 2B，GShard 与 DeepSeekMoE | Pile 损失 1.867 → 1.808，HellaSwag 零样本准确率 50.5% → 54.8%。 | 激活计算量相同，整套设计在这两项上有收益。[表 1](https://aclanthology.org/2024.acl-long.70.pdf#page=5) |
| 已训练的 DeepSeekMoE 2B，关闭共享专家，同时多启用一个路由专家 | Pile 损失 1.808 → 2.414。 | 激活预算保持相同，临时替换仍损害模型表现。该实验能说明这个共享专家在已学到的模型中很重要。[第 4.5 节](https://aclanthology.org/2024.acl-long.70.pdf#page=6) |
| DeepSeekMoE 16B 与稠密 DeepSeek 7B | 每 4K 个 token 分别为 74.4T 和 183.5T FLOPs。HellaSwag 为 77.1% 和 75.4%，MMLU 为 45.0% 和 48.2%。 | 计算量比约为 40.5%，不同任务上的胜负不同。[表 2](https://aclanthology.org/2024.acl-long.70.pdf#page=8) |

<figure class="paper-figure" id="fig-2">
  <a class="paper-panel" href="/images/research/moe-papers/deepseek-fig-2.webp"><img src="/images/research/moe-papers/deepseek-fig-2.webp" width="1531" height="463" loading="lazy" decoding="async" alt="六个基准上的专家结构消融，蓝柱是常规 GShard，其他颜色依次引入共享专家和更细的划分。"></a>
  <figcaption><strong>图 2</strong>　先在同一个任务里比较前两根柱，观察加入共享专家的变化，再读后两根柱看专家变细的影响。纵轴按各任务最佳表现归一化，柱高 1 不能读成准确率 100%。 <a href="https://aclanthology.org/2024.acl-long.70.pdf#page=5">论文原图</a></figcaption>
</figure>

## 优点与不足

验证实验把总参数量与激活参数量控制得很清楚，还对训练好的模型关闭指定专家。前者检验设计组合，后者检验某条分支在现有模型中的作用。关闭分支后的损失变化仍不足以给某个专家贴上具体知识标签。对部署而言，4.3T FLOPs 也无法给出 GPU 耗时。小专家更窄、路由条目更多，这些因素都要单独测量。

## 对我的启发

汇报时可以先画图 1，接着让大家判断“用另一个同宽专家替换共享专家，计算量没变，质量会怎样”。再给出 1.808 和 2.414，实验就回答了开头的问题。若从算子角度继续，可以固定 token 数和激活总宽度，用“2 个宽专家”和“4 个窄专家”分别测量分组、矩阵乘和合并时间。[接着读 ScatterMoE 的数据搬运实验](/research/scattermoe-paper-reading/#fig-4)
