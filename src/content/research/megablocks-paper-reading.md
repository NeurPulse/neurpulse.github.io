---
title: 'MegaBlocks 精读｜当不同专家收到的 token 数量不一样'
description: '从三个专家的槽位手算到块稀疏矩阵乘法，结合两组端到端实验解释 MegaBlocks 的 4.35 倍和 1.18 倍。'
pubDate: 2026-11-19
kind: record
category: '论文精读'
tags: ['论文笔记', 'MoE', 'GPU 算子', '块稀疏矩阵乘法']
---

## 论文信息

- **题名与作者**　*MegaBlocks: Efficient Sparse Training with Mixture-of-Experts*，Trevor Gale、Deepak Narayanan、Cliff Young、Matei Zaharia，来自斯坦福大学、Microsoft Research 和 Google Research。
- **发表**　MLSys 2023。这里选读第 4 节的计算表示、第 5.2 节的路由实现，以及图 7、图 8 的训练对比。[会议收录页](https://proceedings.mlsys.org/paper_files/paper/2023/hash/5a54f79333768effe7e8927bcccffe40-Abstract-mlsys2023.html) · [论文 PDF][paper] · [开源实现](https://github.com/databricks/megablocks)

## 一句话总结

MegaBlocks 用块稀疏矩阵乘法让每条已选路由都能执行，在 8 张 A100 上把 Medium 模型训练到相同验证损失时，相比同样保留全部路由的 Tutel 快 4.35 倍，相比允许丢弃溢出路由并调优容量的 Tutel 快 1.18 倍。[图 7、图 8][results]

## 研究动机

GPU 擅长规则的大矩阵乘法，路由器产生的输入却不规则。比如三个专家各收到 9、2、1 个 token，给每位专家都分配 4 个槽，第一位有 5 个 token 进不去。如果按最忙的专家统一扩成 9 个槽，12 条路由都能算，但要分配 27 个槽。批次一变，最忙的专家和所需槽位数也会改变。[第 3 节与第 4.2 节][formulation]

## 核心方法

1. **按专家整理输入。** 从路由结果记录专家编号和 token 原位置，把同一专家的输入排在一起。后面仍需按原位置把输出写回。
2. **只为有效的专家组合安排矩阵块。** 把所有专家的第一层权重沿输出维度排列，某一组 token 只需要其中属于自己专家的列块。第一步乘法从两个稠密矩阵计算指定位置的块，作者记为 SDD。经过激活函数，第二步用这份块稀疏激活乘第二层权重，得到稠密输出，记为 DSD。第一个字母表示结果的布局，后两个表示两个输入的布局。
3. **恢复 token 顺序。** 把结果送回原 token 位置，并按路由权重合并。块内仍然执行规则的矩阵乘法，块之间有哪些任务则由当前批次的路由决定。[第 4 节][formulation] · [第 5.2 节][impl]

<figure class="paper-figure" id="fig-3">
  <a class="paper-panel" href="/images/research/moe-papers/megablocks-fig-3.webp"><img src="/images/research/moe-papers/megablocks-fig-3.webp" width="1946" height="508" loading="lazy" decoding="async" alt="图 A 的等尺寸批量矩阵乘法、图 B 的等大对角块与图 C 由小块拼出的不等大专家区域。"></a>
  <figcaption><strong>图 3</strong>　对比 B、C。B 中每个专家占同样大的区域，C 中每个专家的区域由多个小块拼成，可以随路由负载改变。沿一个 token 的行去看，空白位置对应这条路由无需计算的列。 <a href="https://proceedings.mlsys.org/paper_files/paper/2023/file/5a54f79333768effe7e8927bcccffe40-Paper-mlsys2023.pdf#page=4">Gale 等，论文原图</a></figcaption>
</figure>

**把前面的 12 条路由再算一次。** 为了手算，设块高为 4，三个组只需各自补到 4 的倍数。于是 $[9,2,1]\rightarrow[12,4,4]$，共分配 20 行。其中 12 行有效、8 行为对齐填充。比按最忙专家统一分配少 7 行，全部路由都得到计算。论文实现采用 128×128 的块，真实节省多少取决于每个专家收到的 token 数和块的利用率。[第 5.2 节][impl]

<figure class="paper-figure" id="fig-4">
  <a class="paper-panel" href="/images/research/scattermoe/megablocks-fig-4.webp"><img src="/images/research/scattermoe/megablocks-fig-4.webp" width="1452" height="1368" loading="lazy" decoding="async" alt="图 4 伪代码依次进行路由、建块索引、收集 token、两次矩阵乘、写回和乘门控权重。"></a>
  <figcaption><strong>图 4</strong>　从第 15 行的 padded_gather 往下读。第 22 和 23 行对应两次矩阵乘，第 26 和 27 行将结果重新排列并乘上路由权重。 <a href="https://proceedings.mlsys.org/paper_files/paper/2023/file/5a54f79333768effe7e8927bcccffe40-Paper-mlsys2023.pdf#page=5">论文原图</a></figcaption>
</figure>

## 相对前作的创新点

- **相对固定容量的 Switch。** Switch 为每个专家预留固定容量，超出的 token 可沿残差路径跳过专家。MegaBlocks 改变 GPU 上矩阵乘法的表示，使实际分配给每个专家的有效行都能参与计算。[Switch 图 3](https://jmlr.org/papers/volume23/21-0998/21-0998.pdf#page=5) · [MegaBlocks 第 4.2 节][formulation]
- **相对 Tutel 的动态容量。** Tutel 已支持随批次改变容量，也能保留全部路由。论文用作对照的 Tutel 无丢弃路径按最忙专家补齐批次，MegaBlocks 只把各专家组补到块边界。这样能减少中间激活占用的显存，并可能容纳更大的微批次。[Tutel 原论文](https://proceedings.mlsys.org/paper_files/paper/2023/file/5616d34cf8ff73942cfd5aa922842556-Paper-mlsys2023.pdf) · [MegaBlocks 第 6.1 节][results]

## 实验设置

作者在 The Pile 上训练解码器语言模型，使用 8 张 A100 SXM4 80 GB。XS、Small、Medium 三种模型都将每个前馈层替换为 64 个专家，采用 top-1 路由。全局批量为 512 个序列，各实现选用显存容得下的最大微批次，并用梯度累积达到全局批量。图 7、图 8 的横轴都是实际训练时间，纵轴是验证损失。[表 2、表 3][impl] · [第 6 节][results]

## 主要结果

| 在相同验证损失下，Tutel 耗时 ÷ MegaBlocks 耗时 | XS | Small | Medium |
| --- | ---: | ---: | ---: |
| 双方都保留全部路由，Tutel 动态补齐（图 7） | 1.38× | 2.00× | 4.35× |
| Tutel 可丢弃溢出路由，各模型取测得的最优容量配置（图 8） | 1.38× | 1.37× | 1.18× |

<figure class="paper-figure" id="fig-7-8">
  <div class="paper-panels">
    <a class="paper-panel" href="/images/research/moe-papers/megablocks-fig-7.webp"><img src="/images/research/moe-papers/megablocks-fig-7.webp" width="830" height="616" loading="lazy" decoding="async" alt="图 7，MegaBlocks 与同样保留全部路由的 Tutel、以及稠密基线的验证损失和训练时间曲线。"></a>
    <a class="paper-panel" href="/images/research/moe-papers/megablocks-fig-8.webp"><img src="/images/research/moe-papers/megablocks-fig-8.webp" width="831" height="615" loading="lazy" decoding="async" alt="图 8，MegaBlocks 与使用不同固定容量因子的 Tutel、以及稠密基线的验证损失和训练时间曲线。"></a>
  </div>
  <figcaption><strong>图 7、图 8</strong>　先看图例中 Tutel 用的是哪种容量策略，再在同一个纵轴损失值上横向比较耗时。两图横轴范围不同。 <a href="https://proceedings.mlsys.org/paper_files/paper/2023/file/5a54f79333768effe7e8927bcccffe40-Paper-mlsys2023.pdf#page=9">论文原图</a></figcaption>
</figure>

为何同一个 Medium 会出现两个倍数？图 7 中，Tutel 为保留全部路由而增加填充，只能容纳微批次 1，而 MegaBlocks 能容纳 8。图 8 允许 Tutel 调容量、跳过溢出路由，它的对照成绩随之改变。因此 4.35 倍与 1.18 倍都描述整段训练达到指定损失的时间，里面包含微批次大小等因素。[表 3][impl] · [图 7、图 8][results]

## 优点与不足

作者把两种容易混淆的基线都测了，还公开微批次大小，读者可以追溯一部分收益的来源。块对齐依然产生填充，维护分组和块索引也需要时间。对小批量推理、每个专家只有少量 token 的情况，需要重新测量。仅凭这组多卡训练曲线无法确定其表现。

## 对我的启发

可以先固定 12 条路由，只改变三个专家的负载分布，分别统计有效行数、按最忙专家补齐的行数和按块补齐的行数。再把这个统计接到矩阵乘法计时上，检验“省了多少行”和“省了多少时间”何时一致。想继续追踪整理 token 产生的内存读写，可以接着读 ScatterMoE 的图 1。[ScatterMoE 精读](/research/scattermoe-paper-reading/#fig-1)

[paper]: https://proceedings.mlsys.org/paper_files/paper/2023/file/5a54f79333768effe7e8927bcccffe40-Paper-mlsys2023.pdf
[formulation]: https://proceedings.mlsys.org/paper_files/paper/2023/file/5a54f79333768effe7e8927bcccffe40-Paper-mlsys2023.pdf#page=4
[impl]: https://proceedings.mlsys.org/paper_files/paper/2023/file/5a54f79333768effe7e8927bcccffe40-Paper-mlsys2023.pdf#page=8
[results]: https://proceedings.mlsys.org/paper_files/paper/2023/file/5a54f79333768effe7e8927bcccffe40-Paper-mlsys2023.pdf#page=9
