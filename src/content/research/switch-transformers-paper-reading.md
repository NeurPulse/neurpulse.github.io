---
title: 'Switch Transformers 精读｜每个 token 只选一个专家，会发生什么'
description: '用四个 token 算清 top-1 路由、专家容量与残差，再用同一张表解释训练吞吐和达到目标质量的时间。'
pubDate: 2026-11-19
kind: record
category: '论文精读'
tags: ['论文笔记', 'MoE', '专家路由', '负载均衡']
---

## 论文信息

- **题名与作者**　*Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*，William Fedus、Barret Zoph、Noam Shazeer，Google。
- **发表**　*Journal of Machine Learning Research*，2022，23(120)，1–39 页。这里选读第 2 节路由机制和表 1 的受控实验。[期刊页面](https://jmlr.org/papers/v23/21-0998.html) · [论文 PDF](https://jmlr.org/papers/volume23/21-0998/21-0998.pdf) · [Mesh TensorFlow 的 MoE 实现](https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/moe.py)

## 一句话总结

Switch 让每个 token 在一个稀疏前馈层只访问一个专家，使 Switch-Base 在 32 个 TPUv3 核心、容量因子为 1 的实验中达到每秒 1000 个样本，高于 top-2 MoE-Base 的 860 个，也更早达到同一质量阈值。[表 1](https://jmlr.org/papers/volume23/21-0998/21-0998.pdf#page=7)

## 研究动机

假设某个 token 的前两个专家分布在两台设备上。用 top-2 时，它可能要被发往两处，运行两套前馈网络，再把结果收回来。只走一条路可以减少专家计算和传输，但也提出两个问题。路由器怎样从一次选择中获得训练信号？当许多 token 同时选中一个专家时，超出的 token 会去哪里？[第 2.1–2.2 节](https://jmlr.org/papers/volume23/21-0998/21-0998.pdf#page=4)

## 核心方法

路由器对所有专家的分数做 softmax，得到概率 $p_i(x)$，选择概率最大的专家 $i$，返回 $p_i(x)E_i(x)$。假设专家 A、B 的概率分别为 0.7、0.3，A 输出 10，专家分支就输出 7。被选专家的概率仍参与乘法，梯度可以通过这一项回到路由器。[公式 1–3](https://jmlr.org/papers/volume23/21-0998/21-0998.pdf#page=4)

<figure class="paper-figure" id="fig-2">
  <a class="paper-panel" href="/images/research/moe-papers/switch-fig-2.webp"><img src="/images/research/moe-papers/switch-fig-2.webp" width="1569" height="787" loading="lazy" decoding="async" alt="两个 token 通过同一个路由器，各自走向一个前馈专家，所选专家的输出乘以路由概率。"></a>
  <figcaption><strong>图 2</strong>　沿一个 token 的箭头看，路由器输出一个专家编号和对应概率。不同 token 可以选同一专家。 <a href="https://jmlr.org/papers/volume23/21-0998/21-0998.pdf#page=4">Fedus 等，论文原图</a> · <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a></figcaption>
</figure>

**用四个 token 算一遍容量。** 一组有 $T$ 个 token、$E$ 个专家，容量因子为 $c$ 时，作者按 $C=(T/E)\times c$ 为每个专家预留位置。设四个 token 依次选 A、A、A、B。容量因子为 1 时，每个专家有 2 个位置。A 的第三个 token 通过残差连接继续到下一层，跳过本层专家计算，B 则有一个位置空着。容量因子改成 1.5，每个专家有 3 个位置，四个 token 都能经过专家，但总共得预留 6 个位置。这里选用能整除的小数字，便于手算。

<figure class="paper-figure" id="fig-3">
  <a class="paper-panel" href="/images/research/scattermoe/switch-fig-3.webp"><img src="/images/research/scattermoe/switch-fig-3.webp" width="1752" height="692" loading="lazy" decoding="async" alt="两种容量因子下三个专家的槽位分配，显示溢出和空槽可以同时出现。"></a>
  <figcaption><strong>图 3</strong>　先找装满的专家，再找其他专家的空位。固定容量按专家分配，两者可以同时出现。 <a href="https://jmlr.org/papers/volume23/21-0998/21-0998.pdf#page=5">论文原图</a> · <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a></figcaption>
</figure>

作者用辅助损失鼓励均衡。对专家 $i$，$f_i$ 是实际选中它的 token 比例，$P_i$ 是全组 token 分给它的平均概率。损失正比于 $\sum_i f_iP_i$，高负载专家如果还持续得到很高的概率，就会增加这一项。它鼓励整个批次分散路由，某一批仍可能溢出。[公式 4–6](https://jmlr.org/papers/volume23/21-0998/21-0998.pdf#page=6)

## 相对前作的创新点

- **GShard 的 top-2。** GShard 的算法 1 选择两个候选专家，并处理容量及第二条路线的派发。Switch 只保留一个目的地，简化派发和汇总。代价是路由到热门专家时，每个 token 只剩这一条专家路径。[GShard 算法 1](https://arxiv.org/pdf/2006.16668#page=7) · [Switch 第 2.1 节](https://jmlr.org/papers/volume23/21-0998/21-0998.pdf#page=4)
- **训练稳定性。** 作者单独测试了选择性精度，把路由器内部部分运算提升到 float32，同时保留主体的 bfloat16 训练。表 2 给出了全 float32、全 bfloat16 与选择性精度三种运行的结果，可以把这个改动和 top-1 的比较分开看。[第 2.4 节及表 2](https://jmlr.org/papers/volume23/21-0998/21-0998.pdf#page=8)

## 实验设置

表 1 比较 T5、top-2 MoE 和 top-1 Switch。MoE-Base 与 Switch-Base 都用 128 个专家，每隔一个前馈层引入专家，硬件相同。预训练用 C4 的文本片段遮盖任务，遮盖约 15% 的 token。作者同时记录每秒样本数、10 万步时的负对数困惑度，以及达到 −1.50 所需的小时数。负对数困惑度越大越好，达到该阈值可以发生在 10 万步之后。[第 2.3 节与表 1](https://jmlr.org/papers/volume23/21-0998/21-0998.pdf#page=7)

## 主要结果

| 容量因子 | 模型 | 样本/秒 ↑ | 10 万步时的负对数困惑度 ↑ | 达到 −1.50 的小时数 ↓ |
| ---: | --- | ---: | ---: | ---: |
| 1.0 | MoE-Base | 860 | −1.572 | 80.1 |
| 1.0 | Switch-Base | 1000 | −1.561 | 62.8 |
| 2.0 | MoE-Base | 840 | −1.547 | 68.7 |
| 2.0 | Switch-Base | 860 | −1.554 | 72.8 |

容量因子为 1 时，Switch 达到阈值少用 17.3 小时。改为 2 后，Switch 每秒仍多处理 20 个样本，但达到阈值多用 4.1 小时。同一张表已经能回答老师可能问的“每步跑得快，一定更早训练好吗”。[原论文表 1](https://jmlr.org/papers/volume23/21-0998/21-0998.pdf#page=7)

## 优点与不足

表 1 同时控制硬件、报告吞吐和达标时间，让路由简化的收益可以量化。固定容量会让某些 token 跳过专家，增大容量又会带来空槽。所有专家的参数仍需存储，跨设备分发仍有成本。这组 TPU 训练实验也无法直接给出单张 GPU 解码时的毫秒延迟。

## 对我的启发

汇报时可以请老师看图 3 的空位，然后追问“空位在 B，能不能直接拿来算选了 A 的 token？”A、B 的权重不同。想保留原路线，就需要一种支持每个专家处理不同数量 token 的计算方式。沿着这个问题读 MegaBlocks，会更容易理解作者为什么使用块稀疏矩阵乘法。[MegaBlocks 精读](/research/megablocks-paper-reading/)
