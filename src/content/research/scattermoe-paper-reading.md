---
title: 'ScatterMoE 精读笔记｜让矩阵乘法按路由读取 token'
description: '从四个 token 的例子理解 ScatterMoE，比较它与 MegaBlocks 的数据搬运方式，并读懂论文的吞吐和显存实验。'
pubDate: 2026-11-18
updatedDate: 2026-11-19
kind: record
category: '论文精读'
tags: ['算子开发', 'MoE', 'Triton', 'GPU']
---

## 论文信息

- **标题**　Scattered Mixture-of-Experts Implementation。作者把实现命名为 ScatterMoE。
- **作者 / 机构**　Shawn Tan、Yikang Shen、Rameswar Panda、Aaron Courville。论文首页的联系邮箱分别对应 Mila、IBM、IBM、蒙特利尔大学。
- **发表**　第一届 Conference on Language Modeling（COLM 2024）。[会议收录列表](https://2024.colmweb.org/AcceptedPapers.html) · [会议版 PDF](https://openreview.net/pdf?id=YDZ7GeFLxq)
- **阅读入口**　[arXiv:2403.08245](https://arxiv.org/abs/2403.08245) · [可直接跳转图表的网页全文](https://arxiv.org/html/2403.08245v2) · [作者的 Triton 实现](https://github.com/shawntan/scattermoe)

## 一句话总结

ScatterMoE 让 GPU 做专家矩阵乘法时按行号读取 token、按下一步需要的顺序写出结果，省去了为分组而单独复制整批向量的开销，在作者的 A100 实验中降低了显存用量，也提高了吞吐。[图 1](#fig-1) · [图 4](#fig-4)

## 研究动机

普通前馈层里，一批 token 的向量组成矩阵 `X`，每一行都可以乘同一个 `W`。稀疏混合专家层（MoE）有多套权重，路由器给每个 token 选出少数几个专家。这样能增加模型容纳的参数，但同一专家收到的行可能隔得很远，各专家收到的行数也随输入变化。[第 3.1 节](https://arxiv.org/html/2403.08245v2#S3.SS1)

用四个 token 就能看清这笔开销。4 个 token 依次去 `[A, B, A, A]`，专家 A 需要第 `0、2、3` 行。`rows_A = [0, 2, 3]` 只记录三个整数，PyTorch 的 `X[rows_A]` 却会复制三条完整向量。假如每行有 4096 个 FP16 元素，仅这份副本的数据就占 `3 × 4096 × 2 = 24,576` 字节，即 **24 KiB**。专家的乘法此时还没开始。论文图 1 左边的 `Copy + Group` 画的就是这种额外搬运。[图 1](#fig-1) · [PyTorch 索引与视图](https://docs.pytorch.org/docs/stable/tensor_view.html)

<figure class="paper-figure" id="fig-1">
  <a class="paper-panel" href="/images/research/scattermoe/fig-1.webp">
    <img src="/images/research/scattermoe/fig-1.webp" width="1522" height="722" loading="lazy" decoding="async" alt="左侧先复制和重排 token 再做线性变换，右侧的专家线性层直接读取自己需要的 token。">
  </a>
  <figcaption><strong>ScatterMoE 图 1</strong>　沿着编号相同的 token 往下看。左侧在第一层乘法前多了一次 Copy + Group，右侧直接连到对应颜色的专家。 <a href="https://arxiv.org/html/2403.08245v2#S1.F1">论文原图</a></figcaption>
</figure>

早期 Switch Transformer 为每个专家预留固定容量。热门专家溢出时，部分 token 会跳过该层的专家计算，冷门专家则留下空位。后来的 MegaBlocks 已能处理全部被路由的 token。ScatterMoE 追问的是，在保留这些计算的前提下，能否连显存里为分组和补齐准备的大张量也省下来？[Switch Transformer 图 3](#switch-fig-3) · [MegaBlocks 图 4](#megablocks-fig-4)

<figure class="paper-figure" id="switch-fig-3">
  <a class="paper-panel" href="/images/research/scattermoe/switch-fig-3.webp">
    <img src="/images/research/scattermoe/switch-fig-3.webp" width="1752" height="692" loading="lazy" decoding="async" alt="相同的一批 token 在容量因子为 1.0 和 1.5 时的路由示意图，红色虚线表示溢出，白格表示空位。">
  </a>
  <figcaption><strong>Switch Transformer 图 3</strong>　中间一组的红色虚线标出因专家已满而溢出的 token。右侧把容量因子从 1.0 提到 1.5，留出了更多空位。 <a href="https://jmlr.org/papers/volume23/21-0998/21-0998.pdf#page=5">Fedus、Zoph 与 Shazeer，JMLR 2022</a> · <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a></figcaption>
</figure>

## 核心方法

设第 $t$ 个 token 的向量为 $x_t$，路由器为它选出 $k$ 个专家。第 $j$ 个被选中的专家编号为 $e_{t,j}$，权重为 $p_{t,j}$。MoE 要算的结果是

$$
y_t = \sum_{j=1}^{k} p_{t,j}\, f_{e_{t,j}}(x_t).
$$

`ParallelLinear` 改变的是数据怎样到达矩阵乘法。实现先把路由记录按专家排序，得到每条记录对应的原始行号。算某个专家时，Triton 内核 `scatter2scatter` 按这些行号把输入的小块读到片上，乘该专家的权重，再写到指定行。论文版对齐的是索引，片上计算块仍可补齐，因此显存中无需先存整份补齐后的输入向量。[第 3.1–3.2 节](https://arxiv.org/html/2403.08245v2#S3.SS2) · [图 2](#fig-2)

<figure class="paper-figure" id="fig-2">
  <div class="paper-panels">
    <a id="fig-2a" class="paper-panel" href="/images/research/scattermoe/fig-2a.webp">
      <img src="/images/research/scattermoe/fig-2a.webp" width="836" height="451" loading="lazy" decoding="async" alt="输入和输出都按专家分组，同一专家的行连续排列。">
      <span>(a) 按组读，按组写</span>
    </a>
    <a id="fig-2b" class="paper-panel" href="/images/research/scattermoe/fig-2b.webp">
      <img src="/images/research/scattermoe/fig-2b.webp" width="817" height="451" loading="lazy" decoding="async" alt="从原始 token 行读入，把同一专家的输出写在一起。">
      <span>(b) 分散读，按组写</span>
    </a>
    <a id="fig-2c" class="paper-panel" href="/images/research/scattermoe/fig-2c.webp">
      <img src="/images/research/scattermoe/fig-2c.webp" width="819" height="450" loading="lazy" decoding="async" alt="按行号读入 token，把结果写回各 token 对应的位置。">
      <span>(c) 分散读，分散写</span>
    </a>
    <a id="fig-2d" class="paper-panel" href="/images/research/scattermoe/fig-2d.webp">
      <img src="/images/research/scattermoe/fig-2d.webp" width="815" height="451" loading="lazy" decoding="async" alt="读取已按专家分组的行，再将输出写回各 token 对应的位置。">
      <span>(d) 按组读，分散写</span>
    </a>
  </div>
  <figcaption><strong>ScatterMoE 图 2</strong>　对照每幅图上下两排的 token 编号。后面的手算例子中，第一层用 (b)，第二层用 (d)，中间的激活函数沿用专家顺序。 <a href="https://arxiv.org/html/2403.08245v2#S3.F2">论文原图</a></figcaption>
</figure>

**四个 token 手算一遍。** 沿用 `[A, B, A, A]`，把每个向量缩成一个数，输入依次为 `[1, 2, -3, 4]`。设 `top-1`、路由权重均为 1。专家 A 的两层权重依次为 `2、3`，专家 B 为 `10、1`，两层之间用 `ReLU(z) = max(0, z)`。下面是教学用的数值，对应论文算法 3。

| 步骤 | 行的顺序 | 手算结果 |
| --- | --- | --- |
| 给路由记录排序 | `0, 2, 3, 1` | 前三行归 A，最后一行归 B。这里只需记录行号。 |
| 第一层按行号读，按专家分组写 | `0, 2, 3, 1` | `[1×2, -3×2, 4×2, 2×10] = [2, -6, 8, 20]` |
| 对隐藏值做 ReLU | `0, 2, 3, 1` | `[2, 0, 8, 20]`，每个数独立计算，可以保持当前行序。 |
| 第二层按组读，直接写回各自的 token 位置 | 输出行 `0, 1, 2, 3` | A 向位置 `0、2、3` 写 `6、0、24`，B 向位置 `1` 写 `20`。最终输出为 **`[6, 20, 0, 24]`**。 |

当 `k > 1` 时，一个 token 有多条专家输出。第二层把每条结果写回对应 token 的路由槽位，再按路由权重加总。训练还要算权重梯度，作者发现先把输入和输出梯度按专家分好组再相乘更快，于是在反向传播中保留必要的分组，并复用临时缓冲区。[算法 1](https://arxiv.org/html/2403.08245v2#alg1) · [算法 2](https://arxiv.org/html/2403.08245v2#alg2) · [算法 3](https://arxiv.org/html/2403.08245v2#alg3)

## 相对前作的创新点

| 最接近的前作 | 沿用了什么 | ScatterMoE 具体改了什么，以及如何核对 |
| --- | --- | --- |
| **MegaBlocks**，Gale 等，MLSys 2023，块稀疏 dMoE 路径 | 保留所有路由分配，把属于同一专家的计算放在一起。 | MegaBlocks 的 `padded_gather` 先复制 token 并把各组补到块大小的倍数，再做块稀疏乘法，最后 `padded_scatter`。ScatterMoE 把分散读写并入专家矩阵乘法，前向显存中省去独立的分组输入副本，中间隐藏值则按专家顺序保存，直接交给下一层。[MegaBlocks 图 4 伪代码](#megablocks-fig-4) · [ScatterMoE 图 1](#fig-1) 与 [算法 3](https://arxiv.org/html/2403.08245v2#alg3) |
| **Sparse Universal Transformer**，Tan 等，EMNLP 2023，多头注意力专家（MoMHA） | 沿用前作的注意力结构，各专家有自己的查询投影，共享键和值投影。 | 新增的是可选择输入、输出行序的 `ParallelLinear` 实现。在注意力前后的两个专家投影都使用「分散读、分散写」，让注意力直接收到按原时间顺序排列的向量，省去单独恢复顺序的步骤。[SUT 第 3.1 节](https://aclanthology.org/2023.emnlp-main.12.pdf#page=3) · [ScatterMoE 图 3](#fig-3) 与 [算法 4](https://arxiv.org/html/2403.08245v2#alg4) |

<figure class="paper-figure" id="megablocks-fig-4">
  <a class="paper-panel" href="/images/research/scattermoe/megablocks-fig-4.webp">
    <img src="/images/research/scattermoe/megablocks-fig-4.webp" width="1452" height="1368" loading="lazy" decoding="async" alt="dMoE 前向伪代码，第 15 行按专家重排，两次块稀疏乘法在第 22、23 行，第 26 行重排输出。">
  </a>
  <figcaption><strong>MegaBlocks 图 4</strong>　第 15 行用 <code>padded_gather</code> 按专家分组，第 22、23 行计算专家层，第 26 行用 <code>padded_scatter</code> 写回。可以与上文的 ScatterMoE 图 1 对照。 <a href="https://proceedings.mlsys.org/paper_files/paper/2023/file/5a54f79333768effe7e8927bcccffe40-Paper-mlsys2023.pdf#page=5">Gale 等，MLSys 2023</a></figcaption>
</figure>

<figure class="paper-figure" id="fig-3">
  <a class="paper-panel" href="/images/research/scattermoe/fig-3.webp">
    <img src="/images/research/scattermoe/fig-3.webp" width="1085" height="993" loading="lazy" decoding="async" alt="注意力层前后的两组专家投影都能分散读写，中间的多头注意力按 token 原顺序接收输入。">
  </a>
  <figcaption><strong>ScatterMoE 图 3</strong>　看中间标有 Multi-head Attention 的紫色横条。两侧的专家投影都能分散读写，因此进入注意力层时，token 仍按时间顺序排列。 <a href="https://arxiv.org/html/2403.08245v2#S3.F3">论文原图</a></figcaption>
</figure>

## 实验设置

- **整模型训练。** 约 15 亿参数的 Mixtral 配置，16 层，模型宽度 1024，专家隐藏宽度 3584，8 个专家、每个 token 选 2 个。单机 8 张 NVIDIA A100，每条序列 2048 个 token，有效 batch 为 256，共训练 100 次更新。Hugging Face 和 ScatterMoE 每批 128、累积 2 次，MegaBlocks 每批 64、累积 4 次。对照含 Hugging Face 原生实现、MegaBlocks Sparse 和 MegaBlocks 的 grouped memory-efficient 两个版本。[图 4(a)](#fig-4a) · [第 4 节](https://arxiv.org/html/2403.08245v2#S4)
- **单层 MLP。** 在一张 A100 上比较 ScatterMoE、MegaBlocks 和 SUT 的 PyTorch 实现。图 4(b)(c) 用 `T = 30 × 2048 = 61,440` 个 token，模型宽度 4096，32 个专家、每个 token 选 4 个，每个专家的隐藏宽度 2048。吞吐以 tokens/s 计，显存以字节计。单层耗时测量运行 100 次，报告中位数，有误差线的曲线给出第 5 和第 95 百分位。[第 4.1 节](https://arxiv.org/html/2403.08245v2#S4.SS1)
- **两种扫参。** 图 5 固定总参数和每个 token 激活的参数，令 `k = 1、2、4、8、16`，`E = 8k`，每个专家宽度变为 `8192/k`，粒度 `G = k`。图 6 固定总专家数 `E = 64`，逐步提高 `k`，观察激活越来越多的专家后还剩多少吞吐优势。[图 5](#fig-5) · [图 6](#fig-6)
- **扩展与输出核对。** 注意力实验用 32 个活跃头、每头 128 维、模型宽度 4096、`T = 16 × 2048`，基线为 MegaBlocks 的 dense 配置。另用同一个已训练的 Mixtral-8x7B-v0.1，在 LM Evaluation Harness 上比较替换实现前后的任务分数。[第 4.4 节](https://arxiv.org/html/2403.08245v2#S4.SS4) · [第 4.5 节](https://arxiv.org/html/2403.08245v2#S4.SS5)

## 主要结果

下表是作者报告的实验。带「约」的吞吐取自图上刻度，百分比写明了分母。

| 实验条件与对照 | 数值和读法 | 原文位置 |
| --- | --- | --- |
| 约 15 亿参数整模型训练，8 张 A100，有效 batch 256 | ScatterMoE 约 **29 万**、MegaBlocks Sparse 约 **21 万** tokens/s。正文报告吞吐提高 **38.1%**。两者实际 batch 和梯度累积次数也不同，这个数衡量整套训练配置。 | [图 4(a)](#fig-4a) · [第 4 节](https://arxiv.org/html/2403.08245v2#S4) |
| 单层 MLP，一张 A100，`T=61,440, E=32, k=4` | 前向吞吐约 **140 万** 对 **120 万** tokens/s（ScatterMoE 对 MegaBlocks）。合计前向和反向后，约 **46 万** 对 **41 万** tokens/s。 | [图 4(b)](#fig-4b) |
| 同一单层 MLP，以 MegaBlocks 显存为 100% | ScatterMoE 在训练（前向加反向）用 **66.2%**，即节省 **33.8%**。仅前向用 **53.6%**，即节省 **46.4%**。 | [图 4(c)](#fig-4c) · [第 4.1 节](https://arxiv.org/html/2403.08245v2#S4.SS1) |
| 增加 MLP 专家粒度，`G=16, k=16, E=128` | 以**相同激活参数量**的稠密 MLP 吞吐为 100%。ScatterMoE 和 MegaBlocks 的训练吞吐约为 **61% 对 51%**，前向约为 **64% 对 45%**。读图时也能看到，在 `G=2` 的前向点，MegaBlocks 的中位数更高。 | [图 5(a)(b)](#fig-5) · [第 4.2 节](https://arxiv.org/html/2403.08245v2#S4.SS2) |
| 固定 `E=64`，增大每个 token 选中的专家数 | 图 6 改以**相同总参数量**的稠密 MLP 为基线。到 `k=30`，两者吞吐都落到该基线的 **1 至 2 倍之间**（读图）。作者报告再增大 `k` 时，MegaBlocks 超出测试设备的显存。 | [图 6](#fig-6) · [第 4.3 节](https://arxiv.org/html/2403.08245v2#S4.SS3) |
| 多头注意力专家，`k=8, E=64`，每个专家 4 个头，32,768 个 token | 对比 MegaBlocks dense 配置实现的同一注意力模块，作者报告 ScatterMoE 的**推理吞吐提高 24.0%**。 | [图 8](#fig-8) · [第 4.4 节](https://arxiv.org/html/2403.08245v2#S4.SS4) |
| Mixtral-8x7B-v0.1，同一套权重，Hugging Face 与 ScatterMoE | WikiText 困惑度为 **5.6135 对 5.6142**，绝对差 **0.0007**。表中 10 项准确率任务的最大绝对差为 **0.0060**，即 **0.6 个百分点**。这项实验衡量替换实现后模型输出在任务指标上的一致性。 | [表 1，第 4.5 节](https://arxiv.org/html/2403.08245v2#S4.T1) |

<figure class="paper-figure" id="fig-4">
  <div class="paper-panels">
    <a id="fig-4a" class="paper-panel paper-panel-wide" href="/images/research/scattermoe/fig-4a.webp">
      <img src="/images/research/scattermoe/fig-4a.webp" width="3321" height="2064" loading="lazy" decoding="async" alt="四种实现训练 15 亿参数模型时的吞吐柱形图，纵轴单位为 tokens/s。">
      <span>(a) 整模型训练吞吐</span>
    </a>
    <a id="fig-4b" class="paper-panel" href="/images/research/scattermoe/fig-4b.webp">
      <img src="/images/research/scattermoe/fig-4b.webp" width="3055" height="2233" loading="lazy" decoding="async" alt="单层 MLP 前向、反向和前向加反向的吞吐，蓝柱代表 ScatterMoE。">
      <span>(b) 单层 MLP 吞吐</span>
    </a>
    <a id="fig-4c" class="paper-panel" href="/images/research/scattermoe/fig-4c.webp">
      <img src="/images/research/scattermoe/fig-4c.webp" width="2970" height="2233" loading="lazy" decoding="async" alt="单层 MLP 仅前向和前向加反向的显存用量，纵轴单位为字节。">
      <span>(c) 单层 MLP 显存</span>
    </a>
  </div>
  <figcaption><strong>ScatterMoE 图 4</strong>　(a) 对应整套训练配置，(b) 和 (c) 对应同一个单层 MLP。(a)(b) 的柱子越高表示吞吐越大，(c) 的柱子越低表示用的显存越少。 <a href="https://arxiv.org/html/2403.08245v2#S4.F4">论文原图</a></figcaption>
</figure>

读原图 4(c) 时，在同一组里对比蓝柱（ScatterMoE）和橙柱（MegaBlocks）。例如用前向蓝柱高度除以前向橙柱高度，得到正文中的约 0.536。第 4.1 节正文有一处把单层实验指向图 4(a)，按三个子图的标题，单层吞吐和显存分别对应 **4(b)、4(c)**。[图 4 全图](#fig-4)

<figure class="paper-figure" id="fig-5">
  <div class="paper-panels">
    <a id="fig-5a" class="paper-panel" href="/images/research/scattermoe/fig-5a.webp">
      <img src="/images/research/scattermoe/fig-5a.webp" width="2843" height="2186" loading="lazy" decoding="async" alt="固定总参数量和激活参数量时，训练相对吞吐随专家粒度 G 的变化。">
      <span>(a) 训练吞吐</span>
    </a>
    <a id="fig-5b" class="paper-panel" href="/images/research/scattermoe/fig-5b.webp">
      <img src="/images/research/scattermoe/fig-5b.webp" width="2843" height="2186" loading="lazy" decoding="async" alt="使用相同配置仅做前向时，两种实现的相对吞吐随专家粒度 G 的变化。">
      <span>(b) 推理吞吐</span>
    </a>
  </div>
  <figcaption><strong>ScatterMoE 图 5</strong>　横轴的粒度 G 越大，专家越多、每个专家越窄。两幅图都以相同激活参数量的稠密 MLP 吞吐为 100%。 <a href="https://arxiv.org/html/2403.08245v2#S4.F5">论文原图</a></figcaption>
</figure>

<figure class="paper-figure" id="fig-6">
  <a class="paper-panel" href="/images/research/scattermoe/fig-6.webp">
    <img src="/images/research/scattermoe/fig-6.webp" width="3145" height="2294" loading="lazy" decoding="async" alt="专家数固定为 64 时，两种实现的相对吞吐都随每个 token 选中的专家数 k 增加而下降。">
  </a>
  <figcaption><strong>ScatterMoE 图 6</strong>　沿横轴从左往右，每个 token 激活的专家越来越多。红色虚线表示相同总参数量的稠密 MLP 吞吐。 <a href="https://arxiv.org/html/2403.08245v2#S4.F6">论文原图</a></figcaption>
</figure>

<figure class="paper-figure" id="fig-8">
  <div class="paper-panels">
    <a id="fig-8a" class="paper-panel paper-panel-wide" href="/images/research/scattermoe/fig-8a.webp">
      <img src="/images/research/scattermoe/fig-8a.webp" width="3100" height="2258" loading="lazy" decoding="async" alt="注意力专家模块在前向、反向和前向加反向时的吞吐对比。">
      <span>(a) 注意力模块吞吐</span>
    </a>
    <a id="fig-8b" class="paper-panel" href="/images/research/scattermoe/fig-8b.webp">
      <img src="/images/research/scattermoe/fig-8b.webp" width="3321" height="2339" loading="lazy" decoding="async" alt="注意力专家粒度 G 改变时，两种实现的训练吞吐。">
      <span>(b) 不同粒度下的训练吞吐</span>
    </a>
    <a id="fig-8c" class="paper-panel" href="/images/research/scattermoe/fig-8c.webp">
      <img src="/images/research/scattermoe/fig-8c.webp" width="3012" height="2421" loading="lazy" decoding="async" alt="注意力专家粒度 G 改变时，两种实现仅做前向的吞吐。">
      <span>(c) 不同粒度下的推理吞吐</span>
    </a>
  </div>
  <figcaption><strong>ScatterMoE 图 8</strong>　(a) 分开看前向和反向，(b)(c) 看粒度变化。这里的纵轴是每秒处理的 token 数。 <a href="https://arxiv.org/html/2403.08245v2#S4.SS4">论文原图</a></figcaption>
</figure>

## 优点与不足

| 优点 | 不足与适用范围 |
| --- | --- |
| 省去了前向分组输入的大副本。输出行序还可以配合下一步计算，MLP 的激活层可以直接消费分组结果。 | 反向计算权重梯度时仍需把部分数据按专家分组。论文用缓冲区复用降低这部分代价。[算法 2、反向传播](https://arxiv.org/html/2403.08245v2#S3.SS2.SSS1) |
| 同一个线性算子能支撑前馈专家和注意力专家。图 4 同时给出单层和整模型结果，便于看出收益出现在哪里。 | 单层与整模型的收益幅度不同。论文测试集中在 A100 和批量处理场景，作者把逐 token 解码的专用内核、多节点并行列为后续工作。[第 5 节](https://arxiv.org/html/2403.08245v2#S5) |

## 对我的启发

- **读算子先数数据。** 看到 `X[rows] @ W`，我可以先算 `len(rows) × 每行宽度 × 每个数的字节数`。这样就能估出一个临时张量有多大，再判断是否值得把按行读取合入矩阵乘法。
- **练习从四行开始。** 上例的 `[6, 20, 0, 24]` 已用 CPU 上的 PyTorch 核对。下一步可以把同一个 token 改成走两个专家，再检查两个路由槽位的加权和。表中的 GPU 性能取自作者的实验。
- **读性能数字先找分母。** 比较「吞吐提高多少」时，先看单层还是整模型、前向还是前反向、实际 batch 是否一致。比如图 4(a) 的 38.1% 可以引用来说明这一配置的端到端收益，评估矩阵乘法内核本身则需要固定形状和批大小另做测量。[图 4](#fig-4)
