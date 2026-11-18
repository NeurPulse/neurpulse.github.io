---
title: '从零读懂 MoE 的教学文档与论文阅读指南'
description: '从两份入门资料开始，按学习阶段选读五份教程和七篇原始论文，并对照 Mixtral、ScatterMoE 和 SonicMoE 的实现。'
pubDate: 2026-11-17
kind: record
category: '论文精读'
tags: ['MoE', 'Mixtral', 'ScatterMoE', 'SonicMoE', '论文阅读']
---

**如果今天只读两份资料，先读 Hugging Face 的中文教程，再读《Mixtral of Experts》第 2.1 节的图 1。** 前者讲清“一个 token 为什么只经过几个专家”，后者把同一件事画成真实模型中的计算流程。能复述这张图之后，再看对应的 PyTorch 代码。第一遍用个人电脑即可。[中文教程](https://huggingface.co/blog/zh/moe) · [Mixtral 图 1 与第 2.1 节](https://arxiv.org/html/2401.04088#S2.SS1)

想先动手理解路由，可以从一个二维向量的手算和 Python 练习开始。[MoE 从零学习讲义与练习](/research/moe-from-scratch/)

## 带着一个问题开始读

一个 token 进入 MoE 层时是一行特征向量。路由器给各个专家打分，选出其中 $k$ 个。被选中的专家分别处理**同一行输入**，各自算出一行输出，再由模型按路由权重合并。读论文时，**跟着这行数据看它被分给谁，以及计算结果怎样回到原来的 token。**

设一批有 $T$ 个 token，模型有 $E$ 个可选专家，专家 $e$ 实际收到 $T_e$ 行。只看路由专家，并假设没有丢弃分配，标准 top-k 下有 $\sum_e T_e=Tk$。固定 $T$ 和 $k$，总分配数就固定了，但每个 $T_e$ 仍会变。**从第一次读 MoE 一直到读 SonicMoE，都用得上这个区别。**

## 五份教学资料

| 资料 | 优先读哪里 | 读完做什么 |
|---|---|---|
| **① Hugging Face《混合专家模型（MoE）详解》** · 中文，首选 | 依次读“什么是混合专家模型？”“什么是稀疏性？”和“混合专家模型中令牌的负载均衡”。对专家怎样形成分工有疑问时，再读“专家如何学习？” | 自己画出“token → 路由器 → 两位专家 → 加权合并”。解释为什么总专家数和每个 token 激活的专家数可以不同。[中文原文](https://huggingface.co/blog/zh/moe) |
| **② NVIDIA《混合专家模型的定义与工作原理》** · 中文，术语速查 | 看“混合专家模型的工作原理”和“专家不平衡（负载均衡问题）”。如果第一份资料已读懂，直接看后一段。 | 用一句话区分“路由器选了谁”和“一整批里谁最忙”。[中文术语表](https://www.nvidia.cn/glossary/mixture-of-experts/) |
| **③ Hugging Face Transformers 的 Mixtral 实现** · 英文，Python 动手首选 | 先看模型文档的简介，再看固定版本 `v4.46.3` 中 `MixtralSparseMoeBlock.forward` 的第 637–675 行。可以只读代码，不下载模型权重。 | 顺着 `self.gate → softmax → topk → expert_layer → index_add_`，找出打分、选专家、专家计算、加权写回分别发生在哪里。[模型文档](https://huggingface.co/docs/transformers/en/model_doc/mixtral) · [固定版本的 39 行前向代码](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/mixtral/modeling_mixtral.py#L637-L675) |
| **④ Keras《Text classification with Switch Transformer》** · 英文，选读的教学代码 | 看 `Implement the router as a layer` 和 `Implement a Switch layer` 两节。熟悉 PyTorch 的读者也可以把这两段当伪代码读。 | 找出 `dispatch_tensor` 如何把 token 分发给专家，以及 `combine_tensor` 如何将输出收回。适合读过 Switch 论文后对照。[Keras 官方示例](https://keras.io/examples/nlp/text_classification_with_switch_transformer/) |
| **⑤ PyTorch 的 Triton 分组矩阵乘教程** · 英文，算子进阶 | 先看图 1 中的三个专家及前后的说明。看懂每个专家的矩阵为什么行数不同后，再往下看分块与缓存优化。 | 画出三个形状不同的矩阵乘，说明为什么可以把它们组织在一次分组矩阵乘（Grouped GEMM）内核调用里。[PyTorch 技术文章](https://pytorch.org/blog/accelerating-moes-with-a-triton-persistent-cache-aware-grouped-gemm-kernel/) |

## 七篇原始论文怎么选

### 先理解路由和专家

**1. Mixtral of Experts**\
Albert Q. Jiang 等，2024 年，作者技术报告。**适合当第一篇读。** 第 2.1 节和图 1 给出一个完整的 top-2 例子。Mixtral 8×7B 的每个 MoE 层有 8 个专家，每个 token 选其中 2 个，两个专家的输出按权重合并。读完图 1，回到上面的 Transformers 代码，找出两个专家的结果如何加回同一个 token。[论文 HTML，第 2.1 节](https://arxiv.org/html/2401.04088#S2.SS1) · [作者论文页](https://arxiv.org/abs/2401.04088)

**2. Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity**\
William Fedus、Barret Zoph、Noam Shazeer，**JMLR 2022**。**适合当第一篇正式发表、能给导师讲清楚的基础论文。** 先看图 2 的 top-1 路由，再看图 3 的专家容量。每个 token 选一位专家，某个专家仍可能收到超出容量的 token。带着“容量留小了会怎样，留大了要付出什么代价”去读负载均衡的部分。[期刊页面](https://jmlr.org/papers/v23/21-0998.html) · [正式 PDF，图 2 和图 3](https://www.jmlr.org/papers/volume23/21-0998/21-0998.pdf#page=4)

一个值得对照的细节是**选完后怎样使用权重**。Switch 的图 2 用选中专家的原路由概率缩放输出。上面固定版本的 Mixtral 代码则把选中两位专家的权重重新归一化，使两项相加为 1。读不同实现时，要分别确认“选几个”和“怎样合并”。[Switch 图 2](https://www.jmlr.org/papers/volume23/21-0998/21-0998.pdf#page=4) · [Mixtral 权重归一化的代码](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/mixtral/modeling_mixtral.py#L644-L650)

**3. Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer**\
Noam Shazeer 等，**ICLR 2017**。读第 2 节、图 1 以及 `Noisy Top-K Gating` 小节。这篇解释了模型怎样在拥有很多套专家参数的同时，让单个输入只用其中几套。先能指出公式中的“门控权重”和“专家输出”，再看训练时加入噪声与均衡约束的动机。[作者提供的会议 PDF](https://www.cs.toronto.edu/~hinton/absps/Outrageously.pdf#page=3) · [arXiv 页面](https://arxiv.org/abs/1701.06538)

### 从模型设计走到 GPU 算子

**4. DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models**\
Damai Dai 等，**ACL 2024 主会长文**。按顺序看图 1 的三个小图，再看第 3.1、3.2 节。沿着图 1 看两项改动怎样生效。第一项把单个专家切得更细，第二项让每个 token 都经过共享专家。画图时，用实线标出每次都要走的共享专家，用另一种线标出路由器选出的专家。这能解释 SonicMoE 为什么关心更窄的专家和更多次 token 分配。[ACL 正式页面](https://aclanthology.org/2024.acl-long.70/) · [正式 PDF，图 1](https://aclanthology.org/2024.acl-long.70.pdf#page=3)

**5. MegaBlocks: Efficient Sparse Training with Mixture-of-Experts**\
Trevor Gale、Deepak Narayanan、Cliff Young、Matei Zaharia，**MLSys 2023**。从图 3 的三种计算布局开始。传统做法给专家预留容量，容量过小会丢分配，预留过多会算到填充位置。MegaBlocks 用块稀疏运算表达不均匀的专家计算。读完试着解释“避免因专家容量不足而丢 token”和“每个硬件小块都恰好填满”为什么是两个问题。[会议页面](https://proceedings.mlsys.org/paper_files/paper/2023/hash/5a54f79333768effe7e8927bcccffe40-Abstract-mlsys2023.html) · [正式 PDF，图 3](https://proceedings.mlsys.org/paper_files/paper/2023/file/5a54f79333768effe7e8927bcccffe40-Paper-mlsys2023.pdf#page=4)

**6. Scattered Mixture-of-Experts Implementation（ScatterMoE）**\
Shawn Tan、Yikang Shen、Rameswar Panda、Aaron Courville，**COLM 2024 正式会议论文**。它也有 arXiv 作者版。读图 2 和算法 1，沿着一次 `ParallelLinear` 前向，追踪输入行的读取索引、每个专家使用的权重，以及结果的写回位置。对会 Python、刚学 GPU 的读者，这是把论文伪代码和 PyTorch/Triton 源码对起来的合适入口。[会议录用名单](https://colmweb.org/2024/AcceptedPapers.html#:~:text=Scattered%20Mixture-of-Experts%20Implementation) · [作者版 PDF，算法 1](https://arxiv.org/pdf/2403.08245#page=4) · [作者源码](https://github.com/shawntan/scattermoe)

**7. SonicMoE: Accelerating MoE with IO and Tile-aware Optimizations**\
Wentao Guo 等，**ICLR 2026 主会**。先看第 2 节的形状与数据量，再读第 3 节和第 5 节图 4。第 3 节回答“为什么可以少保存一份专家输出”，图 4 回答“为什么几个尾部 token 也可能占用整个计算块”。准备讲 GPU 流水线时再读第 4 节和附录图 9。等价的计算图改写可以核对输出和梯度。汇报 token rounding 时还要交代模型质量，因为它改变了训练时的路由。[会议页面](https://proceedings.iclr.cc/paper_files/paper/2026/hash/6ed5bf446f59e2c6646d23058c86424b-Abstract-Conference.html) · [正式 PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/6ed5bf446f59e2c6646d23058c86424b-Paper-Conference.pdf) · [固定版本源码](https://github.com/Dao-AILab/sonic-moe/tree/7396f3e604827d8186c2e16e64b28ee33d3defd0)

## 用个人电脑把三篇资料连起来

下面的例子对应 Mixtral 图 1、Transformers 的 `topk` 与 `index_add_`、PyTorch 分组矩阵乘教程的图 1。它使用已给定的路由概率和三个标量函数，专门演示分配与汇总。只用 Python 3 标准功能即可运行。

```python
def moe_forward(tokens, router_probs):
    experts = [lambda x: 2 * x, lambda x: 5 * x, lambda x: x + 1]
    outputs = []
    counts = [0] * len(experts)

    for x, probs in zip(tokens, router_probs):
        top2 = sorted(range(len(experts)), key=lambda e: probs[e], reverse=True)[:2]
        selected_sum = sum(probs[e] for e in top2)
        output = 0.0
        for e in top2:
            counts[e] += 1
            weight = probs[e] / selected_sum
            output += weight * experts[e](x)
        outputs.append(round(output, 6))
    return outputs, counts


tokens = [2, 4, 7]
spread = [[0.6, 0.3, 0.1], [0.1, 0.6, 0.3], [0.3, 0.1, 0.6]]
focused = [[0.6, 0.3, 0.1] for _ in tokens]

print(moe_forward(tokens, spread))   # ([6.0, 15.0, 10.0], [2, 2, 2])
print(moe_forward(tokens, focused))  # ([6.0, 12.0, 21.0], [3, 3, 0])
```

先用纸笔确认第一个输出。输入是 2，专家 A 算出 4，B 算出 10。选中的权重归一化后为 $2/3$ 和 $1/3$，输出就是 $4\times(2/3)+10\times(1/3)=6$。

再看行数。两组都有 **3 个 token、每个选 2 位专家、总共 6 次分配**。假设 GPU 的一个行块覆盖 2 行，`[2, 2, 2]` 需要 3 个行块，`[3, 3, 0]` 需要 4 个。它解释了为什么同样多的分配可以产生不同数量的尾块。实际时间还要在相同硬件、相同输入条件下测量。

## 针对下一次导师讨论的选读顺序

- **要讲清 MoE 是什么**，用 Hugging Face 中文教程、Mixtral 图 1 和上面的手算。合格标准是能画出一个 token 经过两个专家后，如何重新变回一个输出向量。
- **要解释为什么研究算子**，看 PyTorch 教程图 1 和 MegaBlocks 图 3。合格标准是能对着两组专家行数解释，理论分配总数相同，GPU 做的分块工作为什么仍可能不同。
- **要提出第一个能做的实验**，先读 ScatterMoE 的 `ParallelLinear`，再对照 SonicMoE。可以先固定 token 数、top-k 和专家权重，改变路由分布，检查各实现是否给出一致结果，再测单层耗时。配套的 SonicMoE 精读还附有 GPU 实验边界和梯度自检。[SonicMoE 精读与 CPU 梯度算例](/research/sonicmoe-training-kernels/)
