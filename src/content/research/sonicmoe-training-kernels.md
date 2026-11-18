---
title: 'SonicMoE 如何加速细粒度 MoE 训练'
description: '用四个 token 和一个可手算的梯度，读懂 SonicMoE 的显存优化、GPU 内核和路由策略，再设计一个单卡能验证的分组矩阵乘实验。'
pubDate: 2026-11-17
kind: record
category: '论文精读'
tags: ['SonicMoE', 'MoE', 'GPU 算子', '分组矩阵乘', 'ICLR']
---

同样是四个 token、每个选两个专家，一批可能让四个专家各处理两行，另一批却让一个专家处理四行、另外两个各处理一行。两批的理论乘加次数可以相同，矩阵的形状、显存里的数据流动和 GPU 的利用率却会不同。**这正是 MoE 训练里值得研究的算子问题。**

Wentao Guo、Mayank Mishra、Xinle Cheng、Ion Stoica 和 Tri Dao 在 ICLR 2026 主会发表了 *SonicMoE: Accelerating MoE with IO and Tile-aware Optimizations*。作者改写了反向传播以释放中间激活，把访存和矩阵乘排在一起，也调整了训练路由以减少计算块的浪费。下面的性能数字采用 ICLR 正式版，源码分析固定到提交 `7396f3e`。[ICLR 论文主页](https://proceedings.iclr.cc/paper_files/paper/2026/hash/6ed5bf446f59e2c6646d23058c86424b-Abstract-Conference.html) · [作者实现](https://github.com/Dao-AILab/sonic-moe/tree/7396f3e604827d8186c2e16e64b28ee33d3defd0)

## 从四个 token 看懂动态形状

MoE 是混合专家模型。一个专家可以先理解为一个小型前馈网络。路由器读入每个 token 的特征向量，在 $E$ 个专家中选出 $k$ 个，并为每个选中的专家分配权重。专家完成计算后，模型按这些权重合并结果。

| token | 这次选中的两个专家 |
|---|---|
| 1 | A、B |
| 2 | A、B |
| 3 | A、C |
| 4 | A、D |

四位专家收到的输入行数分别是 **4、2、1、1**。总共 8 份分配，正好等于 4 个 token 各选 2 次。下一批仍然可以有 8 份分配，行数却可能变成 **2、2、2、2**。设整批有 $T$ 个 token，专家 $e$ 实际收到 $T_e$ 个。在标准 top-k 路由下，$\sum_e T_e=Tk$。各个 $T_e$ 的值要等这一批路由结束才知道。

设 $d$ 是模型特征宽度，$n$ 是单个专家的中间宽度。一个采用 SwiGLU 的专家先后执行以下计算。

| 步骤 | 得到的矩阵 | 形状 |
|---|---|---|
| 按路由取出该专家的输入 | $X_e$ | $T_e\times d$ |
| 第一次矩阵乘，生成两组中间特征 | $H_e=X_eW_{1,e}$ | $T_e\times 2n$ |
| 一组特征经过非线性变换后给另一组加门控 | $A_e=\operatorname{SwiGLU}(H_e)$ | $T_e\times n$ |
| 第二次矩阵乘，回到模型宽度 | $Y_e=A_eW_{2,e}$ | $T_e\times d$ |

**分组矩阵乘（Grouped GEMM）** 将多个专家的矩阵乘组织在一起执行。每个专家都有自己的权重，输入行数也可以不同。前向里变化的是输出矩阵的行数 $M=T_e$。反向计算权重梯度时，要沿 token 所在的维度求和，这时变化的是矩阵乘的归约维度 $K=T_e$。后者和路由器的 top-k 是两个不同的量。[论文第 2 节](https://proceedings.iclr.cc/paper_files/paper/2026/file/6ed5bf446f59e2c6646d23058c86424b-Paper-Conference.pdf#page=3)

### 理论算量接近，时间为什么还能差很多

- **专家变细会增加数据搬运。** 把专家宽度 $n$ 从 1024 减到 256，同时把每个 token 选中的专家数 $k$ 从 2 增到 8。两组的 $kn$ 都是 2048，固定 $T,d$ 时，主要矩阵乘的理论计算量接近。但同一个 token 的专家输出从 2 份变为 8 份，每份仍然有 $d$ 个数。一路保存这些宽向量会占用更多显存。
- **整理数据会占用带宽。** 如果先把属于同一专家的行复制到新矩阵，再启动矩阵乘去读取它们，同一批数据便多经历了一次写和一次读。反向传播还会遇到类似的整理和归约。
- **稀疏会放大尾块开销。** 在 $T,k$ 固定时增加专家总数 $E$，每位专家平均分得的行数 $Tk/E$ 下降。GPU 按固定大小的小块（tile）做矩阵乘，一位专家最后剩下的少数几行，仍可能占用一个计算块。

## 改写反向传播，提前释放专家输出

先看一个 token 与一个专家，省略偏置。把激活后的向量记作 $a$，第二次矩阵乘的权重记作 $W_2$，专家输出为 $y=aW_2$。路由器用权重 $s$ 把这个专家的结果加进总输出 $o$。若上游传回梯度 $g=\partial L/\partial o$，就有

$$
\frac{\partial L}{\partial s}
= \langle g,y\rangle
= \langle g,aW_2\rangle
= \langle gW_2^\top,a\rangle .
$$

**手算一次。** 取 $a=2$，$W_2=[3,4]$，$g=[1,2]$。按左边算，前向得到 $y=[6,8]$，反向做内积，得到 $1\times6+2\times8=22$。按右边算，先得到 $gW_2^\top=1\times3+2\times4=11$，再乘上 $a=2$，仍然得到 **22**。

节省显存的理由藏在反向传播里。为了求传回 $a$ 的梯度，本来就要算 $U=gW_2^\top$，因为 $\partial L/\partial a=sU$。一份 $U$ 现在可以做两件事，既能把梯度传给专家内部，也能和 $a$ 做内积求路由权重梯度。**前向照常产生 $Y$ 并用它汇总输出，汇总结束后就可以释放它。** 反向需要 $A$ 时，从已经保存的 $H$ 重做一次成本较低的门控激活即可。

单层为反向保留的主要大张量是 $X$ 和 $H$，另有路由索引和权重。按 BF16 估算，这两个大张量约占 **$2Td+4Tkn$ 字节**。在上述 $kn$ 固定的比较中，这部分缓存不会随专家变细而增大。这个估算覆盖长期保留的主要激活。模型权重、优化器状态和前向临时缓冲区有各自的显存开销。[论文第 3 节、算法 2 和算法 3](https://proceedings.iclr.cc/paper_files/paper/2026/file/6ed5bf446f59e2c6646d23058c86424b-Paper-Conference.pdf#page=4)

## 让数据搬运与矩阵乘配合

**把取行合进加载过程。** 假设一个专家需要原输入的第 2、5、9 行。矩阵乘在加载当前计算块时，可以按路由索引直接取这几行。这样省下一整份重排矩阵的写回和再次读取。反向中按专家收集输出梯度也能用类似方法。

**把后续运算接在矩阵乘收尾。** 一个矩阵块刚算完时，结果已经靠近计算单元。论文把部分逐元素运算与归约放在同一个内核的收尾阶段（epilogue），包括前向的 SwiGLU，以及反向的激活梯度和路由权重梯度。这样一次读来的数据可以接着使用。

**让一组线程算的时候，另一组处理收尾。** Hopper 上，生产者线程准备输入，两组消费者线程交错执行矩阵乘与收尾，之后交换角色。反向计算 $\mathrm{d}H$ 时还需要读取保存的 $H$，作者用额外的 TMA 异步搬运流水线，把一部分读取时间和计算重叠起来。Blackwell 上则利用片上的 Tensor Memory 安排相应的重叠。

一个值得记住的取舍是，作者在 Hopper 上保留了部分独立的 scatter。把这些按索引写回的操作强行融进内核，会牺牲原有的异步写回能力。图 8 中的前向 3 个内核和反向 5 个内核，对应图中画出的专家计算流程。[论文第 4 节](https://proceedings.iclr.cc/paper_files/paper/2026/file/6ed5bf446f59e2c6646d23058c86424b-Paper-Conference.pdf#page=6) · [图 8 的计算图与图 9 的流水线](https://proceedings.iclr.cc/paper_files/paper/2026/file/6ed5bf446f59e2c6646d23058c86424b-Paper-Conference.pdf#page=19)

## 用 token rounding 减少不满的计算块

只看行方向，假设一个 tile 覆盖 128 行。下面是一个教学例子。

| 专家 | 原有行数 | 原来占用的行块 | 调整后的行数 | 调整后的行块 |
|---|---:|---:|---:|---:|
| A | 130 | 2 | 128 | 1 |
| B | 126 | 1 | 128 | 1 |
| 合计 | 256 | 3 | 256 | 2 |

专家 A 原来的第 2 块只有 2 行有效。Token rounding 先执行标准 top-k，统计各专家收到多少 token，再按专家对候选 token 排序，优先保留原先被选中的分配。默认规则把各专家的行数调到最近的 tile 整数倍。向下取整时移除一部分分配，向上取整时补入**真实 token**。每位专家独立按分数决定取舍，上例展示的是行块数量变化，实际加速幅度需要计时。

**三项创新的验证方式由此分开。** 梯度重排和内核融合保留原有数学计算，可以用参考实现检查输出和梯度的数值误差。Token rounding 会改变训练时哪些专家处理哪些 token，部分 token 参与的专家数也可能改变。因此它还要接受训练质量和下游任务评估。[论文第 5 节、算法 4 与图 4](https://proceedings.iclr.cc/paper_files/paper/2026/file/6ed5bf446f59e2c6646d23058c86424b-Paper-Conference.pdf#page=7)

## 怎么读论文的实验结果

以下为作者在 ICLR 正式版中报告的结果。每一行的对照对象、硬件和指标都很重要。

| 实验条件 | 结果 | 数字对应的范围 |
|---|---|---|
| H100，细粒度 7B，BF16，对照 ScatterMoE | $n=256$ 的配置下，单层峰值激活显存约下降 **45%**。摘要报告细粒度 7B 的 MoE 计算吞吐为对照的 **1.86 倍** | 单个 MoE 层的激活显存与计算吞吐 |
| 7B 模型，用 FSDP-2 完整训练 | SonicMoE 用 **64 张 H100** 达到 **2130 亿 token/天**，ScatterMoE 用 **96 张 H100** 达到 **2250 亿 token/天** | 多卡训练在各自卡数下的实际总吞吐 |
| B300，OLMoE 规模的 7B 配置 | 相比作者整合的 DeepGEMM++ 对照，前向和反向分别相对提速 **28.7%**、**22.1%** | 指定硬件、模型配置与对照实现下的结果 |
| Token rounding，128 个专家选 2 个，$n=1024$ | 前向吞吐提高 **16.5%**，反向提高 **6.1%**，两者合计提高 **9.4%** | MoE 主要内核的计时，路由器耗时另计。摘要另报告了高稀疏设置下约 **1.16 倍**的代表性收益 |
| 1.4B 参数、100B 训练 token，比较 token rounding 与标准 top-k | 验证困惑度 **13.22 对 13.32**，11 项下游任务平均准确率均为 **52.0%** | 表 1(c) 的训练条件。每位专家平均 512 个 token，tile 为 128。验证和下游评估统一使用标准 top-k |

汇报时可以把 45% 和 64 张对 96 张这两组数字放在一起。第一组回答“具体省掉了哪类资源”，第二组回答“单层改动放进完整训练还有多少实际价值”。有关模型质量的结论，应和表 1 的模型大小、训练量及评估路由一起引用。[正式版摘要](https://proceedings.iclr.cc/paper_files/paper/2026/hash/6ed5bf446f59e2c6646d23058c86424b-Abstract-Conference.html) · [图 5–7、表 1 与实验设置](https://proceedings.iclr.cc/paper_files/paper/2026/file/6ed5bf446f59e2c6646d23058c86424b-Paper-Conference.pdf#page=8)

## 单卡可以从哪里入手优化

**第一轮研究问题可以定为“动态专家负载下的 MoE 分组矩阵乘配置选择与复用”。** 可以先看前向的第二次矩阵乘 $A_e W_{2,e}$。如果总 token 数相同，一批分布均匀，另一批集中在少数专家，原来选出的分块与调度配置还合适吗？

固定提交的源码里已经有一套复用机制。它对相关张量的首维分桶，例如 65–512 属于一个范围。缓存键还包含其他维度、步长类别、数据类型等信息。在走多候选调优的路径时，首次进入某个键会选择配置，以后命中便复用。**这段分桶代码读取了张量形状，没有直接读取每位专家的 token 数值。** 这使负载分布成为值得测量的变量。是否真的存在配置复用损失，仍是假设。[分桶、缓存键与调优结果复用的源码](https://github.com/Dao-AILab/sonic-moe/blob/7396f3e604827d8186c2e16e64b28ee33d3defd0/sonicmoe/functional/__init__.py#L104)

### 一轮能决定是否继续的实验

1. **建立正确性基线。** 在个人电脑上用 PyTorch 实现一个小型 MoE 层，固定输入、专家权重和路由，核对前向输出、输入梯度和参数梯度。H800 属于 Hopper，可用作原版 SonicMoE 的候选设备，先在实际镜像上通过官方测试。A800 属于 Ampere，可用于 PyTorch 或 Triton 练习。原版与改动版的性能比较始终在同一张通过测试的卡上进行。[作者的环境要求](https://github.com/Dao-AILab/sonic-moe/tree/7396f3e604827d8186c2e16e64b28ee33d3defd0#installation) · [官方正确性测试](https://github.com/Dao-AILab/sonic-moe/blob/7396f3e604827d8186c2e16e64b28ee33d3defd0/tests/moe_test.py) · [NVIDIA 设备架构表](https://docs.nvidia.com/datacenter/tesla/tesla-release-notes-551-61/index.html)
2. **只改变专家负载。** 固定 $T,E,d,n,k$ 和 BF16，生成均匀、明显偏斜、带空专家的三组路由，每种实现重放同一组数据。记录各专家行数、缓存键、命中情况、所选配置、目标矩阵乘和整层耗时。隔离调优缓存，交换不同负载的首次出现顺序，再单独改变总 token 数。[作者的标准 top-k 单层基准](https://github.com/Dao-AILab/sonic-moe/blob/7396f3e604827d8186c2e16e64b28ee33d3defd0/benchmarks/moe-cute.py)
3. **测出可优化空间。** 将原始分桶复用与精确形状调优对照。在固定负载下离线跑同一组候选，用实测最快者估计潜在收益。精确形状也可能对应不同的专家分布，因此要检查实际选中的配置。若确有差距，再尝试利用已有路由信息做低成本选择。
4. **用未见负载作出判断。** 分开记录首次编译、首次调优和预热后的执行时间。把获取负载特征、分派、潜在的 GPU 同步计入整层时间。能在未参与设计的负载上稳定缩短整层时间，并保持输出和梯度正确，才值得深入。如果不同配置都接近最快值，下一步就看时间分解，判断 gather 或专家聚合是否占了更大的比例。

单卡、单层、固定路由足以研究配置选择，也能把正确性和性能问题控制在容易定位的范围内。如果改的是 token rounding，还需要相应规模的模型训练和质量评估。如果转向 NPU，可以沿用梯度等式和分组问题的分析，但需根据目标硬件重新实现数据搬运、同步和计算流水线。**目前完成的实测是下面的 CPU 梯度算例，H800 单层速度和候选策略的收益仍待设备实验。**

### 研一阶段要掌握到什么程度

| 阶段 | 重点 | 自检方式 |
|---|---|---|
| 讲论文前 | top-k、两次专家矩阵乘的形状、梯度等式、tile 尾块 | 脱稿画出 4 个 token 的分配，手算出 22，解释为什么 rounding 还要评估模型质量 |
| 做单卡实验时 | PyTorch 梯度参考，Triton 分块、边界和可靠计时 | 给同一批输入检查输出与梯度，把首次编译和稳定运行分开 |
| 定位到具体慢点后 | CuTe/QuACK 分组 GEMM、Hopper 线程组、TMA 异步搬运 | 对一个实测慢的内核，解释时间花在计算、搬运还是等待上 |

阅读源码可以依次看作者的正确性测试、单层基准、前后向实现与分桶逻辑，然后挑一个归约或聚合算子。带着已经测到的瓶颈进入 QuACK 的矩阵乘实现，更容易看出一个 tile 参数究竟影响了什么。[SonicMoE 实现目录](https://github.com/Dao-AILab/sonic-moe/tree/7396f3e604827d8186c2e16e64b28ee33d3defd0/sonicmoe/functional) · [QuACK 项目](https://github.com/Dao-AILab/quack)

## 在个人电脑上验证路由梯度

下面的 NumPy 代码使用开头的 4 个 token 和 4 位专家，共 8 个 token–专家分配。固定激活后的 $a$ 与专家索引，对**已经选中的聚合权重**逐个作中心差分。取损失 $L=\tfrac12\sum_t\langle o_t,o_t\rangle$，因此 $g_t=o_t$。这一步单独检验 $\partial L/\partial s$，路由器 logits 之前的链式求导属于另一步。

```python
import numpy as np

routes = [[0, 1], [0, 1], [0, 2], [0, 3]]
rng = np.random.default_rng(2026)
a = rng.normal(size=(4, 2, 2))       # [token, choice, n]
w = rng.normal(size=(4, 2, 6))       # [expert, n, d]
s = np.array([[.75, .25], [.65, .35], [.55, .45], [.8, .2]])
y = np.empty((4, 2, 6))
for t, pair in enumerate(routes):
    for j, e in enumerate(pair):
        y[t, j] = a[t, j] @ w[e]

def output(weights):
    return (weights[:, :, None] * y).sum(axis=1)

def loss(weights):
    return np.square(output(weights)).sum() / 2

g = output(s)                       # dL/dO for L = ||O||^2 / 2
eps = 1e-5
algebra_errors, difference_errors = [], []
for t, pair in enumerate(routes):
    for j, e in enumerate(pair):
        old = g[t] @ y[t, j]
        new = (g[t] @ w[e].T) @ a[t, j]
        delta = np.zeros_like(s)
        delta[t, j] = eps
        numeric = (loss(s + delta) - loss(s - delta)) / (2 * eps)
        algebra_errors.append(abs(old - new))
        difference_errors.append(abs(numeric - new))
print(f"pairs={len(algebra_errors)}, algebra={max(algebra_errors):.3e}, finite_diff={max(difference_errors):.3e}")
assert max(algebra_errors) < 1e-12 and max(difference_errors) < 1e-8
```

使用 float64 运行，8 项中两种解析公式的最大绝对差约为 `8.88e-16`，新公式与中心差分的最大绝对差约为 `3.26e-11`。这些是梯度误差。论文的显存和吞吐需要按前面的基准在 GPU 上采集。

## 十分钟汇报提纲

用四个 token 画出动态形状，重点手算路由权重梯度，再用 130 行与 126 行解释尾块。汇报末尾，把论文已测出的结果和自己准备验证的问题分别讲清楚。独立的 Markdown 提纲包含八页的时间分配、可直接使用的开场和收尾，以及导师可能追问的三个问题。[下载十分钟汇报提纲（Markdown）](https://neurpulse.github.io/downloads/sonicmoe-briefing.md)

> 我想研究大模型训练中的动态分组矩阵乘。MoE 每一批把不同数量的 token 分给各位专家，同一个算子会遇到不同的输入形状。SonicMoE 提供了可检查的开源基线。第一阶段，我准备在单卡上测量，当专家负载从均匀变成偏斜时，原有配置复用会损失多少性能。第一次汇报实验时，我会给出单层的正确性结果、不同负载的耗时图，以及这个问题是否值得继续做的判断。

本文依据 Guo 等人的 SonicMoE 论文整理，计算例子与单卡优化方案用于教学和研究讨论。基于原文的教学改编按作者预印本标示的 CC BY-SA 4.0 许可分享。[作者预印本](https://arxiv.org/abs/2512.14080) · [许可说明](https://creativecommons.org/licenses/by-sa/4.0/)
