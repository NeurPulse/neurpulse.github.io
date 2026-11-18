# SonicMoE 十分钟导师汇报提纲

**拟定方向**　动态专家负载下的分组矩阵乘与算子优化。

**希望和导师确定的事**　先以 SonicMoE 为单层基线，在一张兼容的 GPU 上验证专家负载分布是否影响配置复用。根据第一次实验的结果，再确定具体优化模块。[完整论文精读](https://neurpulse.github.io/research/sonicmoe-training-kernels/) · [ICLR 2026 正式论文](https://proceedings.iclr.cc/paper_files/paper/2026/hash/6ed5bf446f59e2c6646d23058c86424b-Abstract-Conference.html)

## 八页如何分配时间

| 页 | 用时 | 一页回答一个问题 | 画什么或展示什么 |
|---:|---:|---|---|
| 1 | 0:40 | 为什么选这篇论文？ | 说清楚研究对象是 MoE 训练中的算子。正式论文、开源实现、正确性测试和单层基准让第一轮研究可以独立开展。 |
| 2 | 1:00 | 动态形状从哪里来？ | 画 4 个 token 各选 2 个专家。专家行数可以是 4、2、1、1，也可以是 2、2、2、2。权重尺寸固定，输入行数随路由改变。 |
| 3 | 1:00 | 理论算量相同，为什么 GPU 耗时不同？ | 比较专家宽度 $n=1024,k=2$ 与 $n=256,k=8$。$kn$ 相同，但路由产生的宽向量变多。再指出取行搬运和矩阵乘尾块的代价。 |
| 4 | 2:00 | 怎么提前释放一份大张量？ | 手算下方的梯度等式。解释 $gW_2^\top$ 本来就是反向需要的结果，复用它便能在前向聚合后释放专家输出 $Y$。 |
| 5 | 1:20 | 怎么让 GPU 少搬数据、少等待？ | 结合图 8、9，讲清按索引取行、在矩阵乘收尾时做激活与归约，以及 Hopper 上交错执行计算和收尾。 |
| 6 | 1:00 | 为什么要改变训练路由？ | 设每块 128 行，两专家原来各有 130、126 行，需要 3 块。各调到 128 行后只需 2 块。真实分配改变了，验证时还要看训练质量。 |
| 7 | 1:00 | 作者测到了什么？ | 报告单层峰值激活显存、单层计算吞吐和完整训练吞吐。若导师关注路由质量，再用表 1(c) 说明条件。 |
| 8 | 2:00 | 下一步能独立验证什么？ | 固定 BF16、单卡、单层、标准 top-k，只改变专家负载。比较原版所选配置与同组候选的离线最快配置，记录整层时间。 |

### 第 4 页可以直接手算的公式

令 $s$ 为聚合权重，$a$ 为激活后的专家向量，$y=aW_2$ 为专家输出，$g$ 为从后续网络传回的梯度。

$$
\frac{\partial L}{\partial s}
=\langle g,y\rangle
=\langle g,aW_2\rangle
=\langle gW_2^\top,a\rangle .
$$

取 $a=2$、$W_2=[3,4]$、$g=[1,2]$。先算 $y$，得到 $6+16=22$。先算 $gW_2^\top$，得到 $11\times2=22$。后者可以复用反向已有的运算。需要 $A$ 时，只需从已保存的 $H$ 重做一次门控激活。[论文计算图与流程示意图](https://proceedings.iclr.cc/paper_files/paper/2026/file/6ed5bf446f59e2c6646d23058c86424b-Paper-Conference.pdf#page=19)

### 第 7 页的数字与适用范围

| 证据 | 在会上怎么说 | 对照位置 |
|---|---|---|
| H100 细粒度 7B，BF16，24,576 个 token，128 位专家选 8 位，每个专家隐藏宽度 256 | 相比 ScatterMoE，MoE **单层峰值激活显存减少约 45%**。图中两者的前向耗时分别为 **1.237 与 2.255 ms**，反向为 **2.173 与 3.968 ms**。摘要另报告 MoE **计算吞吐为 1.86 倍**。 | [图 5 显存](https://proceedings.iclr.cc/paper_files/paper/2026/file/6ed5bf446f59e2c6646d23058c86424b-Paper-Conference.pdf#page=8) · [图 2(a) 耗时](https://proceedings.iclr.cc/paper_files/paper/2026/file/6ed5bf446f59e2c6646d23058c86424b-Paper-Conference.pdf#page=5) · [摘要](https://proceedings.iclr.cc/paper_files/paper/2026/hash/6ed5bf446f59e2c6646d23058c86424b-Abstract-Conference.html) |
| 7B 完整训练，FSDP-2 | SonicMoE **64 张 H100，每天 2130 亿 token**。ScatterMoE **96 张 H100，每天 2250 亿 token**。卡数不同，总吞吐接近。 | [第 6.2 节第 2 段，原文以正文报告](https://proceedings.iclr.cc/paper_files/paper/2026/file/6ed5bf446f59e2c6646d23058c86424b-Paper-Conference.pdf#page=9) |
| 1.4B、1000 亿训练 token、128 位专家选 2 位 | Token rounding 对标准 top-k 的验证困惑度为 **13.22 对 13.32**，11 项下游任务平均准确率均为 **52.0%**。验证和下游评估统一使用标准 top-k。 | [表 1(c)](https://proceedings.iclr.cc/paper_files/paper/2026/file/6ed5bf446f59e2c6646d23058c86424b-Paper-Conference.pdf#page=9) |

以上为论文作者报告的实验。配套的 CPU 示例检查了 8 个路由权重的梯度，与数值差分之间的最大绝对差约为 $3.26\times10^{-11}$。H800 的单层性能仍需另行测量。[完整笔记的 CPU 梯度算例](https://neurpulse.github.io/research/sonicmoe-training-kernels/#在个人电脑上验证路由梯度)

## 可直接使用的开场和收尾

**开场**

> 老师，我想研究大模型训练中的动态分组矩阵乘。在 MoE 里，每一批数据都会把不同数量的 token 分给各个专家。即使理论计算量相同，GPU 上的实际耗时也可能不同。我选 SonicMoE 作基线，是因为它给出了对这个问题的分析、可读的源码，以及能在单层上做对照的基准。

**收尾**

> 我想先做一个能在单卡上验证的小问题。当专家负载从均匀变成偏斜时，复用先前选好的矩阵乘配置，会损失多少性能？我会先在实际租到的 H800 上跑通官方测试，再用单层的正确性结果和不同负载的耗时图，说明这个问题是否值得继续做。

## 三个可能的追问

**既然已经有自动调优，还能做什么？** 先记录实际命中的分桶键和胜出配置。对同样的输入形状分别重放均匀、偏斜、有空专家的负载，比较原版复用与逐负载离线最佳值，并交换首次出现的顺序。只有差距可重复且足以抵消选择开销时，才有理由改配置选择策略。[固定版本的配置复用代码](https://github.com/Dao-AILab/sonic-moe/blob/7396f3e604827d8186c2e16e64b28ee33d3defd0/sonicmoe/functional/__init__.py#L104)

**为什么会 Python 也能开始？** 可以先用 PyTorch 写正确性参考、构造路由、重放输入并测时，再围绕耗时最大的一个算子学习 Triton 或 CuTe。个人电脑能检查梯度，A800 可以练习单层测试。H800 属于 Hopper，是验证 SonicMoE 原实现的候选。原版和改动版应在同一张通过测试的设备上比较。[作者的环境说明](https://github.com/Dao-AILab/sonic-moe/tree/7396f3e604827d8186c2e16e64b28ee33d3defd0#installation)

**什么时候必须训练大模型？** 固定 top-k、只改算子执行时，单层就能对照输出、梯度和耗时。一旦改变 token rounding 等路由行为，或者要提出模型质量结论，就需要补上相应规模的训练和评估。若这轮配置复用的空间很小，可以依据分项耗时转查 gather 或专家聚合。

本提纲据 Wentao Guo 等人的 *SonicMoE: Accelerating MoE with IO and Tile-aware Optimizations* 整理。中文教学改编沿用作者预印本的 CC BY-SA 4.0 许可。[原文与许可信息](https://arxiv.org/abs/2512.14080) · [许可说明](https://creativecommons.org/licenses/by-sa/4.0/)
