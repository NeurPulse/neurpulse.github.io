---
title: 'DeepSeekMoE 精读｜为什么把专家拆小，还要保留共享专家'
description: '用同一个二维输入算一遍宽专家与共享专家的输出，对照 16B 开源模型的张量形状，再用 2B 消融实验检验共享专家的作用。'
pubDate: 2026-11-19
kind: record
category: '论文精读'
tags: ['论文笔记', 'MoE', '专家路由', '大语言模型']
---

## 论文信息

- **题名**　*DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models*。
- **作者与机构**　Damai Dai、Chengqi Deng 等。作者来自 DeepSeek-AI、北京大学、清华大学及南京大学。
- **版本**　ACL 2024 长文，1280–1297 页。图表编号采用 ACL 正式版。代码对照作者发布的 `deepseek-moe-16b-base`，论文里的 2B 模型另用于验证架构。[会议记录与 DOI](https://aclanthology.org/2024.acl-long.70/) · [论文 PDF](https://aclanthology.org/2024.acl-long.70.pdf) · [作者仓库](https://github.com/deepseek-ai/DeepSeek-MoE) · [16B 模型配置][config]

## 一句话总结

DeepSeekMoE 把前馈网络拆成可独立选择的小专家，同时让每个 token 都经过共享专家，在 2B 模型上以与 GShard 相同的激活参数量和计算预算，把 Pile 测试损失从 1.867 降至 1.808。[表 1](https://aclanthology.org/2024.acl-long.70.pdf#page=5)

## 研究动机

在普通 top-2 混合专家层里，路由器读入一个 token 的隐藏向量，为它选两个前馈网络（专家）。每个专家是一个整体。如果它同时学到两种变换，只需要其中一种时，整套权重仍要参与计算。反过来，若许多 token 都需要相似的变换，多位专家可能分别花参数去学同一件事。这是作者提出的两种解释。前一种引出“能否拆开选择”，后一种引出“能否让所有 token 共用一部分权重”。[引言与第 3 节](https://aclanthology.org/2024.acl-long.70.pdf#page=2)

## 核心方法

看图 1 的三栏。左边从 $N$ 个完整专家中选 2 个。中间把每个专家的前馈**中间宽度**减半，从 $2N$ 个小专家中选 4 个。若原来每个专家有 8 个中间通道，同一个 token 经过的通道数从 $2\times8$ 变成 $4\times4$，都是 16。输入和输出向量的长度也保持原样，但原本绑在同一宽专家里的通道，现在可以分开选择。右边把 1 个小专家固定为共享专家，每个 token 都经过它，路由器从其余小专家中再选 3 个。[第 3.1–3.2 节](https://aclanthology.org/2024.acl-long.70.pdf#page=3)

<figure class="paper-figure" id="fig-1">
  <a class="paper-panel" href="/images/research/moe-papers/deepseek-fig-1.webp"><img src="/images/research/moe-papers/deepseek-fig-1.webp" width="1819" height="892" loading="lazy" decoding="async" alt="从普通 top-2，到四个半宽专家，再到一个共享专家和三个按 token 选择的专家。"></a>
  <figcaption><strong>图 1</strong>　沿最右栏的绿色分支往上看，每个输入都会经过这个专家。蓝色分支的选择随输入而变。 <a href="https://aclanthology.org/2024.acl-long.70.pdf#page=3">Dai 等，ACL 2024 原图</a> · <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a></figcaption>
</figure>

设 $x$ 是一个 token 进入专家层的向量，$E_i(x)$ 是专家 $i$ 算出的向量，$p_i(x)$ 是路由器给它的概率。记 $S$ 为共享专家集合，$R(x)$ 为这一次入选的路由专家集合。专家层的输出为

$$
y(x)=\sum_{i\in S}E_i(x)+\sum_{i\in R(x)}p_i(x)E_i(x).
$$

第一项直接相加，第二项按概率加权。论文式 (9) 还加上残差。这里把 $y$ 定义为送回残差分支之前的专家输出，方便和代码里的 `DeepseekMoE.forward` 对照。16B 官方配置对全部路由专家做 softmax，取 top-k 后保留原概率，因此选中项的权重和可以小于 1。[式 (9)–(11)](https://aclanthology.org/2024.acl-long.70.pdf#page=4) · [官方门控实现][gate] · [模型配置][config]

### 用同一个输入算两遍

这是自拟的教学例子。把图 1 缩小到两个宽专家 A、B，每个有两个中间通道。输入取 $x=(1,2)$，用 $e_0,e_1,e_2,e_3$ 表示四个通道对输出向量的贡献。对这一个输入，给定 $e_0=(1,0)$、$e_1=(0,2)$、$e_2=(2,2)$、$e_3=(0,1)$。原来 A 包含 $e_0,e_1$，B 包含 $e_2,e_3$。

| 同一个输入 | 经过哪些通道 | 专家层的输出 |
| --- | --- | --- |
| 两个宽专家选一个 | A、B 的概率为 0.75、0.25，选中 A | $0.75(e_0+e_1)=(0.75,1.5)$ |
| 四个窄专家，每次用两个 | 固定共享专家 $e_0$。其余三位的概率为 0.25、0.50、0.25，选中 $e_2$ | $e_0+0.50e_2=(2,1)$ |

两条路径都只激活两个通道。第二条可以把原属 A 的 $e_0$ 和原属 B 的 $e_2$ 放在一起。给定的概率用来演示前向运算，模型训练时会学习自己的路由与专家权重。下面把四个通道的贡献排成四行，用 NumPy 在 CPU 上核对选中项的加权结果。

```python
import numpy as np

x1, x2 = (1.0, 2.0)
parts = np.array([[x1, 0], [0, x2], [x2, x2], [0, x1]])
wide_experts = np.stack((parts[0] + parts[1], parts[2] + parts[3]))
wide_prob = np.array([0.75, 0.25])
chosen = wide_prob.argmax()
wide_out = wide_prob[chosen] * wide_experts[chosen]

routed_prob = np.array([0.25, 0.50, 0.25])
routed_parts = parts[1:]  # 门控候选依次是 e1、e2、e3
chosen = routed_prob.argmax()
shared_out = parts[0] + routed_prob[chosen] * routed_parts[chosen]
np.testing.assert_allclose(wide_out, [0.75, 1.5])
np.testing.assert_allclose(shared_out, [2.0, 1.0])
```

### 放回真实的 16B 模型

把批量和序列维展平，设一层收到 $T$ 个 token，输入矩阵的形状就是 $[T,2048]$。可以按下面三步跟读作者代码。

1. **分给谁。** 路由器产生 $[T,64]$ 的分数，每行保留 6 个专家编号与权重，一共有 $6T$ 条动态路由记录。两个共享专家处理全部 $T$ 行，共用的是权重，每个 token 都有自己的输出。[`MoEGate.forward`][gate]
2. **各算多少行。** 若路由专家 $i$ 收到 $n_i$ 个 token，它的两次输入投影分别生成 $[n_i,1408]$，其中一路经过激活函数，两路逐元素相乘，再投影回 $[n_i,2048]$。所有 $n_i$ 的和为 $6T$，每个 $n_i$ 由这一批的路由决定。官方实现把两个共享专家合成中间宽度 $2\times1408=2816$ 的一条前馈分支。[模型配置][config] · [专家前馈计算][mlp] · [共享分支的构造][expert]
3. **回到哪个 token。** 公开推理实现的 `moe_infer` 按专家编号排序，收集对应输入行，把加权输出累加回原 token，再加上共享分支。输出恢复为 $[T,2048]$。这三步也指出了算子计时要覆盖的工作，包括分组、矩阵乘法和写回。[`DeepseekMoE.forward` 与 `moe_infer`][expert]

## 相对前作的创新点

- **对照 GShard 的门控。** GShard 的算法 1 为 token 选择至多两个完整专家，并处理容量限制。DeepSeekMoE 沿用按 token 打分和加权的计算方式，把专家宽度和激活个数一起调整，使同等激活宽度有更多组合。[GShard 算法 1](https://arxiv.org/pdf/2006.16668#page=7) · [本文第 3.1 节](https://aclanthology.org/2024.acl-long.70.pdf#page=3)
- **共享路径有前例。** DeepSpeed-MoE 的 Residual-MoE 已让一个固定 MLP 和一个动态专家共同处理输入。DeepSeekMoE 把共享专家纳入细粒度专家预算，固定总参数和激活参数，先比较有无共享专家，再逐步缩小专家宽度、增加激活个数。图 2 的四组柱子正对应这两项改动。[DeepSpeed-MoE 第 4.1.1 节、图 2](https://arxiv.org/pdf/2201.05596#page=9) · [本文第 4.4 节](https://aclanthology.org/2024.acl-long.70.pdf#page=6)

## 实验设置

- **验证架构的 2B 实验。** 模型有 9 层，隐藏维度 1280，用中英为主的语料训练 100B 个 token。GShard 在 16 个完整专家中选 2 个。DeepSeekMoE 每个小专家只有原来四分之一的中间宽度，每个 token 用 **1 个共享专家和从 63 个候选中选出的 7 个路由专家**。二者均约有 2.0B 总参数、0.3B 激活参数，每 2K 个 token 均为 4.3T FLOPs。表 1 的 Pile 指标是测试集交叉熵，越低越好。[第 4.1 节](https://aclanthology.org/2024.acl-long.70.pdf#page=4) · [表 1 与专家配置](https://aclanthology.org/2024.acl-long.70.pdf#page=5)
- **放大到 16B 的实验。** 共 28 层，第一层保留稠密前馈网络，其余各层用 **2 个共享专家和从 64 个候选中选出的 6 个路由专家**。模型有 16.4B 总参数，单个 token 激活约 2.8B。它与稠密 DeepSeek 7B 使用相同训练数据，各训练 2T 个 token。表 2 的 Pile 指标改用每字节比特数（BPB）。附录 B 说明作者在 A100 或 H800 集群上训练，表中的 FLOPs 表示运算量。[第 5.1–5.2 节](https://aclanthology.org/2024.acl-long.70.pdf#page=7) · [表 2](https://aclanthology.org/2024.acl-long.70.pdf#page=8) · [附录 B](https://aclanthology.org/2024.acl-long.70.pdf#page=15)

## 主要结果

| 对照 | 作者报告的结果 | 怎么解读 |
| --- | --- | --- |
| 2B，分别训练 GShard 与 DeepSeekMoE | Pile 损失 1.867 → 1.808，HellaSwag 零样本准确率 50.5% → 54.8%。 | 相同激活预算下，整套设计让损失下降 0.059，准确率提高 4.3 个百分点。[表 1](https://aclanthology.org/2024.acl-long.70.pdf#page=5) |
| 同一个已训练的 DeepSeekMoE 2B，评估时关闭共享专家，同时多启用一个路由专家 | Pile 损失 1.808 → 2.414。 | 都跑 8 个小专家，临时替换使损失升高 0.606。这衡量了该模型对已学共享分支的依赖。[第 4.5 节](https://aclanthology.org/2024.acl-long.70.pdf#page=6) |
| DeepSeekMoE 16B 与稠密 DeepSeek 7B | 每 4K 个 token 分别为 74.4T 和 183.5T FLOPs。HellaSwag 为 77.1% 和 75.4%，MMLU 为 45.0% 和 48.2%。 | 计算量比约为 40.5%，不同任务上的胜负不同。[表 2](https://aclanthology.org/2024.acl-long.70.pdf#page=8) |

<figure class="paper-figure" id="fig-2">
  <a class="paper-panel" href="/images/research/moe-papers/deepseek-fig-2.webp"><img src="/images/research/moe-papers/deepseek-fig-2.webp" width="1531" height="463" loading="lazy" decoding="async" alt="六个基准上的专家结构消融，蓝柱是常规 GShard，其他颜色依次引入共享专家和更细的划分。"></a>
  <figcaption><strong>图 2</strong>　先看 TriviaQA 一组。蓝柱是从 16 个完整专家中选 2 个，黄柱是固定共享 1 个、再从 15 个中选 1 个。绿柱和橙柱进一步拆小专家。纵轴按每项任务的最佳成绩归一化，柱高 1 代表该组最佳。 <a href="https://aclanthology.org/2024.acl-long.70.pdf#page=5">Dai 等，论文原图</a></figcaption>
</figure>

**亲手读一次纵轴。** 表 1 中，GShard 和 DeepSeekMoE 在 TriviaQA 的精确匹配率分别为 10.2% 和 16.6%。用 $10.2/16.6\approx0.61$，就能在图 2 的 TriviaQA 一组找到蓝柱的高度。接着只看蓝柱和黄柱，两种模型都从头训练，专家总数和激活数相同，差别在于是否把一个位置固定给共享专家。[表 1 与图 2](https://aclanthology.org/2024.acl-long.70.pdf#page=5) · [第 4.4 节](https://aclanthology.org/2024.acl-long.70.pdf#page=6)

<figure class="paper-figure" id="fig-3-4">
  <div class="paper-panels">
    <a class="paper-panel" id="fig-3" href="/images/research/moe-papers/deepseek-fig-3.webp"><img src="/images/research/moe-papers/deepseek-fig-3.webp" width="853" height="360" loading="lazy" decoding="async" alt="图 3，按比例屏蔽排名靠前的路由专家时，DeepSeekMoE 和 GShard ×1.5 的 Pile 损失都上升，DeepSeekMoE 上升更多。"><span>图 3　屏蔽路由器最想选的专家</span></a>
    <a class="paper-panel" id="fig-4" href="/images/research/moe-papers/deepseek-fig-4.webp"><img src="/images/research/moe-papers/deepseek-fig-4.webp" width="853" height="359" loading="lazy" decoding="async" alt="图 4，保持共享专家开启，改变激活的路由专家数。选 4 个时，DeepSeekMoE 的 Pile 损失接近 GShard 的水平虚线。"><span>图 4　改变每个 token 可用的路由专家数</span></a>
  </div>
  <figcaption>两图纵轴都是 Pile 损失，越低越好。图 3 的对手是起始损失同为 1.808 的加宽版 GShard ×1.5，图 4 的虚线是标准 GShard 的 top-2 结果。沿虚线找到橙色曲线，交点约在横坐标 4。此处实际用到 4 个路由专家和 1 个共享专家。 <a href="https://aclanthology.org/2024.acl-long.70.pdf#page=6">Dai 等，ACL 2024 图 3、图 4</a> · <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a></figcaption>
</figure>

## 优点与不足

图 3 先选起始损失相同的两个模型，再对每个 token 屏蔽排名靠前的路由专家，给其余专家接手的机会。DeepSeekMoE 的损失上升更多，这为“被选专家更难替代”提供了证据。要进一步判断某位专家究竟处理了哪些语言现象，还需要看具体输入及针对该专家的干预。[第 4.5 节](https://aclanthology.org/2024.acl-long.70.pdf#page=6)

从算子角度看，专家宽度缩小会改变矩阵形状。若热门专家收到许多行、冷门专家只收到一两行，每组矩阵乘法利用 GPU 的程度就可能不同。论文第 3.3 节在训练中加入均衡损失，鼓励路由分散。表 1、表 2 给出模型浮点运算量，要得到整层毫秒数，还需要计入分组、读写、结果合并，以及专家跨卡放置时的通信。[负载均衡公式 (12)–(14)](https://aclanthology.org/2024.acl-long.70.pdf#page=4) · [作者训练系统](https://aclanthology.org/2024.acl-long.70.pdf#page=15)

## 对我的启发

向老师讲时，可以先在图 1 指出一个以前无法选择的通道组合，再指着共享分支提问，“把它关掉，补上一位同样宽的路由专家，预算相同，质量还一样吗？”这时给出第 4.5 节的 **1.808 → 2.414**，并补一句“这是在训练完成后换的专家”。如果老师追问从头训练会怎样，就回到图 2 的前两组架构。[图 1](#fig-1) · [图 2](#fig-2)

**留给算子实验的问题。** 在一张 GPU 上固定 token 数、输入宽度、精度和每个 token 激活的总中间宽度，对比“16 个宽专家选 2 个”和“1 个共享小专家加 63 个候选中选 7 个”。分别构造均匀路由和集中到少数专家的路由，记录每个专家实际收到的行数，同时测分组、专家计算、合并与整层时间。上面的 CPU 例子只核对了前向算术，这项 GPU 性能对照留待实测。[ScatterMoE 的数据搬运实验](/research/scattermoe-paper-reading/#fig-4)

[config]: https://huggingface.co/deepseek-ai/deepseek-moe-16b-base/blob/521d2bc4fb69a3f3ae565310fcc3b65f97af2580/config.json
[mlp]: https://huggingface.co/deepseek-ai/deepseek-moe-16b-base/blob/521d2bc4fb69a3f3ae565310fcc3b65f97af2580/modeling_deepseek.py#L216
[gate]: https://huggingface.co/deepseek-ai/deepseek-moe-16b-base/blob/521d2bc4fb69a3f3ae565310fcc3b65f97af2580/modeling_deepseek.py#L265
[expert]: https://huggingface.co/deepseek-ai/deepseek-moe-16b-base/blob/521d2bc4fb69a3f3ae565310fcc3b65f97af2580/modeling_deepseek.py#L322
