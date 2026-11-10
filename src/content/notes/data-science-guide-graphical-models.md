---
title: "数据科学导引第十一章：概率图模型"
description: "从条件独立与因子分解理解有向图和无向图，用可复算例子掌握 HMM 前后向算法、Viterbi、Baum-Welch 与线性链 CRF"
pubDate: 2026-11-10
updatedDate: 2026-11-10
category: '数据科学'
order: 12
tags: ['数据科学', '数据科学导引', '概率图模型', '隐马尔可夫模型', 'HMM', '条件随机场', 'CRF', 'Viterbi', '前后向算法', '读书笔记', '零基础']
---

> [《数据科学导引》学习地图](/notes/data-science-guide-roadmap/) · 第 11 / 15 章
>
> 上一章：[EM 算法](/notes/data-science-guide-em-algorithm/)
>
> 下一章：[文本分析](/notes/data-science-guide-text-analysis/)

本文对应《数据科学导引》第十一章“概率图模型”，原书第 211—234 页，即所提供扫描版 PDF 第 220—243 页。笔记保留概率图模型概述、隐马尔可夫模型、条件随机场和小结四条主线，补充条件独立的三种基本结构、配分函数手算、HMM 穷举对照、Viterbi 回溯和 CRF 路径概率示例。明确标出的“学习补充”与“实践补充”用于帮助理解，并非教材原文。

## 学习目标

学完本章，应能回答十个问题：

- 生成式模型与判别式模型分别建模什么，朴素贝叶斯、逻辑回归、HMM、CRF 各属于哪一类？
- 有向图怎样通过父节点把联合分布分解为条件概率的乘积？
- 链式、分叉和对撞结构中，条件与边缘独立有什么不同？
- 无向图怎样用团、势函数和配分函数表示联合分布？
- HMM 的初始概率、转移概率、观测概率分别控制什么？
- 前向算法为什么能把指数级的路径求和变成 $O(S^2T)$ 的动态规划？
- Viterbi 与前向算法为何一个取 $\max$、一个求和，它们回答的问题有何不同？
- Baum-Welch 怎样作为 EM 的特例，利用软状态计数和软转移计数更新参数？
- 线性链 CRF 怎样直接建模 $p(\boldsymbol y\mid\boldsymbol x)$，为什么 $Z(\boldsymbol x)$ 依赖输入？
- CRF 的特征工程、条件似然梯度、正则化和序列推断怎样连接起来？

## 先看全局：图是联合分布的压缩说明

数据科学中的许多问题都可以写成：已知观测变量 $\boldsymbol x$，推断目标变量 $\boldsymbol y$ 的分布。变量增多后，直接列出完整联合分布既昂贵，也不利于表达结构。概率图模型用节点表示随机变量，用边及其方向表示依赖关系，再用局部分布或局部势函数恢复整体概率。

先分清四种常见建模方式：

| 模型 | 主要建模对象 | 图的形式 | 本章中的角色 |
|---|---|---|---|
| 朴素贝叶斯 | $p(y,\boldsymbol x)=p(y)\prod_jp(x_j\mid y)$ | 有向图 | 生成式分类 |
| 逻辑回归 | $p(y\mid\boldsymbol x,\boldsymbol\theta)$ | 通常不强调图结构 | 判别式分类 |
| HMM | $p(\boldsymbol x_{1:T},\boldsymbol y_{1:T})$ | 有向链 | 生成式序列模型 |
| 线性链 CRF | $p(\boldsymbol y_{1:T}\mid\boldsymbol x_{1:T})$ | 条件无向链 | 判别式序列模型 |

这里的“生成”不是指模型一定会自动输出文本或图像，而是说它对观测与目标的联合分布建模，因此原则上可以从联合分布采样。判别式模型直接关心预测所需的条件分布，不必额外描述观测数据怎样生成。

学习本章时始终连起四件事：

> **图结构声明依赖关系，因子分解定义概率，推断计算边际或最优路径，学习根据数据估计参数。**

## 11.1 概率图模型概述

### 生成式与判别式：联合分布还是条件分布

若已知参数 $\boldsymbol\theta$，推断目标通常是：

$$
p(\boldsymbol y\mid\boldsymbol x,\boldsymbol\theta)
=\frac{p(\boldsymbol x,\boldsymbol y\mid\boldsymbol\theta)}
{p(\boldsymbol x\mid\boldsymbol\theta)}.
$$

生成式模型先建模联合分布，再通过概率规则得到条件分布。以朴素贝叶斯为例，它假设给定类别 $y$ 后各特征条件独立：

$$
p(y,\boldsymbol x)
=p(y)p(\boldsymbol x\mid y)
=p(y)\prod_{j=1}^{d}p(x_j\mid y).
$$

它的优势是生成机制清楚，也可处理由模型采样的任务；困难在于真实特征之间的依赖可能很复杂，条件独立假设过强时会影响概率估计。

判别式模型直接建模 $p(y\mid\boldsymbol x,\boldsymbol\theta)$。例如逻辑回归用一个线性分数经过 sigmoid 得到类别概率，不需要描述各特征自身的联合分布。它专注于预测边界，但不能仅凭条件模型完整生成观测与标签的联合样本。

二者不是“谁永远更好”的关系：数据生成机制、样本量、特征依赖、是否需要采样以及最终评价任务，都会影响选择。

### 有向图：按父节点分解联合概率

有向图模型又称贝叶斯网络或信念网络。若图是有向无环图，节点 $x_s$ 的父节点集合记为 $\operatorname{pa}(s)$，联合分布写为：

$$
p(\boldsymbol x)=\prod_s p\bigl(x_s\mid\boldsymbol x_{\operatorname{pa}(s)}\bigr).
$$

教材图 11.1 的五变量网络对应：

$$
p(\boldsymbol x)=p(x_1)p(x_2\mid x_1)p(x_3\mid x_1)
p(x_4\mid x_1)p(x_5\mid x_2,x_4).
$$

这个乘积不是看到图后随意拼出的，而是链式法则加局部条件独立假设的结果。在给定拓扑顺序时，一个节点在给定父节点后，与顺序中除父节点外的前驱节点条件独立。更一般地，贝叶斯网络常用“给定父节点后，与非后代节点条件独立”表述局部马尔可夫性质。

朴素贝叶斯的图只有一个父节点 $y$，所有特征节点都由它指出，因此得到 $p(y)\prod_jp(x_j\mid y)$。箭头在这里表示概率分解方向，不应仅凭箭头就宣称已经证明因果关系。

### 学习补充：三种基本结构与条件独立

判断“两个变量是否独立”时，必须说明是否给定了其他变量。以下结论针对只含所示依赖的基本结构：

| 结构 | 图 | 不给定中间节点 | 给定中间节点 $B$ |
|---|---|---|---|
| 链式 | $A\rightarrow B\rightarrow C$ | $A,C$ 通常相关 | 依赖被阻断，$A\perp C\mid B$ |
| 分叉 | $A\leftarrow B\rightarrow C$ | 共同原因使 $A,C$ 通常相关 | 依赖被阻断，$A\perp C\mid B$ |
| 对撞 | $A\rightarrow B\leftarrow C$ | 在无其他连接时 $A,C$ 可边缘独立 | 条件化可引入依赖 |

对撞结构最容易误判。例如 $A$、$C$ 是两个独立原因，$B$ 是共同结果；已知结果 $B$ 后，解释其中一个原因会改变对另一个原因的判断。这就是“解释 away”现象。对撞节点的后代被观测时，也可能打开类似的信息通道。

因此，“图中没有直接边”不等于无条件独立，“两点之间存在路径”也不等于任何条件下都相关。完整有向图推断需要 d-分离规则；本章先掌握这三种局部结构即可。

### 无向图：用团和势函数表示交互

无向图模型又称马尔可夫随机场（MRF）。它没有父子节点方向，更适合表达像素邻接、相互约束等对称关系。

- **团**：两两之间都有边相连的节点集合。
- **极大团**：不被更大的团真包含的团。
- **最大团**：节点数最多的团；“极大”是集合包含关系，“最大”是数量比较，二者不要混淆。

教材图 11.3 的极大团为：

$$
\{x_1,x_3\},\qquad
\{x_1,x_2,x_4\},\qquad
\{x_2,x_4,x_5\}.
$$

对严格为正的联合分布，Hammersley–Clifford 定理给出按极大团势函数的分解：

$$
p(\boldsymbol x)=\frac{1}{Z}\prod_{c\in\mathcal C}\phi_c(\boldsymbol x_c),
$$

其中 $\phi_c$ 是非负势函数，离散变量的配分函数为：

$$
Z=\sum_{\boldsymbol x}\prod_{c\in\mathcal C}\phi_c(\boldsymbol x_c).
$$

连续变量则把求和换成积分。势函数表示局部配置的相对相容程度，**不是已经归一化的条件概率**。只有把所有配置对应的乘积累加起来得到 $Z$，整体才成为概率分布。

无向图的条件独立可通过图分离理解：若删除观测集合后，两组变量之间的所有路径都被阻断，则满足相应条件独立关系。这与有向图的对撞规则不同，不能直接套同一套口诀。

### 手算：势函数为什么必须经过配分函数归一化

设两个二值变量 $X_1,X_2\in\{0,1\}$，只有一个团 $\{X_1,X_2\}$。令：

$$
\phi(x_1,x_2)=
\begin{cases}
2,&x_1=x_2,\\
1,&x_1\ne x_2.
\end{cases}
$$

四个配置 $(0,0),(0,1),(1,0),(1,1)$ 的未归一化权重依次为 $2,1,1,2$，所以：

$$
Z=2+1+1+2=6.
$$

对应概率为：

| 配置 | 未归一化权重 | 概率 |
|---|---:|---:|
| $(0,0)$ | 2 | $1/3$ |
| $(0,1)$ | 1 | $1/6$ |
| $(1,0)$ | 1 | $1/6$ |
| $(1,1)$ | 2 | $1/3$ |

因此两个变量取值相同的概率是 $2/3$。若把势 2 错当成“概率 2”，既可能大于 1，也无法保证总和为 1。

一般模型若有 $n$ 个变量，每个变量有 $M$ 个状态，直接枚举需要处理 $M^n$ 个配置。配分函数正是无向模型学习和推断困难的重要来源；链、树等特殊结构可以利用动态规划降低计算量。

### 对数线性形式：把势函数写成特征加权的指数

实际模型常把势函数写为：

$$
\phi_c(\boldsymbol x_c)
=\exp\{\boldsymbol\theta_c^{T}\boldsymbol f_c(\boldsymbol x_c)\}.
$$

于是：

$$
\ln p(\boldsymbol x)
=\sum_c\boldsymbol\theta_c^{T}\boldsymbol f_c(\boldsymbol x_c)
-\ln Z(\boldsymbol\theta).
$$

这也称最大熵模型或对数线性模型。特征函数描述“什么样的局部配置值得提高或降低权重”，参数控制方向与强度，配分函数随参数变化并重新归一化。CRF 会把同一思想用于条件分布 $p(\boldsymbol y\mid\boldsymbol x)$。

## 11.2 隐马尔可夫模型

### 两个假设、三组参数与生成过程

隐马尔可夫模型（HMM）是处理序列数据的有向生成式模型。时刻 $t$ 的隐状态为 $y_t\in\{s_1,\ldots,s_S\}$，观测为 $x_t$。标准一阶 HMM 有两个核心假设：

1. **一阶马尔可夫假设**：给定当前状态，下一状态与更早状态条件独立：
   $$p(y_{t+1}\mid y_{1:t})=p(y_{t+1}\mid y_t).$$
2. **观测独立假设**：给定当前隐状态，当前观测不依赖其他时刻的状态和观测：
   $$p(x_t\mid\boldsymbol x_{1:T},\boldsymbol y_{1:T})=p(x_t\mid y_t).$$

模型的三组参数为：

| 参数 | 定义 | 约束 |
|---|---|---|
| 初始分布 $\pi_i$ | $p(y_1=s_i)$ | $\sum_i\pi_i=1$ |
| 转移概率 $a_{ij}$ | $p(y_{t+1}=s_j\mid y_t=s_i)$ | 每行和为 1 |
| 观测概率 $b_j(o_k)$ | $p(x_t=o_k\mid y_t=s_j)$ | 每个状态的观测分布和为 1 |

统一记为 $\lambda=(A,B,\pi)$。联合分布为：

$$
p(\boldsymbol x_{1:T},\boldsymbol y_{1:T}\mid\lambda)
=\pi_{y_1}b_{y_1}(x_1)
\prod_{t=2}^{T}a_{y_{t-1},y_t}b_{y_t}(x_t).
$$

生成时先由 $\pi$ 选择 $y_1$，由当前状态生成 $x_1$；随后根据 $A$ 转移到下一状态，再由 $B$ 生成下一观测。语音识别、行为识别、基因区域识别、中文分词和词性标注，都可以在这一框架下定义状态和观测。

### 三个基本问题不要混用

| 问题 | 已知与目标 | 典型算法 | 核心操作 |
|---|---|---|---|
| 估算／评价 | 已知 $\lambda,\boldsymbol x$，求 $p(\boldsymbol x\mid\lambda)$ | 前向或后向算法 | 对所有隐路径求和 |
| 解码 | 已知 $\lambda,\boldsymbol x$，求最可能的整条 $\boldsymbol y$ | Viterbi | 对历史路径取最大并回溯 |
| 学习 | 已知观测，估计 $\lambda$ 使观测似然尽量大 | Baum-Welch | EM 的 E 步与 M 步交替 |

还有一个相关但不同的任务：求每个时刻的状态后验边际 $p(y_t=s_i\mid\boldsymbol x,\lambda)$，使用前向与后向量的乘积。逐时刻取最大边际得到的状态序列，不保证等于联合概率最大的 Viterbi 路径。

### 前向算法：把路径求和改成递推

观测概率需要对所有隐状态路径边缘化：

$$
p(\boldsymbol x_{1:T}\mid\lambda)
=\sum_{\boldsymbol y_{1:T}}
p(\boldsymbol x_{1:T},\boldsymbol y_{1:T}\mid\lambda).
$$

直接枚举有 $S^T$ 条路径，每条还需乘约 $T$ 个因子。前向变量定义为：

$$
\alpha_t(i)=p(x_1,\ldots,x_t,y_t=s_i\mid\lambda).
$$

初始化与递推为：

$$
\alpha_1(i)=\pi_i b_i(x_1),
$$

$$
\alpha_{t+1}(j)
=\left[\sum_{i=1}^{S}\alpha_t(i)a_{ij}\right]b_j(x_{t+1}).
$$

最后：

$$
p(\boldsymbol x_{1:T}\mid\lambda)=\sum_i\alpha_T(i).
$$

直觉是：不必记住到达状态 $j$ 的每条历史路径，只需把所有历史路径的联合概率汇总到 $\alpha_t(i)$，再乘以 $i\rightarrow j$ 的转移和当前观测。时间复杂度为 $O(S^2T)$，存储一个或全部前向切片的空间复杂度为 $O(S)$ 或 $O(ST)$。

### 手算：同一组 HMM 的前向全过程

使用两个状态 $R,S$，观测序列为 $(walk,shop,clean)$，参数为：

$$
\pi=(0.6,0.4),\qquad
A=\begin{pmatrix}
0.7&0.3\\
0.4&0.6
\end{pmatrix},
$$

$$
B=\begin{pmatrix}
0.1&0.4&0.5\\
0.6&0.3&0.1
\end{pmatrix},
$$

其中 $B$ 的列依次对应 $walk,shop,clean$，行依次对应 $R,S$。

初始时刻：

$$
\alpha_1(R)=0.6\times0.1=0.06,\qquad
\alpha_1(S)=0.4\times0.6=0.24.
$$

第二时刻：

$$
\alpha_2(R)=(0.06\times0.7+0.24\times0.4)\times0.4=0.0552,
$$

$$
\alpha_2(S)=(0.06\times0.3+0.24\times0.6)\times0.3=0.0486.
$$

第三时刻：

$$
\alpha_3(R)=(0.0552\times0.7+0.0486\times0.4)\times0.5=0.02904,
$$

$$
\alpha_3(S)=(0.0552\times0.3+0.0486\times0.6)\times0.1=0.004572.
$$

所以观测概率为：

$$
p(walk,shop,clean\mid\lambda)
=0.02904+0.004572=\boxed{0.033612}.
$$

### 后向算法与状态后验边际

标准后向变量是给定当前状态后，未来观测序列的条件概率：

$$
\beta_t(i)=p(x_{t+1},\ldots,x_T\mid y_t=s_i,\lambda).
$$

初始化为 $\beta_T(i)=1$，递推为：

$$
\beta_t(i)=\sum_j a_{ij}b_j(x_{t+1})\beta_{t+1}(j).
$$

观测概率也可写为：

$$
p(\boldsymbol x_{1:T}\mid\lambda)
=\sum_i\pi_i b_i(x_1)\beta_1(i).
$$

本例得到：

$$
\beta_2(R)=0.7\times0.5+0.3\times0.1=0.38,
$$

$$
\beta_2(S)=0.4\times0.5+0.6\times0.1=0.26,
$$

$$
\beta_1(R)=0.1298,\qquad \beta_1(S)=0.1076.
$$

代回初始分布同样得到 $0.033612$。前向量描述“截至当前的观测与当前状态”，后向量描述“当前状态之后的未来观测”，二者相乘再归一化，就得到状态后验边际：

$$
\gamma_t(i)=p(y_t=s_i\mid\boldsymbol x_{1:T},\lambda)
=\frac{\alpha_t(i)\beta_t(i)}
{\sum_j\alpha_t(j)\beta_t(j)}.
$$

本例三个时刻的 $R$ 后验约为 $0.231703,0.624063,0.863977$。它们是逐时刻边际概率，不是某条完整路径的联合概率。

> **记号澄清**：教材第 219 页的后向量文字公式把未来观测与当前状态写成了联合量。为了与随后递推及 $\alpha_t(i)\beta_t(i)$ 的使用一致，应采用上述“给定 $y_t=s_i$ 后未来观测”的条件概率定义。

### Viterbi：把求和换成最大，并保存回溯指针

解码目标是：

$$
\boldsymbol y^*
=\arg\max_{\boldsymbol y}p(\boldsymbol y\mid\boldsymbol x,\lambda)
=\arg\max_{\boldsymbol y}p(\boldsymbol x,\boldsymbol y\mid\lambda).
$$

因为分母 $p(\boldsymbol x\mid\lambda)$ 不随候选路径改变，可直接最大化联合概率。定义：

$$
\delta_t(j)=
\max_{y_1,\ldots,y_{t-1}}
p(x_1,\ldots,x_t,y_1,\ldots,y_{t-1},y_t=s_j\mid\lambda).
$$

递推为：

$$
\delta_{t+1}(j)
=\left[\max_i\delta_t(i)a_{ij}\right]b_j(x_{t+1}),
$$

$$
\psi_{t+1}(j)=\arg\max_i\delta_t(i)a_{ij}.
$$

$\psi$ 保存最优前驱。最后选择 $\arg\max_i\delta_T(i)$，再沿指针反向回溯。若只保存最大值而不保存前驱，只能得到最优分数，不能恢复完整路径。

本例的 $\delta$ 为：

| 时刻 | $\delta_t(R)$ | $\delta_t(S)$ |
|---|---:|---:|
| $t=1$ | 0.06 | 0.24 |
| $t=2$ | 0.0384 | 0.0432 |
| $t=3$ | **0.01344** | 0.002592 |

最优末状态为 $R$；回溯得到 $t=2$ 为 $R$、$t=1$ 为 $S$，因此：

$$
\boldsymbol y^*=(S,R,R),\qquad
p(\boldsymbol x,\boldsymbol y^*\mid\lambda)=0.01344.
$$

该最优路径的后验为 $0.01344/0.033612\approx0.399857$。这并不表示其他路径概率为零；所有路径的后验仍总和为 1。

Viterbi 的时间复杂度为 $O(S^2T)$，保存全部切片与回溯指针的空间复杂度为 $O(ST)$。前向算法计算“所有路径的总证据”，Viterbi 计算“最佳单条路径的分数”，不能互相替代。

### Baum-Welch：HMM 上的 EM

若训练时隐状态也已标注，可以直接统计初始次数、转移次数和观测次数。真正困难的是只有观测序列、没有状态序列，此时使用 Baum-Welch，即 HMM 上的 EM。

先定义相邻状态的后验：

$$
\xi_t(i,j)
=p(y_t=s_i,y_{t+1}=s_j\mid\boldsymbol x_{1:T},\lambda),
$$

$$
\xi_t(i,j)
=\frac{\alpha_t(i)a_{ij}b_j(x_{t+1})\beta_{t+1}(j)}
{p(\boldsymbol x_{1:T}\mid\lambda)}.
$$

分母也可用同一时刻所有 $i,j$ 的未归一化权重之和计算，以减少不同写法带来的混淆。对 $t<T$：

$$
\gamma_t(i)=\sum_j\xi_t(i,j).
$$

最后一个时刻的 $\gamma_T(i)$ 用 $\alpha_T(i)\beta_T(i)$ 归一化得到。于是：

- $\sum_{t=1}^{T-1}\gamma_t(i)$ 是从状态 $s_i$ 出发的软转移总数；
- $\sum_{t=1}^{T-1}\xi_t(i,j)$ 是 $s_i\rightarrow s_j$ 的软转移计数；
- $\gamma_t(j)$ 是时刻 $t$ 属于状态 $s_j$ 的软计数。

M 步更新为：

$$
\pi_i^{\mathrm{new}}=\gamma_1(i),
$$

$$
a_{ij}^{\mathrm{new}}
=\frac{\sum_{t=1}^{T-1}\xi_t(i,j)}
{\sum_{t=1}^{T-1}\gamma_t(i)},
$$

$$
b_j(o_k)^{\mathrm{new}}
=\frac{\sum_{t=1}^{T}\gamma_t(j)\mathbf 1(x_t=o_k)}
{\sum_{t=1}^{T}\gamma_t(j)}.
$$

这些公式与第十章 GMM 的“责任度加权计数”完全同构，只是隐变量从单样本成分归属变成整条状态序列，E 步借助前后向算法高效求得后验。多条训练序列时，分子和分母应跨序列累加，初始分布则对各序列的 $\gamma_1$ 汇总。

标准的精确 E 步与 M 步使训练观测似然单调不减，但只保证收敛到局部解或驻点，不保证全局最优。可使用完整标注数据初始化、简化混合模型初始化或多次随机初始化，并比较独立验证集表现。

### 实践补充：长序列计算与评价边界

1. **防止下溢**：长序列会连续相乘很多小概率，直接计算可能变成 0。常用前向／后向缩放或对数域计算；缩放因子必须保留，才能正确恢复对数似然和后验。
2. **处理零计数**：训练数据未出现某状态、转移或观测时，最大似然会给出零概率，可用适当平滑，但平滑方案也应在训练折内确定。
3. **检查单调性与概率约束**：调试时核对每轮对数似然、转移矩阵行和、观测分布行和及后验边际和。
4. **按时间验证**：金融、行为和语音序列不能任意打乱造成未来信息泄漏，应按时间划分训练、验证与测试，并保持同一实体或说话人的隔离要求。
5. **隐状态不是自动真实机制**：状态编号可交换，拟合良好也不证明状态具有因果意义或可直接转化为交易、诊断决策。

### HMM 的两类拓展

**自回归 HMM** 在 $x_{t-1}\rightarrow x_t$ 之间增加依赖，使当前观测不仅由隐状态决定，也可由最近观测决定。例如：

$$
p(x_t\mid x_{t-1},y_t=s_j,\theta)
=\mathcal N(x_t;w_jx_{t-1}+\mu_j,\sigma_j^2).
$$

它结合隐状态链的长程依赖与观测链的短程依赖，E 步仍可借助前后向思想，M 步则需估计状态相关回归参数。

**可分解 HMM（Factorial HMM）** 用多条隐状态链共同生成观测。例如 10 个二值隐变量可表示 $2^{10}=1024$ 种联合状态，但不必先手工列成一个拥有 1024 个取值的单变量。多条链可描述发音方式、词或不同电器等不同因素；表示更紧凑，不等于推断一定更简单，联合状态耦合仍需专门算法。

## 11.3 条件随机场

### 从结构化预测到线性链 CRF

结构化预测的输出不是单个类别，而是具有内部依赖的标签序列或其他结构。例如中文分词的每个字有位置标签，词性标注的每个词有词性，图像分割的相邻像素标签也相互关联。

线性链条件随机场（CRF）直接建模整条标签序列在输入条件下的概率。教材从 HMM 的条件分布出发，把转移和观测项推广为一般特征函数：

$$
p(\boldsymbol y\mid\boldsymbol x,\boldsymbol\theta)
=\frac{1}{Z(\boldsymbol x)}
\prod_{t=1}^{T}
\exp\left\{\sum_{k=1}^{K}
\theta_kf_k(y_{t-1},y_t,\boldsymbol x,t)\right\}.
$$

也可把整条路径的得分写为：

$$
s(\boldsymbol y,\boldsymbol x;\boldsymbol\theta)
=\sum_t\sum_k\theta_kf_k(y_{t-1},y_t,\boldsymbol x,t),
$$

$$
p(\boldsymbol y\mid\boldsymbol x)
=\frac{\exp s(\boldsymbol y,\boldsymbol x;\boldsymbol\theta)}
{Z(\boldsymbol x)},
\qquad
Z(\boldsymbol x)=\sum_{\boldsymbol y'}
\exp s(\boldsymbol y',\boldsymbol x;\boldsymbol\theta).
$$

这里的归一化是在固定输入 $\boldsymbol x$ 后，对所有可能的标签序列 $\boldsymbol y'$ 求和，不是对所有输入求和。因此 $Z$ 必须写成 $Z(\boldsymbol x)$。特征函数还可读取整个输入序列，而不只局限于当前的 $x_t$，这使 CRF 能灵活使用前后词、窗口特征和其他模型输出。

HMM 与 CRF 都能做序列推断，但训练目标不同：

| 对比项 | HMM | 线性链 CRF |
|---|---|---|
| 建模对象 | 联合分布 $p(\boldsymbol x,\boldsymbol y)$ | 条件分布 $p(\boldsymbol y\mid\boldsymbol x)$ |
| 模型类型 | 生成式、有向图 | 判别式、条件无向图 |
| 局部项 | 初始、转移、观测概率 | 任意实值特征的指数加权势 |
| 特征使用 | 观测生成分布受模型假设约束 | 可使用丰富、重叠、全局输入特征 |
| 无状态标注时学习 | 可用 Baum-Welch | 通常使用有完整标签序列的监督条件似然 |
| 归一化 | 各条件概率局部归一化 | 对每个输入的所有标签路径整体归一化 |

### 手算：四条标签路径的得分与概率

设序列长度为 2，标签取 $\{0,1\}$。定义单点分数：

$$
U_1=(0.4,0.1),\qquad U_2=(0.2,0.8),
$$

以及从前一标签到当前标签的转移分数：

$$
T=\begin{pmatrix}
0.6&-0.2\\
-0.1&0.5
\end{pmatrix}.
$$

路径得分为两个单点分数加一次转移分数：

| 路径 | 计算 | 得分 |
|---|---|---:|
| $(0,0)$ | $0.4+0.2+0.6$ | 1.2 |
| $(0,1)$ | $0.4+0.8-0.2$ | 1.0 |
| $(1,0)$ | $0.1+0.2-0.1$ | 0.2 |
| $(1,1)$ | $0.1+0.8+0.5$ | 1.4 |

因此：

$$
Z(\boldsymbol x)=e^{1.2}+e^{1.0}+e^{0.2}+e^{1.4}
\approx11.3150014762.
$$

四条路径概率约为：

$$
0.293426,\quad0.240237,\quad0.107945,\quad0.358391.
$$

最高得分的 $(1,1)$ 就是最佳解码路径，其条件概率约为 $0.358391$。单看第一个位置，标签 0 的单点分数更高；加入第二个位置和转移分数后，最优整条路径仍可从 1 开始，说明序列解码不能逐位置孤立决策。

### 特征工程：特征函数怎样进入模型

教材重点介绍 label-observation 特征。离散标签下，一类常见形式为：

$$
f_{c k}(\boldsymbol y_c,\boldsymbol x_c)
=\mathbf 1(\boldsymbol y_c=\tilde{\boldsymbol y}_c)q_{c k}(\boldsymbol x_c),
$$

即只有标签满足指定条件时，观测函数 $q$ 才激活。例如“当前词以 ing 结尾且标签为动词”或“当前词首字母大写且标签为实体”。

在一阶链中可区分：

- **节点—观测特征**：依赖当前标签与输入，如 $q_m(\boldsymbol x)\mathbf 1(y_t=y)$。
- **边—观测特征**：依赖相邻标签，也可同时依赖输入，如 $q_m(\boldsymbol x)\mathbf 1(y_{t-1}=y',y_t=y)$。
- **纯转移特征**：只描述标签对，不依赖当前观测。

实践中的特征来源包括：

1. **类别特征变换**：无序类别可二值化；实值特征可按训练集统计做 Z-score 标准化，或分箱后离散化。
2. **高阶与上下文特征**：利用前后位置的词、字形、时间窗口或更远标签，但结构越复杂，学习和推断成本越高。
3. **其他模型的输出**：规则系统、HMM 边际概率或其他预测器的输出可作为输入特征。
4. **领域知识**：词典、实体规则、句法信息、传感器特性等，都可转化为可检验的特征。

特征更多并不必然更好。参数数量、内存、过拟合和推断成本都会增加。若把另一个模型的预测作为特征，应使用训练折内拟合和折外预测等正确堆叠流程，避免目标泄漏；标准化与分箱也只能使用训练数据估计。

### 参数估计：经验特征期望减模型特征期望

给定相互独立的标注序列 $D=\{(\boldsymbol x^{(n)},\boldsymbol y^{(n)})\}_{n=1}^{N}$，条件对数似然为：

$$
L(\boldsymbol\theta)
=\sum_{n=1}^{N}
\left[s(\boldsymbol y^{(n)},\boldsymbol x^{(n)};\boldsymbol\theta)
-\ln Z(\boldsymbol x^{(n)};\boldsymbol\theta)\right].
$$

令 $F_k(\boldsymbol y,\boldsymbol x)=\sum_tf_k(y_{t-1},y_t,\boldsymbol x,t)$ 表示整条序列上的特征总计，则：

$$
\frac{\partial L}{\partial\theta_k}
=\sum_nF_k(\boldsymbol y^{(n)},\boldsymbol x^{(n)})
-\sum_n\mathbb E_{p(\boldsymbol y\mid\boldsymbol x^{(n)},\boldsymbol\theta)}
[F_k(\boldsymbol y,\boldsymbol x^{(n)})].
$$

也就是：

> **训练标签序列上的经验特征计数 − 当前模型预测分布下的期望特征计数。**

梯度为零时二者匹配。线性链可用前后向算法计算特征期望，再用梯度上升或通用数值优化器迭代参数。与 HMM 的无监督 Baum-Welch 不同，这里的标准条件似然训练使用完整标注序列，不需要对真实标签序列做 EM 猜测。

为缓解过拟合，可加入 L2 正则：

$$
L_R(\boldsymbol\theta)=L(\boldsymbol\theta)
-\frac{1}{2\sigma^2}\sum_k\theta_k^2,
$$

$$
\frac{\partial L_R}{\partial\theta_k}
=\frac{\partial L}{\partial\theta_k}
-\frac{\theta_k}{\sigma^2}.
$$

正则强度应通过验证集选择；测试集不能参与特征筛选、调参或提前停止。

### 推断：同样是求和与取最大

定义局部矩阵：

$$
M_t(y_{t-1},y_t\mid\boldsymbol x)
=\exp\left\{\sum_k\theta_kf_k(y_{t-1},y_t,\boldsymbol x,t)\right\}.
$$

加入开始和结束状态后，一条路径的未归一化权重是相应局部矩阵元素的乘积。于是：

- 计算 $Z(\boldsymbol x)$ 和标签边际时，对所有路径做矩阵乘法式求和，即前向／后向动态规划；
- 求最佳标签路径时，把求和换成最大并保存前驱，即 Viterbi；
- 计算训练梯度时，需要模型期望，通常离不开配分函数和边际；
- 只做最佳解码时，$Z(\boldsymbol x)$ 对所有候选路径相同，可以不求它，直接比较路径得分。

这解释了为什么 HMM 与 CRF 虽训练目标不同，却能共享链式动态规划思想：**结构决定推断复用何种递推，概率目标决定局部项和归一化怎样定义。**

## 11.4 小结与实践：结构正确之后，还要验证任务与数据

教材章末给出三个实践方向，可分别对应以下检查重点：

| 案例 | 状态与观测的可能定义 | 关键评价与风险 |
|---|---|---|
| 股票市场分析 | 隐状态为市场状态，观测为收益、波动等特征 | 按时间滚动验证；状态不等于收益保证，警惕未来信息与过拟合 |
| 中文分词 | 隐状态或标签为字在词中的位置，观测为字及上下文 | 比较 HMM 与 CRF 的词级指标、未登录词和跨领域泛化 |
| 语音识别 | 状态为音素或词等，观测为声学特征 | 区分声学模型、语言模型与解码；按说话人隔离并评价词错误率等指标 |

实际建模可按以下顺序推进：

1. 先定义观测、标签、状态空间和单位，不急于选择库。
2. 画出依赖图，写明联合或条件概率公式。
3. 明确任务是似然评价、逐时刻边际、最佳路径还是参数学习。
4. 用短序列穷举结果核对动态规划，再扩展到真实长度。
5. 检查概率归一化、数值稳定性、零计数和实现索引。
6. 最后才比较模型复杂度、特征集合和初始化，并用独立测试评价。

## 可运行示例：核对 MRF、HMM、Baum-Welch 与 CRF

下面只使用 Python 标准库。短序列同时用动态规划和穷举计算，便于发现转移方向、观测下标、归一化和回溯错误。输出中的微小尾差来自二进制浮点表示。

```python
from itertools import product
from math import exp, isclose, log

# 1. 两个二值变量的无向图：相同取值的势为 2，不同取值的势为 1。
mrf_weights = {
    (x1, x2): 2.0 if x1 == x2 else 1.0
    for x1, x2 in product([0, 1], repeat=2)
}
mrf_Z = sum(mrf_weights.values())
mrf_probabilities = {xy: weight / mrf_Z for xy, weight in mrf_weights.items()}
same_probability = sum(
    probability for (x1, x2), probability in mrf_probabilities.items() if x1 == x2
)

# 2. 两状态 HMM：R 表示雨天，S 表示晴天。
states = ["R", "S"]
observation_symbols = ["walk", "shop", "clean"]
observation_sequence = ["walk", "shop", "clean"]
pi = {"R": 0.6, "S": 0.4}
A = {
    "R": {"R": 0.7, "S": 0.3},
    "S": {"R": 0.4, "S": 0.6},
}
B = {
    "R": {"walk": 0.1, "shop": 0.4, "clean": 0.5},
    "S": {"walk": 0.6, "shop": 0.3, "clean": 0.1},
}
T = len(observation_sequence)

# 前向算法。
alpha = [{state: pi[state] * B[state][observation_sequence[0]] for state in states}]
for t in range(1, T):
    alpha.append({
        state: B[state][observation_sequence[t]] * sum(
            alpha[t - 1][previous] * A[previous][state] for previous in states
        )
        for state in states
    })
evidence_forward = sum(alpha[-1].values())

# 后向算法，beta[t][i] = p(x[t+1:] | y[t]=i)。
beta = [None] * T
beta[-1] = {state: 1.0 for state in states}
for t in range(T - 2, -1, -1):
    beta[t] = {
        state: sum(
            A[state][following]
            * B[following][observation_sequence[t + 1]]
            * beta[t + 1][following]
            for following in states
        )
        for state in states
    }
evidence_backward = sum(
    pi[state] * B[state][observation_sequence[0]] * beta[0][state]
    for state in states
)

# 每个时刻的状态后验边际。
gamma = [
    {
        state: alpha[t][state] * beta[t][state] / evidence_forward
        for state in states
    }
    for t in range(T)
]

# Viterbi：保存每个结尾状态的最大路径概率与回溯指针。
delta = [{state: alpha[0][state] for state in states}]
backpointers = []
for t in range(1, T):
    current_delta = {}
    current_backpointers = {}
    for state in states:
        candidates = {
            previous: delta[t - 1][previous] * A[previous][state]
            for previous in states
        }
        best_previous = max(candidates, key=candidates.get)
        current_delta[state] = candidates[best_previous] * B[state][observation_sequence[t]]
        current_backpointers[state] = best_previous
    delta.append(current_delta)
    backpointers.append(current_backpointers)

last_state = max(delta[-1], key=delta[-1].get)
best_joint = delta[-1][last_state]
best_path = [last_state]
for t in range(T - 2, -1, -1):
    best_path.append(backpointers[t][best_path[-1]])
best_path.reverse()
best_path_posterior = best_joint / evidence_forward

# 穷举全部状态路径，独立核对前向结果与 Viterbi 最优值。
def path_joint(path):
    probability = pi[path[0]] * B[path[0]][observation_sequence[0]]
    for t in range(1, T):
        probability *= A[path[t - 1]][path[t]] * B[path[t]][observation_sequence[t]]
    return probability

enumerated_paths = {path: path_joint(path) for path in product(states, repeat=T)}
evidence_enumeration = sum(enumerated_paths.values())
best_enumerated_path = max(enumerated_paths, key=enumerated_paths.get)

# Baum-Welch 的一次 E 步与 M 步。
xi = []
for t in range(T - 1):
    weights = {
        (i, j): alpha[t][i]
        * A[i][j]
        * B[j][observation_sequence[t + 1]]
        * beta[t + 1][j]
        for i in states
        for j in states
    }
    normalizer = sum(weights.values())
    xi.append({edge: value / normalizer for edge, value in weights.items()})

new_pi = gamma[0]
new_A = {}
for i in states:
    denominator = sum(gamma[t][i] for t in range(T - 1))
    new_A[i] = {
        j: sum(xi[t][(i, j)] for t in range(T - 1)) / denominator
        for j in states
    }
new_B = {}
for i in states:
    denominator = sum(gamma[t][i] for t in range(T))
    new_B[i] = {
        symbol: sum(
            gamma[t][i]
            for t in range(T)
            if observation_sequence[t] == symbol
        ) / denominator
        for symbol in observation_symbols
    }

# 3. 长度为 2、两个标签的线性链 CRF：直接枚举四条标签路径。
crf_unary = [[0.4, 0.1], [0.2, 0.8]]
crf_transition = [[0.6, -0.2], [-0.1, 0.5]]

def crf_score(path):
    return sum(crf_unary[t][path[t]] for t in range(2)) + crf_transition[path[0]][path[1]]

crf_scores = {path: crf_score(path) for path in product([0, 1], repeat=2)}
crf_Z = sum(exp(score) for score in crf_scores.values())
crf_probabilities = {path: exp(score) / crf_Z for path, score in crf_scores.items()}
crf_best_path = max(crf_scores, key=crf_scores.get)
crf_log_partition = log(crf_Z)

print("MRF 配分函数、四个配置概率、相同取值概率:")
print(mrf_Z, mrf_probabilities, same_probability)
print("前向变量 alpha:")
print(alpha)
print("后向变量 beta:")
print(beta)
print("前向、后向、穷举得到的观测概率:")
print(evidence_forward, evidence_backward, evidence_enumeration)
print("状态后验边际 gamma:")
print(gamma)
print("Viterbi 最优路径、联合概率、路径后验:")
print(best_path, best_joint, best_path_posterior)
print("穷举核对的最优路径:")
print(best_enumerated_path)
print("一次 Baum-Welch 更新的 pi、A、B:")
print(new_pi, new_A, new_B)
print("CRF 四条路径的分数:")
print(crf_scores)
print("CRF 配分函数、路径概率、最优路径:")
print(crf_Z, crf_probabilities, crf_best_path)

assert isclose(mrf_Z, 6.0)
assert isclose(same_probability, 2 / 3)
assert isclose(evidence_forward, 0.033612)
assert isclose(evidence_forward, evidence_backward)
assert isclose(evidence_forward, evidence_enumeration)
assert best_path == ["S", "R", "R"]
assert best_path == list(best_enumerated_path)
assert isclose(best_joint, enumerated_paths[best_enumerated_path])
assert all(isclose(sum(row.values()), 1.0) for row in gamma)
assert isclose(sum(new_pi.values()), 1.0)
assert all(isclose(sum(new_A[i].values()), 1.0) for i in states)
assert all(isclose(sum(new_B[i].values()), 1.0) for i in states)
expected_crf_scores = {(0, 0): 1.2, (0, 1): 1.0, (1, 0): 0.2, (1, 1): 1.4}
assert all(isclose(crf_scores[path], score) for path, score in expected_crf_scores.items())
assert crf_best_path == (1, 1)
assert isclose(sum(crf_probabilities.values()), 1.0)
```

关键输出为：

```text
MRF 配分函数、四个配置概率、相同取值概率:
6.0 {(0, 0): 0.3333333333333333, (0, 1): 0.16666666666666666, (1, 0): 0.16666666666666666, (1, 1): 0.3333333333333333} 0.6666666666666666
前向变量 alpha:
[{'R': 0.06, 'S': 0.24}, {'R': 0.055200000000000006, 'S': 0.04859999999999999}, {'R': 0.02904, 'S': 0.004572}]
后向变量 beta:
[{'R': 0.1298, 'S': 0.10760000000000002}, {'R': 0.38, 'S': 0.26}, {'R': 1.0, 'S': 1.0}]
前向、后向、穷举得到的观测概率:
0.033612 0.033612 0.033611999999999996
Viterbi 最优路径、联合概率、路径后验:
['S', 'R', 'R'] 0.01344 0.39985719385933594
CRF 四条路径的分数:
{(0, 0): 1.2000000000000002, (0, 1): 1.0000000000000002, (1, 0): 0.20000000000000004, (1, 1): 1.4}
CRF 配分函数、路径概率、最优路径:
11.315001476200438 {(0, 0): 0.2934261148546875, (0, 1): 0.2402369839877247, (1, 0): 0.10794543515784986, (1, 1): 0.358391465999738} (1, 1)
```

完整代码还输出逐时刻后验和一次 Baum-Welch 更新，并断言所有后验、转移分布、观测分布和 CRF 路径概率均正确归一化。CRF 手算分数 $1.2,1.0,0.2,1.4$ 与代码中的 $1.2000000000000002$ 等尾差在数学上相同。

## 常见误区

1. **概率图模型就是画一张依赖示意图。** 不完整。图必须和局部分布或势函数结合，才能定义完整概率模型。
2. **有向箭头已经证明因果关系。** 不对。贝叶斯网络可表示概率分解；因果解释需要额外假设、设计或领域证据。
3. **没有直接边就无条件独立。** 不对。间接路径仍可传递依赖，条件化还可能打开对撞路径。
4. **给定中间节点总会阻断依赖。** 不对。链式和分叉如此，对撞结构反而可能在条件化后打开。
5. **极大团就是节点数最多的最大团。** 不对。极大强调不能被更大团包含，最大强调节点数最多。
6. **势函数就是概率。** 不对。势函数只需提供非负相对权重，必须经配分函数归一化。
7. **无向图也可直接按父节点写条件概率乘积。** 不对。无向图通常按团势函数分解，并需要全局或条件配分函数。
8. **HMM 的观测在各时刻无条件独立。** 不对。它们是在给定相应隐状态及模型结构后满足特定条件独立，边缘上通常仍相关。
9. **前向算法求最可能状态路径。** 不对。前向对所有路径求和得到观测概率；Viterbi 才求最佳完整路径。
10. **Viterbi 只保留每步最大观测概率即可。** 不对。递推必须结合上一步最优路径分数和转移概率，并保存回溯指针。
11. **逐时刻最大后验状态拼起来一定等于 Viterbi 路径。** 不对。边际最优与完整路径联合最优是不同目标。
12. **$\gamma_t(i)$ 与 $\xi_t(i,j)$ 是一回事。** 不对。前者是单时刻状态边际，后者是相邻状态对的联合后验。
13. **Baum-Welch 是独立于 EM 的无关算法。** 不对。它是 HMM 上的 EM，前后向算法用于高效完成 E 步。
14. **Baum-Welch 似然单调不减，所以一定得到全局最大似然。** 不对。它仍可能停在局部解或驻点，初始化很重要。
15. **多条序列做 Baum-Welch 时可以先各自更新再简单平均。** 标准做法应先跨序列累加软计数的分子和分母，再做比例更新。
16. **CRF 对所有输入和标签一起归一化。** 不对。$Z(\boldsymbol x)$ 固定输入，只对所有可能标签序列求和。
17. **CRF 的权重 $\theta_k$ 必须是非负概率。** 不对。权重可为实数，指数变换保证势为正。
18. **CRF 只能看当前观测 $x_t$。** 不对。特征函数可以读取整个输入序列及上下文，具体由模型结构定义。
19. **CRF 和 HMM 的动态规划相同，所以训练目标也相同。** 不对。HMM 通常优化联合生成似然，CRF 优化条件似然。
20. **只解码时也必须先算配分函数。** 不必。$Z(\boldsymbol x)$ 对同一输入的所有候选路径是共同常数，不影响 $\arg\max$。
21. **其他模型输出作为 CRF 特征不需要防泄漏。** 不对。目标编码、模型预测、标准化和分箱都必须在正确训练流程内生成。
22. **隐状态拟合得好，就能直接解释为真实市场或生理机制。** 不对。状态语义、稳定性和决策价值需要独立证据与任务评价。

## 一句话总结

概率图模型用图结构压缩依赖关系，用因子分解定义概率；HMM 通过前向求和、Viterbi 取最大、Baum-Welch 学习隐状态序列，CRF 则用特征加权和输入条件归一化直接完成结构化判别，**所有算法都应先分清“求和、边际、最佳路径与学习”四种目标。**

## 复习卡片

- **生成式**：建模 $p(\boldsymbol x,\boldsymbol y)$；**判别式**：建模 $p(\boldsymbol y\mid\boldsymbol x)$。
- **有向分解**：$p(\boldsymbol x)=\prod_sp(x_s\mid\boldsymbol x_{\operatorname{pa}(s)})$。
- **三种结构**：链式与分叉条件阻断；对撞条件化可能打开。
- **无向分解**：极大团势函数乘积除以 $Z$；势不是概率。
- **HMM 参数**：初始 $\pi$、转移 $A$、观测 $B$。
- **前向**：$\alpha_{t+1}(j)=[\sum_i\alpha_t(i)a_{ij}]b_j(x_{t+1})$。
- **后向**：$\beta_t(i)=\sum_ja_{ij}b_j(x_{t+1})\beta_{t+1}(j)$。
- **状态边际**：$\gamma_t(i)\propto\alpha_t(i)\beta_t(i)$。
- **Viterbi**：前向递推中的求和换成最大，并保存 $\psi$ 回溯。
- **Baum-Welch**：用 $\gamma$ 和 $\xi$ 形成软计数，再更新 $\pi,A,B$。
- **复杂度**：前向与 Viterbi 均为 $O(S^2T)$，穷举路径数为 $S^T$。
- **CRF**：$p(\boldsymbol y\mid\boldsymbol x)=\exp s(\boldsymbol y,\boldsymbol x)/Z(\boldsymbol x)$。
- **CRF 梯度**：经验特征计数减模型期望特征计数；L2 再减 $\theta_k/\sigma^2$。
- **解码边界**：最佳路径无需计算 $Z$；边际与训练梯度通常需要。
- **数值实践**：短序列穷举核对，长序列用缩放或对数域，按时间与实体正确划分数据。

## 自测问题

1. 生成式模型与判别式模型的建模对象有何不同？为什么 HMM 和 CRF 常被放在这一组对比中？
2. 写出教材五变量有向图的联合分布，并解释每个条件因子怎样由父节点决定。
3. 比较链式、分叉和对撞结构在给定中间节点前后的依赖关系。
4. 两个二值变量在相同与不同取值时势分别为 2 和 1，求 $Z$、四个配置概率以及取值相同的概率。
5. 根据正文 HMM 参数计算观测序列 $(walk,shop,clean)$ 的概率，并说明前向递推为什么优于穷举。
6. 前向、后向、状态边际和 Viterbi 分别回答什么问题？为什么逐时刻最大边际不一定构成最佳完整路径？
7. 根据正文 $\delta$ 表恢复 Viterbi 路径，并计算其联合概率与后验。
8. Baum-Welch 中 $\gamma_t(i)$ 和 $\xi_t(i,j)$ 如何形成参数更新的软计数？多条序列应怎样汇总？
9. 根据正文 CRF 单点与转移分数计算四条路径概率，并说明为什么最佳解码可不计算 $Z(\boldsymbol x)$。
10. CRF 条件似然的梯度为什么是经验特征期望减模型特征期望？使用其他模型输出作特征时应怎样防泄漏？

## 参考答案

1. 生成式建模观测与目标的联合分布，可通过概率规则推断和采样；判别式直接建模目标在观测下的条件分布。HMM 是生成式序列模型，CRF 是判别式条件序列模型，二者共享链式推断但训练目标不同。
2. $p(x_1)p(x_2\mid x_1)p(x_3\mid x_1)p(x_4\mid x_1)p(x_5\mid x_2,x_4)$。每个节点只以图中父节点为条件，利用局部条件独立压缩完整链式法则。
3. 链式 $A\to B\to C$ 与分叉 $A\leftarrow B\to C$ 中，给定 $B$ 会阻断 $A,C$ 的依赖；对撞 $A\to B\leftarrow C$ 中，$A,C$ 可在边缘上独立，但给定 $B$ 或其某种后代信息可能产生依赖。
4. $Z=6$，四配置概率依次为 $1/3,1/6,1/6,1/3$，相同取值总概率为 $2/3$。势先提供相对权重，再统一归一化。
5. 前向末值为 $\alpha_3(R)=0.02904$、$\alpha_3(S)=0.004572$，总概率为 $0.033612$。递推把到达同一状态的历史路径汇总，复杂度为 $O(S^2T)$，无需逐条处理 $S^T$ 条路径。
6. 前向计算观测证据，后向计算给定当前状态后的未来观测，二者结合得到逐时刻状态边际，Viterbi 求联合概率最大的完整路径。不同目标的最优量不同，边际拼接不保证路径联合最优。
7. 末状态为 $R$，回溯得到 $(S,R,R)$；联合概率为 $0.01344$，除以观测概率 $0.033612$ 得到后验约 $0.399857$。
8. $\gamma$ 给出状态软计数，$\xi$ 给出相邻转移软计数；转移更新用 $\sum\xi/\sum\gamma$，观测更新用指示观测后的 $\gamma$ 加权比例，初始分布取 $\gamma_1$。多序列先累加各分子和分母再相除。
9. 四路径得分为 $1.2,1.0,0.2,1.4$，$Z\approx11.3150014762$，概率约为 $0.293426,0.240237,0.107945,0.358391$。$Z$ 对所有候选路径相同，不影响最大值对应的标签序列。
10. 对数似然第一项给出训练标签上的特征计数，$-\ln Z$ 的导数给出当前模型分布下的特征期望。其他模型输出应在训练折内拟合并用折外预测生成训练特征，测试数据只用于最终评价，避免目标泄漏。

## 下一章预告

第十二章“文本分析”将讨论文本表示、主题模型与情感分析。概率图模型中的生成机制和条件推断会连接到文本主题建模；本章 CRF 的序列标注思路，也与中文分词、词性标注和命名实体识别直接相关。

## 参考资料

- 欧高炎、朱占星、董彬等：《数据科学导引》第十一章“概率图模型”，高等教育出版社，原书第 211—234 页。
