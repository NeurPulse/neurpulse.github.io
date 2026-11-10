---
title: "数据科学导引第十章：EM 算法"
description: "用可复算的小例子理解隐变量、Jensen 不等式、E 步与 M 步，掌握高斯混合模型的责任度和参数更新"
pubDate: 2026-11-10
updatedDate: 2026-11-10
category: '数据科学'
order: 11
tags: ['数据科学', '数据科学导引', 'EM算法', '隐变量', '高斯混合模型', 'GMM', '最大似然', '软聚类', '读书笔记', '零基础']
---

> [《数据科学导引》学习地图](/notes/data-science-guide-roadmap/) · 第 10 / 15 章
>
> 上一章：[特征选择](/notes/data-science-guide-feature-selection/)

本文对应《数据科学导引》第十章“EM 算法”，原书第 201—210 页，即所提供扫描版 PDF 第 210—219 页。笔记保留 Jensen 不等式、对数似然下界、E 步、M 步和高斯混合模型五条主线，补充下界取等号、责任度、加权参数更新和完整迭代的可复算示例。明确标出的“学习补充”与“实践补充”用于帮助理解，并非教材原文。

## 学习目标

学完本章，应能回答八个问题：

- 什么是隐变量，为什么“先对隐变量求和、再取对数”会让最大似然难以直接求解？
- Jensen 不等式怎样给出对数似然的下界，什么时候不等式取等号？
- 为什么 E 步选择 $Q_i(z_i)=p(z_i\mid x_i,\theta^{(t)})$？
- M 步最大化的是什么，为什么其中与旧参数有关的 $Q$ 必须保持固定？
- 怎样用下界和 KL 散度说明 EM 的观测对数似然不会下降？
- 高斯混合模型中的混合系数、均值、协方差和责任度分别表示什么？
- GMM 的 E 步与 M 步公式怎样从“软计数”得到，和 K-means 有何区别？
- 为什么 EM 可能停在局部解，GMM 还可能出现方差塌缩与标签交换？

## 先看全局：观测不完整时，在归属与参数之间交替

EM 是期望最大化（Expectation-Maximization）算法，用于含隐变量或缺失信息的概率模型参数估计。它面对的不是单纯“数据里有几列空值”，而是一类更一般的问题：

- 能观察到样本 $x_i$；
- 每个样本还有未观察到的隐状态 $z_i$；
- 模型参数 $\theta$ 同时决定隐状态分布和观测分布；
- 如果 $z_i$ 已知，参数估计往往很简单，但现实中并不知道它。

可以把三类对象分开：

| 对象 | 例子 | 在迭代中的角色 |
|---|---|---|
| 观测变量 $X$ | 身高、样本特征、观测序列 | 始终已知，是似然计算的基础 |
| 隐变量 $Z$ | 样本来自哪个成分、隐藏状态 | E 步根据当前参数估计其后验分布 |
| 参数 $\theta$ | 混合比例、均值、协方差、转移概率 | M 步根据隐变量后验重新估计 |

EM 的直觉是：**当前参数给出隐结构的概率判断，隐结构的概率判断再指导参数更新。** 但 E 步不是随便猜一个硬标签，M 步也不是直接最大化原始观测似然；二者通过对数似然下界连接起来。

## 10.1 EM 算法

### 为什么隐变量让对数似然难以处理

设独立观测为 $X=\{x_1,\ldots,x_n\}$，对应隐变量为 $Z=\{z_1,\ldots,z_n\}$。观测似然需要把所有可能的隐状态边缘化：

$$
p(x_i\mid\theta)=\sum_{z_i}p(x_i,z_i\mid\theta).
$$

因此观测对数似然为：

$$
\ell(\theta)
=\sum_i\ln p(x_i\mid\theta)
=\sum_i\ln\sum_{z_i}p(x_i,z_i\mid\theta).
$$

困难来自 $\ln\sum_z$：对数在求和之外，不能像完全数据对数似然那样直接拆成各项之和。不同隐状态对应的参数又共同出现在求和内部，直接求导、解驻点方程通常没有简单闭式解。

若 $z_i$ 已知，完全数据对数似然是：

$$
\ln p(X,Z\mid\theta)
=\sum_i\ln p(x_i,z_i\mid\theta),
$$

很多指数族模型此时可以直接求均值、方差或比例。EM 正是利用“完全数据问题容易”这一特点，但不把未知隐状态当成已经确定的事实。

### 用 Jensen 不等式构造下界

对每个样本引入一个隐变量分布 $Q_i(z_i)$，满足：

$$
Q_i(z_i)\ge0,
\qquad \sum_{z_i}Q_i(z_i)=1.
$$

在联合概率非零处，把观测似然改写为期望：

$$
\sum_{z_i}p(x_i,z_i\mid\theta)
=\sum_{z_i}Q_i(z_i)
\frac{p(x_i,z_i\mid\theta)}{Q_i(z_i)}.
$$

由于 $\ln$ 是凹函数，Jensen 不等式给出“期望的对数不小于对数的期望”：

$$
\ln\sum_{z_i}Q_i(z_i)
\frac{p(x_i,z_i\mid\theta)}{Q_i(z_i)}
\ge
\sum_{z_i}Q_i(z_i)
\ln\frac{p(x_i,z_i\mid\theta)}{Q_i(z_i)}.
$$

对所有样本求和，右侧就是对数似然的一个下界：

$$
LB(Q,\theta)
=\sum_i\sum_{z_i}Q_i(z_i)
\ln\frac{p(x_i,z_i\mid\theta)}{Q_i(z_i)}.
$$

它也可写成：

$$
LB(Q,\theta)
=\underbrace{\sum_i\sum_{z_i}Q_i(z_i)
\ln p(x_i,z_i\mid\theta)}_{\text{完全数据对数似然的期望}}
-\underbrace{\sum_i\sum_{z_i}Q_i(z_i)\ln Q_i(z_i)}_{\text{只与 }Q\text{ 有关的项}}.
$$

E 步固定当前参数并选择 $Q$，让下界在当前点贴紧真实似然；M 步固定这个 $Q$，调整 $\theta$ 使下界尽量增大。

### 手算：为什么取后验时下界刚好贴紧

Jensen 不等式取等号，需要被加权的比值在 $Q$ 的非零支撑上为常数：

$$
\frac{p(x_i,z_i\mid\theta)}{Q_i(z_i)}=c.
$$

对 $z_i$ 求和可得 $c=p(x_i\mid\theta)$，所以：

$$
Q_i(z_i)
=\frac{p(x_i,z_i\mid\theta)}{p(x_i\mid\theta)}
=p(z_i\mid x_i,\theta).
$$

这就是隐变量的后验分布。

用一个学习补充的小例子核对。固定当前参数，对某个观测 $x$，两个隐状态的联合概率为：

$$
p(x,z=A)=0.2,
\qquad p(x,z=B)=0.3.
$$

观测概率为：

$$
p(x)=0.2+0.3=0.5.
$$

所以后验为：

$$
Q(A)=0.2/0.5=0.4,
\qquad Q(B)=0.3/0.5=0.6.
$$

两个比值都等于：

$$
\frac{0.2}{0.4}=\frac{0.3}{0.6}=0.5.
$$

下界为：

$$
0.4\ln\frac{0.2}{0.4}
+0.6\ln\frac{0.3}{0.6}
=\ln0.5
=\ln p(x).
$$

因此下界不只是“小于真实值”，而是在当前参数处与真实对数似然相等。若换成其他 $Q$，两个比值通常不再相同，Jensen 不等式就会留下间隙。

### E 步与 M 步到底做什么

给定当前参数 $\theta^{(t)}$，标准 EM 的两步为：

**E 步（Expectation）**：计算每个样本的隐变量后验：

$$
Q_i^{(t)}(z_i)
=p(z_i\mid x_i,\theta^{(t)}).
$$

由此构造辅助函数：

$$
Q(\theta,\theta^{(t)})
=\sum_i\sum_{z_i}Q_i^{(t)}(z_i)
\ln p(x_i,z_i\mid\theta).
$$

这里前一个 $Q_i^{(t)}$ 是隐变量后验，后一个 $Q(\theta,\theta^{(t)})$ 是辅助函数，教材符号相同但角色不同。为避免混淆，本文把辅助函数称为“完全数据对数似然期望”。

**M 步（Maximization）**：固定后验，更新参数：

$$
\theta^{(t+1)}
=\arg\max_\theta Q(\theta,\theta^{(t)}).
$$

教材直接最大化 $LB$。由于：

$$
LB(Q^{(t)},\theta)
=Q(\theta,\theta^{(t)})
-\sum_i\sum_{z_i}Q_i^{(t)}(z_i)\ln Q_i^{(t)}(z_i),
$$

第二项在 M 步固定不变，所以最大化 $LB$ 与最大化辅助函数完全等价。实际推导更新公式时通常省略这个常数项。

然后重复 E 步和 M 步，直到参数、责任度或观测对数似然的变化足够小，或达到最大迭代次数。

### 为什么观测对数似然不会下降

对任意 $Q_i$，有学习补充的恒等式：

$$
\ell(\theta)
=LB(Q,\theta)
+\sum_i D_{KL}\left(
Q_i(z_i)\,\|\,p(z_i\mid x_i,\theta)
\right).
$$

KL 散度非负，所以 $LB$ 是下界；当 $Q_i$ 等于当前后验时，KL 为 0，下界贴紧。

由此得到一轮更新的证明链：

$$
\ell(\theta^{(t)})
=LB(Q^{(t)},\theta^{(t)})
\le LB(Q^{(t)},\theta^{(t+1)})
\le \ell(\theta^{(t+1)}).
$$

第一步等号来自 E 步选择当前后验；第二个不等号来自 M 步最大化下界；最后一个不等号来自 KL 非负。

这条链保证的是**训练观测似然单调不减**，不是：

- 一定达到全局最大似然；
- 每一步参数都更接近真实参数；
- 测试集表现一定提升；
- 隐状态一定对应真实世界类别；
- 迭代有限次后一定停止。

EM 可能收敛到局部极大值、驻点或鞍点，结果依赖初始化。实践上常从多组初始参数运行，按训练似然选择解，并另用验证数据或任务指标评价模型。

### 停止条件与适用边界

教材建议可用参数变化小于阈值作为停止条件。更完整的实践检查包括：

- 观测对数似然变化是否小于阈值；
- 参数或责任度是否基本稳定；
- 是否达到最大迭代次数；
- 是否出现数值下溢、协方差奇异或空成分；
- 不同初始化是否得到相近结果。

并非所有含隐变量的模型都能直接套用标准精确 EM。E 步需要能够计算或表示后验，M 步需要能够优化辅助函数；若二者不可处理，可能需要近似推断、变分 EM、Monte Carlo EM 或其他优化方法。教材以受限玻尔兹曼机为例，说明其归一化结构使直接 EM 不适用。

## 10.2 EM 的应用：高斯混合模型

### 从生成过程理解模型

高斯混合模型（Gaussian Mixture Model, GMM）假设数据由 $K$ 个高斯成分共同生成。生成一个样本时：

1. 按混合系数 $\pi_1,\ldots,\pi_K$ 选择一个成分；
2. 从该成分的高斯分布 $N(\mu_k,\Sigma_k)$ 生成观测 $x$。

隐变量 $z$ 是 One-Hot 向量，$z_k=1$ 表示选择了第 $k$ 个成分，其余分量为 0。混合系数满足：

$$
\pi_k\ge0,
\qquad \sum_{k=1}^K\pi_k=1.
$$

联合分布为：

$$
p(x,z_k=1\mid\theta)
=\pi_kN(x\mid\mu_k,\Sigma_k).
$$

把未知成分求和，观测分布为：

$$
p(x\mid\theta)
=\sum_{k=1}^K\pi_kN(x\mid\mu_k,\Sigma_k).
$$

独立样本的对数似然因此为：

$$
\ell(\theta)
=\sum_{n=1}^N
\ln\left(
\sum_{k=1}^K\pi_kN(x_n\mid\mu_k,\Sigma_k)
\right),
$$

其中：

$$
\theta=\{\pi_k,\mu_k,\Sigma_k\}_{k=1}^K.
$$

一维高斯密度写为：

$$
N(x\mid\mu,\sigma^2)
=\frac1{\sqrt{2\pi\sigma^2}}
\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right).
$$

教材用班级男女身高说明两个高斯成分混合的直觉。实际应用时，成分编号和形状来自模型拟合，不能在没有外部依据时直接把它们解释成某种真实身份。

### E 步：计算每个成分的责任度

GMM 中隐变量的后验称为责任度（responsibility）：

$$
\gamma_{nk}
=p(z_{nk}=1\mid x_n,\theta^{(t)})
=\frac{\pi_kN(x_n\mid\mu_k,\Sigma_k)}
{\sum_{j=1}^K\pi_jN(x_n\mid\mu_j,\Sigma_j)}.
$$

它满足：

$$
\gamma_{nk}\ge0,
\qquad \sum_{k=1}^K\gamma_{nk}=1.
$$

$\pi_k$ 是看到当前样本前的成分比例，责任度是结合当前样本 $x_n$ 后的条件概率。它既考虑“这个成分总体有多大”，也考虑“这个样本在该成分下有多可能出现”。

**手算：两个等方差成分的责任度。** 设：

$$
\pi_1=\pi_2=0.5,
\qquad \mu_1=0,
\qquad \mu_2=2,
\qquad \sigma_1^2=\sigma_2^2=1.
$$

当 $x=1$ 时，样本到两个均值距离相同，密度相等，所以：

$$
\gamma_1=\gamma_2=0.5.
$$

当 $x=0$ 时，第二个成分的密度相对于第一个成分少一个因子：

$$
\frac{N(0\mid2,1)}{N(0\mid0,1)}
=\exp(-2).
$$

因此：

$$
\gamma_1
=\frac1{1+e^{-2}}
\approx0.880797,
\qquad
\gamma_2
=\frac{e^{-2}}{1+e^{-2}}
\approx0.119203.
$$

责任度不是“最近中心硬分配”。即使 $x=0$ 明显更靠近成分 1，模型仍保留约 0.119 的成分 2 后验概率。

### M 步：把责任度当作软计数

定义第 $k$ 个成分的有效样本量：

$$
N_k=\sum_{n=1}^N\gamma_{nk}.
$$

M 步更新为：

$$
\pi_k^{(t+1)}=\frac{N_k}{N},
$$

$$
\mu_k^{(t+1)}
=\frac1{N_k}\sum_{n=1}^N\gamma_{nk}x_n,
$$

$$
\Sigma_k^{(t+1)}
=\frac1{N_k}\sum_{n=1}^N\gamma_{nk}
(x_n-\mu_k^{(t+1)})
(x_n-\mu_k^{(t+1)})^T.
$$

这些公式与普通最大似然估计形式相似，只是把“每个样本是否属于该成分”的 0/1 计数换成了责任度。注意协方差使用新均值，分母是 $N_k$，不是 $N_k-1$，因为这里求的是辅助目标下的最大似然解，而不是无偏样本协方差修正。

**手算：三个样本的一次 M 步。** 设学习补充数据为：

$$
x_1=0,
\qquad x_2=2,
\qquad x_3=4.
$$

给定两个成分的责任度：

| 样本 | $\gamma_{n1}$ | $\gamma_{n2}$ |
|---|---:|---:|
| $x_1=0$ | 0.8 | 0.2 |
| $x_2=2$ | 0.5 | 0.5 |
| $x_3=4$ | 0.2 | 0.8 |

两个成分的有效样本量为：

$$
N_1=N_2=0.8+0.5+0.2=1.5,
$$

所以新混合系数均为：

$$
\pi_1=\pi_2=1.5/3=0.5.
$$

新均值为：

$$
\mu_1
=\frac{0.8\times0+0.5\times2+0.2\times4}{1.5}
=1.2,
$$

$$
\mu_2
=\frac{0.2\times0+0.5\times2+0.8\times4}{1.5}
=2.8.
$$

以成分 1 为例，新方差为：

$$
\sigma_1^2
=\frac{0.8(0-1.2)^2+0.5(2-1.2)^2+0.2(4-1.2)^2}{1.5}
$$

$$
=\frac{1.152+0.32+1.568}{1.5}
\approx2.026667.
$$

成分 2 的方差相同。计算协方差时必须先得到新均值，不能继续拿旧均值代入。

### 一轮完整 GMM 迭代

把两步连起来就是：

1. 初始化 $\pi_k,\mu_k,\Sigma_k$，保证混合系数归一、协方差合法。
2. 用旧参数计算所有 $\gamma_{nk}$。
3. 用这些责任度计算 $N_k$ 并更新全部参数。
4. 用新参数重新计算观测对数似然和责任度。
5. 检查收敛与数值状态，必要时继续迭代。

E 步的所有责任度必须来自同一组旧参数；M 步则固定这些责任度统一更新参数。不能更新第一个均值后立刻用它重算部分责任度，再把混合结果当作标准 EM 的同一轮更新。

## GMM 与 K-means：软归属和硬归属

第六章的 K-means 交替执行“分配到最近中心”和“更新中心”，GMM 则交替执行“计算成分后验”和“更新加权分布参数”。二者形式相似，但模型输出不同：

| 对比项 | K-means | GMM |
|---|---|---|
| 样本归属 | 每个样本硬分到一个簇 | 每个样本有一组责任度 |
| 簇参数 | 中心 | 混合系数、均值、协方差 |
| 几何结构 | 基于平方欧氏距离 | 可表示不同方向、大小和相关性的椭圆结构 |
| 目标 | 最小化簇内平方距离 | 最大化观测似然或其下界 |
| 主要风险 | 局部最优、尺度与 $K$ 选择 | 局部最优、奇异协方差、似然无界与 $K$ 选择 |

把责任度取最大值可得到硬分配，但这会丢弃不确定性。GMM 的协方差也不是越多越好：完整协方差参数多，样本不足时更容易不稳定，可根据任务考虑对角、共享或其他约束。

## 10.3 小结与实践：似然上升不等于模型可信

EM 不只用于 GMM，也可用于隐马尔可夫模型和含隐变量的主题模型等。教材提到的 LDA 在这里指隐狄利克雷分配主题模型，不是第八章的线性判别分析。是否能使用标准 EM，仍取决于后验与辅助目标是否可处理。

### 初始化与局部解

不同初始化可能导向不同似然和不同成分结构。实践补充可以：

- 用多组随机初始化或 K-means 结果初始化；
- 比较最终训练对数似然，而不只比较一次迭代结果；
- 检查解在不同运行之间是否稳定；
- 保存随机种子、初始参数、收敛阈值和最大迭代次数。

“选训练似然最高的解”仍不保证泛化最好。成分数 $K$、协方差结构和正则强度需要通过合适的验证、信息准则或领域约束确定。

### 方差塌缩与似然无界

若某个高斯成分只解释极少数样本，并且均值靠近其中一点、方差不断趋近 0，一维密度峰值可不断增大，观测似然可能趋向无穷。这样的“似然上升”并不代表发现了合理簇。

需要检查：

- $N_k$ 是否过小，是否出现近乎空成分；
- 方差或协方差特征值是否接近 0；
- 协方差是否病态、不可逆；
- 是否应采用方差下限、正则化、共享协方差或重新初始化。

因此不能只用“对数似然一直增大”判断训练健康，还要同时检查参数合法性和结构稳定性。

### 标签交换与解释边界

把成分 1 与成分 2 的编号整体交换，同时交换其 $\pi_k,\mu_k,\Sigma_k$ 和责任度，观测似然不变。这称为标签交换：模型能识别成分，却不能仅凭似然决定哪个成分应叫“1”。

同样，后验最大的成分不自动等于真实类别。若要把 GMM 用于分类，需要外部标签、映射规则与独立评价；若用于聚类，应报告软归属、参数、稳定性和簇的实际含义。

### 数值计算与章末任务

直接相乘很多小概率容易下溢。实际实现常在对数域计算密度、使用 log-sum-exp 归一化，并处理零概率和协方差正则。正文代码为便于理解直接计算密度，仅适用于教学小例子。

教材章末任务包括：

- 楼宇能源效率数据：768 个样本、10 个字段，其中供暖负荷与冷却负荷是关注目标，要求用 GMM 对能耗不同的建筑聚类；
- 小麦籽粒数据：210 个样本、7 个输入特征和种类标签，每类 70 个样本，要求使用 GMM 做种类预测；
- 裙装款式调查：501 个样本、14 个名义特征，要求预处理后使用 GMM 判断是否推荐。

这些任务不能混用“无监督聚类”和“有监督分类”的评价方式。使用标签训练或映射成分时，必须把拟合、成分数选择和最终测试分开；名义特征也不能不加思考地直接当作等距连续高斯变量。

## 可运行示例：核对后验、责任度、M 步与似然单调性

下面只用 Python 标准库，复算正文的小例子，并对三个样本运行 6 轮一维 GMM。代码为突出公式直接计算普通密度，不替代成熟实现的对数域计算、协方差正则和多种初始化。

```python
import math

# 1. Jensen 下界取等号：Q 是当前隐变量后验
joint = {"A": 0.2, "B": 0.3}
evidence = sum(joint.values())
posterior = {state: value / evidence for state, value in joint.items()}
lower_bound = sum(
    posterior[state] * math.log(joint[state] / posterior[state])
    for state in joint
)
assert evidence == 0.5
assert posterior == {"A": 0.4, "B": 0.6}
assert all(
    math.isclose(joint[state] / posterior[state], evidence)
    for state in joint
)
assert math.isclose(lower_bound, math.log(evidence))
print("联合概率、观测概率、后验与贴紧的下界:")
print(joint, evidence, posterior, lower_bound)

# 2. GMM E 步：两个等混合、等方差成分的责任度
def normal_density(value, mean, variance):
    return (
        math.exp(-0.5 * (value - mean) ** 2 / variance)
        / math.sqrt(2 * math.pi * variance)
    )

def responsibilities(value, weights, means, variances):
    unnormalized = [
        weight * normal_density(value, mean, variance)
        for weight, mean, variance in zip(weights, means, variances)
    ]
    total = sum(unnormalized)
    return [value / total for value in unnormalized]

initial_weights = [0.5, 0.5]
initial_means = [0.0, 2.0]
initial_variances = [1.0, 1.0]
gamma_at_1 = responsibilities(
    1.0, initial_weights, initial_means, initial_variances
)
gamma_at_0 = responsibilities(
    0.0, initial_weights, initial_means, initial_variances
)
expected_at_0 = [1 / (1 + math.exp(-2)), math.exp(-2) / (1 + math.exp(-2))]
assert gamma_at_1 == [0.5, 0.5]
assert all(math.isclose(a, b) for a, b in zip(gamma_at_0, expected_at_0))
print("x=1 与 x=0 时的责任度:", gamma_at_1, gamma_at_0)

# 3. 给定责任度，复算一次 M 步
def m_step(samples, gammas):
    sample_count = len(samples)
    component_count = len(gammas[0])
    effective_counts = [
        sum(gammas[n][k] for n in range(sample_count))
        for k in range(component_count)
    ]
    weights = [count / sample_count for count in effective_counts]
    means = [
        sum(gammas[n][k] * samples[n] for n in range(sample_count))
        / effective_counts[k]
        for k in range(component_count)
    ]
    variances = [
        sum(
            gammas[n][k] * (samples[n] - means[k]) ** 2
            for n in range(sample_count)
        ) / effective_counts[k]
        for k in range(component_count)
    ]
    return weights, means, variances

samples = [0.0, 2.0, 4.0]
given_gammas = [[0.8, 0.2], [0.5, 0.5], [0.2, 0.8]]
updated_weights, updated_means, updated_variances = m_step(samples, given_gammas)
assert updated_weights == [0.5, 0.5]
assert math.isclose(updated_means[0], 1.2)
assert math.isclose(updated_means[1], 2.8)
assert all(math.isclose(value, 152 / 75) for value in updated_variances)
print("一次 M 步的混合系数、均值、方差:")
print(updated_weights, updated_means, updated_variances)

# 4. 完整 EM：同一组旧参数产生责任度，再统一更新
def observed_log_likelihood(samples, weights, means, variances):
    total = 0.0
    for sample in samples:
        density = sum(
            weight * normal_density(sample, mean, variance)
            for weight, mean, variance in zip(weights, means, variances)
        )
        total += math.log(density)
    return total

weights = [0.5, 0.5]
means = [0.0, 4.0]
variances = [1.0, 1.0]
likelihood_trace = [observed_log_likelihood(samples, weights, means, variances)]
for _ in range(6):
    gammas = [
        responsibilities(sample, weights, means, variances)
        for sample in samples
    ]
    weights, means, variances = m_step(samples, gammas)
    likelihood_trace.append(observed_log_likelihood(samples, weights, means, variances))

assert all(
    later >= earlier - 1e-10
    for earlier, later in zip(likelihood_trace, likelihood_trace[1:])
)
assert math.isclose(sum(weights), 1.0)
assert all(variance > 0 for variance in variances)
print("6 轮 EM 的对数似然轨迹:")
print([round(value, 6) for value in likelihood_trace])
print("最终混合系数、均值、方差:")
print(weights, means, variances)
```

手算部分的关键输出为：

```text
联合概率、观测概率、后验与贴紧的下界:
{'A': 0.2, 'B': 0.3} 0.5 {'A': 0.4, 'B': 0.6} -0.6931471805599453
x=1 与 x=0 时的责任度:
[0.5, 0.5] [0.8807970779778824, 0.11920292202211756]
一次 M 步的混合系数、均值、方差:
[0.5, 0.5] [1.2, 2.8000000000000003] [2.026666666666667, 2.026666666666667]
6 轮 EM 的对数似然轨迹:
[-6.142439, -5.461391, -5.461066, -5.461058, -5.461058, -5.461058, -5.461058]
最终混合系数、均值、方差:
[0.49999999999999994, 0.5000000000000001] [0.6744903948218798, 3.3255096051781203] [0.9096909532472108, 0.9096909532472109]
```

完整代码同时断言对数似然轨迹单调不减、混合系数和为 1、方差保持为正。均值 $2.8000000000000003$ 的浮点尾差不影响数学结果 $2.8$；真实报告应统一保留适当小数位。

## 常见误区

1. **EM 是一个只用于高斯混合模型的具体公式。** 不对。EM 是含隐变量模型的迭代估计框架，GMM 是一种应用。
2. **隐变量就是数据表中需要均值填补的缺失值。** 不完全。缺失值可作为隐变量的一种情形，但隐状态、成分归属和未观测结构也属于这一框架。
3. **E 步把每个样本硬分给最可能的成分。** 不对。标准 E 步保留完整后验分布，GMM 中是责任度向量。
4. **观测对数似然就是把所有隐状态的对数概率直接相加。** 不对。必须先对隐状态求和得到观测概率，再取对数。
5. **任何 $Q_i$ 都能让 Jensen 下界在当前点取等号。** 不对。需要 $Q_i(z_i)=p(z_i\mid x_i,\theta^{(t)})$，使相应比值恒定。
6. **M 步还要同时更新 $Q_i$。** 不对。M 步固定 E 步的后验，只优化参数；下一轮 E 步才重新计算后验。
7. **M 步省略 $-\sum Q\ln Q$ 会改变最优解。** 不会。该项相对当前参数是常数，不影响 $\arg\max$。
8. **观测似然单调不减，所以 EM 一定找到全局最优。** 不对。它可能停在局部解或其他驻点，初始化很重要。
9. **训练似然提高，所以测试性能一定提高。** 不对。过复杂成分、方差塌缩和数据泄漏都可能让训练目标与泛化背离。
10. **责任度最大的成分就是样本确定的真实类别。** 不对。责任度是模型后验，硬分配还会丢失不确定性，真实语义需要外部证据。
11. **GMM 的混合系数与责任度是一回事。** 不对。$\pi_k$ 是总体先验比例，$\gamma_{nk}$ 是结合第 $n$ 个样本后的条件概率。
12. **GMM 的协方差更新可以使用旧均值。** 不对。公式使用本轮刚更新的新均值。
13. **GMM 协方差分母应使用 $N_k-1$。** 标准 M 步最大似然更新使用 $N_k$；无偏修正不是这里的辅助目标最优公式。
14. **GMM 与 K-means 只是名称不同。** 不对。GMM 使用软归属并估计协方差和混合比例，K-means 使用最近中心硬分配。
15. **成分协方差越灵活，模型一定越好。** 不对。完整协方差参数更多，小样本下更容易不稳定或奇异。
16. **交换成分编号会改变模型。** 不会。同步交换参数和责任度后，观测分布与似然不变，这就是标签交换。
17. **只要对数似然持续上升，就不需检查方差。** 不对。单点成分和方差趋零可能让似然异常增大甚至无界。
18. **多组初始化得到相同编号和相似图，就证明成分具有真实因果含义。** 不对。稳定性只是证据之一，成分解释仍需领域知识与外部验证。

## 一句话总结

EM 的核心，是在当前参数下计算隐变量后验，再固定后验最大化完全数据对数似然的期望，**让观测似然单调不减，同时用多初始化、参数约束和独立评价防范局部解与病态成分。**

## 复习卡片

- **观测似然**：$p(x\mid\theta)=\sum_zp(x,z\mid\theta)$，困难来自 $\ln\sum_z$。
- **下界**：$LB(Q,\theta)=\sum Q\ln(p/Q)\le\ell(\theta)$。
- **E 步**：令 $Q_i(z_i)=p(z_i\mid x_i,\theta^{(t)})$，使下界在当前点贴紧。
- **M 步**：固定 $Q$，最大化完全数据对数似然期望；$-\sum Q\ln Q$ 是常数。
- **单调性**：旧似然 $=$ 旧下界 $\le$ 新下界 $\le$ 新似然。
- **GMM**：$p(x)=\sum_k\pi_kN(x\mid\mu_k,\Sigma_k)$。
- **责任度**：$\gamma_{nk}=\pi_kN(x_n\mid\mu_k,\Sigma_k)/\sum_j\pi_jN(x_n\mid\mu_j,\Sigma_j)$。
- **软计数**：$N_k=\sum_n\gamma_{nk}$，$\pi_k=N_k/N$。
- **均值更新**：责任度加权平均；**协方差更新**：围绕新均值的加权二阶矩除以 $N_k$。
- **GMM 与 K-means**：软后验与硬分配的区别，不只是迭代形式相似。
- **局部解**：多初始化并比较训练似然，但仍需验证泛化与稳定性。
- **病态风险**：空成分、方差趋零、协方差奇异、标签交换和数值下溢。
- **评价边界**：成分不是自动真实类别，训练似然不是测试性能。

## 自测问题

1. 为什么含隐变量模型的观测对数似然通常难以直接求解析最大值？
2. 根据正文联合概率 $0.2,0.3$ 计算后验，并验证 Jensen 下界为什么恰好等于 $\ln0.5$。
3. E 步为什么使用当前隐变量后验？M 步为什么可以省略 $-\sum Q\ln Q$？
4. 用下界与 KL 散度写出一轮 EM 更新观测似然不下降的证明链，并指出它不能保证什么。
5. 对正文的两个等方差成分，计算 $x=1$ 与 $x=0$ 时的责任度，并解释它与硬分类的区别。
6. 根据正文三个样本的责任度计算 $N_k$、混合系数、均值和方差，说明协方差为什么使用新均值与分母 $N_k$。
7. GMM 与 K-means 的交替步骤有什么相似和不同？何时不能只看最大责任度？
8. 为什么 GMM 会出现方差塌缩和似然无界？应检查哪些信号？
9. 什么是标签交换？为什么成分编号稳定也不能证明真实语义？
10. 若要把小麦籽粒数据用于 GMM 种类预测，应怎样区分聚类、成分到类别的映射和最终测试？

## 参考答案

1. 观测概率要先对隐状态求和，再取对数，形成 $\ln\sum_zp(x,z\mid\theta)$；各成分参数共同处于求和内部，通常不能像完全数据似然那样拆解求解。
2. 观测概率为 0.5，后验为 $0.4,0.6$；联合概率除以后验都为 0.5，因此下界为 $0.4\ln0.5+0.6\ln0.5=\ln0.5$。
3. 当前后验使 Jensen 不等式取等号，下界在当前参数处贴紧。$-\sum Q\ln Q$ 在 M 步只依赖固定的旧后验，与待优化参数无关，不影响最优解。
4. $\ell(\theta^{(t)})=LB(Q^{(t)},\theta^{(t)})\le LB(Q^{(t)},\theta^{(t+1)})\le\ell(\theta^{(t+1)})$。它不保证全局最优、真实参数恢复、测试性能提升或正确类别语义。
5. $x=1$ 时两成分密度相同，责任度均为 0.5；$x=0$ 时约为 $0.880797,0.119203$。软责任度保留归属不确定性，硬分类只保留最大值。
6. $N_1=N_2=1.5$，混合系数均为 0.5，均值为 1.2、2.8，方差均约为 2.026667。协方差定义使用更新后的中心，M 步最大似然解用软计数 $N_k$ 作分母。
7. 两者都交替更新归属和中心，但 K-means 是最近中心硬分配并最小化平方距离，GMM 是后验软归属并更新混合系数与协方差。需要不确定性、异常判断或概率解释时不能只看最大责任度。
8. 某成分可围绕极少样本不断缩小方差，使密度峰值和似然异常增大。应检查有效样本量、方差下限、协方差特征值、可逆性和不同初始化下的稳定性。
9. 交换成分编号及其全部参数不改变观测分布和似然。编号只是表示方式，即使多次运行能对应上，也需要外部标签和领域证据解释真实含义。
10. 若完全忽略标签，就是无监督聚类，不能直接把成分当种类；若用训练标签建立成分到种类的映射或训练各类条件模型，就属于使用标签的流程，必须在训练折拟合映射、选择 $K$ 和预处理，最后只在独立测试集评价。

## 下一章预告

第十一章“概率图模型”将进一步研究变量之间的条件依赖，重点包括隐马尔可夫模型和条件随机场。学习时应把图结构、联合概率、推断任务和学习目标连起来：图不仅是一张示意图，更是在声明哪些变量在已知其他变量后具有条件独立性。

继续阅读：[第十一章：概率图模型](/notes/data-science-guide-graphical-models/)。

## 参考资料

- 欧高炎、朱占星、董彬等：《数据科学导引》第十章“EM 算法”，高等教育出版社，原书第 201—210 页。
