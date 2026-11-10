---
title: "数据科学导引第四章：分类模型"
description: "用可手算的小例子理解逻辑回归、K 近邻、决策树、朴素贝叶斯与支持向量机，并掌握分类指标、模型选择和训练流程"
pubDate: 2026-11-09
updatedDate: 2026-11-09
category: '数据科学'
order: 5
tags: ['数据科学', '数据科学导引', '分类模型', '机器学习', '读书笔记', '零基础']
---

> [《数据科学导引》学习地图](/notes/data-science-guide-roadmap/) · 第 4 / 15 章
>
> 上一章：[回归模型](/notes/data-science-guide-regression/)

本文对应《数据科学导引》第四章“分类模型”，原书第 69—105 页，即所提供扫描版 PDF 第 78—114 页。笔记保留逻辑回归、K 近邻、决策树、朴素贝叶斯和支持向量机五条主线，并补充手算示例、分类指标与模型选择流程。明确标出的“学习补充”与“实践补充”用于帮助理解，并非教材原文。

## 学习目标

学完本章，应能回答七个问题：

- 分类与回归的输出有什么不同，为什么逻辑回归名字中有“回归”却用于分类？
- 逻辑回归怎样把线性分数变成概率，最大似然与梯度下降怎样连接？
- K 近邻中的距离、$k$ 值和特征量纲为什么会改变预测结果？
- 决策树怎样用不纯度下降选择分裂，为什么还需要剪枝？
- 朴素贝叶斯的“朴素”究竟假设了什么，零概率问题怎样处理？
- 支持向量机为什么最大化间隔，软间隔与核函数分别解决什么问题？
- 类别不平衡时，为什么不能只报告正确率？

## 先看全局：分类是在学习离散标签的决策规则

分类是典型的监督学习问题。给定输入向量 $\boldsymbol x$ 和有限类别集合中的标签 $y$，希望学习：

$$
\hat y=f(\boldsymbol x),\qquad y\in\{1,2,\ldots,C\}.
$$

信用违约、垃圾邮件、疾病诊断、图像字符识别都属于分类。训练集中的每个样本既有特征，也有已知标签；模型训练完成后，再对未见样本预测类别。

本章五种模型可以放在同一张比较表中：

| 模型 | 核心思想 | 典型输出 | 主要优势 | 需要特别注意 |
|---|---|---|---|---|
| 逻辑回归 | 对线性分数做 Sigmoid 或 Softmax 变换 | 类别概率 | 简洁、可解释、训练稳定 | 原始特征下决策边界主要是线性的 |
| K 近邻 | 用最近训练样本投票 | 类别或邻居类别比例 | 几乎不需要显式训练，能表达复杂边界 | 距离、量纲、维数和预测成本 |
| 决策树 | 逐步选择特征与分裂点，降低节点不纯度 | 叶节点类别或类别分布 | 规则直观、易可视化 | 容易过拟合，单棵树对数据扰动敏感 |
| 朴素贝叶斯 | 贝叶斯定理加类条件独立假设 | 后验概率 | 训练和预测快，适合高维离散特征 | 条件独立常不完全成立，需处理零概率 |
| 支持向量机 | 找到间隔最大的划分超平面 | 决策分数与类别 | 间隔正则化思想清晰，可配合核方法 | 参数、尺度、核选择和概率校准 |

“训练—预测”只是表面流程。真正需要比较的是：模型对函数形式或数据分布做了什么假设，怎样控制复杂度，以及在独立数据上是否仍然有效。

## 4.1 逻辑回归：把线性分数映射为类别概率

### 为什么不能直接套用线性回归

线性回归的输出可以在整个实数范围内变化，而二分类标签只有两个取值。逻辑回归保留线性组合：

$$
z=\boldsymbol w^T\boldsymbol x,
$$

再用逻辑斯谛函数（Sigmoid）把它映射到 $(0,1)$：

$$
\sigma(z)=\frac1{1+e^{-z}}.
$$

它有三个常用性质：

$$
\sigma(0)=0.5,\qquad
\sigma(-z)=1-\sigma(z),\qquad
\sigma'(z)=\sigma(z)(1-\sigma(z)).
$$

当 $z$ 很大时，输出接近 1；当 $z$ 很小时，输出接近 0。模型把：

$$
p(y=1\mid\boldsymbol x)=\sigma(\boldsymbol w^T\boldsymbol x)
$$

解释为样本属于正类的概率。若采用 $y\in\{-1,+1\}$，两种情况还能统一写成：

$$
p(y\mid\boldsymbol x)=\sigma(y\boldsymbol w^T\boldsymbol x).
$$

这里的 $\boldsymbol x$ 可以包含常数 1，从而把截距并入 $\boldsymbol w$。

### 手算：从 logit 到概率、几率与损失

对上面的概率公式取对数几率，可得：

$$
\log\frac{p}{1-p}=\boldsymbol w^T\boldsymbol x=z.
$$

因此，逻辑回归实际上是在用线性函数建模**对数几率**，而不是直接建模概率。

若某样本的 $z=\ln3$，则：

$$
p=\sigma(\ln3)=\frac34=0.75,
\qquad \frac{p}{1-p}=3.
$$

若真实标签为 $y=1$，该样本的负对数似然为：

$$
-\ln p=-\ln0.75\approx0.287682.
$$

若只按 0.5 阈值分类，它会被预测为正类。但“预测为正类”和“模型给出 75% 概率”包含的信息不同：后者还能用于排序、阈值调整和代价敏感决策。

### 最大似然：让真实标签出现的概率尽量大

给定 $n$ 个独立训练样本，似然函数为：

$$
L(\boldsymbol w)=\prod_{i=1}^{n}p(y_i\mid\boldsymbol x_i).
$$

最大化似然等价于最小化负对数似然（NLL）：

$$
\operatorname{NLL}(\boldsymbol w)
=-\sum_{i=1}^{n}\ln p(y_i\mid\boldsymbol x_i).
$$

取对数把连乘变成求和，也改善数值计算。采用 $y_i\in\{-1,+1\}$ 时，可写成：

$$
\operatorname{NLL}(\boldsymbol w)
=\sum_{i=1}^{n}\ln\left(1+e^{-y_i\boldsymbol w^T\boldsymbol x_i}\right).
$$

教材给出的梯度为：

$$
\nabla\operatorname{NLL}(\boldsymbol w)
=-\sum_{i=1}^{n}
\left(1-\sigma(y_i\boldsymbol w^T\boldsymbol x_i)\right)y_i\boldsymbol x_i,
$$

参数按梯度下降更新：

$$
\boldsymbol w\leftarrow\boldsymbol w-\eta\nabla\operatorname{NLL}(\boldsymbol w).
$$

若改用更常见的 $y\in\{0,1\}$ 记法，令 $p_i=\sigma(\boldsymbol w^T\boldsymbol x_i)$，则单个样本的交叉熵为：

$$
\ell_i=-y_i\ln p_i-(1-y_i)\ln(1-p_i),
$$

矩阵形式的梯度为 $X^T(\boldsymbol p-\boldsymbol y)$。两种标签写法本质一致，推导时不要混用。

### 一步梯度下降到底在做什么

考虑只有一个样本 $x=1,y=1$，没有截距，初始 $w=0$。此时 $p=0.5$，负对数似然为 $\ln2\approx0.693147$。梯度是：

$$
(p-y)x=(0.5-1)\times1=-0.5.
$$

取学习率 $\eta=0.1$，更新后 $w=0.05$。新概率约为 0.512497，新损失约为 0.668460。预测仍然不够自信，但已经朝正确方向移动。

### 多分类：用 Softmax 归一化各类分数

有 $C$ 个类别时，每类各有一个参数向量 $\boldsymbol w_c$，分数为 $z_c=\boldsymbol w_c^T\boldsymbol x$。Softmax 输出：

$$
p(y=j\mid\boldsymbol x)
=\frac{e^{z_j}}{\sum_{c=1}^{C}e^{z_c}}.
$$

所有输出非负且总和为 1。实际计算通常先减去 $\max_c z_c$ 再取指数，避免大数指数溢出；这不会改变归一化结果。多分类参数仍可用最大似然估计。

## 4.2 K 近邻：让最相似的样本参与投票

### 算法只需要三步

K 近邻（KNN）预测一个测试样本时：

1. 选定距离或相似度，以及邻居数 $k$。
2. 找到训练集中距离最近的 $k$ 个样本。
3. 分类时投票，回归时汇总邻居的目标值。

常用距离包括欧氏距离、曼哈顿距离、马氏距离、汉明距离、余弦相似度和杰卡德距离。距离不是模型的附属细节，而是在定义“什么样的样本算相似”。

### 手算：同一个样本为什么会因 $k$ 改变类别

使用学习补充的一维数据：

| 位置 $x$ | 1 | 2 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|
| 类别 | A | A | B | B | B |

测试样本位于 $x=0$，与五个训练样本的距离依次为 1、2、4、5、6。

- $k=2$：最近邻是两个 A，预测为 A。
- $k=5$：全部样本参与投票，B 有 3 票，预测为 B。

较小的 $k$ 更依赖局部样本，边界灵活，但容易受噪声影响；较大的 $k$ 更平滑，却可能抹去局部结构。当 $k$ 等于训练样本数时，所有测试样本都会得到同一个多数类预测。$k$ 应通过训练集内部的交叉验证选择，而不是凭感觉固定。

若使用距离加权投票，可给第 $i$ 个邻居权重 $u_i=1/(d_i+\varepsilon)$，再比较各类别的权重和。做 KNN 回归时，常见加权平均为：

$$
\hat y=\frac{\sum_{i=1}^{k}u_i y_i}{\sum_{i=1}^{k}u_i}.
$$

距离为 0 的样本需要单独处理，避免直接除以 0；并列距离和平票也需要事先约定处理规则。

### 为什么标准化和维数特别重要

若一个特征以元为单位，另一个以万元为单位，欧氏距离可能被数值范围大的特征主导。通常应先根据训练集拟合标准化参数，再变换训练集和测试集。

高维空间中，距离对比可能变得不明显，最近邻未必仍具有直观意义。无关特征越多，越容易掩盖真正的局部结构。

KNN 的训练阶段通常只是保存数据，预测却要反复寻找邻居。教材介绍了 k-d 树：通过按维度递归划分空间建立索引，减少低维数据上的搜索成本；维数很高时，其优势通常会减弱。

## 4.3 决策树：用一连串问题划分样本空间

### 节点、分裂与叶节点

决策树的非叶节点表示对某个特征的一次提问，分支表示不同答案，叶节点给出最终预测。分类树通常返回叶节点中的多数类，回归树通常返回叶节点目标值的平均值。

生成树的核心是：**在哪个特征上、用哪个分裂点，能让子节点比父节点更纯？** 离散特征可以按取值分支，连续特征可以用“年龄 $\le 50$”这样的阈值分裂。

### 三种节点不纯度

设节点 $t$ 中类别 $c$ 的相对频率为 $p(c\mid t)$，共有 $C$ 类。

**Gini 指数**：

$$
\operatorname{Gini}(t)=1-\sum_{c=1}^{C}p(c\mid t)^2.
$$

**信息熵**：

$$
\operatorname{Entropy}(t)
=-\sum_{c=1}^{C}p(c\mid t)\log_2p(c\mid t),
$$

其中约定 $0\log_20=0$。

**误分率**：

$$
\operatorname{Error}(t)=1-\max_c p(c\mid t).
$$

纯节点的不纯度都为 0。二分类中，若正类比例为 $p$，三者分别变为：

$$
\operatorname{Gini}(p)=2p(1-p),
$$

$$
\operatorname{Entropy}(p)=-p\log_2p-(1-p)\log_2(1-p),
$$

$$
\operatorname{Error}(p)=1-\max(p,1-p).
$$

类别越均衡，不纯度通常越大。三者的量纲和曲线不同，不能直接把“Gini 下降 0.2”和“熵下降 0.2”当作同一尺度比较。

### 手算：教材银行贷款例子的收入分裂

根节点有 5 个违约样本和 5 个未违约样本，因此：

$$
\operatorname{Gini}(t_0)=1-(0.5^2+0.5^2)=0.5.
$$

按“收入”分裂后，三个子节点分别为：

| 子节点 | 违约数 | 未违约数 | 样本数 | Gini |
|---|---:|---:|---:|---:|
| 低收入 | 2 | 0 | 2 | 0 |
| 中等收入 | 3 | 2 | 5 | $1-(3/5)^2-(2/5)^2=0.48$ |
| 高收入 | 0 | 3 | 3 | 0 |

分裂后的加权 Gini 为：

$$
\operatorname{Gini}_{\text{split}}
=\frac2{10}\times0+\frac5{10}\times0.48+\frac3{10}\times0
=0.24.
$$

所以 Gini 下降值为：

$$
\Delta\operatorname{Gini}=0.5-0.24=0.26.
$$

同一分裂的加权熵约为 0.485475，信息增益约为 0.514525；加权误分率为 0.2，误分率下降值为 0.3。选择候选分裂时，应在**同一种不纯度指标**下比较下降值，并确保使用样本数加权。

### 信息增益率：修正对多取值特征的偏爱

信息增益为：

$$
\operatorname{InfoGain}
=\operatorname{Entropy}(t_0)
-\sum_{k=1}^{K}\frac{n_k}{n}\operatorname{Entropy}(t_k).
$$

单纯追求信息增益，可能偏爱取值很多、把节点切得很碎的特征。C4.5 引入分裂信息：

$$
\operatorname{SplitInfo}
=-\sum_{k=1}^{K}\frac{n_k}{n}\log_2\frac{n_k}{n},
$$

再计算：

$$
\operatorname{InfoGainRatio}
=\frac{\operatorname{InfoGain}}{\operatorname{SplitInfo}}.
$$

这样能减少对“分支越多越有优势”的倾向，但仍需结合最小叶节点样本数、验证表现等约束。

### ID3、C4.5 与 CART

| 算法 | 分裂依据 | 特征与目标类型 | 每次分裂的子节点 |
|---|---|---|---|
| ID3 | 信息增益 | 离散特征，离散目标 | $K\ge2$ |
| C4.5 | 信息增益率 | 离散或连续特征，离散目标 | $K\ge2$ |
| CART | 分类用 Gini，回归用方差下降 | 离散或连续特征，离散或连续目标 | 二叉分裂，$K=2$ |

这些算法都采用贪心策略：每一步选择当前看来最好的分裂，不保证得到全局最优树。

### 剪枝：不要为训练集中的每一点噪声长出新分支

树太深、叶节点太多时，训练集表现可能很好，测试集却变差。控制复杂度的方法包括：

- **预剪枝**：在生成过程中限制深度、最小样本数或最小不纯度下降。
- **后剪枝**：先生成较完整的树，再删除不能改善整体代价的分支。

教材用整体损失描述拟合与复杂度的权衡：

$$
\operatorname{Cost}_{\alpha}(T)
=\sum_{t=1}^{|T|}n_t\operatorname{Imp}(t)+\alpha|T|,
$$

其中 $|T|$ 表示节点数量，$\alpha$ 越大，对复杂树的惩罚越强。该思想与第三章正则化相同：不是把训练误差压到最低就结束，而是在拟合和复杂度之间取得平衡。

单棵树易解释、易可视化，但决策边界通常由平行于坐标轴的分段构成，而且对训练数据的小扰动可能敏感。这为下一章的随机森林和集成学习留下了改进空间。

## 4.4 朴素贝叶斯：用条件独立简化联合概率

### 贝叶斯定理：把证据与先验结合起来

贝叶斯定理为：

$$
p(Y\mid X)=\frac{p(X\mid Y)p(Y)}{p(X)}.
$$

其中 $p(Y)$ 是先验，$p(X\mid Y)$ 是似然，$p(X)$ 是证据，$p(Y\mid X)$ 是观察到特征后的后验。

教材的球赛例子中，A、B 两队获胜的先验分别为 0.75 和 0.25；B 主场时，A、B 获胜的条件概率分别为 0.20 和 0.60。于是：

$$
p(Y=B\mid X=B\text{ 主场})
=\frac{0.60\times0.25}
{0.60\times0.25+0.20\times0.75}
=0.5.
$$

虽然 B 队总体胜率只有 25%，但“主场”这一证据把其后验胜率更新到了 50%。贝叶斯定理不是只重复先验，而是在新证据下重新分配概率。

### 从联合似然到逐特征连乘

分类时，对同一个测试样本 $\boldsymbol x$，证据 $p(\boldsymbol x)$ 与类别无关，因此：

$$
p(Y=c\mid\boldsymbol x)
\propto p(Y=c)p(\boldsymbol x\mid Y=c).
$$

朴素贝叶斯进一步假设：**给定类别后，各特征条件独立**。于是：

$$
p(\boldsymbol x\mid Y=c)
=\prod_{j=1}^{d}p(X_j=x_j\mid Y=c),
$$

预测规则为：

$$
\hat y=\arg\max_c p(Y=c)
\prod_{j=1}^{d}p(X_j=x_j\mid Y=c).
$$

“条件独立”不等于“特征无条件独立”。特征在总体上可以相关，只是模型假设类别已经解释了它们之间的依赖。这一假设很强，但在文本分类等高维任务中常能得到有用且高效的基线。

### 参数估计与一个小型分类例子

离散特征可以通过计数估计先验和条件概率。假设垃圾邮件与正常邮件各有 2 封，因此先验均为 0.5；某个简化的两类计数模型给出：

$$
p(\text{优惠}\mid\text{垃圾})=p(\text{链接}\mid\text{垃圾})=\frac23,
$$

$$
p(\text{优惠}\mid\text{正常})=p(\text{链接}\mid\text{正常})=\frac13.
$$

对同时出现“优惠”和“链接”的样本，未归一化后验分数为：

$$
s_{\text{垃圾}}=0.5\times\frac23\times\frac23=\frac29,
$$

$$
s_{\text{正常}}=0.5\times\frac13\times\frac13=\frac1{18}.
$$

归一化后：

$$
p(\text{垃圾}\mid\boldsymbol x)
=\frac{2/9}{2/9+1/18}=0.8.
$$

连乘结果本身还不是概率，只有除以所有类别的分数之和才完成归一化。

### 学习补充：零概率、对数与连续特征

若训练集中某类从未出现某个特征值，最大似然估计会让对应条件概率为 0，整个连乘也随之变成 0。常用拉普拉斯平滑：

$$
\hat p(X_j=v\mid Y=c)
=\frac{N_{c,j,v}+\alpha}{N_c+\alpha K_j},
$$

其中 $K_j$ 是该平滑模型下可能的取值数，$\alpha>0$。词频模型、伯努利模型的分母约定不同，应按具体模型实现，不能机械套公式。

很多小概率连乘还可能数值下溢，因此实际实现通常比较对数分数：

$$
\log p(Y=c)+\sum_j\log p(X_j=x_j\mid Y=c).
$$

连续特征可以先离散化，也可以假设类条件分布。例如高斯朴素贝叶斯使用：

$$
p(X_j=x\mid Y=c)
=\frac1{\sqrt{2\pi}\sigma_{cj}}
\exp\left(-\frac{(x-\mu_{cj})^2}{2\sigma_{cj}^2}\right).
$$

这里是**概率密度**，不是“连续变量恰好等于某个点”的概率。每一类、每一个特征都要估计相应参数。

## 4.5 支持向量机：不仅分开，还要尽量远离边界

### 函数间隔与几何间隔

二分类标签取 $y_i\in\{-1,+1\}$，划分超平面为：

$$
\boldsymbol w^T\boldsymbol x+b=0.
$$

样本的函数间隔为：

$$
\hat\gamma_i=y_i(\boldsymbol w^T\boldsymbol x_i+b).
$$

它大于 0 表示分类正确，绝对值越大通常表示离边界越远。但把 $\boldsymbol w,b$ 同时放大相同倍数，函数间隔会改变，超平面却不变。因此使用几何间隔：

$$
\gamma_i=\frac{y_i(\boldsymbol w^T\boldsymbol x_i+b)}{\|\boldsymbol w\|_2}.
$$

硬间隔 SVM 最大化所有样本中的最小几何间隔：

$$
\max_{\boldsymbol w,b}\min_i
\frac{y_i(\boldsymbol w^T\boldsymbol x_i+b)}{\|\boldsymbol w\|_2}.
$$

通过尺度归一化，令最小函数间隔为 1，就得到等价的凸二次规划：

$$
\min_{\boldsymbol w,b}\frac12\|\boldsymbol w\|_2^2,
$$

$$
\text{s.t.}\quad y_i(\boldsymbol w^T\boldsymbol x_i+b)\ge1,
\qquad i=1,\ldots,n.
$$

两侧边界 $\boldsymbol w^T\boldsymbol x+b=\pm1$ 之间的总宽度为 $2/\|\boldsymbol w\|_2$。最小化 $\|\boldsymbol w\|_2$ 就是在最大化间隔。

### 手算：一维四个点的最大间隔分类器

使用学习补充数据：

$$
(-2,-1),\quad(-1,-1),\quad(1,+1),\quad(2,+1),
$$

每对数分别表示 $(x,y)$。最自然的最大间隔边界是 $x=0$。最近样本是 $x=-1$ 和 $x=1$，令它们满足函数间隔为 1：

$$
w\times1+b=1,\qquad -(-w+b)=1.
$$

解得 $w=1,b=0$，决策函数为 $f(x)=x$。两个支持向量到边界的几何间隔都为 1，总间隔宽度为 2；$x=-2$ 和 $x=2$ 不决定最终的硬间隔边界。

若把参数改成 $w=2,b=0$，超平面仍是 $x=0$，但最小函数间隔变为 2。只有除以 $\|w\|$ 后，几何间隔才保持不变。

### 支持向量、对偶与 SMO

拉格朗日对偶推导给出：

$$
\boldsymbol w=\sum_i\alpha_i y_i\boldsymbol x_i,
\qquad \sum_i\alpha_i y_i=0.
$$

对偶目标为：

$$
\max_{\alpha}\sum_i\alpha_i
-\frac12\sum_i\sum_j
\alpha_i\alpha_j y_i y_j(\boldsymbol x_i^T\boldsymbol x_j),
$$

并满足 $\alpha_i\ge0$ 和 $\sum_i\alpha_i y_i=0$。硬间隔中，$\alpha_i>0$ 的样本是支持向量，它们位于间隔边界上并决定 $\boldsymbol w$；其他样本的系数为 0。

在上面的一维例子中，两个支持向量的系数均为 $\alpha=0.5$，于是 $\boldsymbol w=0.5\times1\times1+0.5\times(-1)\times(-1)=1$。

SMO 每次选取两个拉格朗日乘子，固定其余变量，求解小的二次规划并不断迭代。它利用对偶结构避免一次性处理全部变量；对偶形式也自然揭示了模型只依赖样本内积。

### 软间隔：允许少量违规，换取更稳健的边界

现实数据常有噪声或线性不可分。软间隔为每个样本引入松弛变量 $\xi_i\ge0$：

$$
\min_{\boldsymbol w,b,\boldsymbol\xi}
\frac12\|\boldsymbol w\|_2^2+C\sum_i\xi_i,
$$

$$
\text{s.t.}\quad
y_i(\boldsymbol w^T\boldsymbol x_i+b)\ge1-\xi_i,
\qquad \xi_i\ge0.
$$

$\xi_i$ 允许样本进入间隔甚至越过分类边界，但会付出目标函数代价。按本节目标函数写法：

- $C$ 较大：更重视减少训练集间隔违规，可能得到更复杂、更敏感的边界。
- $C$ 较小：允许更多违规，更强调大间隔和简洁性。

$C$ 不是越大越好，应通过验证选择。软间隔对偶中多了上界 $0\le\alpha_i\le C$。等价的学习补充写法是合页损失：

$$
\frac12\|\boldsymbol w\|_2^2
+C\sum_i\max\left(0,1-y_i(\boldsymbol w^T\boldsymbol x_i+b)\right).
$$

它清楚展示了 SVM 同时考虑“间隔正则化”和“间隔违规损失”。

### 核函数：只替换内积，不必显式构造高维特征

对偶目标和预测中都出现样本内积。核方法用：

$$
K(\boldsymbol x_i,\boldsymbol x_j)
=\phi(\boldsymbol x_i)^T\phi(\boldsymbol x_j)
$$

替代原空间内积，从而学习特征空间中的线性边界，对应原空间中的非线性边界。

| 核函数 | 常见形式 |
|---|---|
| 多项式核 | $K(\boldsymbol x,\boldsymbol z)=(\boldsymbol x^T\boldsymbol z+1)^d$ |
| 高斯 RBF 核 | $K(\boldsymbol x,\boldsymbol z)=\exp(-\|\boldsymbol x-\boldsymbol z\|_2^2/(2\delta^2))$ |
| 拉普拉斯核 | $K(\boldsymbol x,\boldsymbol z)=\exp(-\|\boldsymbol x-\boldsymbol z\|_2/\delta)$ |
| Sigmoid 核 | $K(\boldsymbol x,\boldsymbol z)=\tanh(\beta\boldsymbol x^T\boldsymbol z+\theta)$ |

核函数不保证任意数据都能被完美分开，也不消除过拟合。核类型、核参数与 $C$ 仍需联合验证，距离和内积相关的模型通常也需要先做特征缩放。标准 SVM 直接输出决策分数而非天然校准的类别概率，需要概率时应另行校准并验证校准效果。

## 实践补充：分类指标必须对应错误代价

### 混淆矩阵比单一正确率更有信息

二分类中：

|  | 预测正类 | 预测负类 |
|---|---:|---:|
| 实际正类 | TP：真正例 | FN：假反例 |
| 实际负类 | FP：假正例 | TN：真反例 |

常用指标为：

$$
\operatorname{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN},
$$

$$
\operatorname{Precision}=\frac{TP}{TP+FP},\qquad
\operatorname{Recall}=\frac{TP}{TP+FN},
$$

$$
F_1=\frac{2TP}{2TP+FP+FN}.
$$

精确率问“预测为正的样本中有多少是真的”，召回率问“真实正例中找回了多少”。若负类代价也重要，还可报告特异度 $TN/(TN+FP)$。

### 手算：85% 正确率仍可能没有超过多数类基线

设 $TP=20,FP=10,FN=20,TN=150$，总数为 200：

- 正确率：$(20+150)/200=0.85$。
- 精确率：$20/(20+10)=2/3\approx0.666667$。
- 召回率：$20/(20+20)=0.5$。
- $F_1=40/(40+10+20)=4/7\approx0.571429$。
- 特异度：$150/(150+10)=0.9375$。

但全部预测为负类时，也有 $(150+20)/200=0.85$ 的正确率。这个模型的总正确率与多数类基线相同，却找回了半数正例；是否更有价值，取决于任务目标与错误代价。只看“85%”无法作出判断。

阈值降低通常会改变 TP、FP、FN、TN，进而改变精确率和召回率。模型排序能力、阈值后的决策表现和概率是否校准，是三个不同问题，可分别用相应评估方法检查。

## 实践补充：一条可靠的分类训练流程

1. **定义标签与正类**：标签从何而来，正类代表什么，FP 和 FN 哪个代价更高。
2. **先划分数据**：最终测试集不参与特征处理、模型选择、阈值选择或概率校准。
3. **检查类别与分组结构**：报告类别比例；同一用户或同一对象的多条记录要避免跨集合泄漏。
4. **按模型预处理**：KNN、SVM 和逻辑回归通常重视缩放；决策树通常不依赖单调缩放；朴素贝叶斯要选择合适的特征分布模型。
5. **保留简单基线**：多数类、逻辑回归或一棵浅树，能帮助判断复杂模型是否真的有增益。
6. **在训练集内部验证**：比较模型、$k$、树复杂度、$C$、核参数和阈值，不使用测试集反复调参。
7. **报告完整结果**：给出混淆矩阵、适合任务的指标、类别基线，并检查不同群体或数据段上的表现。
8. **最终独立评估**：固定全部选择后，在测试集上评估一次，并保留可复现的数据与参数记录。

本章案例包括玻璃类型识别、光学字符识别、个人信用风险评估和乳腺癌诊断。它们的类别数、错误代价和可解释性要求不同，不能用同一阈值和同一指标机械比较。

## 可运行示例：复现五类模型与评价指标的关键计算

下面只用 Python 标准库，核对逻辑回归一步更新、KNN 投票、决策树不纯度、贝叶斯后验、硬间隔和分类指标。它们是教学用最小实现，不替代成熟的数值优化与模型库。

```python
from math import exp, log, log2, sqrt

# 1. 逻辑回归：概率、对数几率、单样本一步梯度下降
def sigmoid(z):
    return 1 / (1 + exp(-z))

p_from_log3 = sigmoid(log(3))
loss_from_log3 = -log(p_from_log3)
w = 0.0
x, y, learning_rate = 1.0, 1.0, 0.1
loss_before = -log(sigmoid(w * x))
gradient = (sigmoid(w * x) - y) * x
w_after = w - learning_rate * gradient
p_after = sigmoid(w_after * x)
loss_after = -log(p_after)
print("逻辑回归概率与损失:", p_from_log3, round(loss_from_log3, 6))
print("一步更新:", w_after, round(p_after, 6), round(loss_after, 6))

# 2. KNN：k=2 与 k=5 的普通多数投票
knn_points = [(1, "A"), (2, "A"), (4, "B"), (5, "B"), (6, "B")]
query = 0
ordered_neighbors = sorted(knn_points, key=lambda point: abs(point[0] - query))

def knn_vote(k):
    labels = [label for _, label in ordered_neighbors[:k]]
    return max(set(labels), key=labels.count)

knn_k2 = knn_vote(2)
knn_k5 = knn_vote(5)
print("KNN k=2 与 k=5:", knn_k2, knn_k5)

# 3. 决策树：收入分裂前后的三种不纯度
def gini(counts):
    total = sum(counts)
    return 1 - sum((count / total) ** 2 for count in counts)

def entropy(counts):
    total = sum(counts)
    result = 0.0
    for count in counts:
        if count:
            probability = count / total
            result -= probability * log2(probability)
    return result

def misclassification(counts):
    return 1 - max(counts) / sum(counts)

root_counts = (5, 5)
children = [(2, 0), (3, 2), (0, 3)]
root_size = sum(root_counts)

def weighted_impurity(metric):
    return sum(sum(child) / root_size * metric(child) for child in children)

gini_after = weighted_impurity(gini)
entropy_after = weighted_impurity(entropy)
error_after = weighted_impurity(misclassification)
gini_drop = gini(root_counts) - gini_after
information_gain = entropy(root_counts) - entropy_after
error_drop = misclassification(root_counts) - error_after
print("Gini 分裂后与下降值:", gini_after, gini_drop)
print("信息熵分裂后与信息增益:", round(entropy_after, 6), round(information_gain, 6))
print("误分率分裂后与下降值:", error_after, error_drop)

# 4. 贝叶斯定理与朴素贝叶斯逐特征连乘
posterior_b = 0.60 * 0.25 / (0.60 * 0.25 + 0.20 * 0.75)
spam_score = 0.5 * (2 / 3) * (2 / 3)
ham_score = 0.5 * (1 / 3) * (1 / 3)
spam_posterior = spam_score / (spam_score + ham_score)
print("球赛 B 主场获胜后验:", posterior_b)
print("朴素贝叶斯垃圾类后验:", spam_posterior)

# 5. 一维硬间隔：w=1,b=0，支持向量为 x=-1 与 x=1
svm_points = [(-2, -1), (-1, -1), (1, 1), (2, 1)]
w_svm, b_svm = 1.0, 0.0
geometric_margins = [
    label * (w_svm * value + b_svm) / abs(w_svm)
    for value, label in svm_points
]
margin = min(geometric_margins)
margin_width = 2 / abs(w_svm)
alpha_positive = alpha_negative = 0.5
w_from_dual = alpha_positive * 1 * 1 + alpha_negative * (-1) * (-1)
print("SVM 几何间隔与总宽度:", margin, margin_width)
print("SVM 对偶还原 w:", w_from_dual)

# 6. 混淆矩阵与多数类基线
TP, FP, FN, TN = 20, 10, 20, 150
total = TP + FP + FN + TN
accuracy = (TP + TN) / total
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * TP / (2 * TP + FP + FN)
specificity = TN / (TN + FP)
negative_baseline_accuracy = (TN + FN) / total
print("准确率、精确率、召回率:", accuracy, round(precision, 6), recall)
print("F1、特异度、负类基线:", round(f1, 6), specificity, negative_baseline_accuracy)
```

关键输出为：

```text
逻辑回归概率与损失: 0.75 0.287682
一步更新: 0.05 0.512497 0.668460
KNN k=2 与 k=5: A B
Gini 分裂后与下降值: 0.24 0.26
信息熵分裂后与信息增益: 0.485475 0.514525
误分率分裂后与下降值: 0.2 0.3
球赛 B 主场获胜后验: 0.5
朴素贝叶斯垃圾类后验: 0.8
SVM 几何间隔与总宽度: 1.0 2.0
SVM 对偶还原 w: 1.0
准确率、精确率、召回率: 0.85 0.666667 0.5
F1、特异度、负类基线: 0.571429 0.9375 0.85
```

代码中的 KNN 平票处理只适用于该无平票示例；完整实现还需明确并列距离、平票、零距离、数值稳定性、多分类策略和概率校准等约定。

## 常见误区

1. **逻辑回归是回归，所以不能分类。** 不对。它用回归式线性分数建模对数几率，再通过 Sigmoid 或 Softmax 得到类别概率。
2. **Sigmoid 输出就是样本确定的类别。** 不对。概率与阈值后的类别决策不同，阈值应结合验证和错误代价选择。
3. **KNN 的 $k$ 越大越稳定，所以越大越好。** 不对。过大的 $k$ 会抹去局部结构，$k=n$ 时所有预测都退化为同一个多数类。
4. **KNN 不需要训练，所以也不需要预处理和调参。** 不对。距离尺度、特征选择、$k$ 和索引结构都会直接影响结果。
5. **决策树每次分裂都选到当前最优，因此整棵树全局最优。** 不对。它是贪心算法，当前局部分裂最优不保证全局最优。
6. **叶节点越纯，模型泛化一定越好。** 不对。过深的树可能只是在记忆噪声，需要预剪枝、后剪枝和独立验证。
7. **朴素贝叶斯要求所有特征无条件独立。** 不对。它假设的是给定类别后的条件独立。
8. **某个条件概率为 0，只能说明这一类完全不可能。** 不一定。也可能只是训练数据未观察到，需要用平滑避免单次零值抹掉全部证据。
9. **SVM 只追求把训练样本分开。** 不完整。硬间隔还要求最大化最小几何间隔，软间隔则权衡间隔与违规损失。
10. **核函数会自动解决过拟合，SVM 也会天然输出概率。** 都不对。核仍需验证复杂度与参数，标准 SVM 的决策分数也不是天然校准的概率。
11. **分类正确率高就足够好。** 不对。类别不平衡时必须结合混淆矩阵、精确率、召回率、特异度、多数类基线和实际代价。
12. **用测试集选出最佳阈值最省事。** 不可以。阈值也是模型选择的一部分，应在训练数据内部确定，测试集只做最终评估。

## 一句话总结

分类模型的核心，是在不同假设下学习离散标签的决策规则，**并用合适的复杂度控制、独立验证和符合错误代价的指标检查它是否真正可用。**

## 复习卡片

- **逻辑回归**：线性分数 → Sigmoid / Softmax → 类别概率；最大似然对应交叉熵。
- **KNN**：距离定义相似，$k$ 控制局部性；量纲、维数和预测成本不可忽略。
- **决策树**：按样本数加权的子节点不纯度选择分裂，剪枝控制复杂度。
- **朴素贝叶斯**：先验 × 类条件似然；条件独立把联合似然拆成连乘，平滑避免零概率。
- **硬间隔 SVM**：最小化 $\frac12\|w\|^2$，等价于最大化 $2/\|w\|$ 的间隔宽度。
- **软间隔与核**：$C$ 控制违规惩罚，核函数替代特征空间内积。
- **评价**：准确率看整体，精确率看预测正类可信度，召回率看真实正例覆盖，$F_1$ 折中二者。
- **流程**：先基线，训练集内部验证，固定模型和阈值后做最终测试。

## 自测问题

1. 某样本的 logit 为 $\ln2$，正类概率和几率分别是多少？若真实标签为正类，负对数似然约为多少？
2. 本章 KNN 例子中，为什么 $k=2$ 预测 A，而 $k=5$ 预测 B？$k=5$ 时还能体现局部结构吗？
3. 某节点有 4 个正类、4 个负类，Gini、信息熵和误分率分别是多少？
4. 教材收入分裂的加权 Gini 与 Gini 下降值分别是多少？为什么子节点不能直接平均？
5. 为什么拉普拉斯平滑能避免朴素贝叶斯整个后验分数被单个零概率清零？实际为什么还常用对数？
6. “给定类别后特征条件独立”与“特征彼此无条件独立”有什么区别？
7. 一维点 $x=-1$（负类）和 $x=1$（正类）作为支持向量时，若 $f(x)=wx+b$，硬间隔解的 $w,b$ 和总间隔宽度是什么？
8. 软间隔中 $C$ 较大和较小分别更强调什么？为什么 $C$ 不是越大越好？
9. 核函数为什么能让 SVM 在原空间形成非线性边界，而不必显式计算所有高维特征？
10. 对 $TP=20,FP=10,FN=20,TN=150$，计算正确率、精确率、召回率和 $F_1$，并与全部预测负类的正确率比较。

## 参考答案

1. $p=2/3$，几率为 $p/(1-p)=2$；负对数似然为 $-\ln(2/3)\approx0.405465$。
2. 最近的两个样本都是 A，而五个样本中有三个 B。$k=5$ 已包含全部训练样本，退化为全局多数类投票，不能体现测试样本的局部结构。
3. 类别完全均衡，Gini = $2\times0.5\times0.5=0.5$，信息熵 = 1 bit，误分率 = 0.5。
4. 加权 Gini 为 $0.2\times0+0.5\times0.48+0.3\times0=0.24$，下降值为 $0.5-0.24=0.26$。子节点样本数不同，直接平均会让小节点获得不合理的大权重。
5. 平滑给未观察到的取值分配非零概率，避免连乘被 0 清零。很多小概率连乘容易数值下溢，对数把连乘变为求和，更稳定也更便于计算。
6. 条件独立是在固定 $Y=c$ 后讨论特征之间的关系；无条件独立则不考虑类别。前者成立不要求后者成立，类别本身可能正是特征相关性的来源。
7. 约束为 $w+b=1$、$w-b=1$，所以 $w=1,b=0$。两侧边界为 $x=\pm1$，总宽度为 2。
8. 较大的 $C$ 更强调减少训练集间隔违规，较小的 $C$ 更强调大间隔。过大的 $C$ 可能过度迁就噪声和异常点，应通过验证选择。
9. 核函数直接给出特征空间内积 $K(x,z)=\phi(x)^T\phi(z)$，对偶求解和预测只需这些内积，因此可以隐式使用高维表示。
10. 正确率为 0.85，精确率为 $2/3$，召回率为 0.5，$F_1=4/7\approx0.5714$。全部预测负类也有 0.85 的正确率，说明单看准确率无法判断正类识别能力。

## 下一章预告

[第五章“集成模型”](/notes/data-science-guide-ensemble/)将介绍随机森林与 AdaBoost。前者通过多棵去相关的树降低方差，后者通过逐步关注难分样本组合弱学习器；本章的单棵树、分类错误与泛化评估将直接成为下一章的基础。

## 参考资料

- 欧高炎、朱占星、董彬等：《数据科学导引》第四章“分类模型”，高等教育出版社，原书第 69—105 页。
