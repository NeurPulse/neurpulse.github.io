---
title: "数学建模：聚类算法"
description: "通过手算分组理解 K-Means、DBSCAN、层次聚类和 GMM"
pubDate: 2026-08-07
updatedDate: 2026-10-21
order: 21
category: '数学建模'
tags: ["聚类","K-Means","DBSCAN","无监督学习","零基础"]
---

> [专题学习地图](/notes/mathematical-modeling-roadmap/) · 第 21 / 27 篇

## 学习目标与先修知识

目标是手算 K-Means 一轮更新和轮廓系数，理解 DBSCAN 的核心点、边界点与噪声，以及层次聚类和 GMM 的区别。先修为距离、均值和标准化。聚类没有已知类别标签，不代表结果可以不评价。

可回看：[几何模型与线性代数](/notes/mathematical-modeling-geometry/) · [机器学习流程与 KNN](/notes/mathematical-modeling-ml-sklearn-knn/)。

## 问题背景与使用时机

想按客户行为分群，却没有现成的“高价值、中价值、低价值”标签，可以先探索聚类。算法会按选定特征和距离组织样本，业务含义仍需后续解释。

如果本来就有明确标签并希望预测新客户类别，任务可能应该设为监督分类。聚类编号0、1、2没有天然的优劣顺序。

## 核心概念与符号

### K-Means

交替执行“分配到最近中心”和“更新为组均值”，最小化：

$$
J=\sum_{k=1}^K\sum_{i:c_i=k}\|\boldsymbol x_i-\boldsymbol\mu_k\|^2.
$$

中心可以不在任何原始样本上。算法依赖初值，通常只能保证相应迭代目标不增加，不保证找到全局最好分组。K-Means++ 改善初始化，多次启动增加找到较好局部解的机会。空簇需要实现中的明确处理。

轮廓系数为 $s(i)=[b(i)-a(i)]/\max(a(i),b(i))$，a 是到同簇其他点的平均距离，b 是到其他各簇平均距离的最小值。范围[-1,1]，一般越大表示当前距离下更分离，单样本簇通常约定0。

### DBSCAN

eps 邻域内至少有 min_samples 个点的为核心点，scikit-learn 计数包括自身。边界点自身不够密，但能从核心点到达；既非核心也无法从核心到达的标为噪声-1。密度直达通常不对称，密度相连用于组织簇。

### 层次与概率分组

凝聚层次聚类从单点出发，按簇间准则合并。单链接看最近点对、全链接看最远点对、Ward 关注组内平方和增量。Ward 通常配欧氏几何。GMM 假设由多个高斯分布混合产生，可输出软归属概率，不只是一枚硬标签。

## 手算一个最小例子

一维样本0、2、8、10，初始中心0和10。第一次分配为{0,2}、{8,10}，更新中心为1和9；再分配不变，停止。

组内平方和为 $1+1+1+1=4$。对点0，同簇距离 a=2，到另簇的平均距离 b=(8+10)/2=9，轮廓系数为 $(9-2)/9=7/9$。

这并不证明现实中必然有两个群体：K是我们给定的，结果还依赖使用了哪些变量和怎样缩放。若某个变量单位从元换成分，欧氏距离可能改变，分群也可能改变。

## 建模步骤与完整案例

案例：用三团二维教学点比较不同聚类结构。先标准化，在预先给定的 K=2至5 中比较内部轮廓系数，再演示 DBSCAN、Ward 与 GMM。

这里是在描述当前样本，不把全样本的轮廓分数称为对未来标签的预测准确率。真实客户分群还应看不同样本、时间和参数下的稳定性，解释群体特征，并判断分群是否支持行动。

## Python 实现与结果解释

配套文件：[查看或下载本篇完整脚本](/downloads/math-modeling/mathematical-modeling-ml-clustering.py)。

```python
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

small = np.array([[0.], [2.], [8.], [10.]])
hand = KMeans(n_clusters=2, init=np.array([[0.], [10.]]), n_init=1, random_state=42)
hand.fit(small)
assert np.allclose(np.sort(hand.cluster_centers_.ravel()), [1., 9.])
assert np.isclose(hand.inertia_, 4.)

X, _ = make_blobs(n_samples=180, centers=3, cluster_std=0.7, random_state=42)
z = StandardScaler().fit_transform(X)
scores = {}
for k in range(2, 6):
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(z)
    scores[k] = silhouette_score(z, labels)
print("各K轮廓系数:", scores)

density_labels = DBSCAN(eps=0.35, min_samples=5).fit_predict(z)
hierarchy_labels = AgglomerativeClustering(n_clusters=3, linkage="ward").fit_predict(z)
mixture = GaussianMixture(n_components=3, random_state=42).fit(z)
probability = mixture.predict_proba(z)
print("DBSCAN簇标签及数量:", np.unique(density_labels, return_counts=True))
print("层次聚类标签:", np.unique(hierarchy_labels))
print("第一个点的GMM归属概率:", probability[0])
assert np.allclose(probability.sum(axis=1), 1.)
```

轮廓系数需要适当数量的有效簇；如果 DBSCAN 全是噪声或只有一个簇，就不能机械调用它。噪声标签是算法在当前 eps、尺度与密度阈值下的结果，不自动意味着数据错误。

## 方法选择与常见误区

| 特征结构 | 可先尝试 | 局限 |
|---|---|---|
| 紧凑、近似球形且规模相近 | K-Means | 初值、异常值和非球形簇 |
| 非规则形状、有噪声 | DBSCAN | 单一eps难适应变密度 |
| 想观察不同粒度的合并 | 层次聚类 | 链接准则影响明显，计算成本 |
| 椭圆簇及软归属 | GMM | 高斯假设、成分数与协方差估计 |

肘部图和轮廓只是辅助，不会自动给出唯一正确的K。若有外部已知标签，可额外用 ARI 等比较，但不得又用该标签来挑结果，再假装完全无监督发现。

## 练习

1. K-Means 中心一定是样本点吗？
2. 若 DBSCAN 某点邻域数量不足，但紧邻核心点，它一定是噪声吗？
3. 组编号从0变成2是否意味着聚类变差？
4. 为什么把收入和年龄原始值直接计算距离可能不合理？

## 参考答案

1. 不一定，它是组均值，可能位于样本之间。
2. 不一定，可能是边界点。只有无法从核心结构到达等情形才标噪声。
3. 不是，编号可以置换，比较应看成员划分及相应指标。
4. 单位和数值尺度不同，收入可能几乎支配距离。缩放或重新定义相似性应反映分析目的。

## 复习卡片与下一步

聚类先定特征和距离，再定算法与参数，最后检验稳定性和业务含义。下一篇用可训练的多层映射学习非线性关系。

[上一篇 树模型与集成学习](/notes/mathematical-modeling-ml-tree-ensemble/) · [下一篇 神经网络与反向传播](/notes/mathematical-modeling-ml-neural-network/)

## 参考资料

- [Datawhale《数学建模导论》第9章](https://datawhalechina.github.io/intro-mathmodel/#/CH9/第九章-机器学习与统计模型)：用于建立知识目录，建议与本篇的手算和案例对照阅读。
- [scikit-learn 聚类](https://scikit-learn.org/stable/modules/clustering.html)。
- [高斯混合模型](https://scikit-learn.org/stable/modules/mixture.html)。
