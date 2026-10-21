---
title: "数学建模：主成分分析与降维"
description: "从相关特征找到低维表示，理解方差、投影与解释"
pubDate: 2026-10-21
updatedDate: 2026-10-21
order: 12
category: '数学建模'
tags: ["PCA","降维","零基础"]
---

> [专题学习地图](/notes/mathematical-modeling-roadmap/) · 第 12 / 27 篇

## 学习目标与先修知识

目标是从二维手算理解 PCA 的中心化、投影、特征值和解释方差，区分降维与特征选择，并在训练边界内执行缩放、降维和分类。先修为内积、协方差与 SVD。

可回看：[几何模型与线性代数](/notes/mathematical-modeling-geometry/) · [权重生成与综合评价](/notes/mathematical-modeling-evaluation/)。

## 问题背景与使用时机

一张表同时记录厘米与毫米表示的长度，两列几乎重复。再增加多个类似指标，模型和图形会更复杂，但独立信息没有按列数增长。

PCA 寻找一个新坐标系，使前几个方向保留尽量多的样本方差。它不知道标签，也不知道你希望预测什么，因此高方差方向不一定对任务最有用。

## 核心概念与符号

对 $n\times d$ 的数据矩阵 X，先按列减去训练均值得到 $X_c$。样本协方差为：

$$
S=\frac{X_c^TX_c}{n-1}.
$$

第一主方向是在单位长度约束下最大化投影方差的向量：

$$
\max_{\|\boldsymbol v\|=1}\boldsymbol v^TS\boldsymbol v.
$$

最优方向是最大特征值对应的特征向量。后续方向与前面的方向正交，并按剩余方差排序。取前 k 个方向组成 $V_k$，低维得分为 $Z=X_cV_k$，近似重建为 $\hat X=ZV_k^T+\boldsymbol\mu$。

解释方差比为 $\lambda_j/\sum_l\lambda_l$，累计值描述保留了多少样本变异，不是分类准确率。主成分是原特征的线性组合；特征选择则保留原来的某几列。

单位差异会影响方差。例如金额元与万元只差换单位，PCA 却可能改排序，因此是否标准化要由变量含义决定。

## 手算一个最小例子

取点 $(-1,-1),(0,0),(1,1)$。两列均值都为0，样本协方差矩阵为：

$$
S=\frac12\begin{bmatrix}2&2\\2&2\end{bmatrix}
=\begin{bmatrix}1&1\\1&1\end{bmatrix}.
$$

特征值为2和0，第一方向可选 $\boldsymbol v_1=(1,1)^T/\sqrt2$。三个投影得分为 $-\sqrt2,0,\sqrt2$，解释方差比为100%。

只用这一列即可重建原始三个点，因为它们完全位于同一直线上。若第一方向整体变号，得分也变号，重建仍完全相同，不应把符号差异当作算法错误。

## 建模步骤与完整案例

案例：葡萄酒数据有多个不同量纲的化学指标。先留出测试样本，再把标准化、PCA 和分类器放在一个 Pipeline 中。保留95%的训练方差只是一种降维规则；要判断是否有助于分类，还需与不降维模型在训练内部比较。

本例先验证完整流程及输出尺寸，不宣称95%是最佳选择。若特征含强异常值，传统 PCA 可能被异常方向牵引；若关键预测信号方差很小，PCA 还可能把它丢掉。

## Python 实现与结果解释

配套文件：[查看或下载本篇完整脚本](/downloads/math-modeling/mathematical-modeling-pca.py)。

```python
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

small = np.array([[-1., -1.], [0., 0.], [1., 1.]])
hand = PCA(n_components=1).fit(small)
reconstruction = hand.inverse_transform(hand.transform(small))
print("手算案例方差比:", hand.explained_variance_ratio_)
assert np.allclose(reconstruction, small)

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42)
pipeline = make_pipeline(
    StandardScaler(), PCA(n_components=0.95),
    LogisticRegression(max_iter=2000)
)
pipeline.fit(X_train, y_train)
pca = pipeline.named_steps["pca"]
print("原特征数、主成分数:", X.shape[1], pca.n_components_)
print("累计解释方差:", pca.explained_variance_ratio_.sum())
print("留出准确率:", pipeline.score(X_test, y_test))
assert pca.explained_variance_ratio_.sum() >= 0.95
assert pca.n_components_ <= X.shape[1]
```

PCA 的均值、方向和保留成分数都来自训练数据。测试集只经过已有变换；即使降维没有用标签，在全数据上先拟合也会利用测试分布信息。

## 方法选择与常见误区

| 目标 | 可考虑 | 需要说明 |
|---|---|---|
| 保留整体线性方差、压缩冗余 | PCA | 缩放规则与保留方差 |
| 保持原变量含义 | 特征选择 | 选择过程必须在训练内 |
| 利用标签进行投影 | LDA | 类内与类间结构 |
| 展示非线性邻近关系 | t-SNE、UMAP 等 | 图中距离不宜过度解释 |
| 假设由潜在公共因素生成 | 因子分析 | 与 PCA 的建模目标不同 |

累计方差不是模型效果验收；主成分载荷不是因果重要性；正负号与重复特征值下的基变化一般不影响子空间含义。

## 练习

1. 两列完全相同，去掉一列信息是否必然损失？
2. 保留90%方差是否表示预测正确率90%？
3. 为什么训练和测试不能分别拟合两套 PCA？
4. 如果最有预测价值的方向方差很小，PCA 可能怎样？

## 参考答案

1. 数值信息重复，去掉一列不会丢失这两列之间的独立信息；仍需确认两列业务含义和数据质量。
2. 不是，方差是输入分布的量，正确率是预测与标签比较的量。
3. 坐标方向、顺序与均值可能不同，分类器收到的将不是同一套特征；应共享训练拟合的变换。
4. 可能被舍弃，因此降维维数与是否使用 PCA 都应在适当验证中比较。

## 复习卡片与下一步

记住“中心化—找方向—投影—验收”。降维后仍要检查任务指标，不能只看方差曲线。接下来学习有先后顺序的数据，时间顺序会改变验证方法。

[上一篇 权重生成与综合评价](/notes/mathematical-modeling-evaluation/) · [下一篇 时间序列分析与验证](/notes/mathematical-modeling-time-series/)

## 参考资料

- [Datawhale《数学建模导论》第7章](https://datawhalechina.github.io/intro-mathmodel/#/CH7/第7章-权重生成与评价模型)：用于建立知识目录，建议与本篇的手算和案例对照阅读。
- [scikit-learn PCA](https://scikit-learn.org/stable/modules/decomposition.html#pca)。
