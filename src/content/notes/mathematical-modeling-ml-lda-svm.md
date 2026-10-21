---
title: "数学建模：线性判别与支持向量机"
description: "从投影与间隔理解 LDA、软间隔和核函数"
pubDate: 2026-08-07
updatedDate: 2026-10-21
order: 19
category: '数学建模'
tags: ["线性判别","SVM","机器学习","分类","零基础"]
---

> [专题学习地图](/notes/mathematical-modeling-roadmap/) · 第 19 / 27 篇

## 学习目标与先修知识

目标是区分 LDA 的投影思想和 SVM 的最大间隔思想，手算一个方向与分类边界，解释软间隔参数 C、核函数和 gamma，并用标准实现完成验证。先修为内积、矩阵、分类与标准化。

可回看：[几何模型与线性代数](/notes/mathematical-modeling-geometry/) · [机器学习流程与 KNN](/notes/mathematical-modeling-ml-sklearn-knn/)。

## 问题背景与使用时机

两类点可能沿某个方向重叠少，投影后便于分类；也可能适合用一条线隔开，并希望这条线离两侧最近的点都尽量远。这分别引出 LDA 和 SVM。

它们都能给出线性边界，但目标、概率假设与影响边界的数据不同。LDA 使用类别统计结构，SVM 强调间隔及违反间隔的代价。

## 核心概念与符号

### LDA 从类内与类间差异出发

两类均值为 $\boldsymbol\mu_0,\boldsymbol\mu_1$。类内散布矩阵定义为：

$$
S_w=\sum_{c=0}^1\sum_{i:y_i=c}
(\boldsymbol x_i-\boldsymbol\mu_c)(\boldsymbol x_i-\boldsymbol\mu_c)^T.
$$

它是离均差外积的和；若从样本协方差计算，第 c 类要乘 $n_c-1$。投影 $\boldsymbol w^T\boldsymbol x$ 后，希望均值分开、类内波动小：

$$
J(\boldsymbol w)=
\frac{[\boldsymbol w^T(\boldsymbol\mu_1-\boldsymbol\mu_0)]^2}
{\boldsymbol w^TS_w\boldsymbol w}.
$$

当 $S_w$ 可逆且问题非退化时，方向正比于 $S_w^{-1}(\boldsymbol\mu_1-\boldsymbol\mu_0)$。奇异或高维时可用正则化、收缩或合适分解。高斯类条件分布且共享协方差的 LDA 还可产生概率分类边界；类别先验会影响阈值。

### SVM 从间隔出发

设标签为 $-1,+1$，边界 $\boldsymbol w^T\boldsymbol x+b=0$。点到边界距离为 $|\boldsymbol w^T\boldsymbol x+b|/\|\boldsymbol w\|$。软间隔模型为：

$$
\min_{\boldsymbol w,b,\boldsymbol\xi}\frac12\|\boldsymbol w\|^2+C\sum_i\xi_i,
\qquad y_i(\boldsymbol w^T\boldsymbol x_i+b)\ge1-\xi_i,\quad \xi_i\ge0.
$$

C 越大越重罚违反间隔，越小越愿意容忍违反来换取较宽间隔。支持向量是影响边界的关键样本，许多远离间隔的训练点不直接决定最终解。

### 核函数

核通过 $K(x,z)$ 表示某特征空间内积。线性核为 $x^Tz$，多项式核可写 $(\gamma x^Tz+c)^d$，RBF 为 $\exp[-\gamma\|x-z\|^2]$。RBF 的 gamma 大时影响更局部、边界可更复杂。并非任意相似度都是合法半正定核，sigmoid 形式也不是任意参数下都满足要求。

## 手算一个最小例子

LDA 设均值差为 $(2,0)^T$，类内散布为 $\begin{bmatrix}2&0\\0&1\end{bmatrix}$，则方向为 $(1,0)^T$，表示主要沿 x 轴投影。方向乘任何非零常数不会改变投影排序；分类阈值还要结合先验和分布设定。

SVM 一维样本为0、1属于负类，3、4属于正类。最近的异类点为1和3，最大间隔边界在中点2。取 $w=1,b=-2$，两侧支持点的函数分数恰为 -1和1，间隔宽度 $2/|w|=2$。

查询2.5时分数为0.5，预测正类。这个分数不是0.5的类别概率；SVM 决策值与概率需要区分。

## 建模步骤与完整案例

案例：用 Iris 的4个尺寸预测品种。对 LDA，使用带收缩的共享协方差估计；对 SVM，把标准化与 RBF 分类器放入 Pipeline，在训练内部选择 C 与 gamma。最终测试只评价选择后的方案。

LDA 也能用于监督降维，但最多有“类别数减1”个判别方向，并受原特征维数限制；它与不使用标签的 PCA 不同。SVM 可处理非线性边界，但缩放、核与参数选择要保持同样的验证边界。

## Python 实现与结果解释

配套文件：[查看或下载本篇完整脚本](/downloads/math-modeling/mathematical-modeling-ml-lda-svm.py)。

```python
import numpy as np
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

direction = np.linalg.solve([[2., 0.], [0., 1.]], [2., 0.])
assert np.allclose(direction, [1., 0.])
hand = SVC(kernel="linear", C=1e4).fit([[0.], [1.], [3.], [4.]], [-1, -1, 1, 1])
print("手算SVM的w、b:", hand.coef_, hand.intercept_)
assert np.allclose(hand.coef_, [[1.]], atol=1e-3)
assert np.allclose(hand.intercept_, [-2.], atol=1e-3)

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)
lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
lda.fit(X_train, y_train)
search = GridSearchCV(
    make_pipeline(StandardScaler(), SVC(kernel="rbf")),
    {"svc__C": [0.1, 1., 10.], "svc__gamma": ["scale", 0.1, 1.]},
    cv=StratifiedKFold(5, shuffle=True, random_state=42)
)
search.fit(X_train, y_train)
print("SVM参数:", search.best_params_)
print("LDA测试准确率:", lda.score(X_test, y_test))
print("SVM测试准确率:", search.score(X_test, y_test))
```

手算部分验证最简单的边界；完整案例展示真实训练与独立测试。不要为了讲解 SMO 就先让小白实现整个求解器，理解输入、目标与边界后再研究成对更新、KKT 条件和收敛判据。

## 方法选择与常见误区

| 情况 | 可优先比较 | 注意 |
|---|---|---|
| 类别近似共享协方差，想要线性解释 | LDA | 检查类分布与协方差稳定性 |
| 中小规模非线性分类 | RBF SVM | 缩放、C与gamma需联合验证 |
| 稀疏高维线性分类 | 线性分类器 | 计算量与正则化 |
| 想无监督压缩输入 | PCA | 不使用类别标签 |

C 大不等于一定准确，也不等于“去掉了松弛变量”；gamma 大不等于一定能抓住真实规律。若不同类严重重叠，任何硬分隔都可能不合理。输出决策值不自动是已校准概率。

## 练习

1. LDA 与 PCA 都降维，它们最大的目标差别是什么？
2. 上述 SVM 边界对查询1.5给出什么分类？
3. 样本只有10个、特征100个，直接求类内散布逆矩阵为什么危险？
4. 为什么 SVM 的 C 与 gamma 不能只看训练准确率来选？

## 参考答案

1. LDA 使用标签寻找有利于类分离的方向，PCA 不使用标签，保留输入方差。
2. 分数 $1.5-2=-0.5$，预测负类。
3. 矩阵常秩不足或非常不稳定，应考虑收缩、正则化或适当分解。
4. 参数可使模型过度适应训练点，需要训练内部验证评价对未见样本的效果。

## 复习卡片与下一步

LDA 记“类别均值差相对类内波动”，SVM 记“间隔加违反代价”，核记“隐式特征内积”。下一篇用可解释的条件分支建立非线性预测。

[上一篇 朴素贝叶斯](/notes/mathematical-modeling-ml-naive-bayes/) · [下一篇 树模型与集成学习](/notes/mathematical-modeling-ml-tree-ensemble/)

## 参考资料

- [Datawhale《数学建模导论》第9章](https://datawhalechina.github.io/intro-mathmodel/#/CH9/第九章-机器学习与统计模型)：用于建立知识目录，建议与本篇的手算和案例对照阅读。
- [scikit-learn LDA 与 QDA](https://scikit-learn.org/stable/modules/lda_qda.html)。
- [scikit-learn SVM](https://scikit-learn.org/stable/modules/svm.html)。
