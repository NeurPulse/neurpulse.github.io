---
title: "数学建模：机器学习流程与 KNN"
description: "从距离投票理解监督学习、交叉验证和评估"
pubDate: 2026-08-07
updatedDate: 2026-10-21
order: 17
category: '数学建模'
tags: ["scikit-learn","KNN","机器学习","零基础"]
---

> [专题学习地图](/notes/mathematical-modeling-roadmap/) · 第 17 / 27 篇

## 学习目标与先修知识

目标是区分分类、回归和聚类，手算 KNN 投票及分类指标，搭建带缩放、交叉验证和独立测试的监督学习流程。先修为距离、数据预处理与误差评价。这里先学完整流程，再逐步替换为其他模型。

可回看：[数据预处理与插值](/notes/mathematical-modeling-data-preprocessing/) · [回归与曲线拟合](/notes/mathematical-modeling-regression/)。

## 问题背景与使用时机

已知花的萼片、花瓣尺寸，预测品种，是分类；预测花瓣长度，是回归；没有品种标签，想按形态分组，是聚类。任务由需要输出什么及是否有标签决定，不由选中的软件函数决定。

KNN 的直觉是：新样本参考附近训练样本。它不先学习一条复杂公式，但距离定义、尺度和 k 的选择会显著改变结果。

## 核心概念与符号

输入 $X\in\mathbb R^{n\times d}$，每行一个样本、d列特征；标签 y 长度 n；查询样本也要有相同 d 列。欧氏距离为：

$$
d(\boldsymbol x,\boldsymbol z)=\sqrt{\sum_{j=1}^d(x_j-z_j)^2}.
$$

分类 KNN 取最近 k 个邻居投票；回归 KNN 可取邻居目标均值；距离加权版本让近邻影响更大。k 小更灵活但对噪声敏感，k 大更平滑但可能跨越类别边界。

参数由训练学习，如线性模型系数；超参数由训练内部验证选择，如 k。训练集用于拟合，验证用于选择，最终测试用于估计选择后模型的表现。交叉验证反复划分训练折与验证折，但不能使用最终测试挑参数。

偏差表示平均预测与目标规律的系统偏离；方差表示换一份同规模训练数据时预测的波动。模型复杂度与这两者经常存在权衡，但训练更久或模型更大并不在任何情形下都必然过拟合。

### 分类指标

以“有问题商品”为正类，TP是正确检出，FP是误报，FN是漏报，TN是正确排除。精确率 $TP/(TP+FP)$ 关心报警中多少真实；召回率 $TP/(TP+FN)$ 关心实际问题检出多少；F1 为二者调和平均。ROC-AUC 评价分数排序跨阈值表现，类别稀少时还应关注 PR 指标与业务成本。

## 手算一个最小例子

一维训练点为0、1、2，标签分别为A、B、B，查询点0.1。距离依次为0.1、0.9、1.9。k=1预测A，k=3预测B。每个邻居只计一票，不能在循环中反复统计整组标签。

某次检测 TP=8、FP=2、FN=4、TN=6，共20件：

$$
\mathrm{Accuracy}=\frac{8+6}{20}=0.7,\quad
\mathrm{Precision}=\frac8{10}=0.8,\quad
\mathrm{Recall}=\frac8{12}=\frac23,
$$

$$
F1=\frac{2TP}{2TP+FP+FN}=\frac{16}{22}\approx0.7273.
$$

如果漏报代价高，准确率0.7本身不能说明方案可接受，必须进一步看召回和成本。

## 建模步骤与完整案例

案例：Iris 内置数据包含150朵鸢尾花、4个尺寸特征与3种品种。先分层留出测试集，在训练集中用5折交叉验证选择 k，缩放器在每个训练折内拟合，再在最终测试报告准确率和混淆矩阵。

这不是水仙花数据。内置数据只适合演示，不代表现实花卉识别所需的光照、图像或跨设备问题已经解决。先与多数类别基线比较，确认模型是否比“总猜最多类”有价值。

## Python 实现与结果解释

配套文件：[查看或下载本篇完整脚本](/downloads/math-modeling/mathematical-modeling-ml-sklearn-knn.py)。

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def hand_knn(train, labels, query, k):
    train, query = np.asarray(train, float), np.asarray(query, float)
    labels = np.asarray(labels)
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool) or not 1 <= k <= len(train):
        raise ValueError("k必须是有效整数")
    index = np.argsort(np.sum((train-query)**2, axis=1), kind="stable")[:k]
    classes, counts = np.unique(labels[index], return_counts=True)
    return classes[counts.argmax()]  # 平票时按类别排序取前者

assert hand_knn([[0.], [1.], [2.]], ["A", "B", "B"], [0.1], 3) == "B"
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)
search = GridSearchCV(
    make_pipeline(StandardScaler(), KNeighborsClassifier()),
    {"kneighborsclassifier__n_neighbors": [1, 3, 5, 7, 9, 11, 15]},
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring="accuracy"
)
search.fit(X_train, y_train)
prediction = search.predict(X_test)
baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
print("选择的k:", search.best_params_)
print("模型与基线准确率:", accuracy_score(y_test, prediction), baseline.score(X_test, y_test))
print("混淆矩阵:", confusion_matrix(y_test, prediction))
print(classification_report(y_test, prediction, zero_division=0))
```

`classification_report` 的参数顺序是“真实标签、预测标签”；颠倒会交换精确率与召回的含义。测试混淆矩阵的行通常为真实类别、列为预测类别，应在报告中注明。

## 方法选择与常见误区

- KNN 对单位敏感，金额与厘米直接拼在一起会让大尺度列主导距离。
- 奇数 k 只能减少某些二分类平票，多分类仍可能平票；5至15不是通用最优范围。
- 高维时距离区分能力可能下降；KD树在低维可能加速，但不能保证任何维数都比暴力搜索快。
- 同一客户、病人或设备的多条记录不能随意分散到训练测试两边；要按组或时间设计拆分。
- 线性或逻辑回归是常见的简单监督基线。二分类逻辑回归通过 $\sigma(z)=1/(1+e^{-z})$ 把线性分数映射到概率，用对数损失训练；它虽然叫“回归”，通常用于分类。
- 类别不均衡时，不应只报准确率；阈值应按验证集与业务成本选择，不能反复查看测试集来决定。

## 练习

1. 训练点0(A)、2(B)、3(B)，查询0.2，k=1和3各预测什么？
2. TP=3、FP=1、FN=2，精确率和召回率是多少？
3. 先在全部样本上标准化再交叉验证，哪里有泄漏？
4. k越大是否一定越准确？

## 参考答案

1. k=1最近为A；k=3多数为B。
2. 精确率 $3/4=0.75$，召回率 $3/5=0.6$。
3. 验证折已经参与了缩放参数估计，应把缩放和模型一起放在 Pipeline 中。
4. 不一定。过大的邻域会混入其他类别并忽略局部结构，应在训练内部验证。

## 复习卡片与下一步

先确认任务和拆分，再把预处理、模型、选择与测试连成一条流程。后面的朴素贝叶斯、SVM、树和神经网络都可以接入同样的训练边界。

[上一篇 ARIMA 与相关时间序列](/notes/mathematical-modeling-arima/) · [下一篇 朴素贝叶斯](/notes/mathematical-modeling-ml-naive-bayes/)

## 参考资料

- [Datawhale《数学建模导论》第9章](https://datawhalechina.github.io/intro-mathmodel/#/CH9/第九章-机器学习与统计模型)：用于建立知识目录，建议与本篇的手算和案例对照阅读。
- [scikit-learn 最近邻](https://scikit-learn.org/stable/modules/neighbors.html)。
- [模型选择与评价](https://scikit-learn.org/stable/model_selection.html)。
