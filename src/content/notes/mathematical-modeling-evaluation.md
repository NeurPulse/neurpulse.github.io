---
title: "数学建模：权重生成与综合评价"
description: "用 AHP、熵权和 TOPSIS 比较方案并检查排名稳健性"
pubDate: 2026-10-21
updatedDate: 2026-10-21
order: 11
category: '数学建模'
tags: ["综合评价","AHP","熵权","TOPSIS","零基础"]
---

> [专题学习地图](/notes/mathematical-modeling-roadmap/) · 第 11 / 27 篇

## 学习目标与先修知识

目标是区分指标、权重与排名，手算加权分数和 TOPSIS，理解 AHP、熵权的假设，并检查权重改变后排名是否稳定。先修为比例、均值、对数和矩阵。评价不能代替真实预测验证，也不能自动决定价值偏好。

可回看：[必备数学与 Python 基础](/notes/mathematical-modeling-foundations/)。

## 问题背景与使用时机

三种设备中哪一种更适合采购？价格低、速度快、可靠性高通常不能同时最大化。综合评价需要先明确评价对象、指标方向和使用者偏好，再生成权重与总分。

不能先看想要的结果，再倒推一组能让它第一的权重。指标体系应有业务或文献依据，且避免把同一含义重复计权。

## 核心概念与符号

### 先统一方向和尺度

效益指标越大越好，成本指标越小越好。本篇采用 Min-Max 正向化：

$$
z_{ij}=
\begin{cases}
(x_{ij}-\min_i x_{ij})/(\max_i x_{ij}-\min_i x_{ij}),&\text{效益}\\
(\max_i x_{ij}-x_{ij})/(\max_i x_{ij}-\min_i x_{ij}),&\text{成本}
\end{cases}
$$

常量指标没有区分能力，应单独处理。适度指标如温度接近目标最佳，需要以偏离目标的程度定义效用，不能机械按越大越好。

### 权重怎样产生

AHP 用成对重要性判断建立正互反矩阵，$a_{ij}=1/a_{ji}$。可从主特征向量或近似几何均值求权重。$CI=(\lambda_{\max}-m)/(m-1)$，$CR=CI/RI$；m=3 的常用 RI 为0.58。0.1 是常见一致性判断阈值，不证明偏好本身客观或正确。

熵权看各列的数据分布差异。令 $p_{ij}=z_{ij}/\sum_i z_{ij}$：

$$
e_j=-\frac1{\ln n}\sum_i p_{ij}\ln p_{ij},\qquad
w_j=\frac{1-e_j}{\sum_k(1-e_k)}.
$$

约定 $0\ln0=0$。差异更大的指标通常获更高权重；这是数据区分度，不等于业务重要性。

### TOPSIS 怎样排名

令 $v_{ij}=w_jz_{ij}$，理想点为各列最大值，负理想点为各列最小值。计算距离 $D_i^+$、$D_i^-$，接近度为：

$$
C_i=\frac{D_i^-}{D_i^++D_i^-}.
$$

越大表示在当前指标、正向化和权重规则下越接近理想。若两距离都为0，则没有区分能力，不能强行得出排名。

## 手算一个最小例子

三个方案的正向化指标为 A=(1,0)、B=(0.5,0.5)、C=(0,1)，权重为 (0.6,0.4)。

加权后分别为 (0.6,0)、(0.3,0.2)、(0,0.4)。理想点 (0.6,0.4)，负理想点 (0,0)。因此 A 的 $D^+=0.4,D^-=0.6,C=0.6$；B 到两点距离相同，C=0.5；C 的接近度为0.4。

若交换权重为 (0.4,0.6)，A 与 C 的排名反转。不是算错了，而是偏好变了。

AHP 中若三项权重为 (0.5,0.3,0.2)，令 $a_{ij}=w_i/w_j$，则 $Aw=3w$，$\lambda_{\max}=3$、CI=0，一致性完全满足。但这一数学一致性仍不说明实际使用者必须接受这些权重。

## 建模步骤与完整案例

案例：比较三种设备。价格分别为100、80、120；速度8、6、9；可靠评分90、85、80。价格为成本指标，其余为效益指标。假设这些评分可比，权重由使用者明确设为 (0.5,0.3,0.2)。

步骤为：核查指标含义与单位 → 正向化 → 检查常量与重复指标 → 确定权重来源 → 计算 TOPSIS → 改变价格权重观察排名 → 报告稳健性。

设备数量少时，归一化极值可能对某一设备敏感；增加或删除方案还可能改变相对尺度，因此应说明评价范围。

## Python 实现与结果解释

配套文件：[查看或下载本篇完整脚本](/downloads/math-modeling/mathematical-modeling-evaluation.py)。

```python
import numpy as np

raw = np.array([[100., 8., 90.], [80., 6., 85.], [120., 9., 80.]])
benefit = np.array([False, True, True])
span = np.ptp(raw, axis=0)
assert np.all(span > 0)
positive = np.where(benefit,
                    (raw-raw.min(axis=0))/span,
                    (raw.max(axis=0)-raw)/span)

def topsis(z, weights):
    weighted = z*np.asarray(weights)
    best, worst = weighted.max(axis=0), weighted.min(axis=0)
    d_best = np.linalg.norm(weighted-best, axis=1)
    d_worst = np.linalg.norm(weighted-worst, axis=1)
    denominator = d_best+d_worst
    if np.any(denominator == 0):
        raise ValueError("没有区分能力，不能计算接近度")
    return d_worst/denominator

weights = np.array([0.5, 0.3, 0.2])
score = topsis(positive, weights)
print("正向化数据:", positive)
print("A/B/C接近度:", score, "排序:", np.argsort(-score))

# 熵权作为另一种权重来源，不能把它直接称为真实偏好。
probability = positive/positive.sum(axis=0)
terms = np.zeros_like(probability)
mask = probability > 0
terms[mask] = probability[mask]*np.log(probability[mask])
entropy = -terms.sum(axis=0)/np.log(len(raw))
diversity = 1-entropy
entropy_weights = diversity/diversity.sum()
print("熵权及其接近度:", entropy_weights, topsis(positive, entropy_weights))
assert np.isclose(entropy_weights.sum(), 1)

pairwise = weights[:, None]/weights[None, :]
eigenvalues = np.linalg.eigvals(pairwise)
lambda_max = eigenvalues.real.max()
CR = ((lambda_max-3)/2)/0.58
print("一致比较矩阵的CR:", CR)
assert abs(CR) < 1e-10

for cost_weight in [0.2, 0.5, 0.8]:
    w = np.array([cost_weight, (1-cost_weight)*0.6, (1-cost_weight)*0.4])
    print("价格权重:", cost_weight, "排名:", np.argsort(-topsis(positive, w)))
```

给定主观权重时，A、B、C的接近度约为0.584、0.617、0.358，排序为B、A、C；改用本例的熵权后排序为A、B、C。两种权重回答的偏好与区分度问题不同，并不是一方计算错误。CR在浮点运算中可能出现约 $10^{-15}$ 的微小负数，应按数值误差理解为0。

排名用 0、1、2 代表 A、B、C。应把结果翻译为“在这些偏好下，哪个方案得分较高”，并检查权重变动后是否反转，而不是宣称发现了无条件最好的设备。

## 方法选择与常见误区

AHP 适合能明确表达成对判断的场景；熵权适合用样本差异提供权重依据；TOPSIS 是合成评分规则，可以和不同权重来源组合。这三者解决的环节不同。

CRITIC 同时考虑指标波动及相关冗余；模糊综合评价处理分级隶属度；秩和比使用秩信息；DEA 在投入产出假设下比较相对效率。选择它们前，需先说明当前指标和决策结构为何符合假设。不能仅因某方法名字“高级”就替代清楚的偏好讨论。

PCA 的目标是保留方差与减少冗余，不直接产生正确的价值权重；下一篇专门讲这个区别。

## 练习

1. 价格越低越好，应如何处理方向？
2. 某列所有方案数值相同，直接代入 Min-Max 会发生什么？
3. CR 很小能否证明评价客观？
4. 如果两组合理权重给出相反排名，应怎样汇报？

## 参考答案

1. 使用成本型正向化，或先构造有业务依据的价格效用函数。
2. 分母为0。该列不提供区分能力，应按规则移除或处理，不能悄悄添加极小量就声称含义不变。
3. 不能。CR 只检查成对判断的一致程度，偏好来源与指标合理性仍需解释。
4. 报告权重敏感、给出权衡条件及可接受方案，不隐藏不利的权重设置。

## 复习卡片与下一步

复习卡：指标先有依据，方向先统一，权重说明来源，得分只在当前规则下有效，排名必须做敏感性检查。下一篇解释为什么“方差大”与“评价重要”不是同一件事。

[上一篇 图论与网络优化](/notes/mathematical-modeling-graphs/) · [下一篇 主成分分析与降维](/notes/mathematical-modeling-pca/)

## 参考资料

- [Datawhale《数学建模导论》第7章](https://datawhalechina.github.io/intro-mathmodel/#/CH7/第7章-权重生成与评价模型)：用于建立知识目录，建议与本篇的手算和案例对照阅读。
- [Datawhale 第7章](https://github.com/datawhalechina/intro-mathmodel/tree/main/docs/CH7)：可继续对照 CRITIC、模糊评价与 DEA 的适用假设。
