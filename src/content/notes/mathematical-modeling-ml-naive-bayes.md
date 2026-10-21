---
title: "数学建模：朴素贝叶斯"
description: "从条件概率与平滑走到文本分类"
pubDate: 2026-08-07
updatedDate: 2026-10-21
order: 18
category: '数学建模'
tags: ["朴素贝叶斯","分类","机器学习","零基础"]
---

> [专题学习地图](/notes/mathematical-modeling-roadmap/) · 第 18 / 27 篇

## 学习目标与先修知识

目标是从条件概率手算类别后验，理解“给定类别后条件独立”和加性平滑，并完成一个可运行的文本分类例子。先修为概率乘法、对数及训练测试划分。

可回看：[必备数学与 Python 基础](/notes/mathematical-modeling-foundations/) · [机器学习流程与 KNN](/notes/mathematical-modeling-ml-sklearn-knn/)。

## 问题背景与使用时机

邮件含有“中奖”时是否更可能是垃圾邮件？我们可以先统计不同类别中词语出现的规律，再反过来判断看到这些词时属于哪类。

“朴素”指一种简化假设：给定类别后，各特征按模型条件独立。它不是说所有词在总体上完全独立，也不是宣称真实语言没有词序和语义依赖。

## 核心概念与符号

贝叶斯公式为：

$$
P(c\mid\boldsymbol x)=\frac{P(\boldsymbol x\mid c)P(c)}{P(\boldsymbol x)}.
$$

$P(c)$ 是类别先验，$P(\boldsymbol x\mid c)$ 是该类产生特征的似然，$P(c\mid\boldsymbol x)$ 是后验。比较类别时分母相同，可先比较“先验 × 似然”，若要输出概率再对类别归一化。

多项式朴素贝叶斯把文本表示为词频 $x_j$。在类别 c 中，第 j 个词概率估计为：

$$
P(w_j\mid c)=\frac{N_{cj}+\alpha}{N_c+\alpha V},\qquad
\log\mathrm{score}_c=\log P(c)+\sum_jx_j\log P(w_j\mid c).
$$

$N_{cj}$ 是该类词 j 的总次数，$N_c$ 是该类全部词次数，V是训练词表大小。$\alpha=1$ 称为 Laplace 平滑；任意正 $\alpha$ 为加性平滑，分子分母要同步调整。对数把长乘积变成和，避免小概率连乘下溢。

BernoulliNB 使用词是否出现；MultinomialNB 使用非负计数或相应非负权重；GaussianNB 为给定类别后的连续特征建立高斯似然。不能把负的标准化词频直接交给要求非负输入的多项式模型。

## 手算一个最小例子

训练垃圾邮件为“win free”“win deal”，正常邮件为“meeting project”“meeting free”。两类各2篇，先验均为1/2。词表为 win、free、deal、meeting、project，共5词；两类总词数均为4。

加一平滑后，垃圾类的 $P(win)=3/9$、$P(free)=2/9$；正常类分别为 $1/9,2/9$。测试“win free”的未归一化分数：

$$
s_{\mathrm{spam}}=\frac12\frac39\frac29=\frac1{27},\qquad
s_{\mathrm{normal}}=\frac12\frac19\frac29=\frac1{81}.
$$

归一化后垃圾概率为 $(1/27)/(1/27+1/81)=3/4$。不平滑时，正常类没见过 win，会得到零似然；平滑避免“一次未见就完全否定”。这个0.75仍依赖建模假设，不保证真实世界中具有同样校准程度。

## 建模步骤与完整案例

案例先精确重现4篇邮件的手算，再用16篇教学文本演示独立测试流程。词表必须只在训练中建立；交叉验证时词表也需随训练折重建，因此 CountVectorizer 与模型放在同一 Pipeline。

真实文本还要考虑语言切词、重复邮件、发件人分组、时间变化和新词。此处英文空格文本用于免去额外分词依赖，词的业务含义与前面的中文直觉一一对应。

## Python 实现与结果解释

配套文件：[查看或下载本篇完整脚本](/downloads/math-modeling/mathematical-modeling-ml-naive-bayes.py)。

```python
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report

toy = ["win free", "win deal", "meeting project", "meeting free"]
labels = [1, 1, 0, 0]
hand = make_pipeline(CountVectorizer(), MultinomialNB(alpha=1.0)).fit(toy, labels)
probability = hand.predict_proba(["win free"])[0]
print("正常、垃圾概率:", probability)
assert np.allclose(probability, [0.25, 0.75])

texts = [
    "win free prize", "claim cash prize", "win cash now", "free offer now",
    "cheap deal prize", "claim free cash", "win exclusive offer", "free bonus deal",
    "meeting project today", "project report attached", "team meeting tomorrow",
    "review project plan", "team lunch today", "report meeting notes",
    "please review report", "tomorrow project discussion"
]
y = np.array([1]*8 + [0]*8)
train, test, y_train, y_test = train_test_split(
    texts, y, test_size=0.25, stratify=y, random_state=42)
search = GridSearchCV(
    make_pipeline(CountVectorizer(), MultinomialNB()),
    {"multinomialnb__alpha": [0.5, 1., 2.]},
    cv=StratifiedKFold(3, shuffle=True, random_state=42),
    scoring="f1"
)
search.fit(train, y_train)
print("选择:", search.best_params_)
print(classification_report(y_test, search.predict(test), zero_division=0))
```

前半段的概率应与手算完全一致；后半段只有4篇留出文本，指标仅说明流程如何计算，不支持稳定的生产性能结论。未知词在当前词表中不产生特征，完全未知文本可能主要依赖先验。

## 方法选择与常见误区

模型适合作为文本分类的高效基线，对小规模稀疏词频也容易训练。若词间依赖、否定语序或上下文是核心，词袋模型可能遗漏关键含义。

TF-IDF 用文档频率降低常见词权重，得到的非负权重常与多项式模型配合使用，但它不再是原始词次数的严格生成解释。特征计数、词表、先验与平滑参数要一起说明。

分类正确率高不保证预测概率已校准；若后续按概率制定高成本决策，应另行评估概率质量。

## 练习

1. 为什么只比较类别时可以暂不算 $P(\boldsymbol x)$？
2. 某类总词数10、词表5、某词出现2次，$\alpha=1$，平滑概率是多少？
3. 条件独立等于“所有词互不相关”吗？
4. 先把全体邮件建词表，再划分训练测试，为什么仍不理想？

## 参考答案

1. 对同一个待预测样本，分母在所有类别间相同，不影响最大类别；要得到归一化概率才需完整归一化。
2. $(2+1)/(10+5)=3/15=0.2$。
3. 不等于，它只是在给定类别条件下作的分解假设。
4. 词表与可能的文档频率使用了测试分布信息，正确做法是仅在训练部分拟合特征提取。

## 复习卡片与下一步

复习卡：后验来自先验与似然；平滑同时改分子分母；长文本用对数；词表属于训练过程。下一篇用几何方向与间隔来建立分类边界。

[上一篇 机器学习流程与 KNN](/notes/mathematical-modeling-ml-sklearn-knn/) · [下一篇 线性判别与支持向量机](/notes/mathematical-modeling-ml-lda-svm/)

## 参考资料

- [Datawhale《数学建模导论》第9章](https://datawhalechina.github.io/intro-mathmodel/#/CH9/第九章-机器学习与统计模型)：用于建立知识目录，建议与本篇的手算和案例对照阅读。
- [scikit-learn 朴素贝叶斯](https://scikit-learn.org/stable/modules/naive_bayes.html)。
- [文本特征提取](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)。
