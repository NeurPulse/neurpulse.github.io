---
title: '数学建模：朴素贝叶斯'
description: '朴素贝叶斯分类器的原理与应用'
pubDate: 2026-08-07
updatedDate: 2026-10-19
order: 10
category: '数学建模'
tags: ['朴素贝叶斯','分类','机器学习']
---

### [朴素贝叶斯算法](https://blog.csdn.net/phdsky/article/details/95762993)

> 贝叶斯公式：用来描述两个条件概率之间的关系，比如P(A|B)和P(B|A)。它基于贝叶斯定理，该定理指出在已知先验概率的情况下，可以通过新的证据来更新事件的概率。

$$
P(B|A) = \frac{P(A|B)P(B)}{P(A)}
$$

> 引入：
>
> 内容的垃圾邮件识别：
>
> 基于行为的垃圾邮件识别
>
> 基于白名单的垃圾邮件识别
>
> 基于内容的垃圾邮件识别要用到贝叶斯公式
>
> - 具体来说，垃圾邮件分类的任务是根据邮件的特征（如关键词、发件人、主题等）判断该邮件是否为垃圾邮件。首先，我们需要统计训练集中各类别（垃圾邮件、正常邮件）以及各个特征的出现概率。然后，对于待分类的邮件，我们根据其特征计算该邮件属于各个类别的概率。最后，将该邮件归为概率最大的类别。
> - 贝叶斯公式在垃圾邮件分类中的作用原理在于利用条件概率来表示给定特征下各个类别的概率。具体来说，我们使用贝叶斯公式计算给定特征下各个类别的概率，即$P(C|X)$​，其中C表示类别（垃圾邮件或正常邮件），X表示特征（如关键词、发件人、主题等）。在垃圾邮件分类中，贝叶斯公式通常用于实现朴素贝叶斯分类器。该分类器基于贝叶斯定理，通过计算给定特征下各个类别的概率，将待分类的邮件归为概率最大的类别。
> - 朴素贝叶斯算法是一种基于贝叶斯定理的分类方法。它假设给定类别后，各个特征之间条件独立，通过计算每个类别在给定特征下的概率，将待分类的样本归为概率最大的类别。朴素贝叶斯算法在处理文本分类、垃圾邮件过滤等任务中表现出色，并且具有简单、高效的特点。它的核心思想是根据已知的训练数据集，为每个类别计算出特征条件独立的概率，然后利用这些概率来预测新样本的类别。
>
> 朴素贝叶斯算法处理垃圾邮件的基本流程
>
> 1. **特征提取**：从邮件数据集中提取出有意义的特征，通常采用 TF-IDF（词频 - 逆文档频率）方法进行特征提取。
> 2. **训练模型**：将提取出的特征和对应的类别进行训练，计算出每个特征在不同类别下的条件概率。
> 3. **分类**：对未知样本进行分类，根据已知的特征和对应的条件概率计算出每个类别的概率，将样本归为概率最大的类别。

拉普拉斯平滑引入：

若词在某个类别的训练文本中从未出现，最大似然估计会给出零概率，连乘后可能把整个类别分数变为零。多项式朴素贝叶斯常用加性平滑：

$$
P(w_j\mid c)=\frac{N_{cj}+\alpha}{N_c+\alpha V},\qquad\alpha>0.
$$

$N_{cj}$ 是该类中词 $j$ 的计数，$N_c$ 是该类所有词的总计数，$V$ 为词表大小。$\alpha=1$ 称为 Laplace 平滑，其他正数称为 Lidstone 平滑；分子和分母必须一起调整以保持归一化。它主要避免零似然，并不保证任何数据上都更准确。

下列手写示例处理分类特征，要求每一列预先编码为 0、1、2，类别为 0、1；不是通用文本分类器。lambda_=3 是加性平滑，不是严格的加一平滑。

```python
import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import Binarizer

X_train = np.array([[0, 0], [0, 1], [0, 1], [0, 0], [0, 0],

                          [1, 0], [1, 1], [1, 1], [1, 2], [1, 2],

                          [2, 2], [2, 1], [2, 1], [2, 2], [2, 2]])

y_train = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0])

X_test = np.array([[1, 0]])
```

```python
N = len(y_train)

K = len(np.unique(y_train))

S = len(np.unique(X_train[:,0])) # 特征取值

D = X_train.shape[1]  # 维度

n = len(X_test)

d = X_test.shape[1]

prior = np.zeros(K)

condition = np.zeros((K, D, S)) #条件概率

lambda_=3
```

```python
# 朴素贝叶斯训练

def trainNB(X_train, y_train):

    prior = np.zeros(K)
    condition = np.zeros((K, D, S))
    for i in range(0, len(y_train)):

        prior[y_train[i]] += 1

        for j in range(0, D):

            condition[y_train[i]][j][X_train[i][j]] += 1

    prior_probability = (prior + lambda_) / (N + K*lambda_) # 拉普拉斯平滑

    return prior_probability, condition  

def predictNB(prior_probability, condition, X_test):

    predict_label = -1 * np.ones(n)

    for i in range(0, n):

        predict_probability = np.ones(K)

        to_predict = X_test[i]

        for j in range(0, K):

            prior_prob = prior_probability[j]

            for k in range(0, d):

                conditional_probability = (condition[j][k][to_predict[k]] + lambda_) / (sum(condition[j][k]) + S*lambda_)

                predict_probability[j] *= conditional_probability

            predict_probability[j] *= prior_prob

        predict_label[i] = np.argmax(predict_probability)

        print("Sample %d predicted as %d" % (i, predict_label[i]))

    return predict_label
```

```python
print("Start naive bayes training...")   

prior, conditional = trainNB(X_train=X_train, y_train=y_train)

print("Testing on %d samples..." % len(X_test))

predictNB(prior_probability=prior,condition=conditional,X_test=X_test)
```

### 参考资料

- [scikit-learn：朴素贝叶斯](https://scikit-learn.org/stable/modules/naive_bayes.html)：条件独立假设和加性平滑。
