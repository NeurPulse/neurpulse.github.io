---
title: '数学建模：朴素贝叶斯'
description: '朴素贝叶斯分类器的原理与应用'
pubDate: 2026-08-07
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
> - 朴素贝叶斯算法是一种基于贝叶斯定理的分类方法。它假设各个特征之间相互独立，通过计算每个类别在给定特征下的概率，将待分类的样本归为概率最大的类别。朴素贝叶斯算法在处理文本分类、垃圾邮件过滤等任务中表现出色，并且具有简单、高效的特点。它的核心思想是根据已知的训练数据集，为每个类别计算出特征条件独立的概率，然后利用这些概率来预测新样本的类别。
>
> 朴素贝叶斯算法处理垃圾邮件的基本流程
>
> 1. **特征提取**：从邮件数据集中提取出有意义的特征，通常采用 TF-IDF（词频 - 逆文档频率）方法进行特征提取。
> 2. **训练模型**：将提取出的特征和对应的类别进行训练，计算出每个特征在不同类别下的条件概率。
> 3. **分类**：对未知样本进行分类，根据已知的特征和对应的条件概率计算出每个类别的概率，将样本归为概率最大的类别。

拉普拉斯平滑引入：

假设我们正在使用朴素贝叶斯分类器对一组文本进行分类，其中一个类别是“正面情感”。我们的训练数据集中包含了一些正面的文本和负面的文本，但是正面的文本数量较少。在训练过程中，我们发现一个常见的词语“好”在正面的文本中出现了很多次，但在负面的文本中只出现了一次。根据朴素贝叶斯分类器的原理，每个词语的出现概率是独立的，因此在计算正面情感类别的条件概率时，我们不能简单地认为“好”这个词语的出现概率是1（在正面的文本中）和0（在负面的文本中）。这是因为负面的文本中“好”这个词出现的概率虽然很小，但并不为0。

为了解决这个问题，我们可以使用拉普拉斯平滑方法。拉普拉斯平滑的核心思想是在计算概率时，给每个事件添加一个小的常数，以避免出现零概率的情况。具体来说，我们可以给“好”这个词的出现概率增加一个小的常数（例如0.01），这样在计算正面情感类别的条件概率时，就不会出现分母为0的情况。通过引入拉普拉斯平滑，我们可以更准确地计算每个词语的出现概率，从而使得朴素贝叶斯分类器的分类效果更好。这是因为拉普拉斯平滑方法能够处理训练数据中未出现的事件，避免将它们的概率估计为0。

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

    for i in range(0, N):

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





