---
title: '数学建模：树模型与集成学习'
description: '决策树、随机森林、XGBoost等集成方法'
pubDate: 2026-08-07
category: '数学建模'
tags: ['决策树','随机森林','XGBoost','集成学习','机器学习']
---

### 树型结构模型

基本概念：

在决策树的生成中，我们通过信息论里面几个数值的计算判断分类：熵，信息增益，增益率，基尼指数。

1. 信息熵：

   熵是衡量数据集中类别混乱程度的一个指标。在决策树中，我们使用熵来评估一个节点中的数据分布情况。如果一个节点中的数据大部分都属于同一个类别，那么这个节点的熵值就比较低，意味着这个节点的分类比较明确；反之，如果数据分布比较均匀，则熵值较高，表示这个节点的分类比较模糊。通过计算熵，我们可以确定在哪个节点进行分裂，以最大化分类的准确性。如果对于样本集合S，一共可以划分为k个类，每个类概率是$p_k$，那么信息熵定义为：
   $$
   E(S) = -\sum_{i=1}^{k} p_k \log p_k
   $$
   

   如果对数据集S按照某一个属性A对S进行划分，将它划分成v个子集，定义属性A的信息熵为：
   $$
   E(S, A) = \sum_{i=1}^{v} \frac{|A_i|}{|S|} E(A_i)
   $$
   
2. 信息增益：

   数据集本身的k个类是按照最后的返回标签排的，所以数据集本身的信息熵与属性的信息熵并不是一个东西。信息增益就是这个差值，看看按照这个属性划分我的信息到底增长了多少。信息增益用于衡量某个属性对于分类的贡献程度。在决策树中，我们选择信息增益最大的属性作为当前节点的分裂属性，这样可以使得子节点的数据更加集中，从而提高分类的准确性。
   $$
   Gain(A) = E(S) - E(S, A)
   $$
   
3. 增益率：

   增益与信息熵的比值，它解决了信息增益偏向于选择具有多个值的属性问题。增益率结合了信息增益和属性熵，使得决策树在选择分裂属性时既考虑了分类的纯度，又考虑了属性的分散程度。
   $$
   GainRatio(A) = \frac{Gain(A)}{E(S, A)}
   $$
   
4. 基尼指数：

   基尼指数另一种衡量数据纯度的方式。它基于一个假设，即如果一个样本被错误分类，则该样本的基尼指数增加。因此，通过计算基尼指数，我们可以确定数据集的纯度，并选择能够最小化基尼指数的属性进行分裂。假设数据集有n个类，第k类的概率是pk，定义基尼指数：
   $$
   GINI(S) = 1 - \sum_{i=1}^{n} p_k^2
   $$
   

#### ID3决策树

> ID3决策树能够处理自变量和标签都是离散型的分类问题。它是以信息熵和信息增益度为衡量标准，从而实现对数据的归纳分类。它是在已知各种情况发生概率的基础上，通过构成决策树来求取净现值的期望值大于等于零的概率，评价项目风险，判断其可行性的决策分析方法。

训练步骤:

1. 创建根节点，确定属性是什么。
2. 若全部样本都是一类，那就全部落在叶子结点上。否则，根据每个属性计算信息增益，根据最大的信息增益确定划分属性与划分原则。
3. 根据划分属性把属性值不同的样本划到对应边上。
4. 根据不同属性的分类准则递归生成决策树。

##### case1：

用一个非常简单的案例实现ID3决策树。由于它只能处理离散自变量与离散标签的问题，这里选用的案例也比较简单，是一份贷款申请成功表，包含四个自变量：年龄段（青年、中年、老年），有工作（是、否），有自己的房子（是、否）和信贷情况（一般、好、非常好），最终标签是是否贷款成功。

```python
import numpy as np

dataSet = [[0, 0, 0, 0, 'no'],         #数据集

                      [0, 0, 0, 1, 'no'],

                      [0, 1, 0, 1, 'yes'],

                      [0, 1, 1, 0, 'yes'],

                      [0, 0, 0, 0, 'no'],

                      [1, 0, 0, 0, 'no'],

                      [1, 0, 0, 1, 'no'],

                      [1, 1, 1, 1, 'yes'],               

                      [1, 0, 1, 2, 'yes'],

                      [1, 0, 1, 2, 'yes'],

                      [2, 0, 1, 2, 'yes'],

                      [2, 0, 1, 1, 'yes'],

                      [2, 1, 0, 1, 'yes'],

                      [2, 1, 0, 2, 'yes'],

                      [2, 0, 0, 0, 'no']]

dataSet=np.array(dataSet)

labels = ['年龄', '有工作', '有自己的房子', '信贷情况']        #分类属性

# 信息熵
def EntropyData(dataset):

    n = len(dataset)                        #返回数据集的行数

    dataset=dataset[:,-1]

    count = np.unique(dataset, return_counts=True)[1]

    ent = -np.sum([c/n * np.log2(c/n + 0.00001) for c in count]) # 防止出现log0

    return ent


def maxcount(y):

    y,c = np.unique(y,return_counts=True)

    return y[c==max(c)]

def splitdata(dataset, f, value):

    dataset = dataset[dataset[:, f] == value, :]       

    retDataSet=np.delete(dataset,f,1)

    return retDataSet                                   #返回划分后的数据集


def infoGain(fList, i,dataset):

    baseEntropy = EntropyData(dataset)                 #计算数据集的香农熵

    newEntropy = 0.0                                   #经验条件熵

    for value in fList:                           #计算信息增益

        subDataSet = splitdata(dataset, i, value)           #subDataSet划分后的子集

        prob = len(subDataSet) / float(len(dataset))           #计算子集的概率

        newEntropy += prob * EntropyData(subDataSet)        #根据公式计算经验条件熵

    infoGain = baseEntropy - newEntropy                        #信息增益

    return infoGain


def choose(dataset):

    numFeatures = len(dataset[0]) - 1                     #特征数量  

    bestInfoGain = 0.0                                    #信息增益

    bestFeature = -1                                      #最优特征的索引值

    for i in range(numFeatures):                          #遍历所有特征

        #获取dataSet的第i个所有特征

        featList = [example[i] for example in dataset]

        uniqueVals = set(featList)                         #创建set集合{},元素不可重复

        iGain = infoGain(uniqueVals,i,dataset)                        #信息增益

        print("第%d个特征的增益为%.3f" % (i, iGain))             #打印每个特征的信息增益

        if (iGain > bestInfoGain):                              #计算信息增益

            bestInfoGain = iGain                               #更新信息增益，找到最大的信息增益

            bestFeature = i                                      #记录信息增益最大的特征的索引值

    return bestFeature                                           #返回信息增益最大的特征的索引值

def createID3(dataSet, labels, featLabels):

    classList = [example[-1] for example in dataSet]              #取分类标签(是否放贷:yes or no)

    if classList.count(classList[0]) == len(classList):        #如果类别完全相同则停止继续划分

        return classList[0]

    if len(dataSet[0]) == 1:                           #遍历完所有特征时返回出现次数最多的类标签

        return maxcount(classList)

    bestFeat = choose(dataSet)                   #选择最优特征

    bestFeatLabel = labels[bestFeat]                               #最优特征的标签

    featLabels.append(bestFeatLabel)

    myTree = {bestFeatLabel:{}}                                    #根据最优特征的标签生成树

    del(labels[bestFeat])                                          #删除已经使用特征标签

    featValues = [example[bestFeat] for example in dataSet]  #得到训练集中所有最优特征的属性值

    uniqueVals = set(featValues)                                   #去掉重复的属性值

    for value in uniqueVals:

        subLabels=labels[:]

        #递归调用函数createTree(),遍历特征，创建决策树。

        myTree[bestFeatLabel][value] = createID3(splitdata(dataSet, bestFeat, value), subLabels, featLabels)

    return myTree


featLabels = []

myTree = createID3(dataSet, labels, featLabels)

print(myTree)


```

#### C4.5决策树

> 为了处理连续属性的自变量和缺失值问题，对ID3决策树进行改进就得到了C4.5决策树，后来又诞生了C5.0决策树。C4.5决策树在每个节点处都会选择一个最佳的属性进行分支，选择的标准通常是信息增益或增益率。信息增益代表信息不确定性较少的程度，信息增益越大，说明不确定性降低的越多，因此该特征对分类来说越重要。C4.5通过阈值自动把连续变量分成两部分来处理，在数据特性和基本结构上具有很强的灵活性和泛化能力，能够处理各种类型的数据，并构建出准确度高的分类模型。

步骤包括:

1. 将连续数值离散化，创建树。
2. 确定连续属性的阈值，计算信息增益率，确定划分属性。
3. 根据划分属性把属性值不同的样本划到对应边上。
4. 根据不同属性的分类准则递归生成树。

```python
import numpy as np

dataSet = [[0, 0, 0, 0, 'no'],
           [0, 0, 0, 1, 'no'],
           [0, 1, 0, 1, 'yes'],
           [0, 1, 1, 0, 'yes'],
           [0, 0, 0, 0, 'no'],
           [1, 0, 0, 0, 'no'],
           [1, 0, 0, 1, 'no'],
           [1, 1, 1, 1, 'yes'],
           [1, 0, 1, 2, 'yes'],
           [1, 0, 1, 2, 'yes'],
           [2, 0, 1, 2, 'yes'],
           [2, 0, 1, 1, 'yes'],
           [2, 1, 0, 1, 'yes'],
           [2, 1, 0, 2, 'yes'],
           [2, 0, 0, 0, 'no']]

dataSet = np.array(dataSet)
labels = ['年龄', '有工作', '有自己的房子', '信贷情况']

# 信息熵
def EntropyData(dataset):
    n = len(dataset)
    dataset = dataset[:, -1]
    count = np.unique(dataset, return_counts=True)[1]
    ent = -np.sum([c/n * np.log2(c/n + 0.00001) for c in count])
    return ent

def maxcount(y):
    y, c = np.unique(y, return_counts=True)
    return y[c == max(c)][0]  # 修复返回数组问题

# 离散特征划分
def splitdata(dataset, f, value):
    dataset = dataset[dataset[:, f] == value, :]
    retDataSet = np.delete(dataset, f, 1)
    return retDataSet

# 连续特征划分函数 splitdata_C4_5
def splitdata_C4_5(dataset, f, value):
    dataset = dataset[dataset[:, f] <= value, :]
    retDataSet = np.delete(dataset, f, 1)
    return retDataSet

# 信息增益
def infoGain(fList, i, dataset):
    baseEntropy = EntropyData(dataset)
    newEntropy = 0.0
    for value in fList:
        subDataSet = splitdata(dataset, i, value)
        prob = len(subDataSet) / float(len(dataset))
        newEntropy += prob * EntropyData(subDataSet)
    infoGainVal = baseEntropy - newEntropy
    return infoGainVal

# 信息增益率函数
def infoGain_rate(fList, i, dataset):
    H = EntropyData(dataset)
    IG = infoGain(fList, i, dataset)
    # 防止分母接近0
    if H < 1e-6:
        return 0
    return IG / H

# 选择最优特征：依据信息增益率（C4.5）
def choose_C45(dataset):
    numFeatures = len(dataset[0]) - 1
    bestGainRate = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataset]
        uniqueVals = set(featList)
        gainRate = infoGain_rate(uniqueVals, i, dataset)
        print("第%d个特征的信息增益率为%.3f" % (i, gainRate))
        if gainRate > bestGainRate:
            bestGainRate = gainRate
            bestFeature = i
    return bestFeature

# C4.5 递归建树
def createC45(dataSet, labels, featLabels):
    classList = [example[-1] for example in dataSet]
    # 类别全部相同，返回标签
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 没有剩余特征，返回多数类
    if len(dataSet[0]) == 1:
        return maxcount(classList)

    bestFeat = choose_C45(dataSet)
    bestFeatLabel = labels[bestFeat]
    featLabels.append(bestFeatLabel)
    myTree = {bestFeatLabel: {}}
    del(labels[bestFeat])

    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createC45(splitdata(dataSet, bestFeat, value), subLabels, featLabels)
    return myTree


featLabels = []
myTree = createC45(dataSet, labels, featLabels)
print("\nC4.5决策树结果：")
print(myTree)
```



#### [CART决策树](https://blog.csdn.net/vegios/article/details/108509405?ops_request_misc=elastic_search_misc&request_id=2df3eef8e5662a4674f1d8c51e8db6b2&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~ElasticSearch~search_v2-12-108509405-null-null.142^v102^pc_search_result_base6&utm_term=CART&spm=1018.2226.3001.4187)

> CART（Classification and Regression Trees）决策树算法在先前的工作基础上进行了一些重要的改进，使其在处理各种数据和问题时更加有效。顾名思义，CART决策树算法是能够处理回归问题的一种树算法。它适用于各种类型的数据和问题，尤其是处理大规模数据集和多变量输入的情况。CART算法适用于各种类型的数据和问题，包括分类和回归问题，以及连续属性和缺失值的处理，并且有很好的可解释性。Python的sklearn中集成的默认树模型是基于CART结构的。

步骤包括:

1. 将连续数值离散化，创建树。
2. 确定连续属性的阈值，计算基尼指数增长，确定划分属性，按最小者开始划分，注意要二分。
3. 根据划分属性把属性值不同的样本划到对应边上。
4. 根据不同属性递归计算基尼指数增长并划分。

注：CART决策树在处理回归问题中的基尼指数为：
$$
GainGINI = \sqrt{y_{k1}  - \mu_1 } + \sqrt{y_{k2}  - \mu_2} 
$$


相关知识：

> 剪枝。决策树生成中需要剪枝的原因主要是为了防止过拟合。通过剪枝，可以去除决策树中过于复杂的部分，使其在测试数据或实际应用中表现更好。
>
> 剪枝包括预剪枝和后剪枝两种方法。
>
> 预剪枝是指在决策树生成过程中提前停止树的生长，以防止过拟合，可以及早停止树的生长，从而减少过拟合的风险。然而，预剪枝可能会过早地停止树的生长，导致丢失一些重要的特征和样本。
>
> 后剪枝则是在决策树生成完成后对其进行简化，以去除不必要或冗余的节点，可以保留更多的特征和样本，但可能会导致过拟合。
>
> 如果数据集较小，可以采用预剪枝方法，以减少过拟合的风险。如果数据集较大，特征不重要或者存在很多噪声，可以采用后剪枝方法，以保留更多的特征和样本。另外，也可以结合使用预剪枝和后剪枝，先进行预剪枝再进行后剪枝，以获得更好的剪枝效果。

```python

# encoding=utf-8
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.tree import DecisionTreeRegressor
# 准备数据集
boston=fetch_california_housing()
# 探索数据
print(boston.feature_names)
# 获取特征集和房价
features = boston.data
prices = boston.target
# 随机抽取33%的数据作为测试集，其余为训练集
train_features, test_features, train_price, test_price = train_test_split(features, prices, test_size=0.33)
# 创建CART回归树
dtr=DecisionTreeRegressor()
# 拟合构造CART回归树
dtr.fit(train_features, train_price)
# 预测测试集中的房价
predict_price = dtr.predict(test_features)
# 测试集的结果评价
print('回归树二乘偏差均值:', mean_squared_error(test_price, predict_price))
print('回归树绝对值偏差均值:', mean_absolute_error(test_price, predict_price)) 

```





### 集成学习

> 集成学习(Ensemble learning)通过构建并结合多个学习器来完成学习任务，有时也被称为多分类器系统。集成学习的一般结构为：先产生一组“个体学习器”，再用某种策略将它们结合起来。集成中只包含同种类型的个体学习器，称为同质，当中的个体学习器亦称为“基学习器”，相应的算法称为“基学习算法”。集成中包含不同类型的个体学习器，称为“异质”，当中的个体学习器称为“组建学习器”。
>
> 集成学习是一种机器学习方法，通过结合多个学习器的预测结果来提高整体的预测精度。它通常包含三个步骤：训练多个基学习器，组合这些学习器的预测结果，以及评估整个集成系统的性能。基学习器是构成集成学习的“基础个体”。比如，你有一组数据，你想用这组数据预测一个结果。基学习器就是那些你用来做预测的“个体”模型，比如决策树等。集成就是将多个基学习器的预测结果结合起来，形成一个更强大的“集体”。这个“集体”的预测结果往往比任何一个基学习器都要好。通过集成这些基学习器，集成学习方法可以在一定程度上降低模型过拟合的风险，提高模型的泛化能力。
>
> 在集成学习中，常见的[算法](https://blog.csdn.net/albert201605/article/details/147553840?ops_request_misc=&request_id=&biz_id=102&utm_term=Bagging&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-3-147553840.142^v102^pc_search_result_base6&spm=1018.2226.3001.4187)包括[bagging](https://blog.csdn.net/qq_61260911/article/details/130249924?ops_request_misc=elastic_search_misc&request_id=01b36e4b24dfb14a5d1a8b753140a68c&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-130249924-null-null.142^v102^pc_search_result_base6&utm_term=Bagging&spm=1018.2226.3001.4187)、boosting和stacking等。Bagging是一种基于自助采样法的集成学习技术，通过从原始数据集中有放回地随机抽取样本构建多个子数据集，并分别训练基模型。Boosting是一种基于加权平均思想的集成方法，通过为不同的基学习器分配不同的权重来组合它们的预测结果。Stacking是一种分层集成方法，通过将多个基学习器的预测结果作为新的特征输入到另一层模型中进行训练。

#### Boosting

> Boosting是一种特殊的集成方法。它首先训练一个基学习器，然后用这个学习器的错误来调整数据，再训练下一个学习器。这样，每个后续的基学习器都会特别关注那些第一个学习器搞错的地方。Boosting就像一个老师，它不断给学生们出更难的题目，直到学生们都掌握知识为止。
>
> boosting方法的基学习器必须得是一个一个生成，后一个在前一个的基础上生成，故又名串行生成。它的工作机制是初始给每个样本赋予一个均衡的权重值1/m，然后训练第一个弱学习器，根据该弱学习器的学习误差率来更新权重值，使得该学习器中的误差率高的训练样本的权重值变高。这就好像一场考试后学霸再进行学习，即使他是绩点满分的神仙它也不会在意做对了哪些，只会盯着自己究竟还有哪些错误并不断改进。以此方法来依次学习各个弱学习器，直到弱学习器的数量达到预先指定的值为止，最后通过某种策略将这些弱学习器进行整合，得到最终的强学习器。
>
> 关注的三特征：
>
> 1. 关注错误：当我们有一个分类问题，Boosting方法并不会忽略那些被第一个分类器错误分类的样本。相反，它会特别关注这些“困难”的样本，并给予它们更大的权重。这样，在下一次迭代中，这些样本会被给予更多的关注，增加了被正确分类的机会。
> 2. 迭代训练：Boosting方法不是一个单一的学习器，而是一系列学习器的集合。每个学习器在训练时都考虑到了前一个学习器的错误。这意味着，每个学习器都是在上一个学习器的基础上进行的训练。
> 3. 加权投票：不同于普通的投票系统，Boosting中的每个分类器都有不同的权重。这个权重是基于它在前一轮的表现来决定的。表现得越好，权重越大；反之，则权重越小。最后的预测结果是所有分类器的加权平均，而不是简单的多数投票。

##### [AdaBoost](https://zhuanlan.zhihu.com/p/343196025)

> AdaBoost算法是Boosting方法中的一种经典算法，主要用于解决二分类问题。该算法通过调整样本权重来实现对样本分布的调整，即提高前一轮个体学习器错误分类的样本权重，降低正确分类的样本权重。这样可以使错误分类的样本在下一轮中受到更多关注，从而在下一轮中被正确分类，增加分类的准确性。对于组合方式，AdaBoost采用加权多数表决的方法，具体地，加大分类误差率小的弱分类器的权值，减小分类误差率大的弱分类器的权值，从而调整它们在表决中的作用。
>
> Adaboost算法基本原理就是将多个弱分类器（决策树桩或逻辑回归等）进行合理的结合，使其成为一个强分类器。它每次迭代只训练一个弱分类器，训练好的弱分类器将参与下一次迭代的使用。第N个弱分类器更可能分对前N-1个弱分类器没分对的数据，最终分类输出要看这N个分类器的综合效果。

[Adaboost](https://blog.csdn.net/yaoyao2024/article/details/138467790?ops_request_misc=elastic_search_misc&request_id=5fc21b222b2354201e2da4aac35b69a6&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-138467790-null-null.142^v102^pc_search_result_base6&utm_term=adaboost&spm=1018.2226.3001.4187)的算法流程:

Step1：初始化训练数据的权值分布 每一个训练样本最开始时都被赋予相同的权值：$w_i=\frac{1}{N}$，这样得到训练样本集的初始权值分布。
$$
D_1(i)=(w_1,w_2,\dots,w_N)=\left(\frac{1}{N},\dots,\frac{1}{N}\right) $
$$
Step2：进行多轮迭代 用 $t = 1,2,\dots,T$ 表示迭代的第几轮

1. 选取一个当前误差率最低的弱分类器 $h$ 作为第 $t$ 个基本分类器 $H_t$，并计算该弱分类器在分布 $D_t$ 上的误差率 $e_t$：
   $$
   e_t = P(H_t(x_i)\neq y_i)=\sum_{i=1}^N w_{ti}I(H_t(x_i)\neq y_i) $
   $$

   > 由上述式子可知，$H_t(x)$ 在训练数据上的误差率 $e_t$ 就是被 $H_t(x)$ 误分类样本的权值之和。

2. 计算本次迭代的基本分类器 $H_t(x)$ 的系数，即该弱分类器在最终分类器所占权重： 
   $$
    \alpha_t = \frac{1}{2}\ln\left(\frac{1-e_t}{e_t}\right) 
   $$
   由上述式子可知，分类误差率越小的基本分类器在最终分类器中的作用越大。

3. 更新训练数据集的权值分布（目的：得到样本的新的权值分布），用于下一轮迭代
   $$
   D_{t+1}(i)=\frac{D_t(i)\exp\left(-\alpha_t y_i H_t(x_i)\right)}{Z_t}
   $$
   

     其中 $Z_t$ 为归一化常数： $$ Z_t = 2\sqrt{e_t(1-e_t)} $$ 

   

 Step3：组合各个弱分类器 $H_m\ (m\in 1\sim M)$ 可以看到其实就是将每轮得到的弱分类器乘以其权重系数叠加
$$
f(x)=\sum_{t=1}^T \alpha_t H_t(x)
$$
通过符号函数 $\text{sign}$​ 的作用，从而得到最终分类器，如下：
$$
H_{final}=\text{sign}(f(x))=\text{sign}\left(\sum_{t=1}^T \alpha_t H_t(x)\right)
$$
基本分类器的个数的确定：基本分类器的个数，也就是迭代次数，通常是一个超参数，需要通过交叉验证或其他模型选择技术来确定。在训练过程中，随着基本分类器数量的增加，整体模型的性能通常会提高，但也可能出现过拟合的情况。因此，需要找到一个平衡点，使得模型在验证集上有最佳的性能。过多的基本分类器不仅会导致过拟合，还会增加模型的计算复杂度。

基本分类器的具体形式的确定：AdaBoost算法可以与多种类型的弱学习器结合使用，但最常用的是决策树，尤其是一层的决策树（也称为决策树桩）。选择哪种类型的弱学习器通常取决于问题本身的特性。例如，对于文本数据，可能选择朴素贝叶斯分类器作为基本分类器；而对于有大量数值型特征的数据，使用决策树桩可能更合适。基本分类器的选择也是一个超参数，需要基于实验结果来选择。在实际应用中，会通过试验不同的基本分类器类型，然后选择在验证集上性能最好的一种。

在 AdaBoost 算法中，弱学习器通常被选取为**容易训练且性能稍差**的分类器，它们通常会随着训练过程进行改变。这是因为每次训练弱学习器时，都会根据上一轮迭代中错误的样本调整数据集的权重，使得**新的弱学习器更专注于之前分类错误的样本**。常见的弱学习器选择有：

1. 决策树桩（Decision Stumps）：一种非常简单的一棵决策树，通常只包含一个特征的判断。

2. 感知机（Perceptrons）：线性分类器，适用于简单的线性可分问题。

3. 核机（Nueral Networks）：简单的神经网络结构，例如单层感知器。

4. 朴素贝叶斯分类器（Naive Bayes Classifiers）：基于概率的简单分类器，适用于文本分类等任务。

5. K-近邻分类器（K-Nearest Neighbors）：一种非参数方法，根据数据点之间的距离来分类。

6. 逻辑回归（Logistic Regression）：虽然是常规的分类器，但其模型的正则化和限制可以在某些情况下被视作弱学习器。

7. 支持向量机（Support Vector Machines）：尽管本身是一个强分类器，但是如果适当设置参数（如不使用核技巧或降低正则化），可以降低其性能。

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 创造一些模拟的二分类数据集
X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=0,
                           random_state=42, n_clusters_per_class=1)

# 将数据集分割成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建AdaBoost模型实例
ada_clf = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1), 
    n_estimators=200,
    learning_rate=0.5, 
    random_state=42
)

# 训练AdaBoost模型
ada_clf.fit(X_train, y_train)

# 进行预测
y_pred = ada_clf.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2f}")
```

```python
import numpy as np

def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    """
    对数据进行分类的单层决策树分类函数。
    Args:
      dataMatrix: 数据矩阵
      dimen: 需要考虑的特征的维度
      threshVal: 阈值
      threshIneq: 不等式，可以是'lt'（小于）或'gt'（大于）

    Returns:
      retArray: 分类结果
    """
    retArray = np.ones((np.shape(dataMatrix)[0], 1))  # 默认所有样本分类结果为1
    # 根据阈值和不等式标记分类结果
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray

def buildStump(dataArr, classLabels, D):
    """
    在加权数据集中找到最佳的单层决策树。
    Args:
      dataArr: 数据集
      classLabels: 类别标签
      D: 数据点的权重

    Returns:
      bestStump: 最佳的单层决策树信息
      minError: 最小的错误率（标量）
      bestClasEst: 最佳的分类结果
    """
    dataMatrix = np.asmatrix(dataArr)
    labelMat = np.asmatrix(classLabels).T
    m, n = np.shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClasEst = np.asmatrix(np.zeros((m, 1)))
    minError = np.inf
    for i in range(n):
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1, int(numSteps) + 1):
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = np.asmatrix(np.ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError.item(), bestClasEst  # ✅ 取出标量！

def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    """
    使用AdaBoost算法训练模型。
    Args:
      dataArr: 数据集
      classLabels: 类别标签
      numIt: 迭代次数

    Returns:
      weakClassArr: 训练好的弱分类器数组
    """
    weakClassArr = []
    m = np.shape(dataArr)[0]
    D = np.asmatrix(np.ones((m, 1)) / m)
    aggClassEst = np.asmatrix(np.zeros((m, 1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        # error现在是普通float标量，不会报错
        alpha = 0.5 * np.log((1.0 - error) / max(error, 1e-16))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        # 更新样本权重D
        expon = np.multiply(-1 * alpha * np.asmatrix(classLabels).T, classEst)
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()
        # 累计预测值，计算训练误差
        aggClassEst += alpha * classEst
        aggErrors = np.multiply(np.sign(aggClassEst) != np.asmatrix(classLabels).T, np.ones((m, 1)))
        errorRate = aggErrors.sum() / m
        if errorRate == 0.0:
            break
    return weakClassArr

def adaClassify(datToClass, classifierArr):
    """
    使用AdaBoost算法对数据进行分类。
    Args:
      datToClass: 待分类的数据
      classifierArr: 训练好的弱分类器数组

    Returns:
      分类结果
    """
    dataMatrix = np.asmatrix(datToClass)
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.asmatrix(np.zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'],
                                 classifierArr[i]['thresh'],
                                 classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
    return np.sign(aggClassEst)


# 测试数据
dataMat = np.array([
    [1.0, 2.1],
    [2.0, 1.1],
    [1.3, 1.0],
    [1.0, 1.0],
    [2.0, 1.0]
])
classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]

classifierArray = adaBoostTrainDS(dataMat, classLabels, 9)
prediction = adaClassify(dataMat, classifierArray)

print("Predictions:\n", prediction)
print("Actual labels:", classLabels)
```



#### Bagging

> Bagging是另一种集成学习技术，与Boosting不同，它通过从原始数据集中随机抽取样本（有放回地）来训练基学习器。这种方式能够降低模型对数据集的过度拟合，提高模型的泛化能力。Bagging通过引入随机性来降低各个基学习器之间的相关性，从而获得更好的集成效果。

特点：

1. 自助采样：Bagging采用有放回的自助采样方法从原始数据集中随机抽取样本，每个样本被选中的概率相等。这样每个基学习器都从不同的数据子集中进行训练，降低了它们之间的相关性。
2. 简单平均：Bagging通过简单平均的方式将多个基学习器的预测结果结合起来，形成一个更稳定的预测结果。与Boosting不同，Bagging不会对每个基学习器的预测结果赋予不同的权重，而是平等对待它们。
3. 降低方差：Bagging通过降低基学习器之间的相关性来降低集成学习模型的方差，从而提高泛化性能。这使得Bagging对于噪声数据和异常值具有一定的鲁棒性。

##### 随机森林

[随机森林](https://blog.csdn.net/m0_61404163/article/details/145799914?ops_request_misc=elastic_search_misc&request_id=2b5550150b2ded43dcb9ea955f7a78a7&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_click~default-2-145799914-null-null.142^v102^pc_search_result_base6&utm_term=%E9%9A%8F%E6%9C%BA%E6%A3%AE%E6%9E%97&spm=1018.2226.3001.4187)是Bagging最有代表性的算法。随机森林的名称中有两个关键词，一个是“随机”，一个就是“森林”。“森林”我们很好理解，一棵叫做树，那么成百上千棵就可以叫做森林了，这样的比喻还是很贴切的，其实这也是随机森林的主要思想——集成思想的体现。那随机是什么？就是在生成决策树的时候会随机选择一些属性。一般的决策树在选择划分属性时是在当前结点的所有d属性中选择一个最优属性，而在随机森林的基学习器上不一定用完全部的属性而是抽样抽出一部分k，然后从该属性集中选择最优的划分属性。

优势：

1. 能够处理很高维度的数据。
2. 在训练完成后，可以给出哪些属性比较重要，这一方法也被用于自动化特征工程。
3. Bagging是一种并行化方法，训练速度快。
4. 方便进行可视化展示，便于后续分析。

过程：

1. 从样本集里面Bootstrap采集n个样本。
2. 在树的每个节点上，从所有属性中随机选择k个属性，选择出一个最佳分割属性作为节点,建立决策树，一般,d表示属性的总个数。
3. 重复以上两步m次，建立m棵决策树。重复的过程可以并行化。
4. 通过简单平均或加权平均形成随机森林。



```PYTHON
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
 
# 加载鸢尾花数据集
iris = load_iris()
 
X = iris.data
y = iris.target
 
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
 
# 创建随机森林模型
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)
 
# 模型训练
model.fit(X_train, y_train)
 
# 预测
y_pred = model.predict(X_test)
 
# 计算准确率
acc = accuracy_score(y_test, y_pred)
 
print("准确率：", acc)
```



#### GBDT

> [GBDT](https://blog.csdn.net/qq_44665283/article/details/137559749?ops_request_misc=elastic_search_misc&request_id=ce9e8ac7ebe498e9388dbadba27f77ac&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-137559749-null-null.142^v102^pc_search_result_base6&utm_term=GBDT&spm=1018.2226.3001.4187) 是一种基于梯度提升（Gradient Boosting）的集成学习算法。它通过不断学习一系列的决策树模型，并将这些模型组合起来，形成一个强大的集成模型。在每一次迭代中，GBDT 会对上一次迭代的残差进行建模，也就是预测值与实际值之间的差。这样，GBDT 能够逐步地提高模型的预测精度。
>
> 由来：在传统集成学习中，Boosting系列算法的基学习器往往是串行生成，Bagging系列算法（例如随机森林）往往是并行生成。能不能提出一种可以并行或部分并行的Boosting方法呢？传统的集成学习算法（如bagging和随机森林）是对原始数据进行重采样和特征重选择来构建多个基模型。而GBDT是直接对原始的损失函数进行优化，学习的是残差函数，这使得GBDT能更好地逼近真实函数。在每一步，GBDT 都会根据上一轮的误差来更新样本权重，这样在下一轮迭代中，模型会对之前难以预测的样本给予更大的关注。这种动态调整样本权重的策略使得 GBDT 在处理非平衡数据、噪声数据和连续特征等方面有更好的鲁棒性。并且GBDT是可以并行化训练的。
>
> GBDT有三个典型的实现：[XGBoost](https://blog.csdn.net/CXDNW/article/details/141272585?ops_request_misc=&request_id=&biz_id=102&utm_term=XGBoost&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-1-141272585.142^v102^pc_search_result_base6&spm=1018.2226.3001.4187)是2017年的一种相当强力的机器学习方法，梯度提升树框架的第一个里程碑；CatBoost针对类别型数据进行了改进，对离散特征数据进行了优化；LightGBM针对XGB的效率进行改进，通过梯度采样和直方图算法支持并行化。

原理：

1. 所有弱分类器的结果相加等于预测值。

2. 每次都以当前预测为基准，下一个弱分类器去拟合残差（预测值与真实值之间的误差）。

   > GBDT的弱分类器使用的是树模型。
   > 注意：实际上每个决策树拟合的都是负梯度，只是当损失函数是均方损失时，负梯度刚好是残差，所以其实残差只是负梯度的一种特例而已，我们后面会详细说明。

```python


import xgboost as xgb

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score

from matplotlib import pyplot as plt

from sklearn.metrics import classification_report

from sklearn import preprocessing

X=data[['X%d'%i for i in range(1,65)]]

Y=data['class']

xgb_n_clf = xgb.XGBClassifier(

        max_depth=12

        ,learning_rate=0.1

        ,reg_lambda=1

        ,n_estimators=150

        ,subsample = 0.9

        ,colsample_bytree = 0.9

        ,random_state=0

        ,eval_metric='logloss')

xgb_n_clf.fit(X,Y)

Y_test=xgb_n_clf.predict(X)

print(classification_report(Y,Y_test))

pd.DataFrame(xgb_n_clf.predict_proba(X)).to_csv("2026-predict.csv")

```

```python
import lightgbm as lgb

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score

from matplotlib import pyplot as plt

from sklearn.metrics import classification_report

from sklearn import preprocessing

X=data[['X%d'%i for i in range(1,65)]]

Y=data['class']

lgb_n_clf = lgb.LGBMClassifier(

        max_depth=12

        ,learning_rate=0.1

        ,reg_lambda=1

        ,n_estimators=150

        ,subsample = 0.9

        ,colsample_bytree = 0.9

        ,random_state=0

        ,eval_metric='logloss')

lgb_n_clf.fit(X,Y)

Y_test=lgb_n_clf.predict(X)

print(classification_report(Y,Y_test))

pd.DataFrame(lgb_n_clf.predict_proba(X)).to_csv("2026-predict.csv")

```

