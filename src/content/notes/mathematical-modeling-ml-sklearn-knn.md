---
title: '数学建模：scikit-learn与KNN'
description: 'scikit-learn入门与基于距离的KNN模型'
pubDate: 2026-08-07
category: '数学建模'
tags: ['scikit-learn','KNN','机器学习']
---

## 机器学习

定义：

> *A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P if its performance at tasks in T, as measured by P, improves with experience E.*
>
> 机器学习本质上就是根据数据训练一个数学模型，并对未知数据能够成功进行合理预测。换言之，机器学习的要素包括训练材料与算法（数据、限制条件）、训练目的（分类、回归或其他）、训练效果的衡量标准（目标函数）。但业界更习惯于采用这三个要素：模型、学习准则、优化算法。

一个好模型：

1. 选择一个合适的模型：这通常需要依据实际问题而定，针对不同的问题和任务需要选取恰当的模型，模型就是一组函数的集合。
2. 判断一个函数的好坏：这需要确定一个衡量标准，也就是通常说的损失函数，损失函数的确定也需要依据具体问题而定，如回归问题一般采用欧式距离，分类问题一般采用交叉熵代价函数。
3. 找出“最好”的函数：如何从众多函数中最快的找出“最好”的那一个，这一步是最大的难点，做到又快又准往往不是一件容易的事情。常用的方法有梯度下降算法、最小二乘法等。

机器学习的分类：

> 机器学习的任务大体上按照有无标签区分。有标签指导训练过程的任务被称作有监督学习，包含目标为离散标签的分类问题和目标为连续标签的回归问题。无标签指导的任务被称作无监督学习，主要包括聚类问题和降维问题等
>
> 还有半监督和强化学习等问题

* 分类问题

  > 比如机器学习中著名的鸢尾花数据集（鸢尾花也就是水仙花），数据集收集了150朵花的萼片长度、萼片宽度、花瓣长度和花瓣宽度等四个形状指标，最后有一列标注数据表明每一条数据数据是哪种鸢尾花（数据集的最后一列是字符串，有setosa，versicolor和virginica三种类型）。数据的最后一列显然是离散变量，而前面四列自变量都是连续变量。这种数据可以用来预测一朵新的鸢尾花是三种鸢尾花中的哪一种，但是分类模型需要基于后续的标签来指导，这叫分类问题。
  >
  > 已知自变量的数值，求因变量数据属于ABC三类中哪一类的选择题。分类问题本身就像做选择题一样，有标准答案

* 回归问题

  > 在已知自变量的情况下若因变量是连续数据如何去进行建模与预测。
  >
  > 例如，著名的波士顿房价数据集，在这个数据集中存在不同的自变量，但因变量房价却是连续的数值。对房价做预测本质上也是一种回归。
  >
  > 已知自变量的数值，求因变量的计算结果，使其与实际结果偏差不大。回归问题本身就像做计算题一样，也有标准答案。

* 聚类问题

  > 比如在一家服装专卖店内老板会收集每个VIP用户的信息，例如年龄、性别、消费次数、卡内充值、消费偏好等，但这些都是自变量没有因变量。现在老板需要根据这一系列自变量对用户进行画像，将其分为若干个群，至于分多少群、每个群体有什么样的特征是由数据自变量所决定的。
  >
  > 只有自变量没有因变量将数据分群的过程就是聚类，它也可以被抽象为根据自变量的数值做一个论述题，是没有标准答案的。

* 降维问题

  > 主要探究如何用更少的变量表示原始数据，并且尽可能保留更多的信息。
  >
  > 常见的降维方法包括主成分分析、因子分析、独立成分分析、t-SNE等

一些概念：

> 一个机器学习模型本质上就是一个数学模型，它是在处理这个问题的过程中列出的一个带参数的优化问题。例如，机器学习的模型就是两个部分：第一是线性回归方程的基本形式，第二是均方误差函数的最优化。这里我们把这个误差函数又叫损失函数。我们记得，损失函数是带有参数的，有些参数是可以优化的自变量，但有些参数是不可以优化的，它是我们手动设置参数值的。我们把必须手动设置的这些参数又叫超参数，寻找效果最好的机器学习模型一个非常重要的过程就是找到最好的一套超参数配置。官方说法叫调参
> 偏差和方差用于评估模型的泛化能力。
>
> 偏差是指模型预测的期望值与真实值之间的差。偏差反映了模型拟合训练数据的能力，即算法的拟合能力。如果偏差较大，说明模型在训练数据上的表现不佳，这可能是由于模型过于简单或者模型未充分拟合训练数据所导致的。
>
> 方差是指在给定不同训练数据集的情况下，模型预测的期望值的变化量。方差反映了模型对于训练数据的变化的敏感度。如果方差较大，说明模型容易受到训练数据的影响，即训练数据的变化会导致模型预测结果的大幅度变化，这可能导致模型在新的数据集上表现不佳。
>
> 注：偏差和方差是有冲突的。偏差反映了学习算法的拟合能力，而方差则反映了同样大小的训练集的变动所导致的学习性能的变化。在训练不足的情况下，学习器的拟合能力不够强，训练数据的扰动不足以使学习器产生显著变化，此时偏差占主导。随着训练程度的加深，学习器的拟合能力逐渐增强，训练数据发生的扰动被学习器学到，方差逐渐占主导。在训练程度充足后，学习器拟合能力已非常强，训练数据的轻微扰动都会导致学习器发生显著变化，若继续学习，则将发生过拟合
>
> 欠拟合和过拟合
>
> ![image-15](/images/math-modeling/image-15.png)
>
> 欠拟合是指模型在训练数据上的表现不佳，无法充分拟合训练数据，导致在新的数据集上表现也不佳，就像第一张图。这通常是因为模型过于简单，无法捕捉到数据的复杂模式和规律。
>
> 过拟合是指模型在训练数据上的表现非常好，但在新的数据集上表现不佳，就像第三张图。这是因为模型过于复杂，对训练数据进行了过度的拟合，导致模型泛化能力下降。
>
> 选择和使用评估指标是机器学习项目中的关键步骤。
>
> 分类问题：准确率、精确率、召回率和F1得分
>
> 准确率：准确率是正确分类的样本数占总样本数的比例。例如，如果你有100个水果，你正确判断了90个，那么你的准确率是90%。
>
> 精确率：精确率是正确分类的正类样本数占所有被判定为正类的样本数的比例。例如，如果你有10个香蕉，你正确判断了8个，那么你的精确率是80%。
>
> 召回率：召回率是正确分类的正类样本数占所有实际正类样本数的比例。例如，如果你有10个香蕉，其中有3个是坏的，你正确判断了2个，那么你的召回率是66.6%。
>
> F1得分：F1得分是精确率和召回率的调和平均值，它综合反映了模型在精确率和召回率方面的表现。F1得分越高，模型的性能越好。
>
> 二分类问题：ROC曲线和AUC值
>
> ROC曲线：ROC曲线是一张反映模型在不同阈值下的性能的图。横轴表示假阳性率（即误判为正类的负类样本比例），纵轴表示真阳性率（即正确判为正类的正类样本比例）。ROC曲线越靠近左上角，表示模型的性能越好。
>
> AUC值：AUC值是ROC曲线下的面积，它反映了模型在整个阈值范围内的平均性能。AUC值越高，表示模型的效果越好。
>
> 回归问题：均方误差（MSE）、平均绝对误差（MAE）和R方值
>
> 均方误差（MSE）是预测值与真实值之差的平方的平均值。MSE越低，表示模型的预测误差越小。
>
> 平均绝对误差（MAE）是预测值与真实值之差的绝对值的平均值。MAE越低，表示模型的预测误差越小。
>
> R方值：R方值是模型解释的变异与总变异的比率。R方值越高，表示模型的解释能力越强。



### scikit-learn

```python
import numpy as np
import pandas as pd
# 鸢尾花数据集，红酒数据集，乳腺癌数据集，糖尿病数据集
from sklearn.datasets import load_iris,load_wine,load_breast_cancer,load_diabetes
# 回归重要指标
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# 分类重要指标
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_curve, roc_auc_score,classification_report
from sklearn.model_selection import train_test_split #训练集训练集分类器
import graphviz #画文字版决策树的模块
import pydotplus #画图片版决策树的模块
from IPython.display import Image #画图片版决策树的模块
iris = load_iris()
print(iris.data) # 数据
print(iris.target_names) # 标签名
print(iris.target)  # 标签值
print(iris.feature_names) # 特征名(列名)
iris_dataframe = pd.concat([pd.DataFrame(iris.data),pd.DataFrame(iris.target)],axis=1)
print(iris_dataframe)
Xtrain, Xtest, Ytrain,Ytest = train_test_split(iris.data,iris.target,test_size=0.3)
```



```python
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.ensemble import ExtraTreesRegressor,ExtraTreesClassifier
from sklearn.ensemble import AdaBoostRegressor,AdaBoostClassifier
from sklearn.ensemble import GradientBoostingRegressor,GradientBoostingClassifier
clf = DecisionTreeClassifier()
clf.fit(Xtrain, Ytrain)
Ypredict=clf.predict(Xtest)
print(classification_report(Ypredict,Ytest))
```



```python
from sklearn import tree
tree_data = tree.export_graphviz(
    clf
    ,feature_names =iris.feature_names
    ,class_names = iris.feature_names#也可以自己起名
    ,filled = True #填充颜色
    ,rounded = True #决策树边框圆形/方形
)
graph1 = graphviz.Source(tree_data.replace('helvetica','Microsoft YaHei UI'), encoding='utf-8')
graph1.render('./iris_tree',view=True,format='png')
```



```python
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

xgb = XGBClassifier()
xgb.fit(Xtrain, Ytrain)
Ypredict = xgb.predict(Xtest)
print(classification_report(Ypredict, Ytest))
```

### 基于距离的KNN模型

> 原理：
>
> KNN是监督学习算法，你只要找出你的“邻居”是什么标签，然后看哪个标签的“邻居”最多，就给这个样本贴上哪个标签。它和“懒惰学习”有点像，因为它在学习的时候并不会做很多工作，只是把数据存起来，等到要分类的时候才来用。它主要是看样本和谁最像，就像人一样，总爱找和自己像的人玩。怎么才算最像呢？就是看各个样本之间的距离，距离越近越像。就像我们平时说的“**物以类聚，人以群分**”，这个分类方法也是这样，它通过距离来判断样本的相似度。所以，只要找到与测试样本最近的k个训练样本，看这k个样本里哪个类别最多，就认为这个类别是测试样本的类别

步骤：

1. 计算距离：对于测试样本，计算其与训练集中每个样本的距离，距离的度量方式可以是欧式距离、曼哈顿距离等。
2. 选择k个最近样本：选择与测试样本距离最近的k个样本。
3. 投票并返回结果：根据k个最近样本的类别标签进行投票，多数决定原则，即哪个类别标签的多数就选择哪个标签作为测试样本的分类结果。

模型好坏的三因素：**距离的计算方式，K值的选取和数据集**。

距离的计算方式:欧几里得距离、曼哈顿距离、车比雪夫距离。用的最多的计算方法是欧几里得距离。

k值如何选择?

1. 如果K值太小，那测试样本就只会听“隔壁邻居”的意见，要是这个邻居是个“噪音制造者”，那测试样本的分类就会出错。
2. 如果K值太大，那远处的邻居也会插嘴，虽然这样可以让分类更“稳重”，但也可能分得不准确，也就是“没分清楚”。
3. K值得试了才知道，不是我们一开始就能决定的。通常，我们会选一个5-15之间的奇数。但具体选哪个，除了实验和交叉验证，还得看看你的数据量有多大，以及训练集的标签有什么特点。

注：为了加快最近邻居的搜索可以利用KD树进行数据结构的优化。KD树是对数据点在k维空间中划分的一种数据结构。在KD 的构造中，每个节点都是k维数值点的二叉树。既然是二叉树，就可以采用二叉树的增删改查操作，这样就大大提升了搜索效率。

```python
from sklearn.datasets import load_iris

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

x,y=load_iris(return_X_y=True)

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,random_state=42)
# 距离的计算
def distance(a,b):
    return np.sqrt(np.sum(np.square(a-b)))

def KNN(x_train,y_train,x_test,k):

    def get_label(x):

        dist=list(map(lambda a:distance(a,x),x_train))

        ind=np.argsort(dist)

        ind=ind[:k]

        labels=np.zeros(3)

        for i in ind:

            label=y_train[ind].astype(int)

            labels[label]+=1

        return np.argmax(labels)

    y_predict=np.zeros(len(x_test))

    for i in range(len(x_test)):

        y_predict[i]=get_label(x_test[i])

    return y_predict

# 使用不同的K值进行测试
for k in range(1,10):

    y_predict=KNN(x_train,y_train,x_test,k)

    print(accuracy_score(y_test,y_predict))

```



