---
title: '数学建模：聚类算法'
description: 'K-Means、DBSCAN等无监督聚类方法'
pubDate: 2026-08-07
category: '数学建模'
tags: ['聚类','K-Means','DBSCAN','无监督学习']
---

### 聚类算法

#### k-means 算法

> K-means 算法，也称为 K 平均或者 K 均值，一般作为掌握聚类算法的第一个算法。这里的 k 为常数，需事先设定，通俗地说该算法是将没有标注的 M 个样本通过迭代的方式聚集成 k 个簇。在对样本进行聚集的过程往往是以样本之间的距离作为指标来划分。

> 它和 KNN 有很多共同之处，同样需要计算距离，同样会进行近邻寻找，但它的过程没有标签指导，所以具体流程也就有所不同。

通俗一点来说，聚类如果希望我们把数据分成 k 堆，那么首先会随机抽 k 个幸运儿作为样本的中心点，然后对于数据中每个样本，找到它最近的那个中心点作为 “大哥”，然后加入他的帮派；每个小弟都找到自己大哥以后帮派内会 “比武” 选出新的 “大哥”，这个 “比武” 就是比帮派内谁与这个帮派的中心点最近；然后所有的小弟集中起来，再去给他们重新选择自己想加入的堂口的机会，于是诞生了一批二五仔换到其他的帮派里面去…… 当这个数据集里面不再存在二五仔的时候聚类就结束了，每个堂口的老大也确定了

k-means 算法是根据给定的 n 个数据对象的数据集，构建 k 个划分聚类的方法，每个划分聚类即为一个簇。该方法将数据划分为 n 个簇，每个簇至少有一个数据对象，每个数据对象必须属于而且只能属于一个簇。同时要满足同一簇中的数据对象相似度高，不同簇中的数据对象相似度较小。聚类相似度是利用各簇中对象的均值来进行计算的。

K-Means 的执行流程

1. 选取 K 个点做为初始聚集的簇心（也可选择非样本点）。

2. 分别计算每个样本点到 K 个簇核心的距离，找到离该点最近的簇核心，将它归属到对应的簇。

3. 所有点都归属到簇之后，M 个点就分为了 K 个簇。之后重新计算每个簇的中心，也就是每个堂口的老大。

4. 反复迭代 2 - 3 步骤，直到达到某个中止条件。常用的中止条件有迭代次数、最小平方误差 MSE、簇中心点变化率等。

对于 KMean 算法来说有三个比较重要的因素要考虑，分别如下：

- k 值的选择：k 值对最终结果的影响至关重要，而它却必须要预先给定。比较合适的方法有肘部图法、轮廓系数法等。
- 异常点的干扰：K-means 算法在迭代的过程中使用所有点的均值作为新的中心，如果簇中存在异常点，将导致均值偏差比较严重。
- 初值敏感：K-means 算法是初值敏感的，选择不同的初始值可能导致不同的簇划分规则。为了避免这种敏感性导致的最终结果异常性，可以采用初始化多套初始节点构造不同的分类规则，然后选择最优的构造规则。针对这点后面因此衍生了：二分 K-Means 算法、K-Means++ 算法、Canopy 算法等。

衡量聚类好坏的标准可以用轮廓系数来描述。轮廓系数的定义为:
$$
s(i)=\frac{b(i)-a(i)}{\max\{a(i),b(i)\}}
$$
轮廓系数在 [-1,1] 之间，越大越合理。

判断最优的 k 值会采取肘部图策略。肘部法则的计算原理是损失函数，损失函数是每个变量点到其类别中心的位置距离平方和。在选择类别数量上，肘部法则会把不同值的成本函数值画出来。肘部就是指这个图的拐点，下降从快到慢的点。

```python
def init_cent(dataset,K):

    idx=np.random.choice(np.arange(len(dataset)),size=K,replace=False)

    return dataset[idx]

def Kmeans(dataset,K,init_cent):

    centroids=init_cent(dataset,K)

    cluster=np.zeros(len(dataset))

    changed=True

    while changed:

        changed=False

        loss=0

        for i,data in enumerate(dataset):

            dis=np.sum((centroids-data)**2,axis=-1)

            k=np.argmin(dis)

            if cluster[i]!=k:

                cluster[i]=k

                changed=True

            loss+=np.sum((data-centroids[k])**2)

        for i in range(K):

            centroids[i]=np.mean(dataset[cluster==i],axis=0)

    return centroids,cluster
```

```python
from sklearn.cluster import KMeans  

import numpy as np  

# 创建一个随机数据集  

X = np.random.rand(100, 2)   

# 创建KMeans聚类模型，设置簇的数量为3  

kmeans = KMeans(n_clusters=3)  

# 使用数据集拟合模型  

kmeans.fit(X)  

# 输出聚类中心点  

print("Cluster centers:")  

print(kmeans.cluster_centers_)  

# 对数据集进行预测，得到每个数据点的聚类标签  

labels = kmeans.predict(X)  

print("Labels of data points:")  

print(labels)
```

#### DBSCAN算法

DBSCAN 算法是密度聚类算法，所谓密度聚类算法就是说这个算法是根据样本的密集程度来进行聚类。

要根据样本中的数据密度进行聚类，首先定义样本中数据密度大的地方应该怎样表示，这两个参数引出了两个概念：

- ε- 邻域：若 $x_i $是一个样本点，邻域就是指距离 $x_i$ 不超过 ε 的范围。本质上就是衡量离 A 样本有多远。
- 核心对象：如果$ x_i$ 的 ε- 邻域内至少含有 M 个样本，则$ x_i$ 是一个核心对象。这个 M 也被定义为密集的阈值

定义核心对象之间的几何关系：

- 密度直达：如果 B 样本位于 A 样本的 ε- 邻域内，则称 AB 密度直达。
- 若 A 样本和 B 样本密度直达，B 样本和 C 样本密度直达，A 和 C 之间不是密度直达，则称 A 样本和 C 样本密度可达。
- 若 B 样本和 A、C 样本均密度可达，则称 A 和 C 密度相连。

本质上这个算法就是统计数据中的核心对象并将其归类，但这个算法有一个特点就是非核心对象会被判定为离群点，所以安排一个离群类 - 1 给它。

DBSCAN 虽然能比较好地分类出离群点，但是算法稳定性很不好，参数稍微改变一点会对性能产生很大影响。

步骤：

1. 选择合适的ε值和M值。

2. 计算每个样本点的ε-邻域。

3. 确定核心对象。

4. 建立密度连接关系。

5. 合并密度相连的核心对象为同一聚类。

6. 将离群点分类。

```python
from sklearn.cluster import DBSCAN  

from sklearn.datasets import make_moons  

import matplotlib.pyplot as plt  

# 生成半月形数据集  

X, y = make_moons(n_samples=200, noise=0.05, random_state=0)  

# 执行DBSCAN聚类  

dbscan = DBSCAN(eps=0.3, min_samples=5)  

dbscan.fit(X)  

labels = dbscan.labels_  

# 绘制聚类结果  

unique_labels = set(labels)  

colors = [plt.cm.Spectral(each)  

          for each in np.linspace(0, 1, len(unique_labels))]  

for k, col in zip(unique_labels, colors):  

    if k == -1:  

        # Black used for noise.  

        col = [0, 0, 0, 1]  

    class_member_mask = (labels == k)  

    xy = X[class_member_mask & (labels != -1)]  

    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),  

             markeredgecolor='k', markersize=14)  

plt.title('DBSCAN')  

plt.show()
```



#### 层次聚类法

层次聚类法（计算距离进行聚类），不过这种聚类是一种树状结构的聚类。层次聚类有两种顺序：自上而下和自底而上。层次聚类比较新的算法有例如 BIRCH，ROCK 等。

自上而下的层次聚类输入样本数据和聚类数量后进行操作：

- 将样本归为一类。
- 在一个类里面计算样本的距离，找到距离最远的 a,b。
- 将 a,b 分到两个不同的簇里面。
- 对剩下样本，到 a 和到 b 哪个距离更小就去哪一堆。
- 递归生成聚类树。

自底而上的层次聚类则按照这个顺序进行：

- 一个样本作为一个类。
- 计算两两之间的距离，最小的两个点合并为一个类别。
- 重复上一个操作直到所有数据被归为一类。

```python
import numpy as np  

import matplotlib.pyplot as plt  

from scipy.cluster.hierarchy import dendrogram, linkage  

# 生成随机数据  

np.random.seed(0)  

X = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], size=500)  

# 执行层次聚类  

linked = linkage(X, 'ward')  

# 绘制层次聚类模型图  

fig = plt.figure(figsize=(10, 7))  

dendrogram(linked,  

                      orientation='top',  

                      distance_sort='descending',  

                      show_leaf_counts=True)  

plt.show()
```

```python
from sklearn.cluster import AgglomerativeClustering  

import matplotlib.pyplot as plt  

# 生成随机数据  

X = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], size=500)   

# 执行层次聚类  

cluster = AgglomerativeClustering(n_clusters=2, metric='euclidean')

cluster.fit(X)  

labels = cluster.labels_

# 绘制层次聚类模型图  

fig = plt.figure(figsize=(10, 7))  

plt.scatter(X[:, 0], X[:, 1], c=labels)  

plt.show()
```



