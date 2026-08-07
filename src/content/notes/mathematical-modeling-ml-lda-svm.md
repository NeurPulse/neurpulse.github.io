---
title: '数学建模：线性判别与支持向量机'
description: '线性判别分析（LDA）与支持向量机（SVM）'
pubDate: 2026-08-07
category: '数学建模'
tags: ['线性判别','SVM','机器学习','分类']
---

### [线性判别](https://blog.csdn.net/xiaoyingxixi1989/article/details/142370417?ops_request_misc=elastic_search_misc&request_id=3100290b811576dc65b3b19e8be2f8c6&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_click~default-1-142370417-null-null.142^v102^pc_search_result_base6&utm_term=%E7%BA%BF%E6%80%A7%E5%88%A4%E5%88%AB%E5%88%86%E6%9E%90&spm=1018.2226.3001.4187)与[支持向量机](https://blog.csdn.net/panss__/article/details/148094354?ops_request_misc=elastic_search_misc&request_id=ad0a5d74058ad32ddd708add90c0ea80&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-148094354-null-null.142^v102^pc_search_result_base6&utm_term=%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA&spm=1018.2226.3001.4187)

#### 线性判别分析：

线性判别分析（LDA），由于是费舍尔所发明，故又名费舍尔判别。LDA在模式识别领域（比如人脸识别，舰艇识别等图形图像识别领域）中有非常广泛的应用。LDA是一种监督学习的降维技术，也就是说它可以同时实现降维和分类两个操作。LDA的思想可以用一句话概括，就是“投影后类内方差最小，类间方差最大”，如图所示。 我们要将数据在低维度上进行投影，投影后希望每一种类别数据的投影点尽可能的接近，而不同类别的数据的类别中心之间的距离尽可能的大。


线性判别分析的流程：

- 对于给定有标签数据集$(x_i,y_i)$，计算出均值和协方差	。

- 投影到直线$y=\boldsymbol{w}^T\boldsymbol{x}$以后均值和协方差变成了$\boldsymbol{w}^T \boldsymbol{\mu}_i$和$\boldsymbol{w}^T \Sigma_i \boldsymbol{w}$。

- 类内差别尽量小，类间差别尽可能大，列出目标函数：
  $$
  J = \frac{\boldsymbol{w}^T (\boldsymbol{\mu}_0 - \boldsymbol{\mu}_1)(\boldsymbol{\mu}_0 - \boldsymbol{\mu}_1)^T \boldsymbol{w}}{\boldsymbol{w}^T (\Sigma_0 + \Sigma_1)\boldsymbol{w}}
  $$

- 记$S_w = \Sigma_0 + \Sigma_1$，$S_b = (\boldsymbol{\mu}_0 - \boldsymbol{\mu}_1)(\boldsymbol{\mu}_0 - \boldsymbol{\mu}_1)^T$，那目标函数就变成了一个广义瑞利商 $$J = \frac{\boldsymbol{w}^T S_b \boldsymbol{w}}{\boldsymbol{w}^T S_w \boldsymbol{w}}$$ 于是问题等价成一个凸优化问题，再将分母规约得到：
  $$
  \begin{cases} \min \quad -\boldsymbol{w}^T S_b \boldsymbol{w} \\ \text{s.t.}\quad \boldsymbol{w}^T S_w \boldsymbol{w} = 1 \end{cases} ​
  $$







线性可分:

```python
import numpy as np

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

from tqdm import tqdm, trange

data = np.loadtxt('linear.csv', delimiter=',')

print('数据集大小：', len(data))

x = data[:, :2]

y = data[:, 2]

# 数据集可视化

plt.figure()

plt.scatter(x[y == -1, 0], x[y == -1, 1], color='red', label='y=-1')

plt.scatter(x[y == 1, 0], x[y == 1, 1], color='blue', marker='x', label='y=1')

plt.xlabel(r'$x_1$')

plt.ylabel(r'$x_2$')

plt.legend()

plt.show()

```



```python
def SMO(x, y, ker, C, max_iter):

    '''

    SMO算法

    x，y：样本的值和类别

    ker：核函数，与线性回归中核函数的含义相同

    C：惩罚系数

    max_iter：最大迭代次数

    '''

    # 初始化参数

    m = x.shape[0]

    alpha = np.zeros(m)



    # 预先计算所有向量的两两内积，减少重复计算

    K = np.zeros((m, m))

    for i in range(m):

        for j in range(m):

            K[i, j] = ker(x[i], x[j])



    for l in trange(max_iter):

        # 开始迭代

        for i in range(m):

            # 有m个参数，每一轮迭代中依次更新

            # 固定参数alpha_i与另一个随机参数alpha_j，并且保证i与j不相等

            j = np.random.choice([l for l in range(m) if l != i])



            # 用-b/2a更新alpha_i的值

            eta = K[j, j] + K[i, i] - 2 * K[i, j] # 分母

            e_i = np.sum(y * alpha * K[:, i]) - y[i] # 分子

            e_j = np.sum(y * alpha * K[:, j]) - y[j]

            alpha_i = alpha[i] + y[i] * (e_j - e_i) / (eta + 1e-5) # 防止除以0

            zeta = alpha[i] * y[i] + alpha[j] * y[j]

            # 将alpha_i和对应的alpha_j保持在[0,C]区间

            # 0 <= (zeta - y_j * alpha_j) / y_i <= C

            if y[i] == y[j]:

                lower = max(0, zeta / y[i] - C)

                upper = min(C, zeta / y[i])

            else:

                lower = max(0, zeta / y[i])

                upper = min(C, zeta / y[i] + C)

            alpha_i = np.clip(alpha_i, lower, upper)

            alpha_j = (zeta - y[i] * alpha_i) / y[j]



            # 更新参数

            alpha[i], alpha[j] = alpha_i, alpha_j

    return alpha

# 设置超参数

C = 1e8 # 由于数据集完全线性可分，我们不引入松弛变量

max_iter = 1000

np.random.seed(0)

alpha = SMO(x, y, ker=np.inner, C=C, max_iter=max_iter)

# 用alpha计算w，b和支持向量

sup_idx = alpha > 1e-5 # 支持向量的系数不为零

print('支持向量个数：', np.sum(sup_idx))

w = np.sum((alpha[sup_idx] * y[sup_idx]).reshape(-1, 1) * x[sup_idx], axis=0)

wx = x @ w.reshape(-1, 1)

b = -0.5 * (np.max(wx[y == -1]) + np.min(wx[y == 1]))

print('参数：', w, b)

# 绘图

X = np.linspace(np.min(x[:, 0]), np.max(x[:, 0]), 100)

Y = -(w[0] * X + b) / (w[1] + 1e-5)

plt.figure()

plt.scatter(x[y == -1, 0], x[y == -1, 1], color='red', label='y=-1')

plt.scatter(x[y == 1, 0], x[y == 1, 1], marker='x', color='blue', label='y=1')

plt.plot(X, Y, color='black')

# 用圆圈标记出支持向量

plt.scatter(x[sup_idx, 0], x[sup_idx, 1], marker='o', color='none', 

    edgecolor='purple', s=150, label='support vectors')

plt.xlabel(r'$x_1$')

plt.ylabel(r'$x_2$')

plt.legend()

plt.show()

```

线性不可分:

```python
data = np.loadtxt('spiral.csv', delimiter=',')

print('数据集大小：', len(data))

x = data[:, :2]

y = data[:, 2]

# 数据集可视化

plt.figure()

plt.scatter(x[y == -1, 0], x[y == -1, 1], color='red', label='y=-1')

plt.scatter(x[y == 1, 0], x[y == 1, 1], marker='x', color='blue', label='y=1')

plt.xlabel(r'$x_1$')

plt.ylabel(r'$x_2$')

plt.legend()

plt.axis('square')

plt.show()
	
```



核函数：

```python
# 简单多项式核

def simple_poly_kernel(d):

    def k(x, y): 

        return np.inner(x, y) ** d

    return k

# RBF核

def rbf_kernel(sigma):

    def k(x, y):

        return np.exp(-np.inner(x - y, x - y) / (2.0 * sigma ** 2))

    return k

# 余弦相似度核

def cos_kernel(x, y):

    return np.inner(x, y) / np.linalg.norm(x, 2) / np.linalg.norm(y, 2)

# sigmoid核

def sigmoid_kernel(beta, c):

    def k(x, y):

        return np.tanh(beta * np.inner(x, y) + c)

    return k

测试不同核函数下样本会被如何核化：

kernels = [

    simple_poly_kernel(3), 

    rbf_kernel(0.1), 

    cos_kernel, 

    sigmoid_kernel(1, -1)

]

ker_names = ['Poly(3)', 'RBF(0.1)', 'Cos', 'Sigmoid(1,-1)']

C = 1e8

max_iter = 500

# 绘图准备，构造网格

plt.figure()

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

axs = axs.flatten()

cmap = ListedColormap(['coral', 'royalblue'])

# 开始求解 SVM

for i in range(len(kernels)):

    print('核函数：', ker_names[i])

    alpha = SMO(x, y, kernels[i], C=C, max_iter=max_iter)

    sup_idx = alpha > 1e-5 # 支持向量的系数不为零

    sup_x = x[sup_idx] # 支持向量

    sup_y = y[sup_idx]

    sup_alpha = alpha[sup_idx]

    # 用支持向量计算 w^T*x

    def wx(x_new):

        s = 0

        for xi, yi, ai in zip(sup_x, sup_y, sup_alpha):

            s += yi * ai * kernels[i](xi, x_new)

        return s

    # 计算b*

    neg = [wx(xi) for xi in sup_x[sup_y == -1]]

    pos = [wx(xi) for xi in sup_x[sup_y == 1]]

    b = -0.5 * (np.max(neg) + np.min(pos))

    # 构造网格并用 SVM 预测分类

    G = np.linspace(-1.5, 1.5, 100)

    G = np.meshgrid(G, G)

    X = np.array([G[0].flatten(), G[1].flatten()]).T # 转换为每行一个向量的形式

    Y = np.array([wx(xi) + b for xi in X])

    Y[Y < 0] = -1

    Y[Y >= 0] = 1

    Y = Y.reshape(G[0].shape)

    axs[i].contourf(G[0], G[1], Y, cmap=cmap, alpha=0.5)

    # 绘制原数据集的点

    axs[i].scatter(x[y == -1, 0], x[y == -1, 1], color='red', label='y=-1')

    axs[i].scatter(x[y == 1, 0], x[y == 1, 1], marker='x', color='blue', label='y=1')

    axs[i].set_title(ker_names[i])

    axs[i].set_xlabel(r'$x_1$')

    axs[i].set_ylabel(r'$x_2$')

    axs[i].legend()

plt.show()

```



