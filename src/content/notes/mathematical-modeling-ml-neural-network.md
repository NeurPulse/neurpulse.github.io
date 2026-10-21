---
title: "数学建模：神经网络与反向传播"
description: "从单个神经元的手算更新走到可验证的 MLP"
pubDate: 2026-08-07
updatedDate: 2026-10-21
order: 22
category: '数学建模'
tags: ["神经网络","MLP","深度学习","零基础"]
---

> [专题学习地图](/notes/mathematical-modeling-roadmap/) · 第 22 / 27 篇

## 学习目标与先修知识

目标是手算一个神经元的前向、损失与梯度更新，读懂多层感知机的尺寸、激活和训练循环，并完成带缩放与早停的分类案例。先修为内积、导数、分类指标与验证集。框架语法不是本篇的先修要求。

可回看：[回归与曲线拟合](/notes/mathematical-modeling-regression/) · [机器学习流程与 KNN](/notes/mathematical-modeling-ml-sklearn-knn/)。

## 问题背景与使用时机

线性模型只能表达一类相对简单的关系。多个仿射变换之间加入非线性激活后，可以组合出更丰富的模式。网络更灵活，也更需要数据、正则化和验证。

多层感知机 MLP 是神经网络的一类，神经网络还包括卷积、循环等结构。小数据也可使用小网络，但不能据此认为一定优于线性或树模型。

## 核心概念与符号

### 前向计算与形状

一批 n 个样本、每个 d 个特征，用 $X\in\mathbb R^{n\times d}$ 表示。隐藏单元数 h、类别数 c：

$$
H=\sigma(XW_1+b_1),\quad W_1\in\mathbb R^{d\times h},\quad b_1\in\mathbb R^h,
$$

$$
Z=HW_2+b_2,\quad W_2\in\mathbb R^{h\times c},\quad b_2\in\mathbb R^c.
$$

H 是 $n\times h$，Z 是 $n\times c$。偏置沿样本行广播。若没有非线性，多个仿射层仍能合并为一个仿射层。

![多层感知机的形状与训练流程](/images/math-modeling/learning/neural-network.svg)

### 激活与损失

ReLU 为 $\max(0,z)$，负区间梯度为0；0点不可导，框架通常约定反传值为0。sigmoid 把分数映射到(0,1)，适用于二分类概率；tanh 映射到(-1,1)。sigmoid、tanh 在饱和区导数很小，深层训练可能困难。

多分类常用 softmax：$p_k=e^{z_k}/\sum_j e^{z_j}$，计算时减去最大分数提高数值稳定性。交叉熵为 $-\sum_k y_k\log p_k$；整数标签可由框架在内部处理，不一定手动独热。

反向传播用链式法则计算损失对各参数的导数；优化器据此更新参数。两者不是同一步。梯度下降为 $\theta\leftarrow\theta-\eta\nabla_\theta L$；SGD 用小批量，Adam 等自适应方法用额外状态调节更新。

## 手算一个最小例子

一个二分类神经元：输入 x=2、权重 w=0.5、偏置 b=0、真实标签 y=1。

前向 $z=wx+b=1$，$p=\sigma(1)\approx0.7311$；交叉熵 $L=-\ln p\approx0.3133$。对 sigmoid 加二分类交叉熵，链式求导简化为：

$$
\frac{\partial L}{\partial z}=p-y,\quad
\frac{\partial L}{\partial w}=(p-y)x,\quad
\frac{\partial L}{\partial b}=p-y.
$$

所以梯度约为 $-0.5379,-0.2689$。学习率0.1时，$w_{\mathrm{new}}\approx0.5538$、$b_{\mathrm{new}}\approx0.0269$；重新前向得到更大的正类概率和更小的损失。

这是用同一个样本优化损失的演示，不是泛化验证。多隐藏层仍用同样的链式法则，只是中间导数更多。

## 建模步骤与完整案例

案例：用 Wine 数据训练一个小型 MLP。13个特征进入16个隐藏单元，再输出3类概率。参数量为 $13\times16+16+16\times3+3=275$。

先分层留出测试，再从剩余样本分出早停验证集。缩放器只在实际用于梯度训练的部分拟合，验证集与测试集都只 transform。每轮用验证交叉熵决定是否保存当前参数，连续30轮没有足够改善就停止；最终测试只使用保存的最好模型。

完整训练循环为：取输入 → 前向 → 计算损失 → 清理旧梯度 → 反向求梯度 → 优化器更新 → 在验证集检查 → 达到条件停止。不能在训练后直接把同一批样本的准确率当作未见数据效果。

## Python 实现与结果解释

配套文件：[查看或下载本篇完整脚本](/downloads/math-modeling/mathematical-modeling-ml-neural-network.py)。

```python
import numpy as np
from copy import deepcopy
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import log_loss, accuracy_score

x, y, w, b = 2., 1., 0.5, 0.
sigmoid = lambda z: 1/(1+np.exp(-z))
p = sigmoid(w*x+b)
old_loss = -np.log(p)
gradient_w, gradient_b = (p-y)*x, p-y
w -= 0.1*gradient_w
b -= 0.1*gradient_b
new_loss = -np.log(sigmoid(w*x+b))
print("更新后的w、b、损失:", w, b, new_loss)
assert new_loss < old_loss

X, labels = load_wine(return_X_y=True)
X_development, X_test, y_development, y_test = train_test_split(
    X, labels, test_size=0.3, stratify=labels, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_development, y_development, test_size=0.2,
    stratify=y_development, random_state=42)
scaler = StandardScaler().fit(X_train)
z_train, z_valid = scaler.transform(X_train), scaler.transform(X_valid)
network = MLPClassifier(
    hidden_layer_sizes=(16,), activation="relu", solver="adam",
    alpha=0.01, learning_rate_init=0.01, early_stopping=False, random_state=42)
classes = np.unique(y_train)
best_loss, stale, best_network, best_epoch = np.inf, 0, None, 0
for epoch in range(500):
    network.partial_fit(z_train, y_train, classes=classes)
    validation_loss = log_loss(y_valid, network.predict_proba(z_valid), labels=classes)
    if validation_loss < best_loss-1e-4:
        best_loss, stale = validation_loss, 0
        best_network, best_epoch = deepcopy(network), epoch+1
    else:
        stale += 1
    if stale >= 30:
        break
probability = best_network.predict_proba(scaler.transform(X_test))
prediction = best_network.classes_[probability.argmax(axis=1)]
baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
print("网络与基线测试准确率:", accuracy_score(y_test, prediction), baseline.score(X_test, y_test))
print("停止轮数、保存轮数:", epoch+1, best_epoch)
assert probability.shape == (len(X_test), 3)
assert np.allclose(probability.sum(axis=1), 1.)

```

最多训练500轮，打印的停止轮数和保存轮数可以不同：验证效果最好的参数可能出现在更早的一轮。随机种子固定只是帮助复现，真实结论还应看多个种子和数据划分的稳定性。

## 方法选择与常见误区

- 所有隐藏单元用相同零权重初始化会产生对称性，通常需要合适随机初始化；偏置是否初始化为零是另一件事。
- 过拟合时可减小网络、增加正则化、早停或补充数据；优化失败时还要检查尺度、学习率和梯度。
- ReLU 的负区间可能长期没有梯度；sigmoid 饱和可能梯度过小；损失定义和输出激活必须匹配。
- 图像常考虑 CNN，时间依赖可考虑 RNN/LSTM 等结构；只有真正需要时再增加结构复杂度。
- PyTorch 中常见 `CrossEntropyLoss` 接收原始 logits，不要又先 softmax 再当 logits 传入。不同接口约定应对照文档。
- 大网络不是自动优势；必须与线性、树等合理基线在相同数据边界和指标下比较。

## 练习

1. 输入5维、隐藏4维、输出2维，两层全连接共多少参数？
2. 为什么连续两层线性映射没有非线性时不能得到任意复杂关系？
3. 反向传播与优化器分别做什么？
4. 单样本更新后损失下降能证明模型泛化好吗？
5. 多分类概率行和应满足什么条件？

## 参考答案

1. 第一层 $5\times4+4=24$，第二层 $4\times2+2=10$，共34。
2. 仿射函数的复合仍是仿射函数，可以合并成一个权重与偏置。
3. 前者计算导数，后者根据导数与自身状态更新参数。
4. 不能，只表明对该样本的目标下降；需独立验证。
5. 各项非负，行和约为1；这检查概率计算形式，不代表概率已校准。

## 复习卡片与下一步

先能手算一个神经元，再看层与层的尺寸，最后学习框架。框架实践可继续读站内 PyTorch 的张量、模型构建和自动求导笔记。下一篇在不能方便求导或存在离散编码时学习随机优化。

[上一篇 聚类算法](/notes/mathematical-modeling-ml-clustering/) · [下一篇 进化计算与群体智能](/notes/mathematical-modeling-evolutionary/)

## 参考资料

- [Datawhale《数学建模导论》第9章](https://datawhalechina.github.io/intro-mathmodel/#/CH9/第九章-机器学习与统计模型)：用于建立知识目录，建议与本篇的手算和案例对照阅读。
- [scikit-learn MLP](https://scikit-learn.org/stable/modules/neural_networks_supervised.html)。
- [PyTorch 交叉熵接口](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)。
