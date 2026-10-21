---
title: "数学建模：回归与曲线拟合"
description: "从手算最小二乘到带验证的预测模型"
pubDate: 2026-10-21
updatedDate: 2026-10-21
order: 6
category: '数学建模'
tags: ["回归","最小二乘","拟合","零基础"]
---

> [专题学习地图](/notes/mathematical-modeling-roadmap/) · 第 6 / 27 篇

## 学习目标与先修知识

目标是独立手算简单直线拟合，理解残差、损失、系数与泛化误差，再用训练、验证和测试完成一个预测案例。先修为均值、方差、矩阵乘法与数据划分。回归既可描述变量关系，也可预测连续数值，因果解释还需要额外假设。

可回看：[统计模型与假设检验](/notes/mathematical-modeling-statistics/) · [数据预处理与插值](/notes/mathematical-modeling-data-preprocessing/)。

## 问题背景与使用时机

知道练习时长后，想估计成绩，可以先尝试一条直线。这条线不必穿过每个点，因为观测含噪声，特征也不完整。拟合要在数据波动中提取稳定关系。

如果目标是解释，关心系数含义、区间、混杂和模型设定；如果目标是预测，关心新数据上的误差与稳定性。训练集 $R^2$ 很高不能替代独立预测评估。

## 核心概念与符号

简单线性回归为：

$$
y_i=b+w x_i+\varepsilon_i,\qquad
\hat y_i=\hat b+\hat w x_i,\qquad e_i=y_i-\hat y_i.
$$

$\varepsilon_i$ 是模型中的未知误差，$e_i$ 是拟合后可计算的残差。最小二乘选择使 $\sum e_i^2$ 最小的参数；MSE 除以 $n$，最优参数相同。

$$
\hat w=\frac{\sum(x_i-\bar x)(y_i-\bar y)}
{\sum(x_i-\bar x)^2},\qquad
\hat b=\bar y-\hat w\bar x.
$$

分母要求 x 有变化。多元回归写成 $\boldsymbol y=X\boldsymbol\beta+\boldsymbol\varepsilon$；截距需一列 1。满列秩时可形式上写 $(X^TX)^{-1}X^Ty$，实际优先用最小二乘分解，避免显式求逆放大数值问题。

平方损失会更重地惩罚大误差。MAE 是绝对误差平均，单位与 y 相同；RMSE 是 MSE 的平方根；$R^2=1-\mathrm{SSE}/\mathrm{SST}$ 比较与常数均值基线的差别，测试集上可以为负。

Ridge 在平方损失外加入 $\lambda\sum\beta_j^2$，压缩系数、减少不稳定性；Lasso 用 $\lambda\sum|\beta_j|$，可产生零系数。惩罚尺度受特征单位影响，通常配合训练集内标准化。

## 手算一个最小例子

用三个点 $(1,2),(2,3),(3,5)$ 拟合。$\bar x=2$，$\bar y=10/3$。

分子为 $(-1)(-4/3)+0(-1/3)+1(5/3)=3$，分母为 $1+0+1=2$，故 $\hat w=1.5$，$\hat b=1/3$。

三个拟合值为 $11/6,10/3,29/6$，残差为 $1/6,-1/3,1/6$。残差平方和：

$$
\mathrm{SSE}=\frac1{36}+\frac1{9}+\frac1{36}=\frac16,
\qquad \mathrm{MSE}=\frac1{18}.
$$

对 $x=4$ 的预测为 $19/3\approx6.3333$。它已经超出训练 x 的区间，是外推；手算得到一个数不等于已经证明外推可靠。

## 建模步骤与完整案例

案例：根据练习时长预测成绩。用教学数据先留出最后两个点作测试，再在训练部分用交叉验证选择一次或二次关系。这里样本按时长排列，拆分只为演示外推难度；若是真实时间序列，应严格按时间切分。

比较对象包括只预测训练均值的基线与候选回归。流程为：

1. 明确 x 是预测时可获得的输入，y 是结果。
2. 检查单位、异常和范围，先画散点。
3. 预先固定候选模型与评价指标。
4. 在训练内部选择复杂度，最终测试不参与选择。
5. 报告误差及数据范围，解释模型可外推到哪里。

## Python 实现与结果解释

配套文件：[查看或下载本篇完整脚本](/downloads/math-modeling/mathematical-modeling-regression.py)。

```python
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error

x = np.arange(1., 13.).reshape(-1, 1)
y = 40 + 3*x.ravel() + np.array([0., 1., -1., 2., 0., -2., 1., 0., -1., 1., 0., -1.])
X_train, X_test = x[:10], x[10:]
y_train, y_test = y[:10], y[10:]
pipeline = make_pipeline(
    PolynomialFeatures(include_bias=False),
    StandardScaler(),
    Ridge()
)
search = GridSearchCV(
    pipeline,
    {"polynomialfeatures__degree": [1, 2], "ridge__alpha": [0.1, 1., 10.]},
    cv=KFold(5, shuffle=True, random_state=42),
    scoring="neg_mean_absolute_error"
)
search.fit(X_train, y_train)
prediction = search.predict(X_test)
baseline = np.full_like(y_test, y_train.mean())
print("选择的参数:", search.best_params_)
print("测试预测:", prediction)
print("模型MAE、基线MAE:",
      mean_absolute_error(y_test, prediction),
      mean_absolute_error(y_test, baseline))
assert np.isfinite(prediction).all()

# 重现手算：加入一列1表示截距。
A = np.column_stack([np.ones(3), [1., 2., 3.]])
coefficient = np.linalg.lstsq(A, [2., 3., 5.], rcond=None)[0]
print("手算的截距、斜率:", coefficient)
assert np.allclose(coefficient, [1/3, 1.5])
```

`neg_mean_absolute_error` 只是为了适配“得分越大越好”的接口，实际报告仍应转回非负误差。测试仅有两个点，不能据此做稳定的性能结论；真实应用应增加独立样本或滚动评估。

## 方法选择与常见误区

### 非线性曲线怎样选

多项式回归对 x 可以非线性，但对系数仍线性。一般参数曲线可用 `scipy.optimize.curve_fit`；返回的 `pcov` 是参数估计的近似协方差，不是拟合误差。拟合误差需另算残差、MAE 或 RMSE。

对 $y=a e^{bx}$ 取对数可线性化，但要求 y 为正；最小化对数误差与原尺度平方误差通常不是同一个目标。反变换的均值还可能有偏差，不应机械地指数变回去就称为原尺度无偏预测。

统计 OLS 的常规标准误需要适当误差条件；异方差或相关误差下可用相应稳健或相关结构模型。预测用得好也不自动保证每个系数有因果意义。

当残差呈弯曲时考虑变换或非线性；特征高度相关时考虑正则化；训练好、验证差时先检查泄漏、样本量和复杂度。

## 练习

1. 残差为 $1,-1,2$，MAE、MSE、RMSE 各是多少？
2. 只把所有 x 从米改成厘米，斜率的数值怎样变化，预测是否应变化？
3. 增加多项式次数后训练误差下降，能否直接认为更好？
4. 为什么不能把“练习时长系数为 3”直接说成“多练一小时必然提高 3 分”？

## 参考答案

1. MAE 为 $4/3$，MSE 为 $6/3=2$，RMSE 为 $\sqrt2$。
2. 厘米数是米数的 100 倍，斜率应变为原来的 $1/100$，同一物理输入的预测保持不变。
3. 不能。更复杂模型更容易适应训练噪声，需比较未参与拟合的验证数据。
4. 回归系数是在模型及其他输入给定条件下描述的关系；基础能力、自我选择等因素可能混杂，因果结论需要研究设计。

## 复习卡片与下一步

记住“先残差，再损失，再估计，最后验证”。手算直线是后续神经网络与梯度提升的基础。接着可读几何与矩阵课，或进入机器学习流程比较不同预测机制。

[上一篇 统计模型与假设检验](/notes/mathematical-modeling-statistics/) · [下一篇 几何模型与线性代数](/notes/mathematical-modeling-geometry/)

## 参考资料

- [Datawhale《数学建模导论》第6章](https://datawhalechina.github.io/intro-mathmodel/#/CH6/第六章-数据处理与拟合模型)：用于建立知识目录，建议与本篇的手算和案例对照阅读。
- [NumPy 最小二乘](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html)。
- [scikit-learn 线性模型](https://scikit-learn.org/stable/modules/linear_model.html)。
- [SciPy 曲线拟合](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html)。
