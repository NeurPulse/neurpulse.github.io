---
title: "数学建模：ARIMA 与相关时间序列"
description: "从差分和自相关理解 AR、MA、ARIMA 的选择与诊断"
pubDate: 2026-10-21
updatedDate: 2026-10-21
order: 16
category: '数学建模'
tags: ["ARIMA","时间序列","零基础"]
---

> [专题学习地图](/notes/mathematical-modeling-roadmap/) · 第 16 / 27 篇

## 学习目标与先修知识

目标是读懂 AR、MA、ARIMA 的符号，手算一个均值回复预测，理解差分、ACF、残差诊断和回测之间的关系。先修为时间序列、概率波动与一步/多步预测。这里的 MA 不是直接平均最近观测值。

可回看：[时间序列分析与验证](/notes/mathematical-modeling-time-series/) · [时间序列之指数平滑法](/notes/mathematical-modeling-exponential-smoothing/)。

## 问题背景与使用时机

有些序列今天偏高，明天还倾向偏高，但影响会逐渐衰减。简单水平模型可能漏掉这种相关结构；ARIMA 用历史状态与随机冲击描述它。

模型选择不应只看一张 ACF 图或最低训练误差。参数阶数、差分和季节设定会影响样本需求与稳定性。

## 核心概念与符号

### 三种基本结构

AR(p) 用过去 p 个值解释当前值。以中心化的 AR(1) 为例：

$$
y_t-\mu=\phi(y_{t-1}-\mu)+\varepsilon_t.
$$

若 $|\phi|<1$ 且冲击满足适当条件，可有平稳均值回复结构。MA(q) 用当前和过去 q 个随机冲击：

$$
y_t=\mu+\varepsilon_t+\theta_1\varepsilon_{t-1}+\cdots+\theta_q\varepsilon_{t-q}.
$$

冲击通常不是直接观测，需与模型一起估计。不同软件对 MA 符号约定可能不同，解释前要看定义。

ARIMA(p,d,q) 先进行 d 次普通差分，再对差分序列建立 ARMA(p,q)。$d=1$ 不等于“做一次移动平均”。季节模型还可用 $(P,D,Q)_m$ 描述周期 m 的结构。

### 诊断与选择

ACF、PACF 可提示简单模型候选，但不是自动识别唯一正确阶数的规则。AIC 等信息准则平衡似然与参数复杂度；应比较在相同数据、相同目标和合适估计条件下的候选，最终仍需预测验证。

拟合后检查残差是否仍有自相关，可使用 Ljung–Box 等检验并考虑已估参数对自由度的影响。没有显著残差相关也不证明模型已经完整或因果正确。

## 手算一个最小例子

设长期均值 $\mu=10$、$\phi=0.6$，当前观测 $y_t=15$，未来冲击条件均值为0。

一步预测为：

$$
\hat y_{t+1\mid t}=10+0.6(15-10)=13.
$$

两步预测再让偏离均值的部分乘0.6，得到 $10+0.6^2\times5=11.8$。h 步为 $10+0.6^h\times5$，长期接近10。

这不表示每次真实观测都单调回到10，因为新冲击仍会发生。若冲击方差为 $\sigma^2$，h步误差方差为 $\sigma^2\sum_{j=0}^{h-1}\phi^{2j}$，预测范围会随步长扩大。

## 建模步骤与完整案例

案例：构造具有已知 AR(1) 结构的序列，先模拟较长预热段再保留180个观测，减少随意初值的影响。前108个拟合，随后36个验证，在 AR(1)、AR(2) 和均值基线之间选择，最后对36个留出点作固定起点多步预测。

教学数据只用于检查流程及机制；真实序列是否差分，要结合图、领域知识、平稳性分析和验证。趋势不明显时也不能一概设置 d=1。

## Python 实现与结果解释

配套文件：[查看或下载本篇完整脚本](/downloads/math-modeling/mathematical-modeling-arima.py)。

```python
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

rng = np.random.default_rng(42)
values = np.empty(280)
values[0] = 10.
for i in range(1, len(values)):
    values[i] = 10 + 0.6*(values[i-1]-10) + rng.normal()
y = values[100:]
train, valid, test = y[:108], y[108:144], y[144:]

orders = [(1, 0, 0), (2, 0, 0)]
errors = {"mean": float(np.mean(np.abs(valid-train.mean())))}
for order in orders:
    fit = ARIMA(train, order=order, trend="c").fit()
    prediction = np.asarray(fit.forecast(len(valid)))
    errors[order] = float(np.mean(np.abs(valid-prediction)))
choice = min(errors, key=errors.get)
history = y[:144]
if choice == "mean":
    prediction = np.full(len(test), history.mean())
else:
    fit = ARIMA(history, order=choice, trend="c").fit()
    prediction = np.asarray(fit.forecast(len(test)))
print("验证误差:", errors, "选择:", choice)
print("最终多步MAE:", np.mean(np.abs(test-prediction)))
print("已知参数的两步预测:", 10+0.6**2*(15-10))
assert np.isclose(10+0.6**2*5, 11.8)
assert np.isfinite(prediction).all()
```

最终选择可能是简单均值，这是允许且有价值的结论：当固定起点预测较远时，AR(1) 的预测也会迅速接近均值。不能为了展示复杂模型而隐藏基线。

## 方法选择与常见误区

- ARIMA 描述条件均值；波动聚集可能需要另建条件方差模型，如 GARCH，不能把均值与方差任务混为一谈。
- 统计平稳性检验有自己的原假设与功效限制，不应只凭一次检验机械决定差分。
- 灰色 GM(1,1) 等小样本模型也有结构假设，样本少不等于任何趋势都可可靠预测。
- 多步预测需要递归处理未知未来，预测区间依赖模型及噪声假设。
- 趋势、季节、结构突变与外生变量应有领域解释；同一个阶数不适合所有业务。

## 练习

1. AR(1) 的 $\mu=20,\phi=0.5$，当前24，一步、两步预测各是多少？
2. 观测 $3,5,8,12$ 的一阶差分是什么？
3. MA(1) 是否就是最近两次观测的平均？
4. 为什么训练残差很小仍需留出验证？

## 参考答案

1. 一步22，两步21，按偏离均值每期缩半计算。
2. $2,3,4$。差分后仍可能有趋势，不能自动认定平稳。
3. 不是。MA(1) 组合当前与上次随机冲击，系数也不必是各0.5。
4. 训练拟合可能适应噪声，模型阶数与参数选择还会产生选择偏差；未来误差需要独立或按时间的回测。

## 复习卡片与下一步

先解释为什么有相关，再决定是否差分，最后按实际步长验证。现在可以转向有标签的分类问题，注意时间序列任务仍要保留时间边界。

[上一篇 时间序列之指数平滑法](/notes/mathematical-modeling-exponential-smoothing/) · [下一篇 机器学习流程与 KNN](/notes/mathematical-modeling-ml-sklearn-knn/)

## 参考资料

- [Datawhale《数学建模导论》第8章](https://datawhalechina.github.io/intro-mathmodel/#/CH8/第8章-时间序列)：用于建立知识目录，建议与本篇的手算和案例对照阅读。
- [ARIMA 原理与选择](https://otexts.com/fpp3/arima.html)。
- [statsmodels ARIMA](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html)。
