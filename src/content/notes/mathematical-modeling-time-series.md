---
title: "数学建模：时间序列分析与验证"
description: "识别趋势、季节和相关性，建立按时间验证的流程"
pubDate: 2026-08-10
updatedDate: 2026-10-21
order: 13
category: '数学建模'
tags: ["时间序列","零基础"]
---

> [专题学习地图](/notes/mathematical-modeling-roadmap/) · 第 13 / 27 篇

## 学习目标与先修知识

目标是识别趋势、季节、循环与残差，理解平稳性和自相关，设计没有未来信息泄漏的滚动验证，并与简单预测基线比较。先修为数据清洗、均值、方差与 MAE。读完再进入移动平均、指数平滑和 ARIMA。

可回看：[数据预处理与插值](/notes/mathematical-modeling-data-preprocessing/) · [统计模型与假设检验](/notes/mathematical-modeling-statistics/)。

## 问题背景与使用时机

销量、温度、流量、价格按时间排列后，昨天与今天通常有关联。把记录随机打乱再分训练测试，可能让未来模式提前进入训练，得到难以部署的成绩。

在预测某天之前，只能用那个时点已经掌握的信息。昨天的真实销量可用于今天的一步预测；明天的真实销量不能用于今天发出的多步预测。这两个预测场景的评价方式不同。

## 核心概念与符号

### 先认识数据结构

单个对象按时间记录是一元或多元时间序列；同一时点多个对象是截面；多个对象在多个时点重复观测是面板。采样间隔必须明确，例如日历日、工作日和交易日并不等价。

趋势是长期水平变化；季节项有固定周期；循环的长度通常不固定；剩余项是模型未解释的部分，不应在未检验前就断言完全不可预测。加法分解常写 $y_t=T_t+S_t+R_t$，乘法写 $y_t=T_tS_tR_t$，经典乘法分解通常要求 y 为正。

经典分解中的趋势一般把趋势与较慢循环合并，不会自动分出两个独立分量。默认居中滤波还会使用后续观测，所以全序列分解图只能作描述，不能直接作为历史预测时可用的特征。

### 平稳与相关

弱平稳要求均值恒定、方差有限且协方差只依赖时滞。图像大致水平不是充分证据。自相关 $\rho(h)$ 描述序列与滞后 h 的线性关联；ACF 看各滞后相关，PACF 进一步控制中间滞后的线性影响。

一阶差分 $\Delta y_t=y_t-y_{t-1}$ 可消除某些趋势，但不保证任何序列都平稳；季节差分为 $y_t-y_{t-m}$。过度差分也会引入不必要结构。

### 预测评价

同一预测起点发出未来 H 步预测，和每天更新信息的一步滚动预测不是同一个任务。MAE 与 RMSE 必须在相同目标尺度、时段和预测步长下比较。

## 手算一个最小例子

销量为 $10,12,11,13$。差分为 $2,-1,2$；持久性预测“明天等于今天”在后三个时点的预测为 $10,12,11$，误差为 $2,-1,2$。

因此 MAE 为 $(2+1+2)/3=5/3$，RMSE 为 $\sqrt{(4+1+4)/3}=\sqrt3$。这是一项一步预测基线，而不是对原序列做平滑。

如果固定在第 2 天预测第 3、4 天，持久性模型两步都预测12；不能在预测第4天时偷用第3天真实值11，再把它称为同一个起点的两步预测。

## 建模步骤与完整案例

案例：构造140天带周周期的日销量。最后28天用于最终评估，前112天用于探索和模型选择。假设每天都营业，预测时前一天销量已经结算。

读取真实数据时按日期解析、排序，检查重复、缺口、有限值和单位；不能直接删掉缺失天就把不等间隔序列假装成等间隔。对每个验证起点，仅使用此前数据。

本篇比较训练均值、昨天销量和上周同日销量三个基线。季节周期7是教学数据设定；真实数据需要由业务与诊断支持，不能只因一年365天就预设价格具有年周期。

## Python 实现与结果解释

配套文件：[查看或下载本篇完整脚本](/downloads/math-modeling/mathematical-modeling-time-series.py)。

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

rng = np.random.default_rng(42)
t = np.arange(140)
y = 30 + 0.05*t + 3*np.sin(2*np.pi*t/7) + rng.normal(0, 1, len(t))
series = pd.Series(y, index=pd.date_range("2026-01-01", periods=len(y), freq="D"))
cut = 112
train = series.iloc[:cut]
# 分解只在训练段做描述，不把居中趋势作为预测输入。
decomposition = seasonal_decompose(train, model="additive", period=7)
reconstructed = decomposition.trend+decomposition.seasonal+decomposition.resid
valid = reconstructed.notna()
assert np.allclose(reconstructed[valid], train[valid])

truth = y[cut:]
predictions = {
    "training mean": np.full(len(truth), train.mean()),
    "yesterday": y[cut-1:-1],
    "last week": y[cut-7:-7],
}
for name, prediction in predictions.items():
    mae = np.mean(np.abs(truth-prediction))
    rmse = np.sqrt(np.mean((truth-prediction)**2))
    print(name, "MAE", round(mae, 3), "RMSE", round(rmse, 3))
print("一阶差分均值:", np.diff(train).mean())
assert len(predictions["last week"]) == len(truth)
```

测试段内使用前一天或上周已观察到的销量，符合“每天滚动发出一步预测”的设定；这些测试值没有被用来调参。若业务要求一次提前28天备货，就要重新设计固定起点的28步预测，不能直接复用这份误差结果。

## 方法选择与常见误区

- 无趋势无季节时可比较均值、移动平均和简单指数平滑。
- 平滑趋势可比较 Holt；稳定固定周期可比较季节朴素与 Holt–Winters。
- 差分后仍有自相关时，可考虑 ARIMA 或季节 ARIMA。
- 有外部已知驱动变量时，可做时序回归，但未来外部变量本身也可能需要预测。
- 不规则观测、结构突变、节假日、缺货造成的销量截断，需要先改数据和任务设定。
- 回测只能说明历史规则下的表现；更换业务机制后可能失效。残差仍有明显相关，表示预测模型可能遗漏可利用结构。

## 练习

1. 记录某店连续100天的收入和客流，是面板数据吗？
2. 为什么随机划分时间序列可能高估效果？
3. 训练有趋势，去掉一条直线后能保证平稳吗？
4. 预测7天后的销量与预测明天，误差能直接比较吗？

## 参考答案

1. 这是单个对象的多元时间序列；多个门店跨日期重复观察才形成典型面板。
2. 未来模式、相邻记录与预处理信息可能进入训练；真实预测时这些信息并不可用。
3. 不能，仍可能有随机趋势、季节性、方差变化或结构突变。
4. 通常不能，难度和预测时可用信息不同。应在相同步长、起点和尺度下比较方法。

## 复习卡片与下一步

预测先写清“在何时、用哪些已知信息、预测多远”。下一篇用同样的信息边界手算移动平均，体会平滑和预测之间的一期差别。

[上一篇 主成分分析与降维](/notes/mathematical-modeling-pca/) · [下一篇 时间序列之移动平均法](/notes/mathematical-modeling-moving-average-method/)

## 参考资料

- [Datawhale《数学建模导论》第8章](https://datawhalechina.github.io/intro-mathmodel/#/CH8/第8章-时间序列)：用于建立知识目录，建议与本篇的手算和案例对照阅读。
- [Forecasting Principles and Practice 第3版](https://otexts.com/fpp3/)：分解、基线与滚动评价。
- [statsmodels seasonal_decompose](https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.seasonal_decompose.html)。
