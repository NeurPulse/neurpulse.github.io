---
title: "数学建模：时间序列之指数平滑法"
description: "从递推权重理解水平、趋势与季节模型"
pubDate: 2026-08-10
updatedDate: 2026-10-21
order: 15
category: '数学建模'
tags: ["指数平滑法","零基础"]
---

> [专题学习地图](/notes/mathematical-modeling-roadmap/) · 第 15 / 27 篇

## 学习目标与先修知识

目标是手算一次指数平滑，理解初始化、权重衰减和一步预测，并区分 Holt、Brown 重复平滑与 Holt–Winters。先修为加权平均、时间切分与移动平均。知道什么时候需要趋势或季节项，比背“几次平滑”的名字更重要。

可回看：[时间序列之移动平均法](/notes/mathematical-modeling-moving-average-method/)。

## 问题背景与使用时机

移动平均中，刚超出窗口的旧观测权重会突然从 $1/N$ 变为0。指数平滑让旧信息逐步衰减，同时只需维护少量状态。

近期信息更有用是模型假设，不是所有序列的定律。若未来主要由节假日、促销或价格决定，只更新历史水平可能不足以预测。

## 核心概念与符号

### 简单指数平滑

记 $\ell_t$ 为看过第 t 期观测后的水平：

$$
\ell_t=\alpha y_t+(1-\alpha)\ell_{t-1},
\qquad \hat y_{t+1\mid t}=\ell_t,\quad0<\alpha<1.
$$

展开后近期观测权重依次为 $\alpha,\alpha(1-\alpha),\alpha(1-\alpha)^2$，还要加上初始化项的剩余权重。$\alpha$ 大则响应快，小则更平滑；一次更新的预测误差为 $y_t-\ell_{t-1}$。

### 带趋势的 Holt

$$
\ell_t=\alpha y_t+(1-\alpha)(\ell_{t-1}+b_{t-1}),\qquad
b_t=\beta(\ell_t-\ell_{t-1})+(1-\beta)b_{t-1},
$$

$$
\hat y_{t+h\mid t}=\ell_t+h b_t.
$$

水平和斜率分开更新，阻尼趋势版本会减少远期线性外推的增长幅度。

### 季节与重复平滑

Holt–Winters 再引入周期 m 的季节状态，可用加法或乘法季节项；乘法形式通常要求正值及幅度随水平变化。

Brown 双重平滑对同一序列重复两次平滑，设结果为 $S_t^{(1)},S_t^{(2)}$，可取 $a_t=2S_t^{(1)}-S_t^{(2)}$、$b_t=\alpha(S_t^{(1)}-S_t^{(2)})/(1-\alpha)$。三重 Brown 再平滑一次，可构造局部二次趋势。重复次数与季节项不是一回事，不能把 Brown 三次平滑直接等同于 Holt–Winters。

## 手算一个最小例子

观测 $10,12,14$，取初始水平 $\ell_1=10$、$\alpha=0.5$。

第2期到来前预测为10，误差 $12-10=2$；更新后 $\ell_2=0.5\times12+0.5\times10=11$。第3期预测为11，误差3；更新后 $\ell_3=0.5\times14+0.5\times11=12.5$，所以第4期预测为12.5。

展开最后水平：

$$
\ell_3=0.5\times14+0.25\times12+0.25\times10=12.5.
$$

最后0.25是初始状态的剩余影响，不应漏掉。平滑值12.5已经看过14，而预测14时只能用11。

若 Holt 初始化为 $\ell_1=10,b_1=2$，且观测保持每期加2，那么预测12、14并保持斜率2；这体现了趋势状态的作用。

## 建模步骤与完整案例

案例：对带周季节的教学销量，在前84天拟合状态，用后28天验证固定起点的28步预测，选择简单水平、Holt 或加法 Holt–Winters；最后用前112天重新拟合选中的结构，再一次预测最后28天。

这里与前面的滚动一步示例任务不同：本篇明确评估“一次提前28天”。所有模型都用同一预测起点、步长和误差尺度。复杂模型可能更合适，也可能因样本少或季节不稳定而更差，结论由验证决定。

## Python 实现与结果解释

配套文件：[查看或下载本篇完整脚本](/downloads/math-modeling/mathematical-modeling-exponential-smoothing.py)。

```python
import numpy as np
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing

level = 10.
one_step_errors = []
for observation in [12., 14.]:
    one_step_errors.append(observation-level)
    level = 0.5*observation + 0.5*level
assert np.isclose(level, 12.5)
assert np.allclose(one_step_errors, [2., 3.])

rng = np.random.default_rng(42)
t = np.arange(140)
y = 30 + 0.05*t + 3*np.sin(2*np.pi*t/7) + rng.normal(0, 1, len(t))
def fit_kind(kind, data):
    if kind == "level":
        return SimpleExpSmoothing(data, initialization_method="estimated").fit()
    if kind == "trend":
        return Holt(data, initialization_method="estimated").fit()
    return ExponentialSmoothing(
        data, trend="add", seasonal="add", seasonal_periods=7,
        initialization_method="estimated"
    ).fit()

validation = {}
for kind in ["level", "trend", "seasonal"]:
    forecast = np.asarray(fit_kind(kind, y[:84]).forecast(28))
    validation[kind] = np.mean(np.abs(y[84:112]-forecast))
selected = min(validation, key=validation.get)
final_forecast = np.asarray(fit_kind(selected, y[:112]).forecast(28))
print("验证MAE:", validation, "选择:", selected)
print("28步最终测试MAE:", np.mean(np.abs(y[112:]-final_forecast)))
assert final_forecast.shape == (28,)
assert np.isfinite(final_forecast).all()
```

第一项误差没有计入，因为用第一个观测初始化水平，不能把初始化当作一次成功预测。打印的误差是 MAE；若改用平方误差平均再开根号，应称 RMSE，不是标准误。

## 方法选择与常见误区

| 观察到的结构 | 可比较的模型 |
|---|---|
| 水平稳定，无固定季节 | 简单指数平滑与均值基线 |
| 局部近似线性趋势 | Holt、阻尼 Holt |
| 固定周期季节 | 季节朴素、Holt–Winters |
| 趋势叠加季节 | 带趋势与季节的状态模型 |
| 外部变量主导或明显自相关遗漏 | 时序回归或 ARIMA 等 |

平滑不自动填补缺失日期。季节模型需要足够周期、稳定频率及合适初始化。估计出的平滑参数只是当前数据与损失下的结果，不是永恒规律。拟合曲线与原序列对齐，也不等于它们都属于真正的事前预测。

## 练习

1. 初始水平20，$\alpha=0.2$，新观测30，更新后是多少？
2. $\alpha$ 接近1会怎样？
3. Brown 三重平滑一定能表达周季节吗？
4. 为什么不能用最后28天挑出最佳模型，再报告同一段是独立测试？

## 参考答案

1. $0.2\times30+0.8\times20=22$，但事前预测该次观测仍是20。
2. 新观测影响很大，反应快，抑制噪声的能力弱；简单指数平滑接近持久性预测。
3. 不一定，它描述重复平滑和局部趋势，没有显式的7期季节状态。
4. 测试误差已参与模型选择，独立性被破坏。应在更早验证段选择，再只用最终段评价。

## 复习卡片与下一步

复习卡：状态在观察后更新，预测只用更新前的信息；先明确水平、趋势和季节，再选择模型。下一篇学习从相关结构和差分出发的 ARIMA。

[上一篇 时间序列之移动平均法](/notes/mathematical-modeling-moving-average-method/) · [下一篇 ARIMA 与相关时间序列](/notes/mathematical-modeling-arima/)

## 参考资料

- [Datawhale《数学建模导论》第8章](https://datawhalechina.github.io/intro-mathmodel/#/CH8/第8章-时间序列)：用于建立知识目录，建议与本篇的手算和案例对照阅读。
- [指数平滑与 ETS](https://otexts.com/fpp3/expsmooth.html)。
- [statsmodels Holt–Winters](https://www.statsmodels.org/stable/examples/notebooks/generated/exponential_smoothing.html)。
