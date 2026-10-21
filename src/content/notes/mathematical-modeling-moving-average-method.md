---
title: "数学建模：时间序列之移动平均法"
description: "手算平滑与一步预测，理解窗口、滞后和趋势修正"
pubDate: 2026-08-10
updatedDate: 2026-10-21
order: 14
category: '数学建模'
tags: ["移动平均法","零基础"]
---

> [专题学习地图](/notes/mathematical-modeling-roadmap/) · 第 14 / 27 篇

## 学习目标与先修知识

目标是手算简单、加权和二次移动平均，区分平滑值与预测值，理解窗口为什么产生滞后，并用时间验证选择窗口。先修为均值、数组切片和时间序列信息边界。

可回看：[时间序列分析与验证](/notes/mathematical-modeling-time-series/)。

## 问题背景与使用时机

每天销量有波动，想看最近一周的水平，可以平均最近7天。平均会削弱偶然起伏，但同时会落后于变化中的趋势；这是一种明确的权衡。

移动平均既可作描述性的平滑工具，也可构造短期预测基线。不能因为画出来曲线很贴近观测，就把包含当前值的平滑误差当成预测误差。

## 核心概念与符号

窗口长度为 N，包含当前观测的一次平滑值为：

$$
M_t^{(1)}=\frac1N\sum_{j=0}^{N-1}y_{t-j},\qquad t\ge N.
$$

滚动更新为 $M_t^{(1)}=M_{t-1}^{(1)}+(y_t-y_{t-N})/N$，新数据进入，最旧数据离开。预测下一时刻可取 $\hat y_{t+1\mid t}=M_t^{(1)}$。

加权移动平均为 $\sum_jw_jy_{t-j}$，权重非负且和为1；应明确最近值对应哪个权重。

再次平均得到 $M_t^{(2)}$。当局部趋势近似线性时，可构造：

$$
a_t=2M_t^{(1)}-M_t^{(2)},\qquad
b_t=\frac2{N-1}(M_t^{(1)}-M_t^{(2)}),\qquad
\hat y_{t+h\mid t}=a_t+b_th.
$$

要求 $N>1$，二次平均最早在 $t=2N-1$ 可计算。这里“二次”指重复两次平均，不是拟合二次曲线，也不自动包含季节项。

## 手算一个最小例子

序列为 $10,12,14,16,18,20$，N=3。一次平滑从第3项开始为12、14、16、18；二次平滑在第5项为 $(12+14+16)/3=14$，第6项为 $(14+16+18)/3=16$。

最后一期 $M_6^{(1)}=18$，$M_6^{(2)}=16$，故：

$$
a_6=2\times18-16=20,\qquad
b_6=\frac2{2}(18-16)=2.
$$

趋势预测下一期为22，而简单移动平均预测为18。序列每期增长2，简单平均落后于当前水平，趋势修正则在这个理想例子中恢复线性增长。

若用最近三个数10、12、14，按从旧到新权重0.2、0.3、0.5，加权平均为 $2+3.6+7=12.6$。

## 建模步骤与完整案例

案例：比较日销量一步预测的多个窗口。教学序列共140天，前84天为初始训练，随后28天用于选窗口，最后28天只作最终检验。

每个预测只用前 N 个观测。窗口候选预先定为3、7、14，用验证 MAE 选取；到了测试段，每天允许观察前一天已实现销量，但不再根据测试误差换窗口。

如果验证目标是“一次预测未来28天”，应另写多步模型。本篇只评估滚动一步。

## Python 实现与结果解释

配套文件：[查看或下载本篇完整脚本](/downloads/math-modeling/mathematical-modeling-moving-average-method.py)。

```python
import numpy as np

def moving_average(values, window):
    values = np.asarray(values, dtype=float)
    if (values.ndim != 1 or not np.isfinite(values).all()
            or isinstance(window, bool) or not isinstance(window, (int, np.integer))
            or not 1 <= window <= len(values)):
        raise ValueError("需要有限一维序列，窗口为有效整数")
    out = np.full(len(values), np.nan)
    out[window-1:] = np.convolve(values, np.ones(window)/window, mode="valid")
    return out

small = np.arange(10., 22., 2.)
first = moving_average(small, 3)
second = moving_average(first[2:], 3)
a = 2*first[-1]-second[-1]
b = 2*(first[-1]-second[-1])/(3-1)
assert np.isclose(a+b, 22.)

rng = np.random.default_rng(42)
t = np.arange(140)
y = 30 + 0.05*t + 3*np.sin(2*np.pi*t/7) + rng.normal(0, 1, len(t))
def predict_at(values, time, window):
    return np.mean(values[time-window:time])

validation = {}
for window in [3, 7, 14]:
    prediction = np.array([predict_at(y, i, window) for i in range(84, 112)])
    validation[window] = np.mean(np.abs(y[84:112]-prediction))
best_window = min(validation, key=validation.get)
test_prediction = np.array([predict_at(y, i, best_window) for i in range(112, 140)])
print("验证MAE:", validation, "选中窗口:", best_window)
print("最终测试MAE:", np.mean(np.abs(y[112:]-test_prediction)))
assert np.isnan(first[:2]).all()
assert np.allclose(first[2:], [12., 14., 16., 18.])
```

开头不满窗口的位置保留 NaN，表示没有完整历史，而不是填一个看似平滑的数字。N 等于序列长度时可以计算最后一个平滑值，但序列内没有可评价的下一期预测，应避免对空误差数组求平均。

## 方法选择与常见误区

窗口小，反应快但噪声大；窗口大，平滑强但滞后更明显。没有对所有序列都最好的 N，也不要求所有预测先去趋势。

强趋势时可比较趋势移动平均或 Holt；固定季节时应保留季节信息；突变时长窗口可能长期跟不上。居中移动平均使用后面的观测，只适合相应的离线分析，不能直接充当实时预测。

不要把这里的“观测移动平均”与 ARIMA 中的 MA 混淆：后者线性组合的是历史随机冲击。

## 练习

1. 数列 $2,4,6,8$，窗口2的一次平均从哪里开始，分别是多少？
2. 预测第4项8时，合法的一步移动平均预测是多少？
3. 为什么用第4项参与其自身预测会得到过于乐观的误差？
4. N=1 时能否使用上面的趋势修正公式？

## 参考答案

1. 从第2项开始为3、5、7；第1项缺少窗口。
2. 只平均第2、3项，得到 $(4+6)/2=5$。
3. 预测目标已经进入输入，减小的误差一部分来自泄漏，而不是提前预测能力。
4. 不能，公式分母 $N-1$ 为0；N=1 的一次平滑也仅等于当前观测。

## 复习卡片与下一步

复习卡：先确定窗口是否包含当前值；要预测就后移一期；窗口在验证段选择；趋势与季节需要额外结构。下一篇用指数衰减权重代替突然进出窗口。

[上一篇 时间序列分析与验证](/notes/mathematical-modeling-time-series/) · [下一篇 时间序列之指数平滑法](/notes/mathematical-modeling-exponential-smoothing/)

## 参考资料

- [Datawhale《数学建模导论》第8章](https://datawhalechina.github.io/intro-mathmodel/#/CH8/第8章-时间序列)：用于建立知识目录，建议与本篇的手算和案例对照阅读。
- [预测工具箱与误差评价](https://otexts.com/fpp3/accuracy.html)。
