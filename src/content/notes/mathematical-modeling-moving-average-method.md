---
title: '数学建模：时间序列之移动平均法'
description: '常用于大趋势建模'
pubDate: 2026-08-10
updatedDate: 2026-10-19
order: 7
category: '数学建模'
tags: ['移动平均法']
---

# 移动平均法

## 一、方法概述
移动平均法是时间序列短期预测中常用的平滑预测技术，核心思路为按时间顺序逐期推移，选取固定数量的近期实际数据计算序列均值，以此削弱周期变动、随机波动带来的数据起伏，提取序列长期趋势线，依托趋势开展预测。

### 适用条件

序列无快速大幅增减、无显著季节性因素时，消除随机波动的效果最佳；按预测权重可分为**简单移动平均**与**加权移动平均**两类。

## 二、分场景模型与公式
### 1. 一次移动平均法
**适用场景**：数据围绕固定水平上下浮动，趋势线为水平线，无明显线性趋势

- 初始均值公式：
$$M_{(1)}^N=\frac{1}{N}\sum_{i=1}^N y_i$$
- 递推公式：
$$M_{(1)}^t=\frac{1}{N}\sum_{i=0}^{N-1}y_{t-i}=M_{(1)}^{t-1}+\frac{y_t-y_{t-N}}{N},\qquad t>N$$
参数说明：$N$为移动平均窗口长度，$y_t$为$t$时刻真实时序值

### 2. 二次移动平均法
**适用场景**：序列呈现线性趋势（趋势线为一次函数），对一次移动平均的结果再次做移动平均

- 递推公式：
$$M_{(2)}^t=\frac{1}{N}\sum_{i=0}^{N-1}M_{(1)}^{t-i},\qquad t\ge 2N-1$$

### 3. 趋势移动平均法
**适用场景**：局部趋势近似线性时作多期外推；这个趋势公式本身不包含季节项，要求 $N>1$

- 预测方程：
$$\hat{y}_{T+m}=a_T+b_T\cdot m,\quad m=1,2,\dots$$
- 参数计算公式：
$$a_T=2M_{(1)}^T-M_{(2)}^T,\quad b_T=\frac{2}{N-1}\big(M_{(1)}^T-M_{(2)}^T\big)$$
参数说明：$T$为当前最新时刻，$m$为未来预测期数

## 三、预测效果衡量指标：均方根误差

$$S=\sqrt{\frac{\sum_{t=N+1}^T (\hat{y}_t-y_t)^2}{T-N}}$$
- $\hat{y}_t$：$t$时刻预测值，$y_t$：$t$时刻真实值，$T$：总样本量，$N$：窗口长度
- 误差$S$越小，代表模型拟合预测效果越好

## 四、核心分析要点
1. 趋势分离原则：若时间序列存在明显直线/曲线趋势，必须先分离趋势成分再开展后续分析；二次移动平均、趋势移动平均的核心作用就是拟合并分离趋势项。
2. 平稳性优势：剥离趋势后的剩余序列更贴近平稳序列，平稳时间序列的分析建模难度远低于非平稳序列。

## 五、方法适配对照表
| 方法类型 | 适配趋势形态 |
| ---- | ---- |
| 一次移动平均 | 水平平稳、无明显趋势 |
| 二次移动平均 | 单向线性趋势 |
| 趋势移动平均 | 局部线性趋势，不直接建模季节性 |

## 六、优劣总结
- 优点：计算简便，平滑随机波动效果好，适合短期预测
- 局限：不适用于长期预测；强季节性、剧烈波动序列需先做预处理再使用

## 七、时序知识联动拓展
以比特币价格这类具备长期上涨趋势的非平稳序列为例，不可直接使用一次移动平均，需通过二次/趋势移动平均剥离上涨趋势，再对剩余残差做平稳性分析，与时序分解（T趋势/S季节/I噪声）的分析逻辑相互契合。



```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
y = 20 + rng.normal(size=180)  # 可替换为按时间排序的单变量序列

def MoveAverage(y, N):
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or not np.isfinite(y).all():
        raise ValueError("y 必须是一维有限数值序列")
    if isinstance(N, bool) or not isinstance(N, (int, np.integer)) or not 1 <= N <= len(y):
        raise ValueError("N 必须是 1 到序列长度之间的整数")
    result = np.full(len(y), np.nan)
    result[N-1:] = np.convolve(y, np.ones(N) / N, mode="valid")
    return result

smooth = MoveAverage(y, 30)
# 平滑值包含当前观测；预测 y[t] 时只能用截至 t-1 的平滑值。
prediction = np.concatenate(([np.nan], smooth[:-1]))
valid = np.isfinite(prediction)
rmse = np.sqrt(np.mean((y[valid] - prediction[valid]) ** 2))
print("滚动一步预测 RMSE:", rmse)
plt.plot(y, label="observed")
plt.plot(smooth, label="30-step trailing average")
plt.legend()
plt.show()
```

开头不足窗口的位置保留为 `NaN`，避免用第一条数据伪造历史窗口；输出与原序列同长度。滚动评估每一步只能访问过去数据，窗口大小应在验证段选择，最终测试段不参与选择。

