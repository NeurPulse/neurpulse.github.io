---
title: '数学建模：时间序列之指数平滑法'
description: '常用于大趋势建模'
pubDate: 2026-08-10
updatedDate: 2026-10-19
order: 8
category: '数学建模'
tags: ['指数平滑法']

---

# 指数平滑法

## 一、核心基本思想
指数平滑法的核心逻辑：**距离预测时刻越近的历史数据，赋予越高权重，权重随时间按指数规律衰减**。
以股市预测为例：多年前的历史数据参考价值低，近期数据影响更大，模型会自动给近期数据分配更高权重，契合时序预测的现实规律。

## 二、三级形式与适用场景
| 平滑类型 | 适配序列特征 |
| ---- | ---- |
| 一次指数平滑 | 无趋势、无季节性的平稳序列 |
| 二次指数平滑 | 存在趋势、无季节性的序列 |
| Brown 三次平滑 | 局部二次趋势，不含显式季节项 |
| Holt–Winters | 水平、趋势与固定周期季节项 |

## 三、核心公式推导
### 1. 一次指数平滑
递推公式：
$$S_{(1)}^t = \alpha y_t + (1-\alpha)S_{(1)}^{t-1}$$
有限历史展开（包括初始化项）：
$$S_{(1)}^t=(1-\alpha)^tS_{(1)}^0+\alpha\sum_{j=0}^{t-1}(1-\alpha)^j y_{t-j}$$
参数说明：
- $\alpha$：平滑系数（修正幅度），$0<\alpha<1$，调节新旧数据的权重占比；
- $y_t$：$t$时刻真实时序值；$S_{(1)}^{t-1}$：上一期一次平滑结果。

### 2. Brown 三次指数平滑（重复平滑）
$$
\begin{cases}
S_t^{(1)} =\alpha y_t + (1-\alpha)S_{t-1}^{(1)}\\
S_t^{(2)} =\alpha S_t^{(1)} + (1-\alpha)S_{t-1}^{(2)}\\
S_t^{(3)} =\alpha S_t^{(2)} + (1-\alpha)S_{t-1}^{(3)}
\end{cases}
$$

### 3. Brown 三次平滑的二次趋势预测
可做多期外推预测：
$$\hat{y}_{t+m}=a_t + b_t m + c_t m^2$$
$m$ 代表预测步数，$a_t,b_t,c_t$ 由三阶平滑值计算。这套重复平滑是 Brown 的二次趋势形式，不能自动拟合季节性；Holt–Winters 通常也被称为“三次指数平滑”，但使用独立的水平、趋势和季节更新方程，二者不能混用。

## 四、关键对比注意事项
与移动平均法的数据长度差异：
1. 移动平均：受窗口长度限制，预测得到的趋势序列**数据量会少于原始序列**；
2. 指数平滑：全程逐期递推计算，最终趋势线长度**和原始时间序列长度完全对齐**，不会出现数据缺失。

## 五、方法特点总结
1. 权重特性：天然实现近大远小的指数衰减加权，无需手动设置窗口权重；
2. 模型选择：区分一次平滑、趋势模型与 Holt–Winters 季节模型，不按重复平滑的次数直接判断是否包含季节项；
3. 实操要点：调节平滑系数$\alpha$即可控制修正幅度，可通过代码批量测试不同$\alpha$的预测效果；
4. 效果验证：是否存在稳定季节性应由数据支持；复杂模型不保证优于一次平滑或持久性基线，应按时间做验证。



```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========一次指数平滑函数==========
def ExpMove(y, a):
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or len(y) < 2 or not np.isfinite(y).all() or not 0 < a < 1:
        raise ValueError("要求至少两个有限观测，且 0 < a < 1")
    n = len(y)
    M = np.zeros(n)
    M[0] = y[0]   # 初始值：第一期平滑值等于第一个真实值
    for i in range(1, n):
        M[i] = a * y[i-1] + (1 - a) * M[i-1]  # 用前一期真实值 + 前一期平滑值
    return M

# =========读取比特币数据==========
df = pd.read_csv("Bitcoin.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.set_index("Date", drop=True).sort_index()
y = df["Closing Price (USD)"].values   # 转为numpy数组

# =========三种平滑系数对比==========
yt1 = ExpMove(y, 0.2)   # α=0.2，平滑程度高，曲线平缓
yt2 = ExpMove(y, 0.5)   # α=0.5，中等平滑
yt3 = ExpMove(y, 0.8)   # α=0.8，平滑程度低，跟随性强

# =========计算预测标准误差S==========
s1 = np.sqrt(((y[1:] - yt1[1:])**2).mean())
s2 = np.sqrt(((y[1:] - yt2[1:])**2).mean())
s3 = np.sqrt(((y[1:] - yt3[1:])**2).mean())

print(f"α=0.2 标准误差: {s1:.2f}")
print(f"α=0.5 标准误差: {s2:.2f}")
print(f"α=0.8 标准误差: {s3:.2f}")

# =========保存到Excel==========
d = pd.DataFrame(np.c_[y, yt1, yt2, yt3], 
                 columns=['真实值', 'α=0.2', 'α=0.5', 'α=0.8'])
d.to_excel('exp_smooth_example.xlsx')

# =========绘图==========
plt.figure(figsize=(12, 5))
plt.plot(y, label='真实值', linewidth=1.5)
plt.plot(yt1, label='α=0.2（平滑）', linestyle='--')
plt.plot(yt2, label='α=0.5（中等）', linestyle='--')
plt.plot(yt3, label='α=0.8（跟随）', linestyle='--')
plt.title('一次指数平滑 — 不同α对比')
plt.xlabel('时间（天）')
plt.ylabel('比特币价格（美元）')
plt.legend()
plt.tight_layout()
plt.show()

print(d)
```

代码需要本地 `Bitcoin.csv`，包含 `Date` 与 `Closing Price (USD)` 两列，价格为有限数值。`ExpMove` 返回的是使用前一期观测的**一步预测**，并非包含当前值的平滑序列；第一项只是初始化，RMSE 已排除。示例是历史滚动误差，不是独立测试结果；选择 $\alpha$ 时应使用较早的训练、验证段，再在未来留出段评估。

参考：[Forecasting: Principles and Practice — Exponential smoothing](https://otexts.com/fpp3/expsmooth.html)。
