---
title: '数学建模：时间序列'
description: '介绍什么是时间序列，时间序列的应用'
pubDate: 2026-08-10
category: '数学建模'
tags: ['时间序列']
---

## 时间序列

**定义**：

> 以时间作为索引、具备先后顺序的序列数据。采样间隔可以为日、小时、分钟、秒；传统时序模型大多默认等时间间隔采样。

**应用**：

1. 天气预报中每一天的天气按照时间构成了一个序列，这属于离散的时间序列（可选用马尔可夫、ARIMA、深度学习等多种模型建模)
2. 股票每日开盘价、收盘价随时间变化，也是时间序列。

时间序列两大目标：特征分析和预测

**何为预测？**

基于历史时序数据，对未来时刻数值做推断。

例：利用历史气象数据预测未来24小时天气；利用历史股价预测未来一周股价走势。

**何为特征分析？**

通过模型估计参数，挖掘序列内在统计特征，结合业务/领域知识解读现象。

> 参数学习：是模型的手段，不是最终任务，通过拟合得到模型参数，刻画趋势、周期、相关性等序列特征。

**时间序列的分解**：

> 为了能更好的对一个时间序列进行分析，我们需要用到分解

基础概念：

1. 截面数据：某一个时间点（单日）的一整行全部观测指标，例如某天河流的 pH、溶解氧、重金属、细菌含量，一行就是一个截面。
2. 面板数据：把不同日期的多个截面，按时间顺序拼接整合，同时包含时间维度 + 多指标维度的数据集。
3. 时间序列数据：单独**提取某一项指标**，按日期先后排列形成的数据（如连续每日 pH 值序列），核心特征是数值随时间变化。

特殊的时间序列：平稳型时间序列

序列的**均值、方差、协方差不随时间发生改变**

1. 长期趋势线呈水平状态，无持续上涨 / 下跌走向
2. 全程波动幅度均匀，不存在某段剧烈波动、某段极度平稳的情况
3. 数值分布疏密一致，不会局部密集、局部稀疏

时间序列四个要素：

- **长期趋势 T**：较长时间段内整体持续变化走向，多为平滑曲线，代表长期发展态势。 
- **季节波动 S**：固定短周期带来的周期性变动，不限于一年四季，24小时日内周期也属于季节项。周期固定且较短。 
- **循环波动 C**：中长期周期性变动，**没有严格固定周期**，周期长度大于季节波动。
- **不规则波动 I（噪声）**：偶然随机因素带来的扰动，无规律，不可预测。

时序分解模型（用于拆分四大要素）

> 联系：时间序列是待分析的数据对象，分解模型是拆分解析时序数据的数学方法，把原始序列$Y[t]$拆解为T、S、C、I四个组分。

1. 加法模型：
   $$
   \boldsymbol{Y[t] = T[t] + S[t] + C[t] + I[t]}
   $$
   特点：四个组成成分相互独立，所有量纲一致；波动幅度不会随趋势大小改变。

2. 乘法模型:
   $$
   \boldsymbol{Y[t] = T[t] \times S[t] \times C[t] \times I[t]}
   $$
   特点：仅趋势项T与最终序列Y量纲一致；季节、循环项为比例系数，不规则项服从正态分布；波动幅度会随趋势增减同步放大 / 缩小。

   > 理论假设：不规则项I为独立随机变量，服从正态分布；**现实数据不一定满足该假设**。
   >
   > 适合：波动幅度会跟随趋势同步放大/缩小的数据。



```python
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose


df = pd.read_csv("Bitcoin.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.set_index("Date", drop=True)
df["Bitcoin"] = df["Closing Price (USD)"]
y = df.Bitcoin

# 绘制时间序列图
plt.figure(figsize=(12, 4))
plt.plot(y, label='比特币价格')
plt.title('比特币时间序列图')
plt.xlabel('日期')
plt.ylabel('价格（美元）')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# 时序分解
decompose_result = seasonal_decompose(y.dropna(), model="multiplicative", period=365)
plt.rcParams["figure.figsize"] = (14, 10)
fig = decompose_result.plot()

axes = fig.axes
axes[0].set_title('原始序列')
axes[1].set_title('长期趋势 T')
axes[2].set_title('季节波动 S')
axes[3].set_title('不规则噪声 I')

axes[0].set_ylabel('价格')
axes[1].set_ylabel('趋势')
axes[2].set_ylabel('季节')
axes[3].set_ylabel('残差噪声')

plt.tight_layout()
plt.show()

T = decompose_result.trend
S = decompose_result.seasonal
I = decompose_result.resid
```



```python
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# 读取数据
df = pd.read_csv("Bitcoin.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.set_index("Date", drop=True)
df["Bitcoin"] = df["Closing Price (USD)"]



# 加法模型时序分解
res = sm.tsa.seasonal_decompose(df.Bitcoin, model='additive', period=365)

fig = res.plot()
# 设置总标题
plt.suptitle('时序分解 — 加法模型', y=1.02)


axes = fig.axes
axes[0].set_title('原始序列')
axes[1].set_title('长期趋势 T')
axes[2].set_title('季节波动 S')
axes[3].set_title('不规则残差 I')

axes[0].set_ylabel('价格')
axes[1].set_ylabel('趋势')
axes[2].set_ylabel('季节')
axes[3].set_ylabel('残差')

plt.tight_layout()
plt.show()

# 提取各分量
trend_add = res.trend
seasonal_add = res.seasonal
resid_add = res.resid


# ----------------乘法模型对比（中文）----------------
res_mul = sm.tsa.seasonal_decompose(df.Bitcoin, model='multiplicative', period=365)
fig2 = res_mul.plot()
plt.suptitle('时序分解 — 乘法模型', y=1.02)

axes2 = fig2.axes
axes2[0].set_title('原始序列')
axes2[1].set_title('长期趋势 T')
axes2[2].set_title('季节波动 S')
axes2[3].set_title('不规则残差 I')

axes2[0].set_ylabel('价格')
axes2[1].set_ylabel('趋势')
axes2[2].set_ylabel('季节')
axes2[3].set_ylabel('残差')

plt.tight_layout()
plt.show()

trend_mul = res_mul.trend
seasonal_mul = res_mul.seasonal
resid_mul = res_mul.resid
```