---
title: '数学建模：时间序列'
description: '介绍什么是时间序列，时间序列的应用'
pubDate: 2026-08-10
updatedDate: 2026-10-19
order: 6
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

1. 截面数据：同一时点多个观察单位的数据，如同一天多个河流监测站的各项指标。
2. 面板数据：多个观察单位在多个时点的重复观测，同时包含个体维度与时间维度。单个站点多个指标的日序列属于多元时间序列，不因指标多就成为面板数据。
3. 时间序列：一个或多个变量按时间排列的观测。

弱平稳（协方差平稳）要求有限二阶矩、均值为常数，且协方差只依赖时滞：

$$
\mathbb E[Y_t]=\mu,\qquad\operatorname{Cov}(Y_t,Y_{t+h})=\gamma(h).
$$

这不等于一条样本路径的波动幅度处处完全相同，也不能只凭图像水平就确定平稳性。

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
   特点：各项使用相同量纲相加，适合波动幅度不随水平明显变化的情况；分解形式本身并不保证各项统计独立。

2. 乘法模型:
   $$
   \boldsymbol{Y[t] = T[t] \times S[t] \times C[t] \times I[t]}
   $$
   特点：仅趋势项T与最终序列Y量纲一致；季节、循环和不规则项为无量纲的比例因子；波动幅度会随趋势增减同步放大 / 缩小。

   > 乘法分解不要求不规则因子服从正态分布。经典乘法分解通常要求序列严格为正；取对数后可转成加法形式。
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

示例需要按日等间隔、无缺失的 `Bitcoin.csv`，含 `Date` 与正的 `Closing Price (USD)` 列，且至少覆盖两轮设定周期。`period=365` 只是年度周期的示范设定，不代表已证明数据有年度季节性。先排序并检查重复日期、缺口；直接 `dropna()` 可能破坏等间隔条件。

`seasonal_decompose` 的 `trend` 通常是趋势与循环的合并项，并不分别估计上述 T 与 C；默认居中滤波会使用未来数据，不能直接把全序列分解结果作为在线预测特征。
