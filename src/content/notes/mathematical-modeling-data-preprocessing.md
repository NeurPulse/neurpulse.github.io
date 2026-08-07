---
title: '数学建模：数据预处理与插值'
description: '数据清洗、标准化与插值方法'
pubDate: 2026-08-07
category: '数学建模'
tags: ['数据预处理','插值','Pandas','Scikit-learn']
---

## 数据处理与拟合模型



何为数据？

- 数值类数据，例如结构化的excel表格和SQL文件。
- 文本类数据，例如新闻报道、微博评论、餐饮点评等文字。
- 图像类数据，以一定尺寸的黑白或彩色图像在计算机内存储。
- 音频类数据，例如音乐、电话录音等。
- 信号类数据，例如地震波的波形、电磁波信号、脑电信号等。

### 数据预处理

在现实生活中，我们得到的数据往往很混乱，不全面，这对一个模型来说是不利的。无法从中有效的识别并提取信息。

在数据处理中会遇到的问题：空缺，重复，异常记录

怎么处理？

1. 重复数据：直接删除

2. 缺失数据：

   根据缺失率来选择处理方法

   * 缺失数据占比较少（5%)    删除行
   * 缺失数据占比少（5% - 20%)  可以进行填充（常数填充、均值填充），插值（线性插值，三次样条插值，拉格朗日插值）
   * 缺失数据占比较多（20% - 40%)  可以用预测方法填充缺失数据（机器学习）
   * 缺失数据占比多（50%)   删除列

数据处理借助pandas

相应的说明

```python
（1）创建pandas dataframe
df = pd.DataFrame({'From_To': ['LoNDon_paris', 'MAdrid_miLAN', 'londON_StockhOlm',
                               'Budapest_PaRis', 'Brussels_londOn'],
              'FlightNumber': [10045, np.nan, 10065, np.nan, 10085],
              'RecentDelays': [[23, 47], [], [24, 43, 87], [13], [67, 32]],
                   'Airline': ['KLM(!)', '<Air France> (12)', '(British Airways. )',
                               '12. Air France', '"Swiss Air"']})
df

（2）FlightNumber列中有某些缺失值，缺失值常用nan表示，请在该列中添加10055与10075填充该缺失值。
df['FlightNumber'] = df['FlightNumber'].interpolate().astype(int)

（3）由于列From_To 代表从地点A到地点B，因此可以将这列拆分成两列，并赋予为列From与To。
temp = df['From_To'].str.split("_", expand=True)
temp.columns = ['From', 'To']

（4）将列From和To转化成只有首字母大写的形式。
temp['From'] = temp['From'].str.capitalize()
temp['To'] = temp['To'].str.capitalize()

（5）将列From_To从df中去除，并把列From和To添加到df中
df.drop('From_To', axis=1, inplace=True)
df[['From', 'To']] = temp

（6）清除列中的特殊字符，只留下航空公司的名字。
df['Airline'] = df['Airline'].str.extract(r'([a-zA-Z\s]+)', expand=False).str.strip()

（7）在 RecentDelays 列中，值已作为列表输入到 DataFrame 中。我们希望每个第一个值在它自己的列中，每个第二个值在它自己的列中，依此类推。如果没有第 N 个值，则该值应为 NaN。将 Series 列表展开为名为 的 DataFrame delays，重命名列delay_1，delay_2等等，并将不需要的 RecentDelays 列替换df为delays。
delays = df['RecentDelays'].apply(pd.Series)
delays.columns = ['delay_%s' % i for i in range(1, len(delays.columns)+1)]
df = df.drop('RecentDelays', axis=1).join(delays, how='left')

（8）将delay_i列的控制nan都填为自身的平均值。
for i in range(1, 4):
    df[f'delay_{i}'] = df[f'delay_{i}'].fillna(np.mean(df[f'delay_{i}']))

（9）在df中增加一行，值与FlightNumber=10085的行保持一致。
df = df._append(df.loc[df['FlightNumber'] == 10085, :], ignore_index=True)

（10）对df进行去重，由于df添加了一行的值与FlightNumber=10085的行一样的行，因此去重时需要去掉。
df = df.drop_duplicates()
```

### 插值模型

#### 线性插值

利用两个点求线性方程来进行插值

比如有原始数据列{y}和数据的下标{x} 注：*数据下标x可能并不是固定频率的连续取值而是和y一样存在缺失的*

有数据点(xk,yk)和(xk+1,yk+1)，需要对两个点之间构造直线进行填充。

根据直线的点斜式方程  $\to$ 已知直线过点 $P_0(x_0,y_0)$，斜率为 k  方程：$\boldsymbol{y-y_0=k(x-x_0)}$

得到：
$$
L_1(x) = y_k + \frac{y_{k + 1} - y_k}{x_{k + 1} - x_k}(x - x_k)
$$

```python
import numpy as np
#数据准备
X = np.arange(-2*np.pi, 2*np.pi, 1) # 定义样本点X，从-pi到pi每次间隔1
Y = np.sin(X) # 定义样本点Y，形成sin函数
new_x = np.arange(-2*np.pi, 2*np.pi, 0.1) # 定义插值点
# 进行样条插值
import scipy.interpolate as spi
# 进行一阶样条插值
ipo1 = spi.splrep(X, Y, k=1)  # 样本点导入，生成参数
iy1 = spi.splev(new_x, ipo1)  # 根据观测点和样条参数，生成插值
```



#### 三次样条插值

两个数据点间的填充为三次多项式。

在一组离散的点中间插值。在**每两个相邻点 $x_i$与 $x_{i+1}$ 之间**，用一段**三次多项式** $S_i(x)=a_ix^3+b_ix^2+c_ix+d_i ，\text{定义域}x \in [x_i,x_{i + 1}]$去连接这两个点

插值满足以下条件：

1. 函数值相等：$S_i(x_i)=S_{i+1}(x_i)$   左右两段在节点处函数值相等
   $$
   a_i x_i^3 + b_i x_i^2 + c_i x_i + d_i = a_{i+1} x_{i}^3 + b_{i+1} x_{i}^2 + c_{i+1} x_{i} + d_{i+1}
   $$

2. 一阶导数相等$S_i'(x_i)=S_{i+1}'(x_i)$   斜率相等
   $$
   3a_i x_i^2 + 2b_i x_i + c_i = 3a_{i+1} x_{i}^2 + 2b_{i+1} x_{i} + c_{i+1}
   $$

3. 二阶导数相等$$S_i''(x_i)=S_{i+1}''(x_i)$$   曲率相等
   $$
   6a_i x_i + 2b_i = 6a_{i+1} x_{i} + 2b_{i+1}
   $$

注：内部节点只能给出部分方程，还需要2 个边界条件才能唯一确定整套样条。

* 自然样条：两端二阶导数 \(S''=0\)
* 夹持样条：指定两端一阶导数值

```python
import scipy.interpolate as spi

# X,Y：原始离散样本点
# k=3 代表使用三次样条
ipo3 = spi.splrep(X, Y, k=3)    # 计算样条全部参数
# new_x：你想要插值的一系列新横坐标
iy3 = spi.splev(new_x, ipo3)   # 代入新x，算出插值得到的y值
```



#### 拉格朗日插值







