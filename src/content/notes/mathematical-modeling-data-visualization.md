---
title: '数学建模：数据可视化'
description: 'Matplotlib与Plotnine绘图：柱状图、折线图、堆叠图'
pubDate: 2026-08-07
category: '数学建模'
tags: ['数据可视化','Matplotlib','Plotnine']
---

### 数据可视化

1. 分析过程中的数据可视化
2. 分析结果表达中的数据可视化

一个好的数据可视化图表

1. 图表展示的信息全面且无歧义
2. 图表表达的信息越多、越全面越好
3. 通俗易懂，不能太专业

```python
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
%matplotlib inline 
plt.rcParams['font.sans-serif']=['SimHei','Songti SC','STFangsong']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
import seaborn as sns

df = pd.DataFrame({
    'variable': ['gender', 'gender', 'age', 'age', 'age', 'income', 'income', 'income', 'income'],
    'category': ['女性', '男性', '1‑24岁', '25‑54岁', '55岁以上', '低收入', '中低收入', '中等收入', '高收入'],
    'value': [60, 40, 50, 30, 20, 10, 25, 25, 40],
})
# 设置分类顺序，保证坐标轴顺序不变
df['variable'] = pd.Categorical(df['variable'], categories=['gender', 'age', 'income'])
df['category'] = pd.Categorical(df['category'], categories=df['category'])

from plotnine import *

p = (
    ggplot(df, aes(x='variable', y='value', fill='category')) +
    geom_col() +
    # 修改坐标轴标签、标题、图例标题为中文
    xlab("变量类型") +
    ylab("数值") +
    ggtitle("堆叠柱状图") +
    labs(fill="类别") +
    theme(dpi=100, figure_size=(8,5))
)
p.draw()
```

![0888992627a39a805051af5e895a00ad](/images/math-modeling/0888992627a39a805051af5e895a00ad.png)

```python
# 柱状图
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei','Songti SC','STFangsong']
plt.rcParams['axes.unicode_minus'] = False

df = pd.DataFrame({
    'variable': ['gender', 'gender', 'age', 'age', 'age', 'income', 'income', 'income', 'income'],
    'category': ['女性', '男性', '1‑24岁', '25‑54岁', '55岁以上', '低收入', '中低收入', '中等收入', '高收入'],
    'value': [60, 40, 50, 30, 20, 10, 25, 25, 40],
})
df['variable'] = pd.Categorical(df['variable'], categories=['gender', 'age', 'income'])
df['category'] = pd.Categorical(df['category'], categories=df['category'])

from plotnine import *

p = (
    ggplot(df, aes(x='variable', y='value', fill='category')) +
    geom_col(stat='identity', position='dodge') +
    scale_x_discrete(labels=["性别","年龄","收入"]) +
    xlab("变量类型") +
    ylab("数值") +
    ggtitle("并列柱状图") +
    labs(fill="类别") +
    theme(dpi=100, figure_size=(8,5))
)
p.draw()
```

![f5d42be59fab57626b120b19999ba90e](/images/math-modeling/f5d42be59fab57626b120b19999ba90e.png)

#### python中数据可视化的3个工具库：

##### Matplotlib

调用Matplotlib中的pyplot绘图接口

```python
import matplotlib.pyplot as plt
```

绘制散点图:

参考流程：

- 创建一个图形对象，并设置图形对象的大小（可以想象成在白纸中添加一个图，并设置图的大小）：plt.figure(figsize=(6,4))
- 在纸上的坐标系中绘制散点：plt.scatter(x=x, y=y)
- 设置x轴的标签label：plt.xlabel('x')
- 设置y轴标签的label：plt.ylabel('y')
- 设置图表的标题：plt.title('y = sin(x)')
- 展示图标：plt.show() 

例：

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-2*np.pi, 2*np.pi, 100)
y = np.sin(x)

plt.figure(figsize=(8,6))
plt.scatter(x,y)
plt.xlabel("x")
plt.ylabel("y", labelpad=12)

# 在整张图底部居中放置标题
plt.figtext(0.5, 0.03, 'y = sin(x)', ha="center", fontsize=13)
plt.subplots_adjust(bottom=0.15)

plt.show()
```

![4e3571892914cee0170fae07c5a250ea](/images/math-modeling/4e3571892914cee0170fae07c5a250ea.png)



例：画y = sin(x) 和 y=cos(x)的散点图：

流程：

1. 创建第一个绘图对象： fig1 = plt.figure(figsize=(6,4), num='first')
2.  在纸上的坐标系中绘制散点：plt.scatter(x=x, y=sin(x)) 
3. 设置x轴的标签label：plt.xlabel('x') 
4. 设置y轴标签的label：plt.ylabel('y') 
5. 设置图表的标题：plt.title('y = sin(x)')



1. 创建第二个绘图对象： fig2 = plt.figure(figsize=(6,4), num='second')
2. 在纸上的坐标系中绘制散点：plt.scatter(x=x, y=cos(x))
3.  设置x轴的标签label：plt.xlabel('x') 
4. 设置y轴标签的label：plt.ylabel('y') 
5. 设置图表的标题：plt.title('y = cos(x)')





##### Seaborn

> Seaborn主要用于统计分析绘图的，它是基于Matplotlib进行了更高级的API封装。

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 准备数据
x = np.linspace(-10, 10, 100)
y = 2 * x + 1 + np.random.randn(100)
df = pd.DataFrame({'x':x, 'y':y})

# 使用Seaborn绘制带有拟合直线效果的散点图
sns.lmplot(x="x", y="y", data=df)

plt.show()
```

![a2e1f9a4c74ce3717daa2aec3152ba47](/images/math-modeling/a2e1f9a4c74ce3717daa2aec3152ba47.png)

##### Plotnine

> 原理：采用了“图层”的概念，每一句代码都是相当于往图像中添加一个图层，一个图层就是一类绘图动作

```python
from plotnine import *     # 讲Plotnine所有模块引入
from plotnine.data import *   # 引入PLotnine自带数据集
mpg.head()
```

mpg数据集记录了美国1999年和2008年部分汽车的制造厂商，型号，类别，驱动程序和耗油量。

* manufacturer 生产厂家
* model 型号类型
* year 生产年份
* cty 和 hwy分别记录城市和高速公路驾驶耗油量
* cyl 气缸数
* displ 表示发动机排量
* drv 表示驱动系统：前轮驱动、(f),后轮驱动®和四轮驱动(4)
* class 表示车辆类型，如双座汽车，suv，小型汽车
* fl (fuel type)，燃料类型

```python
# 绘制汽车在不同驱动系统下，发动机排量与耗油量的关系
import pandas as pd
import matplotlib.pyplot as plt
from plotnine import *
# 直接导入内置mpg数据集
from plotnine.data import mpg

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 绘制汽车在不同驱动系统下，发动机排量与耗油量的关系
p1 = (
    ggplot(mpg, aes(x='displ', y='hwy', color='drv'))     # 设置数据映射图层，数据集使用mpg，x数据使用mpg['displ']发动机排量，y数据使用mpg['hwy']高速油耗，颜色映射使用mpg['drv']驱动类型
    + geom_point()       # 绘制散点图图层
    + geom_smooth(method='lm', se=False)        # 绘制线性回归平滑线图层，se=False关闭置信区间阴影
    + labs(x='发动机排量', y='高速油耗', color='驱动系统', title="不同驱动系统下发动机排量与耗油量关系")     # 设置x、y轴标签、图例标题、图标题
    + theme(dpi=100, figure_size=(9, 5))
)
p1.draw()  
```

![13fef4176ca2dd2ed0fe8d6614f9f407](/images/math-modeling/13fef4176ca2dd2ed0fe8d6614f9f407.png)

缺点：只能实现直角坐标系



总结：

直角坐标系用Plotnine

极坐标图表用Matplotlib和Seaborn



#### 图表 Quick Start

##### 类别型图表

适用： X类别下Y数值之间的比较   也就是 X为类别型数据、Y为数值型数据。

形式： 柱状图、横向柱状图（条形图）、堆叠柱状图、极坐标的柱状图、词云、雷达图、桑基图



例：

```python
## Matplotlib绘制单系列柱状图：不同城市的房价对比
import pandas as pd
import matplotlib.pyplot as plt

data = pd.DataFrame({
    'city':['深圳', '上海', '北京', '广州', '成都'],
    'house_price(w)':[3.5, 4.0, 4.2, 2.1, 1.5]
})

fig = plt.figure(figsize=(10,6))
# [left, bottom, width, height]
ax1 = fig.add_axes([0.15,0.15,0.7,0.7])

ax1.bar(data['city'], data['house_price(w)'], width=0.6, align='center', orientation='vertical', label='城市')
"""
x 表示x坐标，数据类型为int或float类型，也可以为str
height 表示柱状图的高度，也就是y坐标值，数据类型为int或float类型
width 表示柱状图的宽度，取值在0~1之间，默认为0.8
bottom 柱状图的起始位置，也就是y轴的起始坐标
align 柱状图的中心位置，"center","lege"边缘
color 柱状图颜色
edgecolor 边框颜色
linewidth 边框宽度
tick_label 下标标签
log 柱状图y周使用科学计算方法，bool类型
orientation 柱状图是竖直还是水平，竖直："vertical"，水平条："horizontal"
"""
ax1.set_title("不同城市的房价对比图")
ax1.set_xlabel("城市")
ax1.set_ylabel("房价/w")


ax1.grid(False, which='both')


for i in range(len(data)):
    val = data.iloc[i]['house_price(w)']
    ax1.text(i-0.05, val+0.05, val, fontsize=13)

ax1.legend()
plt.show()
```

![ba69d96bfe3e520e9e71e396c3822102](/images/math-modeling/ba69d96bfe3e520e9e71e396c3822102.png)



```python
## Plotnine绘制单系列柱状图：不同城市的房价对比
data = pd.DataFrame({'city':['深圳', '上海', '北京', '广州', '成都'], 'house_price(w)':[3.5, 4.0, 4.2, 2.1, 1.5]})

p_single_bar = (
    ggplot(data, aes(x='city', y='house_price(w)', fill='city', label='house_price(w)'))+
    geom_bar(stat='identity')+
    labs(x="城市", y="房价(w)", title="不同城市的房价对比图")+
    geom_text(nudge_y=0.08)+
    theme(text = element_text(family = "SimHei"),
         panel_grid = element_blank(),# 去除全部网格线（横向+纵向）
         panel_background = element_blank()# 把灰色绘图背景也去掉)
         )
)
p_single_bar.draw()
```

![43767ffad8340325d5dbc47b80ac77cf](/images/math-modeling/43767ffad8340325d5dbc47b80ac77cf.png)



```python
## Matplotlib绘制多系列柱状图：不同城市在不同年份的房价对比
data = pd.DataFrame({
    '城市':['深圳', '上海', '北京', '广州', '成都', '深圳', '上海', '北京', '广州', '成都'],
    '年份':[2021,2021,2021,2021,2021,2022,2022,2022,2022,2022],
    '房价(w)':[3.5, 4.0, 4.2, 2.1, 1.5, 4.0, 4.2, 4.3, 1.6, 1.9]
})

fig = plt.figure(figsize=(10,6))
ax1 = fig.add_axes([0.15,0.15,0.7,0.7])  # [left, bottom, width, height], 它表示添加到画布中的矩形区域的左下角坐标(x, y)，以及宽度和高度
plt.bar(
    np.arange(len(np.unique(data['城市'])))-0.15, 
    data.loc[data['年份']==2021,'房价(w)'], 
    width=0.3, 
    align='center', 
    orientation='vertical', 
    label='年份：2021'
    )
plt.bar(
    np.arange(len(np.unique(data['城市'])))+0.15, 
    data.loc[data['年份']==2022,'房价(w)'], 
    width=0.3, 
    align='center', 
    orientation='vertical', 
    label='年份：2022'
    )
plt.title("不同城市的房价对比图")   # 在axes1设置标题
plt.xlabel("城市")    # 在axes1中设置x标签
plt.ylabel("房价/w")    # 在axes1中设置y标签
plt.xticks(np.arange(len(np.unique(data['城市']))), np.array(['深圳', '上海', '北京', '广州', '成都']))
plt.grid(False, which='both')  # 在axes1中设置设置网格线

data_2021 = data.loc[data['年份']==2021,:]
for i in range(len(data_2021)):
    plt.text(i-0.15-0.05, data_2021.iloc[i,2]+0.05, data_2021.iloc[i,2],fontsize=13)   # 添加数据注释

data_2022 = data.loc[data['年份']==2022,:]
for i in range(len(data_2022)):
    plt.text(i+0.15-0.05, data_2022.iloc[i,2]+0.05, data_2022.iloc[i,2],fontsize=13)   # 添加数据注释
plt.legend()
plt.show()
```

![e14cb84ea5a8e800bc6863e6e23e9959](/images/math-modeling/e14cb84ea5a8e800bc6863e6e23e9959.png)



```python
## Plotnine绘制多系列柱状图：不同城市在不同年份的房价对比
import pandas as pd
from plotnine import *

data = pd.DataFrame({
    '城市':['深圳', '上海', '北京', '广州', '成都', '深圳', '上海', '北京', '广州', '成都'],
    '年份':[2021,2021,2021,2021,2021,2022,2022,2022,2022,2022],
    '房价(w)':[3.5, 4.0, 4.2, 2.1, 1.5, 4.0, 4.2, 4.3, 1.6, 1.9]
})
data['年份'] = pd.Categorical(data['年份'], ordered=True, categories=[2021,2022])

# 预计算标签高度，不使用nudge_y
data['label_y'] = data['房价(w)'] + 0.08


bar_width = 0.6        # 单根柱子粗细
dodge_total_width = 0.6 # 同一城市一组柱子的总宽度！减小这个，两根柱子就贴紧


p_mult_bar = (
    ggplot(data, aes(x='城市', y='房价(w)', fill='年份'))
    + geom_bar(stat='identity', width=bar_width, position=position_dodge(dodge_total_width))
    + scale_fill_manual(values = ["#f6e8c3", "#5ab4ac"])
    + labs(x="城市", y="房价(w)", title="不同城市的房价对比图")
    + geom_text(
        aes(label='房价(w)', y='label_y'),
        position=position_dodge(dodge_total_width), # 和柱子dodge数值必须一模一样
        size=9
    )
    + theme(
        text = element_text(family = "SimHei"),
        panel_grid = element_blank(),
        panel_background = element_blank()
    )
)
p_mult_bar
```

![58f947779f67e2678e43e00bd805f961](/images/math-modeling/58f947779f67e2678e43e00bd805f961.png)



```python
## Matplotlib绘制堆叠柱状图：不同城市在不同年份的房价对比
data = pd.DataFrame({
    '城市':['深圳', '上海', '北京', '广州', '成都', '深圳', '上海', '北京', '广州', '成都'],
    '年份':[2021,2021,2021,2021,2021,2022,2022,2022,2022,2022],
    '房价(w)':[3.5, 4.0, 4.2, 2.1, 1.5, 4.0, 4.2, 4.3, 1.6, 1.9]
})
tmp=data.set_index(['城市','年份'])['房价(w)'].unstack()
data=tmp.rename_axis(columns=None).reset_index()
data.columns = ['城市','2021房价','2022房价']
print(data)

plt.figure(figsize=(10,6))
plt.bar(
    data['城市'], 
    data['2021房价'], 
    width=0.6, 
    align='center', 
    orientation='vertical', 
    label='年份：2021'
    )
plt.bar(
    data['城市'], 
    data['2022房价'], 
    width=0.6, 
    align='center', 
    orientation='vertical', 
    bottom=data['2021房价'],
    label='年份：2022'
    )
plt.title("不同城市2121-2022年房价对比图")   # 在axes1设置标题
plt.xlabel("城市")    # 在axes1中设置x标签
plt.ylabel("房价/w")    # 在axes1中设置y标签
plt.legend()
plt.show()
```

![d3d50ca430755231980e0801751b0db4](/images/math-modeling/d3d50ca430755231980e0801751b0db4.png)



```python
## Matplotlib绘制百分比柱状图：不同城市在不同年份的房价对比
## Matplotlib绘制堆叠柱状图：不同城市在不同年份的房价对比
data = pd.DataFrame({
    '城市':['深圳', '上海', '北京', '广州', '成都', '深圳', '上海', '北京', '广州', '成都'],
    '年份':[2021,2021,2021,2021,2021,2022,2022,2022,2022,2022],
    '房价(w)':[3.5, 4.0, 4.2, 2.1, 1.5, 4.0, 4.2, 4.3, 1.6, 1.9]
})
tmp=data.set_index(['城市','年份'])['房价(w)'].unstack()
data=tmp.rename_axis(columns=None).reset_index()
data.columns = ['城市','2021房价','2022房价']
print(data)

plt.figure(figsize=(10,6))
plt.bar(
    data['城市'], 
    data['2021房价']/(data['2021房价']+data['2022房价']), 
    width=0.4, 
    align='center', 
    orientation='vertical', 
    label='年份：2021'
    )
plt.bar(
    data['城市'], 
    data['2022房价']/(data['2021房价']+data['2022房价']), 
    width=0.4, 
    align='center', 
    orientation='vertical', 
    bottom=data['2021房价']/(data['2021房价']+data['2022房价']),
    label='年份：2022'
    )
plt.title("不同城市2121-2022年房价对比图")   # 设置标题
plt.xlabel("城市")    # 在axes1中设置x标签
plt.ylabel("房价/w")    # 在axes1中设置y标签
plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1))

# 给右边图例留出空白，防止图例被裁剪
plt.tight_layout()
plt.show()
```

![6bf8643e0b62adc21a146b5003a699d9](/images/math-modeling/6bf8643e0b62adc21a146b5003a699d9.png)



```python
# 使用Matplotlib绘制雷达图：英雄联盟几位英雄的能力对比
data = pd.DataFrame({
    '属性': ['血量', '攻击力', '攻速', '物抗', '魔抗'],
    '艾希':[3, 7, 8, 2, 2],
    '诺手':[8, 6, 3, 6, 6]
})

plt.figure(figsize=(8,8))
theta = np.linspace(0, 2*np.pi, len(data), endpoint=False)   # 每个坐标点的位置
theta = np.append(theta, theta[0])  # 让数据封闭
aixi = np.append(data['艾希'].values,data['艾希'][0])  #让数据封闭
nuoshou = np.append(data['诺手'].values,data['诺手'][0])  # 让数据封闭
shuxing = np.append(data['属性'].values,data['属性'][0])  # 让数据封闭

plt.polar(theta, aixi, 'ro-', lw=2, label='艾希') # 画出雷达图的点和线
plt.fill(theta, aixi, facecolor='r', alpha=0.5) # 填充
plt.polar(theta, nuoshou, 'bo-', lw=2, label='诺手')  # 画出雷达图的点和线
plt.fill(theta, nuoshou, facecolor='b', alpha=0.5) # 填充
plt.thetagrids(theta/(2*np.pi)*360, shuxing)  # 为每个轴添加标签
plt.ylim(0,10)
plt.legend()
plt.show()
```

![4470edade1f6d49b881ce49dada9ed3b](/images/math-modeling/4470edade1f6d49b881ce49dada9ed3b.png)

##### 关系型图表

适用： X数值与Y数值之间的关系  例：是否是线性关系、是否有正向相关关系

关系有：数值型关系、层次型关系和网络型关系



```python
# 使用Matplotlib和四个图说明相关关系：
x = np.random.randn(100)*10
y1 = np.random.randn(100)*10
y2 = 2 * x + 1 + np.random.randn(100)
y3 = -2 * x + 1 + np.random.randn(100)
y4 = x**2 + 1 + np.random.randn(100)

plt.figure(figsize=(12, 12))

plt.subplot(2,2,1)  #创建两行两列的子图，并绘制第一个子图
plt.scatter(x, y1, c='dodgerblue', marker=".", s=50)
plt.xlabel("x")
plt.ylabel("y1")
plt.title("y1与x不存在关联关系")

plt.subplot(2,2,2)  #创建两行两列的子图，并绘制第二个子图
plt.scatter(x, y2, c='tomato', marker="o", s=10)
plt.xlabel("x")
plt.ylabel("y2")
plt.title("y2与x存在关联关系")

plt.subplot(2,2,3)  #创建两行两列的子图，并绘制第三个子图
plt.scatter(x, y3, c='magenta', marker="o", s=10)
plt.xlabel("x")
plt.ylabel("y3")
plt.title("y3与x存在关联关系")

plt.subplot(2,2,4)  #创建两行两列的子图，并绘制第四个子图
plt.scatter(x, y4, c='deeppink', marker="s", s=10)
plt.xlabel("x")
plt.ylabel("y3")
plt.title("y4与x存在关联关系")

plt.show()
```

![953223de7d32553e51b30a79525c87d0](/images/math-modeling/953223de7d32553e51b30a79525c87d0.png)



```python
# 使用Plotnine和四个图说明相关关系：
x = np.random.randn(100)*10
y1 = np.random.randn(100)*10
y2 = 10 * x + 1 + np.random.randn(100)
y3 = -10 * x + 1 + np.random.randn(100)
y4 = x**2 + 1 + np.random.randn(100)

df = pd.DataFrame({
    'x': np.concatenate([x,x,x,x]),
    'y': np.concatenate([y1, y2, y3, y4]),
    'class': ['y1']*100 + ['y2']*100 + ['y3']*100 + ['y4']*100
})

p1 = (
    ggplot(df)+
    geom_point(aes(x='x', y='y', fill='class', shape='class'), color='black', size=2)+
    scale_fill_manual(values=('#00AFBB', '#FC4E07', '#00589F', '#F68F00'))+
    theme(text = element_text(family = "SimHei"))
)
p1.draw()
```

![bfc177762e662524b5299297e9bccfd7](/images/math-modeling/bfc177762e662524b5299297e9bccfd7.png)



```python
# 使用Matplotlib绘制具备趋势线的散点图
from sklearn.linear_model import LinearRegression  #线性回归等参数回归
import statsmodels.api as sm

from sklearn.preprocessing import PolynomialFeatures  # 构造多项式
x = np.linspace(-10, 10, 100)
y = np.square(x) + np.random.randn(100)*100
x_poly2 = PolynomialFeatures(degree=2).fit_transform(x.reshape(-1, 1))
y_linear_pred = LinearRegression().fit(x.reshape(-1, 1), y).predict(x.reshape(-1, 1))
y_poly_pred = LinearRegression().fit(x_poly2, y).predict(x_poly2)
y_exp_pred = LinearRegression().fit(np.exp(x).reshape(-1, 1), y).predict(np.exp(x).reshape(-1, 1))
y_loess_pred = sm.nonparametric.lowess(x, y, frac=2/3)[:, 1]

plt.figure(figsize=(8, 8))
plt.subplot(2,2,1)
plt.scatter(x, y, c='tomato', marker="o", s=10)
plt.plot(x, y_linear_pred, c='dodgerblue')
plt.xlabel("x")
plt.ylabel("y")
plt.title("带线性趋势线的散点图")

plt.subplot(2,2,2)
plt.scatter(x, y, c='tomato', marker="o", s=10)
plt.plot(x, y_poly_pred, c='dodgerblue')
plt.xlabel("x")
plt.ylabel("y")
plt.title("带二次趋势线的散点图")

plt.subplot(2,2,3)
plt.scatter(x, y, c='tomato', marker="o", s=10)
plt.plot(x, y_exp_pred, c='dodgerblue')
plt.xlabel("x")
plt.ylabel("y")
plt.title("带指数趋势线的散点图")

plt.subplot(2,2,4)
plt.scatter(x, y, c='tomato', marker="o", s=10)
plt.plot(x, y_loess_pred, c='dodgerblue')
plt.xlabel("x")
plt.ylabel("y")
plt.title("带 loess平滑线的散点图")

plt.show()
```

![3ee3c7577d8d41f2f9b521b7c3fc7402](/images/math-modeling/3ee3c7577d8d41f2f9b521b7c3fc7402.png)



```python
# 使用Matplotlib绘制聚类散点图
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris  # 加载鸢尾花数据集

plt.rcParams['font.sans-serif'] = ['SimHei']   # 设置中文黑体
plt.rcParams['axes.unicode_minus'] = False     # 解决负号显示方块问题

iris = load_iris()
X = iris.data
label = iris.target
feature = iris.feature_names
df = pd.DataFrame(X, columns=feature)
df['label'] = label

label_unique = np.unique(df['label']).tolist()
plt.figure(figsize=(10, 6))

# 鸢尾花类别中文映射
class_name = {0: '山鸢尾', 1: '变色鸢尾', 2: '维吉尼亚鸢尾'}

for i in label_unique:
    df_label = df.loc[df['label'] == i, :]
    plt.scatter(x=df_label['sepal length (cm)'], 
                y=df_label['sepal width (cm)'], 
                s=20, 
                label=class_name[i])

plt.xlabel('花萼长度 (厘米)')
plt.ylabel('花萼宽度 (厘米)')
plt.title('花萼宽度 ~ 花萼长度')
plt.legend(title="鸢尾花类别")
plt.show()
```

![8107d5f13d2892c350715984f6ebe073](/images/math-modeling/8107d5f13d2892c350715984f6ebe073.png)



```python
# 使用plotnine绘制相关系数矩阵图：
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotnine import *
from plotnine.data import mtcars

# matplotlib中文全局设置
plt.rcParams['font.sans-serif'] = ['SimHei']   # 黑体
plt.rcParams['axes.unicode_minus'] = False     # 负号正常显示

# 处理mtcars：车型名称设为索引，只保留数值列计算相关系数
mtcars_num = mtcars.set_index("name")
corr_mat = np.round(mtcars_num.corr(), 1).reset_index()

df = pd.melt(corr_mat, id_vars='index', var_name='variable', value_name='corr_xy')
df['abs_corr'] = np.abs(df['corr_xy'])

# 圆点版相关系数图
p1 = (
    ggplot(df, aes(x='index', y='variable', fill='corr_xy', size='abs_corr'))+
    geom_point(shape='o', color='black')+
    scale_size_area(max_size=11, guide=None)+
    scale_fill_cmap(name='RdYlBu_r')+
    coord_equal()+
    labs(x="变量", y="变量", title="相关系数矩阵（圆点）", fill="相关系数")+
    theme(dpi=100, figure_size=(4.5,4.55))
)

# 方块散点版
p2 = (
    ggplot(df, aes(x='index', y='variable', fill='corr_xy', size='abs_corr'))+
    geom_point(shape='s', color='black')+
    scale_size_area(max_size=10, guide=None)+
    scale_fill_cmap(name='RdYIBu_r')+
    coord_equal()+
    labs(x="变量", y="变量", title="相关系数矩阵（方块）", fill="相关系数")+
    theme(dpi=100, figure_size=(4.5,4.55))
)

# 热力图瓷砖版本，显示相关系数数值
p3 = (
    ggplot(df, aes(x='index', y='variable', fill='corr_xy', label='corr_xy'))+
    geom_tile(color='black')+
    geom_text(size=8, color='white')+
    scale_fill_cmap(name='RdYIBu_r')+
    coord_equal()+
    labs(x="变量", y="变量", title="相关系数热力图", fill="相关系数")+
    theme(dpi=100, figure_size=(4.5,4.55))
)

p1.draw()
p2.draw()
p3.draw()
```

![b094cc3e9f9078d53da55a8e0c9cc154](/images/math-modeling/b094cc3e9f9078d53da55a8e0c9cc154.png)

```python
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 中文设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 生成随机 10行12列 0~1之间数据
uniform_data = np.random.rand(10, 12)

plt.figure(figsize=(8, 6))
ax = sns.heatmap(uniform_data)
ax.set_title("热力图示例")
ax.set_xlabel("X轴")
ax.set_ylabel("Y轴")
plt.show()
```

![c1f93357b0ef5c628be95f2fb2f5c4e5](/images/math-modeling/c1f93357b0ef5c628be95f2fb2f5c4e5.png)



```python
# 使用Matplotlib/Seaborn绘制相关系数矩阵图
uniform_data = np.random.rand(10, 12)
sns.heatmap(uniform_data)
```





##### 分布型图表

描述数据的密集或者稀疏 ： 数据在哪里比较密集，那里比较稀疏

可以用频率或者概率表示





```python
# 使用matplotlib绘制直方图：
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取mtcars数据集，pandas内置可直接加载
mtcars = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/mtcars.csv')

# 使用matplotlib绘制直方图：汽车每加仑里程mpg分布
plt.figure(figsize=(8, 6))
plt.hist(mtcars['mpg'], bins=20, alpha=0.85)
plt.xlabel("每加仑里程(mpg)")
plt.ylabel("频数")
plt.title("汽车油耗mpg分布直方图")
plt.show()
```

![e7261ab42ca13e7392d6b1a6201e7f3e](/images/math-modeling/e7261ab42ca13e7392d6b1a6201e7f3e.png)



```python
# 使用matplotlib绘制箱线图
# 使用matplotlib绘制箱线图
import matplotlib.pyplot as plt
import seaborn as sns
from plotnine.data import mtcars

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

data = mtcars
data['carb'] = data['carb'].astype('category')
plt.figure(figsize=(8, 6))


sns.boxenplot(
    x='carb',
    y='mpg',
    hue='carb',
    data=mtcars,
    linewidth=0.2,
    palette=sns.husl_palette(6, s=0.9, l=0.65, h=0.0417),
    legend=False
)

plt.xlabel('carb（化油器）')
plt.ylabel('mpg（每加仑里程）')
plt.title('boxenplot增强箱线图')
plt.show()
```

![63ad3cb681e8b10551761416e6a3cd97](/images/math-modeling/63ad3cb681e8b10551761416e6a3cd97.png)



```python
# 使用Matplotlib绘制饼状图：
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm, colors

# 中文设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

df = pd.DataFrame({
    '己方': ['寒冰', '布隆', '发条', '盲僧', '青钢影'],
    '敌方': ['女警', '拉克丝', '辛德拉', '赵信', '剑姬'],
    '己方输出': [26000, 5000, 23000, 4396, 21000],
    '敌方输出': [25000, 12000, 21000, 10000, 18000]
})

df_our = df[['己方', '己方输出']].sort_values(by='己方输出', ascending=False).reset_index(drop=True)
df_other = df[['敌方', '敌方输出']].sort_values(by='敌方输出', ascending=False).reset_index(drop=True)

color_list = [cm.Set3(i) for i in range(len(df))]

plt.figure(figsize=(16, 10))

# 左图：己方输出饼图
plt.subplot(1, 2, 1)
plt.pie(
    df_our['己方输出'].values,
    startangle=90,
    shadow=True,
    colors=color_list,
    labels=df_our['己方'].tolist(),
    explode=(0, 0, 0, 0, 0.3),
    autopct='%.2f%%'
)
plt.title("己方输出占比", y=-0.1)

# 右图：敌方输出饼图
plt.subplot(1, 2, 2)
plt.pie(
    df_other['敌方输出'].values,
    startangle=90,
    shadow=True,
    colors=color_list,
    labels=df_other['敌方'].tolist(),
    explode=(0, 0, 0, 0, 0.3),
    autopct='%.2f%%'
)
plt.title("敌方输出占比", y=-0.1)

plt.tight_layout()
plt.show()
```

![73a1464273dc6fcb2b9e6c5775587300](/images/math-modeling/73a1464273dc6fcb2b9e6c5775587300.png)



```python
# 使用Matplotlib绘制环状图：
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm, colors

# 中文设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

df = pd.DataFrame({
    '己方': ['寒冰', '布隆', '发条', '盲僧', '青钢影'],
    '敌方': ['女警', '拉克丝', '辛德拉', '赵信', '剑姬'],
    '己方输出': [26000, 5000, 23000, 4396, 21000],
    '敌方输出': [25000, 12000, 21000, 10000, 18000]
})

df_our = df[['己方', '己方输出']].sort_values(by='己方输出', ascending=False).reset_index(drop=True)
df_other = df[['敌方', '敌方输出']].sort_values(by='敌方输出', ascending=False).reset_index(drop=True)

color_list = [cm.Set3(i) for i in range(len(df))]
# 环状图核心参数：width 环的厚度
wedgeprops = {'width': 0.3, 'edgecolor': 'black', 'linewidth': 3}

plt.figure(figsize=(16, 10))

# 左图 己方环状图
plt.subplot(1, 2, 1)
plt.pie(
    df_our['己方输出'].values,
    startangle=90,
    shadow=True,
    colors=color_list,
    wedgeprops=wedgeprops,
    labels=df_our['己方'].tolist(),
    explode=(0, 0, 0, 0, 0.3),
    autopct='%.2f%%'
)
# 圆环中心文字
plt.text(0, 0, '己方', ha='center', va='center', fontsize=30)
plt.title("己方输出占比", y=-0.1)

# 右图 敌方环状图
plt.subplot(1, 2, 2)
plt.pie(
    df_other['敌方输出'].values,
    startangle=90,
    shadow=True,
    colors=color_list,
    wedgeprops=wedgeprops,
    labels=df_other['敌方'].tolist(),
    explode=(0, 0, 0, 0, 0.3),
    autopct='%.2f%%'
)
plt.text(0, 0, '敌方', ha='center', va='center', fontsize=30)
plt.title("敌方输出占比", y=-0.1)

plt.tight_layout()
plt.show()
```

![5739e877005fb2aa97c0425d986d17aa](/images/math-modeling/5739e877005fb2aa97c0425d986d17aa.png)



##### 时间序列型图表

```python
# 使用Plotnine绘制时间序列线图：
import pandas as pd
import matplotlib.pyplot as plt
from plotnine import *

# 中文设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# AirPassengers 航空乘客原始数据
data_list = [
    ["1949-01",112],["1949-02",118],["1949-03",132],["1949-04",129],["1949-05",121],["1949-06",135],
    ["1949-07",148],["1949-08",148],["1949-09",136],["1949-10",119],["1949-11",104],["1949-12",118],
    ["1950-01",115],["1950-02",126],["1950-03",141],["1950-04",135],["1950-05",125],["1950-06",149],
    ["1950-07",170],["1950-08",170],["1950-09",158],["1950-10",133],["1950-11",114],["1950-12",140],
    ["1951-01",145],["1951-02",150],["1951-03",178],["1951-04",163],["1951-05",172],["1951-06",178],
    ["1951-07",199],["1951-08",199],["1951-09",184],["1951-10",162],["1951-11",146],["1951-12",166],
    ["1952-01",171],["1952-02",180],["1952-03",193],["1952-04",181],["1952-05",183],["1952-06",218],
    ["1952-07",230],["1952-08",242],["1952-09",209],["1952-10",191],["1952-11",172],["1952-12",194],
    ["1953-01",196],["1953-02",196],["1953-03",236],["1953-04",235],["1953-05",229],["1953-06",243],
    ["1953-07",264],["1953-08",272],["1953-09",237],["1953-10",211],["1953-11",180],["1953-12",201],
    ["1954-01",204],["1954-02",188],["1954-03",235],["1954-04",227],["1954-05",234],["1954-06",264],
    ["1954-07",302],["1954-08",293],["1954-09",259],["1954-10",229],["1954-11",203],["1954-12",229],
    ["1955-01",242],["1955-02",233],["1955-03",267],["1955-04",269],["1955-05",270],["1955-06",315],
    ["1955-07",364],["1955-08",347],["1955-09",312],["1955-10",274],["1955-11",237],["1955-12",278],
    ["1956-01",284],["1956-02",277],["1956-03",317],["1956-04",313],["1956-05",318],["1956-06",374],
    ["1956-07",413],["1956-08",405],["1956-09",355],["1956-10",306],["1956-11",271],["1956-12",306],
    ["1957-01",315],["1957-02",301],["1957-03",356],["1957-04",348],["1957-05",355],["1957-06",422],
    ["1957-07",465],["1957-08",467],["1957-09",404],["1957-10",347],["1957-11",305],["1957-12",336],
    ["1958-01",340],["1958-02",318],["1958-03",362],["1958-04",348],["1958-05",363],["1958-06",435],
    ["1958-07",491],["1958-08",505],["1958-09",404],["1958-10",359],["1958-11",310],["1958-12",337],
    ["1959-01",360],["1959-02",342],["1959-03",406],["1959-04",396],["1959-05",420],["1959-06",472],
    ["1959-07",548],["1959-08",559],["1959-09",463],["1959-10",407],["1959-11",362],["1959-12",405],
    ["1960-01",417],["1960-02",391],["1960-03",419],["1960-04",461],["1960-05",472],["1960-06",535],
    ["1960-07",622],["1960-08",606],["1960-09",508],["1960-10",461],["1960-11",390],["1960-12",432]
]

df = pd.DataFrame(data_list, columns=["date","value"])
df['date'] = pd.to_datetime(df['date'])

p1 = (
    ggplot(df, aes(x='date', y='value'))+
    geom_line(size=1, color='red')+
    scale_x_date(date_labels="%Y", date_breaks="1 year")+
    xlab('日期')+
    ylab('乘客数量')+
    ggtitle("航空乘客时间序列线图")+
    theme(dpi=100, figure_size=(10,5))
)

p1.draw()
```

![91f2ac6e889735915c585abe2df03e11](/images/math-modeling/91f2ac6e889735915c585abe2df03e11.png)



```python
# 使用Matplotlib绘制时间序列折线图
import pandas as pd
import matplotlib.pyplot as plt

# 中文设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


df = pd.DataFrame(data_list, columns=["date","value"])
df['date'] = pd.to_datetime(df['date'])

plt.figure(figsize=(8,6))
plt.plot(df['date'], df['value'], color='red', linewidth=1.2)
plt.xlabel("日期")
plt.ylabel("乘客数量")
plt.title("航空乘客时间序列折线图")

# x轴年份显示优化，避免文字重叠
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

![cb520095c8f09659bde26355252b9815](/images/math-modeling/cb520095c8f09659bde26355252b9815.png)





```python
# Plotnine绘制多系列折线图
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotnine import *

# 中文设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

date_list = pd.date_range('2022-01-01', '2022-03-31').astype('str').tolist() * 2
value_list = np.random.randn(len(date_list))
Class = [1] * (len(date_list) // 2) + [2] * (len(date_list) // 2)

data = pd.DataFrame({
    'date_list': date_list,
    'value_list': value_list,
    'Class': Class
})

data['date_list'] = pd.to_datetime(data['date_list'])

p1 = (
    ggplot(data, aes(x='date_list', y='value_list', color='factor(Class)')) +
    geom_line(size=1) +
    scale_x_date(
        date_breaks="1 week",   # 每一周一个刻度
        date_labels="%m-%d"     # 月‑日，也可以用 "%D"
    ) +
    labs(x='date', y='value', title="多系列时间序列折线图", color="类别") +
    theme(dpi=100, figure_size=(10, 5))
)

p1.draw()
```

![678e6ff31b87e217ca6d80d0e12a7e35](/images/math-modeling/678e6ff31b87e217ca6d80d0e12a7e35.png)



```python
# Matplotlib 绘制多系列折线图
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

date_list = pd.date_range('2022-01-01', '2022-03-31').astype('str').tolist() * 2
value_list = np.random.randn(len(date_list))
Class = [1] * (len(date_list) // 2) + [2] * (len(date_list) // 2)

data = pd.DataFrame({
    'date_list': date_list,
    'value_list': value_list,
    'Class': Class
})
data['date_list'] = pd.to_datetime(data['date_list'])

plt.figure(figsize=(10,5))
for c in data['Class'].unique():
    sub = data[data['Class'] == c]
    plt.plot(sub['date_list'], sub['value_list'], label=f"Class {c}")

ax = plt.gca()
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

plt.xlabel("date")
plt.ylabel("value")
plt.title("多系列时间序列折线图")
plt.legend(title="类别")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

![d0b2fe3928e9b232f2283e41947c3201](/images/math-modeling/d0b2fe3928e9b232f2283e41947c3201.png)



