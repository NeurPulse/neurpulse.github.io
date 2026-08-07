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
    'category': ['Female', 'Male', '1-24', '25-54', '55+', 'Lo', 'Lo-Med', 'Med', 'High'],
    'value': [60, 40, 50, 30, 20, 10, 25, 25, 40],
})
df['variable'] = pd.Categorical(df['variable'], categories=['gender', 'age', 'income'])
df['category'] = pd.Categorical(df['category'], categories=df['category'])
# 堆叠柱状图
from plotnine import *
(
    ggplot(df, aes(x='variable', y='value', fill='category'))+
    geom_col()
)
```

```python
# 柱状图
from plotnine import *
(
    ggplot(df, aes(x='variable', y='value', fill='category'))+
    geom_col(stat='identity', position='dodge')
)
```



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
# 创建数据
x = np.linspace(-2*np.pi, 2*np.pi, 100) # 在前面的范围内平均分为100份
y = np.sin(x)  # 得到y值
import matplotlib.pyplot as plt 
plt.figure(figsize=(8,6))  # 得到画布
plt.scatter(x=x, y=y)  # 在画布上画点
plt.xlabel('x')  # 设置x轴标签
plt.ylabel('y')  # 设置y轴标签
plt.title('y = sin(x)') # 设置主题
plt.show()
```





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
# 准备数据
x = np.linspace(-10, 10, 100)
y = 2 * x + 1 + np.random.randn(100)
df = pd.DataFrame({'x':x, 'y':y})

# 使用Seaborn绘制带有拟合直线效果的散点图
sns.lmplot("x","y",data=df)
```

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
p1 = (
    ggplot(mpg, aes(x='displ', y='hwy', color='drv'))     # 设置数据映射图层，数据集使用mpg，x数据使用mpg['displ']，y数据使用mpg['hwy']，颜色映射使用mog['drv']
    + geom_point()       # 绘制散点图图层
    + geom_smooth(method='lm')        # 绘制平滑线图层
    + labs(x='displacement', y='horsepower')     # 绘制x、y标签图层
)
print(p1)   # 展示p1图像
```

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
data = pd.DataFrame({'city':['深圳', '上海', '北京', '广州', '成都'], 'house_price(w)':[3.5, 4.0, 4.2, 2.1, 1.5]})

fig = plt.figure(figsize=(10,6))
ax1 = fig.add_axes([0.15,0.15,0.7,0.7])  # [left, bottom, width, height], 它表示添加到画布中的矩形区域的左下角坐标(x, y)，以及宽度和高度
plt.bar(data['city'], data['house_price(w)'], width=0.6, align='center', orientation='vertical', label='城市')
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
plt.title("不同城市的房价对比图")   # 在axes1设置标题
plt.xlabel("城市")    # 在axes1中设置x标签
plt.ylabel("房价/w")    # 在axes1中设置y标签
plt.grid(b=True, which='both')  # 在axes1中设置设置网格线
for i in range(len(data)):
    plt.text(i-0.05, data.iloc[i,]['house_price(w)']+0.01, data.iloc[i,]['house_price(w)'],fontsize=13)   # 添加数据注释
plt.legend()
plt.show()
```





```python
## Plotnine绘制单系列柱状图：不同城市的房价对比
data = pd.DataFrame({'city':['深圳', '上海', '北京', '广州', '成都'], 'house_price(w)':[3.5, 4.0, 4.2, 2.1, 1.5]})

p_single_bar = (
    ggplot(data, aes(x='city', y='house_price(w)', fill='city', label='house_price(w)'))+
    geom_bar(stat='identity')+
    labs(x="城市", y="房价(w)", title="不同城市的房价对比图")+
    geom_text(nudge_y=0.08)+
    theme(text = element_text(family = "Songti SC"))
)
print(p_single_bar)
```





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
plt.grid(b=True, which='both')  # 在axes1中设置设置网格线

data_2021 = data.loc[data['年份']==2021,:]
for i in range(len(data_2021)):
    plt.text(i-0.15-0.05, data_2021.iloc[i,2]+0.05, data_2021.iloc[i,2],fontsize=13)   # 添加数据注释

data_2022 = data.loc[data['年份']==2022,:]
for i in range(len(data_2022)):
    plt.text(i+0.15-0.05, data_2022.iloc[i,2]+0.05, data_2022.iloc[i,2],fontsize=13)   # 添加数据注释
plt.legend()
plt.show()
```





```python
## Plotnine绘制多系列柱状图：不同城市在不同年份的房价对比
data = pd.DataFrame({
    '城市':['深圳', '上海', '北京', '广州', '成都', '深圳', '上海', '北京', '广州', '成都'],
    '年份':[2021,2021,2021,2021,2021,2022,2022,2022,2022,2022],
    '房价(w)':[3.5, 4.0, 4.2, 2.1, 1.5, 4.0, 4.2, 4.3, 1.6, 1.9]
})

data['年份'] = pd.Categorical(data['年份'], ordered=True, categories=data['年份'].unique())
p_mult_bar = (
    ggplot(data, aes(x='城市', y='房价(w)', fill='年份'))+
    geom_bar(stat='identity',width=0.6, position='dodge')+
    scale_fill_manual(values = ["#f6e8c3", "#5ab4ac"])+
    labs(x="城市", y="房价(w)", title="不同城市的房价对比图")+
    geom_text(aes(label='房价(w)'), position = position_dodge2(width = 0.6, preserve = 'single'))+
    theme(text = element_text(family = "Songti SC"))
)
print(p_mult_bar)
```





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
plt.legend()
plt.show()
```





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
    theme(text = element_text(family = "Songti SC"))
)
print(p1)
```





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





```python
# 使用Matplotlib绘制聚类散点图
from sklearn.datasets import load_iris  #家在鸢尾花数据集
iris = load_iris()
X = iris.data
label = iris.target
feature = iris.feature_names
df = pd.DataFrame(X, columns=feature)
df['label'] = label

label_unique = np.unique(df['label']).tolist()
plt.figure(figsize=(10, 6))
for i in label_unique:
    df_label = df.loc[df['label'] == i, :]
    plt.scatter(x=df_label['sepal length (cm)'], y=df_label['sepal width (cm)'], s=20, label=i)
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.title('sepal width (cm)~sepal length (cm)')
plt.legend()
plt.show()
```





```python
# 使用plotnine绘制相关系数矩阵图：
from plotnine.data import mtcars
corr_mat = np.round(mtcars.corr(), 1).reset_index() #计算相关系数矩阵
df = pd.melt(corr_mat, id_vars='index', var_name='variable', value_name='corr_xy') #将矩阵宽数据变成长数据
df['abs_corr'] = np.abs(df['corr_xy'])
p1 = (
    ggplot(df, aes(x='index', y='variable', fill='corr_xy', size='abs_corr'))+
    geom_point(shape='o', color='black')+
    scale_size_area(max_size=11, guide=False)+
    scale_fill_cmap(name='RdYIBu_r')+
    coord_equal()+
    labs(x="Variable", y="Variable")+
    theme(dpi=100, figure_size=(4.5,4.55))
)
p2 = (
    ggplot(df, aes(x='index', y='variable', fill='corr_xy', size='abs_corr'))+
    geom_point(shape='s', color='black')+
    scale_size_area(max_size=10, guide=False)+
    scale_fill_cmap(name='RdYIBu_r')+
    coord_equal()+
    labs(x="Variable", y="Variable")+
    theme(dpi=100, figure_size=(4.5,4.55))
)
p3 = (
    ggplot(df, aes(x='index', y='variable', fill='corr_xy', label='corr_xy'))+
    geom_tile(color='black')+
    geom_text(size=8, color='white')+
    scale_fill_cmap(name='RdYIBu_r')+
    coord_equal()+
    labs(x="Variable", y="Variable")+
    theme(dpi=100, figure_size=(4.5,4.55))
)
print([p1, p2, p3])
```





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
plt.figure(figsize=(8, 6))
plt.hist(mtcars['mpg'], bins=20, alpha=0.85)
plt.xlabel("mpg")
plt.ylabel("count")
plt.show()
```





```python
# 使用matplotlib绘制箱线图
import seaborn as sns 
from plotnine.data import mtcars

data = mtcars
data['carb'] = data['carb'].astype('category')
plt.figure(figsize=(8, 6))
sns.boxenplot(x='carb', y='mpg', data=mtcars, linewidth=0.2, palette=sns.husl_palette(6, s=0.9, l=0.65, h=0.0417))
plt.show()
```





```python
# 使用Matplotlib绘制饼状图：
from matplotlib import cm, colors
df = pd.DataFrame({
    '己方': ['寒冰', '布隆', '发条', '盲僧', '青钢影'],
    '敌方': ['女警', '拉克丝', '辛德拉', '赵信', '剑姬'],
    '己方输出': [26000, 5000, 23000, 4396, 21000],
    '敌方输出': [25000, 12000, 21000, 10000, 18000]
})

df_our = df[['己方', '己方输出']].sort_values(by='己方输出', ascending=False).reset_index()
df_other = df[['敌方', '敌方输出']].sort_values(by='敌方输出', ascending=False).reset_index()
color_list = [cm.Set3(i) for i in range(len(df))]
plt.figure(figsize=(16, 10))
plt.subplot(1,2,1)
plt.pie(df_our['己方输出'].values, startangle=90, shadow=True, colors=color_list, labels=df_our['己方'].tolist(), explode=(0,0,0,0,0.3), autopct='%.2f%%')


plt.subplot(1,2,2)
plt.pie(df_other['敌方输出'].values, startangle=90, shadow=True, colors=color_list, labels=df_other['敌方'].tolist(), explode=(0,0,0,0,0.3), autopct='%.2f%%')

plt.show()
```





```python
# 使用Matplotlib绘制环状图：
from matplotlib import cm, colors
df = pd.DataFrame({
    '己方': ['寒冰', '布隆', '发条', '盲僧', '青钢影'],
    '敌方': ['女警', '拉克丝', '辛德拉', '赵信', '剑姬'],
    '己方输出': [26000, 5000, 23000, 4396, 21000],
    '敌方输出': [25000, 12000, 21000, 10000, 18000]
})

df_our = df[['己方', '己方输出']].sort_values(by='己方输出', ascending=False).reset_index()
df_other = df[['敌方', '敌方输出']].sort_values(by='敌方输出', ascending=False).reset_index()
color_list = [cm.Set3(i) for i in range(len(df))]
wedgeprops = {'width':0.3, 'edgecolor':'black', 'linewidth':3}
plt.figure(figsize=(16, 10))
plt.subplot(1,2,1)
plt.pie(df_our['己方输出'].values, startangle=90, shadow=True, colors=color_list, wedgeprops=wedgeprops, labels=df_our['己方'].tolist(), explode=(0,0,0,0,0.3), autopct='%.2f%%')
plt.text(0, 0, '己方' , ha='center', va='center', fontsize=30)

plt.subplot(1,2,2)
plt.pie(df_other['敌方输出'].values, startangle=90, shadow=True, colors=color_list, wedgeprops=wedgeprops, labels=df_other['敌方'].tolist(), explode=(0,0,0,0,0.3), autopct='%.2f%%')
plt.text(0, 0, '敌方' , ha='center', va='center', fontsize=30)
plt.show()
```





##### 时间序列型图表

```python
# 使用Plotnine绘制时间序列线图：
df = pd.read_csv(
    './data/AirPassengers.csv'
    )  # 航空数据1949-1960
df['date'] = pd.to_datetime(df['date'])
p1 = (
    ggplot(df, aes(x='date', y='value'))+
    geom_line(size=1, color='red')+
    scale_x_date(date_labels="%Y", date_breaks="1 year")+
    xlab('date')+
    ylab('value')
)
print(p1)
```





```python
# 使用Matplotlib绘制时间序列折线图
plt.figure(figsize=(8,6))
plt.plot(df['date'], df['value'], color='red')
plt.xlabel("date")
plt.ylabel("value")
plt.show()
```







```python
# Plotnine绘制多系列折线图
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
    ggplot(data, aes(x='date_list', y='value_list', group='factor(Class)', color='factor(Class)'))+
    geom_line(size=1)+
    scale_x_date(date_labels="%D")+
    xlab('date')+
    ylab('value')
)
print(p1)
```





```python
# Matplotlib 绘制多系列折线图
date_list = pd.date_range('2022-01-01', '2022-03-31').astype('str').tolist()
value_list1 = np.random.randn(len(date_list))
value_list2 = np.random.randn(len(date_list))
data = pd.DataFrame({
    'date_list': date_list,
    'value_list1': value_list1,
    'value_list2': value_list2
})
data['date_list'] = pd.to_datetime(data['date_list'])

plt.figure(figsize=(8, 6))
plt.plot(data['date_list'], data['value_list1'], color='red', alpha=0.86, label='value1')
plt.plot(data['date_list'], data['value_list2'], color='blue', alpha=0.86, label='value2')
plt.legend()
plt.xlabel('date')
plt.ylabel('value')
plt.show()
```





