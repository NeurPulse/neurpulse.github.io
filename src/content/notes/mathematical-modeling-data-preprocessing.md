---
title: "数学建模：数据预处理与插值"
description: "从缺失、重复和量纲走到可用数据，理解插值与信息泄漏"
pubDate: 2026-08-07
updatedDate: 2026-10-21
order: 3
category: '数学建模'
tags: ["数据预处理","插值","Pandas","Scikit-learn","零基础"]
---

> [专题学习地图](/notes/mathematical-modeling-roadmap/) · 第 3 / 27 篇

## 学习目标与先修知识

学完后，你应能区分错误重复与真实重复、缺失与零、插值与外推，能在训练数据上拟合填补器和缩放器，并保持测试信息独立。先看数学与 Python 基础，至少理解“行是样本、列是变量”。

可回看：[必备数学与 Python 基础](/notes/mathematical-modeling-foundations/)。

## 问题背景与使用时机

预测学生后续成绩时，表里可能有漏填的练习时长、重复导出的学号、不同量纲的特征。模型不会自动知道某个 0 是“没有练习”还是“缺失时误填”。

清洗的目标是让数据表达实际观察，不是把所有数都变得平滑好看。保留原始表，记录每个处理规则及受影响行数；真实的极端事件也可能是最有价值的观测。

## 核心概念与符号

### 先判断数据的含义

主键用于识别观察单位，例如“学生编号 + 日期”。只有确认同一观察被重复录入，才删除重复；同一学生不同日期的记录不能随意去掉。缺失机制包括完全随机缺失、依赖已观测信息的缺失和依赖未观测值的缺失；简单均值填补不能自动解决选择偏差。

数值特征可用均值、中位数或模型填补；类别特征可增设“未知”。整列都缺失时没有可估计的列均值，应删除、补采或使用事先定义的规则。目标标签缺失的训练行通常不能直接参与普通监督拟合。

### 缩放与编码

标准化 $z=(x-\mu_{\mathrm{train}})/s_{\mathrm{train}}$ 把尺度转为相对训练数据的距离。Min-Max 缩放为 $(x-a)/(b-a)$，其中极值也只能来自训练集。常量列分母为零，需要明确处理。类别如城市不能随意编码为 1、2、3 后当作等距数字；独热编码为每种类别建立一个指示列。

### 插值及其边界

线性插值在两个已知位置之间连直线；样条在各区间用低次多项式拼接；拉格朗日插值构造一个通过所有节点的多项式。坐标可以不等间隔，但必须已知且互不重复。超出已知区间是外推，风险不同。

三次样条还需边界条件；自然样条指定两端二阶导数为零，not-a-knot 是另一种条件。高次多项式即使穿过所有观测，也可能在节点间剧烈振荡。

## 手算一个最小例子

已知 1 日记录 10、3 日记录 14，想填 2 日：

$$
\hat y(2)=10+\frac{14-10}{3-1}(2-1)=12.
$$

这里额外假设区间内变化近似线性。若 2 日恰好发生突发事件，12 不一定接近真实值。

训练时长为 $1,2,3$ 小时，均值 2；用总体式标准差 $s=\sqrt{2/3}\approx0.8165$，则 3 小时标准化为 $(3-2)/0.8165\approx1.2247$。scikit-learn 的 StandardScaler 使用这种分母为 $n$ 的方差定义，不等于统计推断中分母 $n-1$ 的样本方差。

训练集为 $1,3,\text{缺失}$ 时，中位数是 2。测试集即使出现 100，填补训练缺失的值仍应为 2；把测试值混进估计会泄漏信息。

## 建模步骤与完整案例

案例：用练习时长预测成绩。先按学生分出训练和测试，再做填补、标准化和回归。假设各学生相互独立，练习时长在预测时已知；真实研究中不能把考试后才能获得的字段作为输入。

流程为：检查编号唯一性 → 区分特征与标签 → 留出测试学生 → 在训练折估计预处理参数 → 拟合模型 → 对测试行只做 transform → 评价误差。交叉验证时这些步骤也要在每个训练折内部重做，Pipeline 可以自动保持边界。

![先划分再拟合预处理的边界](/images/math-modeling/learning/validation-boundary.svg)

## Python 实现与结果解释

配套文件：[查看或下载本篇完整脚本](/downloads/math-modeling/mathematical-modeling-data-preprocessing.py)。

```python
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

X_train = np.array([[1.], [2.], [np.nan], [4.], [5.]])
y_train = np.array([52., 54., 56., 58., 60.])
X_test = np.array([[3.], [6.]])
y_test = np.array([56., 62.])
model = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    LinearRegression()
)
model.fit(X_train, y_train)
prediction = model.predict(X_test)
imputer = model.named_steps["simpleimputer"]
print("训练填补值:", imputer.statistics_)
print("预测值:", prediction)
print("测试MAE:", np.mean(np.abs(prediction-y_test)))
assert np.allclose(imputer.statistics_, [3.])
assert np.allclose(prediction, [56., 62.])

# 独立的插值案例：只在已知区间内查询。
from scipy.interpolate import interp1d, CubicSpline, BarycentricInterpolator
x = np.array([1., 2., 3., 4.])
y = np.array([1., 4., 9., 16.])
linear = interp1d(x, y, kind="linear", bounds_error=True)
natural_spline = CubicSpline(x, y, bc_type="natural", extrapolate=False)
polynomial = BarycentricInterpolator(x, y)
print("2.5处三种插值:", linear(2.5), natural_spline(2.5), polynomial(2.5))
assert np.isclose(linear(2.5), 6.5)
assert np.isclose(polynomial(2.5), 6.25)
```

训练填补值为 3，测试 MAE 为 0。这是人为构造的直线数据，目的是验证流程，不代表现实成绩可完美预测。2.5 处真实平方函数为 6.25；线性插值得到 6.5，说明“通过节点”不等于“节点间也完全正确”。

## 方法选择与常见误区

| 情况 | 可优先尝试 | 必须核查 |
|---|---|---|
| 少量数值缺失、分布有极端值 | 中位数与缺失指示 | 缺失是否含额外信息 |
| KNN、SVM、神经网络输入 | 训练集内缩放 | 测试是否只 transform |
| 树模型 | 通常不依赖尺度缩放 | 缺失支持随实现而异 |
| 离线传感器缺口 | 有序坐标下插值 | 变化规律是否合理 |
| 未来预测 | 仅用过去填补 | 双向插值可能用到未来 |
| 类别变量 | 独热或明确的类别处理 | 测试出现新类别怎么办 |

异常检测只标出待核查记录，不代表必须删除。不要根据测试集的预测误差反复修改清洗规则后仍称它是“独立测试”。

## 练习

1. 训练值 $2,4,\text{缺失},8$，中位数填补是多少？
2. 1 时刻为 4、4 时刻为 10，线性插值的 2 时刻是多少？
3. 为什么用整张表标准化后再划分数据有问题？
4. 城市编码“北京=1，上海=2，广州=3”直接送入 KNN，会暗含什么假设？

## 参考答案

1. 有效值排序为 $2,4,8$，中位数为 4。
2. 斜率 $(10-4)/(4-1)=2$，所以 2 时刻为 $4+2=6$。
3. 均值、方差等已经使用测试信息。正确顺序是先划分，并在训练集或交叉验证训练折拟合。
4. 暗含城市之间存在人为的等距顺序，例如北京到广州的距离是北京到上海的两倍；这种数字距离通常没有业务依据。

## 复习卡片与下一步

记住四条：确认观察单位；处理规则来自训练；预测时只能用当时可用的信息；插值需要坐标和变化假设。下一篇通过图检查数据分布、异常和关系。

[上一篇 必备数学与 Python 基础](/notes/mathematical-modeling-foundations/) · [下一篇 数据可视化与探索分析](/notes/mathematical-modeling-data-visualization/)

## 参考资料

- [Datawhale《数学建模导论》第6章](https://datawhalechina.github.io/intro-mathmodel/#/CH6/第六章-数据处理与拟合模型)：用于建立知识目录，建议与本篇的手算和案例对照阅读。
- [scikit-learn 常见错误与信息泄漏](https://scikit-learn.org/stable/common_pitfalls.html)。
- [SciPy 插值指南](https://docs.scipy.org/doc/scipy/tutorial/interpolate.html)。
