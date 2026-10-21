---
title: "数学建模：数据可视化与探索分析"
description: "围绕问题选择图表，读懂分布、关系和异常"
pubDate: 2026-08-07
updatedDate: 2026-10-21
order: 4
category: '数学建模'
tags: ["数据可视化","Matplotlib","Plotnine","零基础"]
---

> [专题学习地图](/notes/mathematical-modeling-roadmap/) · 第 4 / 27 篇

## 学习目标与先修知识

学完后应能根据问题选择图形，解释坐标、单位、样本与误差，并用图发现需要进一步核查的现象。先修为数据预处理及基本数组操作。本篇先学怎样读图，再学怎样写绘图代码。

可回看：[数据预处理与插值](/notes/mathematical-modeling-data-preprocessing/)。

## 问题背景与使用时机

假设你看到“练习越多，成绩越高”的散点图。它可能显示真实关系，也可能受年级、基础成绩或样本选择影响。图形帮助提出问题，不能单凭一条趋势线完成因果判断。

探索分析中的图用来找规律与问题；对外展示中的图需要明确回答一个问题。两者都应保留单位和来源，不能用截断坐标、面积或颜色制造夸张印象。

## 核心概念与符号

| 想回答的问题 | 合适的起点 | 读图时看什么 |
|---|---|---|
| 各类别多少 | 条形图 | 类别与长度，数值轴通常从 0 开始 |
| 数值如何分布 | 直方图、箱线图 | 集中、偏斜、尾部与异常候选 |
| 两变量怎样关联 | 散点图 | 方向、弯曲、分群、离群点 |
| 随时间怎样变化 | 折线图 | 先后顺序、缺口、趋势与周期 |
| 总量由什么组成 | 堆叠图 | 各部分是否可加、是否同一总体 |
| 相关矩阵是什么 | 相关热图 | 方向、强度及变量是否重复 |

直方图把数值分箱，纵轴可为频数或密度。密度图中“柱高 × 区间宽度”对应概率，全部面积约为 1；不同箱宽时不能只比较高度。箱线图通常显示中位数、四分位数和按规则确定的须，须外的点是异常候选，不是已确认的错误。

相关系数位于 $[-1,1]$，Pearson 相关衡量线性联系；$y=x^2$ 可能有强关联而线性相关接近 0。

Matplotlib 控制画布和坐标轴；Seaborn 提供统计绘图接口；Plotnine 按映射与图层组织图形。先选清楚图的含义，换库不会改变统计问题。

## 手算一个最小例子

四位学生成绩为 $60,70,80,90$，按区间 $[60,70),[70,80),[80,90]$ 分箱，频数是 $1,1,2$。每个箱宽 10，密度分别为：

$$
\frac1{4\times10}=0.025,\quad0.025,\quad\frac2{4\times10}=0.05.
$$

总面积 $0.025\times10+0.025\times10+0.05\times10=1$。密度 0.05 不是“90 分的概率为 5%”。

再看两年房价单价 3 万与 4 万。两者相加为 7 万，并不代表一个有业务意义的年度总量，因此不宜用百分比堆叠图解释“价格构成”。同一年收入中工资与奖金可以相加，才适合讨论构成。

## 建模步骤与完整案例

案例：检查一组练习时长与成绩记录。先画时长分布，了解样本集中在哪；再画散点，检查关系与异常；最后画简单线性拟合的残差，观察模型遗漏。

假设数据为教学构造，时间单位为小时、成绩单位为分。若某个长时长样本成绩偏低，先核查录入、缺失、样本情况和模型形态，不能只因它偏离趋势线就删除。

残差定义为 $e_i=y_i-\hat y_i$。如果残差随时长呈弯曲，可能需要非线性关系；如果波动随时长增加，则可能存在异方差。图是诊断起点，后续要用模型与验证确认。

## Python 实现与结果解释

配套文件：[查看或下载本篇完整脚本](/downloads/math-modeling/mathematical-modeling-data-visualization.py)。

```python
import numpy as np
import matplotlib.pyplot as plt

hours = np.array([1., 2., 3., 4., 5., 6., 7., 8.])
scores = np.array([51., 55., 56., 60., 61., 65., 66., 60.])
slope, intercept = np.polyfit(hours, scores, 1)
fitted = slope*hours + intercept
residual = scores-fitted

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
axes[0].hist(hours, bins=[0.5, 2.5, 4.5, 6.5, 8.5], edgecolor="black")
axes[0].set(xlabel="practice hours", ylabel="count", title="sample distribution")
axes[1].scatter(hours, scores)
axes[1].plot(hours, fitted, color="tomato")
axes[1].set(xlabel="practice hours", ylabel="score", title="relationship")
axes[2].scatter(hours, residual)
axes[2].axhline(0, color="gray", linestyle="--")
axes[2].set(xlabel="practice hours", ylabel="observed - fitted", title="residuals")
plt.tight_layout()
plt.show()
print("最后一个样本残差:", residual[-1])
assert abs(residual.mean()) < 1e-10
```

最后一个样本位于拟合线下方，残差为负；这只能说明模型高估了它。含截距的最小二乘模型在训练集上的残差和约为零，因此这个断言是在检查计算一致性，不是证明模型预测可靠。

如果使用 LOWESS，statsmodels 的参数顺序为 `lowess(y, x)`。默认结果按 x 排序，画图时应使用返回的两列；使用 `return_sorted=False` 才按原始顺序返回拟合值。

## 方法选择与常见误区

- 条形图适合类别比较，折线图中的相邻连接暗含顺序；没有顺序的城市不应仅靠连线强调变化。
- 雷达图需要统一量纲和优劣方向。原始血量、速度、价格直接放一起，图形面积没有统一意义。
- 相关热图要先计算 `data.corr()`，任意随机矩阵热图不能称为相关矩阵。
- 汽车数据中的 mpg 是“每加仑行驶英里数”，数值越大通常越省油，不等于每百公里耗油量。
- 多系列比较尽量共享刻度；置信区间、不确定性带和个体波动范围是不同对象，图例必须说明。
- 饼图适合少数、非负、互斥且构成同一总体的类别。类别很多时，排序条形图通常更便于比较。

## 练习

1. 想比较 6 个地区的平均通勤时长，首先选什么图？还应补什么信息？
2. 横轴是收入、纵轴是支出，散点向右上方延伸，能否证明提高收入一定导致支出增加？
3. 箱线图须外一点可以直接删吗？
4. 直方图某箱密度为 0.2、宽度为 0.5，对应的比例约是多少？

## 参考答案

1. 条形图或带区间的点图，注明分钟、样本量和均值的不确定性；若关注分布差异，再加箱线图或分布图。
2. 不能。可能存在共同影响因素、反向关系或选择偏差，因果结论需要额外设计与假设。
3. 不可以。须外只是按当前规则标出的候选异常，应查证来源和业务含义。
4. 面积为 $0.2\times0.5=0.1$，约为 10%，而不是 20%。

## 复习卡片与下一步

复习顺序：先说问题，再选图；先读单位，再解释方向；先核查异常，再决定处理。下一篇学习如何给差异和估计结果加上不确定性。

[上一篇 数据预处理与插值](/notes/mathematical-modeling-data-preprocessing/) · [下一篇 统计模型与假设检验](/notes/mathematical-modeling-statistics/)

## 参考资料

- [Datawhale《数学建模导论》第6章](https://datawhalechina.github.io/intro-mathmodel/#/CH6/第六章-数据处理与拟合模型)：用于建立知识目录，建议与本篇的手算和案例对照阅读。
- [Matplotlib 教程](https://matplotlib.org/stable/tutorials/index.html)。
- [statsmodels LOWESS](https://www.statsmodels.org/stable/generated/statsmodels.nonparametric.smoothers_lowess.lowess.html)。
