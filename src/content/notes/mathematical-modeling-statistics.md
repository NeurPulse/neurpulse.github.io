---
title: "数学建模：统计模型与假设检验"
description: "从样本波动理解区间、p 值和检验选择"
pubDate: 2026-08-07
updatedDate: 2026-10-21
order: 5
category: '数学建模'
tags: ["统计","假设检验","t检验","回归","SciPy","零基础"]
---

> [专题学习地图](/notes/mathematical-modeling-roadmap/) · 第 5 / 27 篇

## 学习目标与先修知识

目标是分清总体与样本、标准差与标准误、置信区间与 p 值，并能根据数据设计选择检验。先修为均值、方差、条件概率和平方根。所有示例是教学数据，结论只针对相应设定，不外推成真实实验结论。

可回看：[必备数学与 Python 基础](/notes/mathematical-modeling-foundations/)。

## 问题背景与使用时机

两组平均成绩不同，可能是教学方式不同，也可能只是抽到的学生不同。统计推断研究“观察到的差异，相对于抽样波动有多大”。

先确定观察单位和抽样方式，再选择公式。把同一个人多次测量当作独立样本、把相关时序当作独立数据，即使软件返回很多小数位，也不会自动获得可信结论。

## 核心概念与符号

### 从样本到区间

总体是想研究的全部对象；样本是实际观察到的一部分。统计量是由样本计算的数，如 $\bar x$。重复抽样时统计量也会变化，描述这种变化的就是抽样分布。

对独立正态样本、未知总体方差，样本均值的标准误估计为 $SE=s/\sqrt n$。总体均值的双侧 $1-\alpha$ 置信区间为：

$$
\bar x\pm t_{1-\alpha/2,n-1}\frac{s}{\sqrt n}.
$$

$1-\alpha=0.95$ 称为 95% 置信水平；它描述重复抽样构造区间的方法有约 95% 覆盖固定真值，不是观测完这一个区间后直接给固定参数赋予 95% 概率。贝叶斯可信区间使用另一套概率解释。

### 四种常见分布

- 正态 $N(\mu,\sigma^2)$：对称钟形，$\sigma>0$；标准正态为 $N(0,1)$。
- 卡方 $\chi^2_\nu$：$\nu$ 个独立标准正态变量平方和，常用于正态总体方差的推断。
- t 分布：$Z/\sqrt{U/\nu}$，其中 $Z\sim N(0,1)$、$U\sim\chi^2_\nu$ 且独立；尾部较厚。
- F 分布：$(U/\nu_1)/(V/\nu_2)$，两个卡方量独立；常用于方差比和 ANOVA。

这些是统计量在规定条件下的分布，不是“所有原始数据都要先变成正态”。

### 检验的三个量

原假设 $H_0$ 表示待检验的基准，如均值等于 9；备择假设 $H_1$ 表示关心的偏离。显著性水平 $\alpha$ 应预先确定。p 值是在 $H_0$ 和检验条件成立时，获得同样或更极端统计量的概率。

p 小表示当前数据与 $H_0$ 相容性较弱；p 大只能说证据不足。p 不等于原假设为真的概率，不衡量效果有多大，也不能替代实际意义。

## 手算一个最小例子

样本 $8,9,10,11,12$，样本量 $n=5$，均值 $\bar x=10$。离均差平方和 $4+1+0+1+4=10$，故：

$$
s^2=\frac{10}{4}=2.5,\quad SE=\sqrt{\frac{2.5}{5}}=\sqrt{0.5}\approx0.7071.
$$

检验 $H_0:\mu=9$，双侧 t 统计量为：

$$
t=\frac{10-9}{0.7071}\approx1.4142,\qquad df=4.
$$

查 t 分布得到 p 约为 0.2302。在 $\alpha=0.05$ 下不拒绝 $H_0$，但不能说“已经证明总体均值就是 9”。

95% 双侧临界值约为 2.7764，区间为 $10\pm2.7764\times0.7071$，约 $[8.04,11.96]$。样本均值比 9 高 1 个单位，这是效应的原始量纲描述；区间宽提醒我们样本很少。

## 建模步骤与完整案例

案例：一家包装线抽取 5 件商品的重量，研究平均重量是否偏离标称值。假设这些件来自相同过程、相互独立且总体近似正态。样本量小时，这些假设尤其需要领域依据和图形辅助，不能靠一个正态性检验“担保”。

若比较两条独立生产线，通常可从 Welch t 检验开始；若同一批设备改造前后测量，应按设备配对。统计模型必须对应数据的生成设计。

| 研究目标与设计 | 常用入口 | 关键条件或限制 |
|---|---|---|
| 一个总体均值与目标比较 | 单样本 t 检验 | 独立，正态或适当大样本近似 |
| 两组独立总体均值比较 | Welch t 检验 | 独立，不要求等方差 |
| 同一对象前后变化 | 配对 t 检验 | 对差值建模，存在真实配对 |
| 多个独立组均值比较 | ANOVA | 经典版本要求等方差和误差近似正态 |
| 分类变量是否独立 | 卡方独立性检验 | 计数而非均值，期望频数需适当 |
| 方差是否相同 | Levene 检验 | 不是所有 t 检验的必经门槛 |
| 单调或线性关联 | Spearman 或 Pearson | 关联不等于因果 |

ANOVA 显著只表示至少一组不同；两两比较需要控制多重检验，如 Tukey。多因素模型的交互项描述“一个因素的差异随另一个因素而变化”，其统计显著性本身不解释因果机制。

## Python 实现与结果解释

配套文件：[查看或下载本篇完整脚本](/downloads/math-modeling/mathematical-modeling-statistics.py)。

```python
import numpy as np
from scipy import stats

weight = np.array([8., 9., 10., 11., 12.])
n = len(weight)
se = stats.sem(weight)
result = stats.ttest_1samp(weight, popmean=9)
interval = stats.t.interval(0.95, n-1, loc=weight.mean(), scale=se)
print("均值、样本标准差、标准误:", weight.mean(), weight.std(ddof=1), se)
print("t与p:", result.statistic, result.pvalue)
print("95%双侧区间:", interval)
assert np.isclose(result.statistic, np.sqrt(2))
assert interval[0] < 9 < interval[1]

before = np.array([10., 12., 11., 14., 13.])
after = before + np.array([1., 2., 1., 3., 2.])
paired = stats.ttest_rel(after, before)
difference_test = stats.ttest_1samp(after-before, 0)
assert np.isclose(paired.statistic, difference_test.statistic)
print("配对变化:", paired)
print("两条独立生产线的Welch检验:",
      stats.ttest_ind(weight, [10., 11., 14., 13., 12., 15.], equal_var=False))
print("分类列联表:", stats.chi2_contingency([[20, 10], [10, 20]], correction=False))
```

程序应重现前面的 t 值和区间。配对检验等价于对“每对差值”做均值为 0 的单样本检验；这正是它保留配对信息的原因。独立组不能仅因为数组长度相同就套配对公式。

## 方法选择与常见误区

### 正态性与相关性

Shapiro–Wilk、QQ 图、Jarque–Bera 等用于检查与正态假设的相容性；不拒绝不能证明正态。普通 KS 对一个完全指定的连续分布做检验；`kstest(x, "norm")` 指定的是 $N(0,1)$。用同一批数据估计均值、方差后直接套普通 KS 的 p 值不正确。

Pearson 系数为协方差除以两变量标准差之积，主要度量线性关系，对异常值敏感。Spearman 对秩计算相关，适合单调关系；有并列值时应由正确的秩处理实现计算。Kendall 比较样本对的顺序一致程度。没有统一的“超过某值就一定强相关”阈值。

### 常见解释错误

- 统计显著不保证业务效果重要，小效应在大样本中也可能显著。
- p 大不证明两组等效；等效与非劣需要预定界限及相应方法。
- 两个各 95% 的单侧界限拼起来，通常对应 90% 双侧区间，不能改名为 95%。
- 反复尝试分组和检验再挑最小 p，会改变错误率。
- 观测数据中的显著相关不能自动识别因果。报告效应大小、区间、样本设计和假设比只报一个星号更有用。

## 练习

1. 样本标准差为 6、独立样本数为 36，均值标准误是多少？
2. p=0.08、预设 $\alpha=0.05$，该怎样表述？
3. 同一学生考试前后成绩，用配对还是独立检验？若是两个不同班级呢？
4. ANOVA 显著是否表示每两组都不同？
5. 95% 区间变窄可能来自哪些变化？

## 参考答案

1. $SE=6/\sqrt{36}=1$。
2. 在该水平下不拒绝原假设；不能说原假设有 92% 的概率成立，也不能说一定无差异。
3. 前后测按学生一一对应，考虑配对差值；不同班级若独立抽样，则考虑独立组方法。若班内强相关，还应考虑聚类或层级结构。
4. 不是，只表示“所有组均值相同”受到证据反对；具体组对需进一步比较并控制多重性。
5. 更多独立样本、较低数据波动或更低置信水平都可能使区间变窄；不能只凭区间窄就断言研究更可靠。

## 复习卡片与下一步

复习时能回答：单位是什么、观察独立吗、比较的是均值还是分布、效应多大、区间多宽。下一篇用回归把多个变量与连续结果联系起来，并分开解释和预测的目标。

[上一篇 数据可视化与探索分析](/notes/mathematical-modeling-data-visualization/) · [下一篇 回归与曲线拟合](/notes/mathematical-modeling-regression/)

## 参考资料

- [Datawhale《数学建模导论》第9章](https://datawhalechina.github.io/intro-mathmodel/#/CH9/第九章-机器学习与统计模型)：用于建立知识目录，建议与本篇的手算和案例对照阅读。
- [SciPy 统计接口](https://docs.scipy.org/doc/scipy/reference/stats.html)。
- [Welch 与独立样本 t 检验](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html)。
- [Shapiro–Wilk](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html)。
