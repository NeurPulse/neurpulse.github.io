---
title: "数学建模：综合案例 从销量预测到备货决策"
description: "完整连接数据检查、基线、验证、成本优化和报告"
pubDate: 2026-10-21
updatedDate: 2026-10-21
order: 26
category: '数学建模'
tags: ["综合案例","预测","库存优化","零基础"]
---

> [专题学习地图](/notes/mathematical-modeling-roadmap/) · 第 26 / 27 篇

## 学习目标与先修知识

目标是独立完成一个从问题重述到决策报告的案例：检查销量数据、建立基线、选择预测模型、校准需求情景、选择备货量、评估真实成本和敏感性。先修为回归、时间验证、优化与随机模拟。本篇代码自带教学数据，不依赖本地 CSV。

可回看：[回归与曲线拟合](/notes/mathematical-modeling-regression/) · [时间序列分析与验证](/notes/mathematical-modeling-time-series/) · [线性规划与整数优化](/notes/mathematical-modeling-optimization/)。

## 问题背景与使用时机

小店要每天提前准备午餐。准备少了会失去销售机会，准备多了会浪费。预测销量最准确的点估计，不一定就是最赚钱的备货量，因为缺货与剩余的损失不同。

业务问题应写成：在出餐前只能知道日期、预定促销和历史销量，选择非负整数备货量 Q，使预期经营损失低，同时报告缺货与浪费。不能使用当天结束才知道的需求。

## 核心概念与符号

定义需求 D、售价 p、单位成本 c、剩余回收价值 s，假设 $p>c>s$，无额外库存跨日、无需求替代，且不会出现负需求。

当天利润：

$$
\Pi(Q,D)=p\min(Q,D)+s(Q-D)_+-cQ.
$$

相对“事先知道需求且正好备齐”的损失可写为：

$$
L(Q,D)=(p-c)(D-Q)_++(c-s)(Q-D)_+.
$$

$(a)_+=\max(a,0)$。缺货单位代价 $c_u=p-c$，剩余单位代价 $c_o=c-s$。连续分布下的最优临界分位为 $F_D(Q^*)=c_u/(c_u+c_o)$；离散需求采用适当的分位或直接枚举核查。

这个公式来自边际比较：多备一件在需求超过库存时节约 $c_u$，在需求已经满足时增加 $c_o$。当缺货更贵时，应偏向较高需求分位，而不是总取平均。

## 手算一个最小例子

设 p=10、c=4、s=1，则 $c_u=6,c_o=3$，临界分位为 $6/9=2/3$。需求可能为10、12、14，概率分别为0.25、0.5、0.25。

- Q=10：期望损失 $0.5\times6\times2+0.25\times6\times4=12$。
- Q=12：$0.25\times3\times2+0.25\times6\times2=4.5$。
- Q=14：$0.25\times3\times4+0.5\times3\times2=6$。

因此这三个候选中12最好，完整枚举也得到12。不能因为缺货单位损失更大，就不计代价地把全部可能需求都备齐。

若缺货代价增加为18、剩余仍为3，临界分位升为18/21，最优转为14。经营偏好改变会改变决策，即使预测分布完全相同。

## 建模步骤与完整案例

### 数据与边界

构造210天数据，含缓慢趋势、7天周期、预先计划的促销和随机波动。划分为112天初始训练、28天模型选择、28天情景校准和42天最终测试。

验证段用于选择均值、星期均值或 Ridge 的参数；选定后用前140天重训。校准段只用于估计该冻结模型的残差分布；最后42天既不选模型，也不改成本参数。这样将模型选择、需求情景估计与最终评价分开。

![训练、验证、校准和经营评价的边界](/images/math-modeling/learning/forecast-to-stock.svg)

### 从预测到行动

1. 检查日期、唯一性、非负需求与预测时可用的促销字段。
2. 在相同验证时段比较简单基线与候选模型。
3. 由校准残差构造“预测均值 + 残差”的经验需求情景。
4. 在有限整数 Q 上最小化情景平均损失，得到每日备货。
5. 与“直接四舍五入预测值备货”比较利润、缺货和剩余。
6. 使用事先设定的成本情景做敏感性分析，不能看完测试再挑有利的代价。

残差跨时间可交换、加法误差近似稳定是情景生成的额外假设；若误差随促销或销量水平明显变化，应分层或建条件分布。

## Python 实现与结果解释

配套文件：[查看或下载本篇完整脚本](/downloads/math-modeling/mathematical-modeling-integrated-case.py)。

```python
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

rng = np.random.default_rng(42)
t = np.arange(210)
promotion = rng.binomial(1, 0.15, len(t))  # 预先计划，预测时可知
demand = np.maximum(np.rint(
    50 + 0.08*t + 5*np.sin(2*np.pi*t/7) + 6*promotion + rng.normal(0, 3, len(t))
), 0).astype(int)
table = pd.DataFrame({
    "date": pd.date_range("2026-01-01", periods=len(t), freq="D"),
    "day": t, "promotion": promotion, "demand": demand
})
assert table["date"].is_unique and table["date"].is_monotonic_increasing
assert table["demand"].ge(0).all()
X = np.column_stack([t, np.sin(2*np.pi*t/7), np.cos(2*np.pi*t/7), promotion])

def fit_predict(kind, end, query_indices):
    if kind == "mean":
        return np.full(len(query_indices), demand[:end].mean()), None
    if kind == "weekday":
        means = np.array([demand[:end][t[:end] % 7 == d].mean() for d in range(7)])
        return means[query_indices % 7], None
    model = make_pipeline(StandardScaler(), Ridge(alpha=float(kind)))
    model.fit(X[:end], demand[:end])
    return model.predict(X[query_indices]), model

candidates = ["mean", "weekday", 0.1, 1., 10.]
valid_indices = np.arange(112, 140)
errors = {}
for candidate in candidates:
    predicted, _ = fit_predict(candidate, 112, valid_indices)
    errors[candidate] = np.mean(np.abs(demand[valid_indices]-predicted))
selected = min(errors, key=errors.get)
print("验证MAE与选择:", errors, selected)

all_indices = np.arange(140, 210)
forecast, fitted_model = fit_predict(selected, 140, all_indices)
calibration_residuals = demand[140:168]-forecast[:28]
test_forecast = forecast[28:]
test_demand = demand[168:]
assert len(test_forecast) == len(test_demand) == 42

def loss(quantity, scenarios, under=6., over=3.):
    return under*np.maximum(scenarios-quantity, 0) + over*np.maximum(quantity-scenarios, 0)

def choose_stock(prediction, residuals, under=6., over=3.):
    scenarios = np.maximum(np.rint(prediction+residuals), 0).astype(int)
    choices = np.arange(scenarios.max()+1)
    expected = np.array([loss(q, scenarios, under, over).mean() for q in choices])
    return int(choices[expected.argmin()])

stock = np.array([choose_stock(p, calibration_residuals) for p in test_forecast])
point_stock = np.maximum(np.rint(test_forecast), 0).astype(int)
def summarize(quantity):
    profit = 10*np.minimum(quantity, test_demand) + np.maximum(quantity-test_demand, 0)-4*quantity
    return {
        "mean_profit": float(profit.mean()),
        "mean_shortage": float(np.maximum(test_demand-quantity, 0).mean()),
        "mean_leftover": float(np.maximum(quantity-test_demand, 0).mean()),
        "mean_loss": float(loss(quantity, test_demand).mean())
    }
print("最终预测MAE:", np.mean(np.abs(test_forecast-test_demand)))
print("点预测备货:", summarize(point_stock))
print("情景优化备货:", summarize(stock))
print("缺货代价18时平均备货:",
      np.mean([choose_stock(p, calibration_residuals, under=18.) for p in test_forecast]))

# 独立核验手算例子。
scenarios = np.array([10, 12, 12, 14])  # 以重复次数表达概率
best_q = min(range(16), key=lambda q: loss(q, scenarios).mean())
assert best_q == 12 and np.isclose(loss(12, scenarios).mean(), 4.5)
assert np.all(stock >= 0) and np.issubdtype(stock.dtype, np.integer)
```

代码会打印验证选择、最终预测误差和两种备货策略的经营结果。情景策略优化的是估计分布下的期望损失，不保证在每个有限测试段都更好；若实际表现变差，应检查校准样本量与分布假设，而不是偷偷改测试数据。

在给出的固定种子和复现环境下，示例选择 Ridge 的 $\alpha=10$，最终 MAE 约2.46份。点预测备货的平均损失约10.21，情景优化约9.93；缺货减少的同时，平均剩余从约1.50份增至2.69份。这说明优化体现了成本权衡，而不是让所有指标同时改善。浮点与库版本可能使末位略有变化。

真实报表应保存逐日预测、Q、D、利润与缺货，不只汇报平均数；只有在模型决策时已经知道的量才可进入输入。

## 方法选择与常见误区

### 可交付的模型报告

报告应包含：业务目标与单位、数据时间范围、可用信息、划分示意、基线与选型规则、误差与经营指标、参数敏感性、适用条件和失效后的复核机制。

下面几种变化需要重新建模：库存可跨日、缺货会产生延期订单、促销并未预先确定、销量被历史缺货截断、不同午餐可互相替代、供货有提前期。不要把“已售数量”无条件当作完整需求。

训练精度、预测误差、需求分布校准与经营收益是四个层面。预测稍差的模型可能在某种成本下决策更好，必须以任务目标进行评价。

## 练习

1. 为什么这里需要单独的校准段？
2. 若剩余可按采购成本回收，$s=c$，剩余单位代价是什么，模型会有什么变化？
3. 为什么不能用当天实际销量决定当天备货？
4. 若需求10、12、14等概率，缺货6、剩余3，Q=12与14的期望损失各是多少？
5. 如果只报告“最终MAE=某个数”，还缺少哪些决策信息？

## 参考答案

1. 用来估计冻结模型未参与拟合的预测残差，避免把训练残差过小误当作未来风险。
2. 当 s=c 时剩余代价0，当前无容量上限模型会倾向覆盖高需求；需确认回收条件是否真实，必要时加入容量与其他成本。
3. 当天结束才知道的销量在决策时不可用，使用它属于未来信息泄漏。
4. Q=12 为 $(6+0+12)/3=6$；Q=14 为 $(12+6+0)/3=6$，两者在此离散分布下并列最优，说明最优可能不唯一。
5. 缺货与剩余代价、备货规则、利润、缺货量、剩余量、时间划分和成本敏感性等。

## 复习卡片与下一步

本篇完成了“数据—预测—不确定性—优化—经营评价”的闭环。复习时不看代码，先写变量、目标、约束和信息时点，再重现手算。最后读赛题阅读篇，把这条流程迁移到更长的真实任务。

[上一篇 图像文本与信号建模](/notes/mathematical-modeling-multimodal/) · [下一篇 研究生数学建模赛题的阅读与方法选择](/notes/mathematical-modeling-machine-learning-data-processing/)

## 参考资料

- [Datawhale《数学建模导论》第3章](https://datawhalechina.github.io/intro-mathmodel/#/CH3/第三章-函数极值与规划模型)：用于建立知识目录，建议与本篇的手算和案例对照阅读。
- [时间序列预测的评价](https://otexts.com/fpp3/accuracy.html)。
- [scikit-learn 数据泄漏边界](https://scikit-learn.org/stable/common_pitfalls.html)。
