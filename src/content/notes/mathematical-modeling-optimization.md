---
title: "数学建模：线性规划与整数优化"
description: "把资源分配写成目标与约束，再检查解的可行性"
pubDate: 2026-10-21
updatedDate: 2026-10-21
order: 9
category: '数学建模'
tags: ["线性规划","整数规划","优化","零基础"]
---

> [专题学习地图](/notes/mathematical-modeling-roadmap/) · 第 9 / 27 篇

## 学习目标与先修知识

需要从符号、导数和课本例题开始逐步学习时，可先读[最优化方法第一章绪论零基础学习笔记](/notes/mathematical-modeling-optimization-introduction/)。该篇完整推导生产、水槽、运输等模型，并详解六道课后习题；本篇继续关注通用建模与 Python 求解。

学完应能把资源分配写成目标与约束，区分连续、整数与非线性模型，并检查求解是否可行、有界和成功。先修为一次函数、矩阵与不等式。小例子先手算和枚举，再使用 SciPy，避免软件输出无法解释。

可回看：[学习地图与完整流程](/notes/mathematical-modeling-roadmap/) · [必备数学与 Python 基础](/notes/mathematical-modeling-foundations/)。

## 问题背景与使用时机

给定原料、工时和利润，怎样安排生产？目标是选择行动，而不是仅预测一个结果。建模时尤其要区分“最多”“至少”“恰好”和“必须整件”。

本篇沿用学习地图中的面包计划：A、B 每批利润 3、2，耗料 2、1，最多 5 批，原料容量为 M。参数给定，生产批数是决策变量。

## 核心概念与符号

线性规划要求目标和约束对决策变量都是线性的。一般最小化写法为：

$$
\min_{\boldsymbol x}\boldsymbol c^T\boldsymbol x,\qquad
A_{\mathrm{ub}}\boldsymbol x\le\boldsymbol b_{\mathrm{ub}},\quad
A_{\mathrm{eq}}\boldsymbol x=\boldsymbol b_{\mathrm{eq}},\quad
\boldsymbol l\le\boldsymbol x\le\boldsymbol u.
$$

最大化可以把目标取负。$A$ 的每行是一条约束，每列对应一个变量；向量的顺序必须一致。若批数必须为整数，需额外指定整性。直接把连续解四舍五入可能违反约束。

**可行**表示满足全部约束；**不可行**表示约束互相冲突；**无界**表示目标可无限改善，常与遗漏资源或变量界有关。有解也可能有多个最优方案。

线性目标在有界非空多面体上的最优值可由某个顶点达到，但整性要求不能仅靠顶点性质解决。非线性目标或约束可能有多个局部最优，需区分凸问题与一般非凸问题。

## 手算一个最小例子

连续模型的两个资源边界相交时：

$$
2x+y=M,\qquad x+y=5
\Rightarrow x=M-5,\quad y=10-M.
$$

当 $M=7.5$ 时，交点为 $(2.5,2.5)$，利润 12.5，是连续松弛的最优方案。若只生产整批，就不能采用半批。

枚举整数点发现 $(2,3)$ 耗料 7、利润 12；$(3,1)$ 耗料 7、利润 11，$(0,5)$ 利润 10。最大整数利润为 12。连续松弛值 12.5 是最大化整数模型的上界，不是可执行方案。

若把连续解直接分别四舍五入成 $(3,3)$，批数为 6、耗料为 9，同时违反两条约束。

## 建模步骤与完整案例

完整案例采用 M=7.5：

1. 假设固定单耗和利润，需求足够，忽略启动成本。
2. 用两变量表示批数，写原料与批次数限制。
3. 先解连续松弛，了解可达到的理论上限。
4. 加整性求可执行计划，并用小规模枚举独立核对。
5. 报告目标值、资源松弛量和整性要求。
6. 改 M 或单位利润，观察最优方案怎样变化。

同一资源增加一点是否有价值，要看当前是否紧缺以及是否跨越整数门槛。连续模型的影子价格有局部适用范围，不能机械外推到大幅变更或整数模型。

## Python 实现与结果解释

配套文件：[查看或下载本篇完整脚本](/downloads/math-modeling/mathematical-modeling-optimization.py)。

```python
import numpy as np
from scipy.optimize import linprog, linear_sum_assignment, minimize

M = 7.5
cost = np.array([-3., -2.])  # 最大利润转为最小负利润
A = np.array([[2., 1.], [1., 1.]])
b = np.array([M, 5.])
relaxed = linprog(cost, A_ub=A, b_ub=b, bounds=[(0, None)]*2, method="highs")
integer = linprog(cost, A_ub=A, b_ub=b, bounds=[(0, None)]*2,
                  integrality=[1, 1], method="highs")
assert relaxed.success and integer.success
print("连续解:", relaxed.x, "利润:", -relaxed.fun)
print("整数解:", integer.x, "利润:", -integer.fun)
print("整数资源剩余:", b-A@integer.x)
oracle = max(3*x+2*y for x in range(6) for y in range(6)
             if 2*x+y <= M and x+y <= 5)
assert np.isclose(-integer.fun, oracle)
assert np.all(A@integer.x <= b+1e-8)
assert np.allclose(integer.x, np.round(integer.x))

# 指派：每个工人、每个任务各使用一次，使总成本最小。
assignment_cost = np.array([[4., 1.], [2., 3.]])
workers, tasks = linear_sum_assignment(assignment_cost)
print("指派总成本:", assignment_cost[workers, tasks].sum())
assert assignment_cost[workers, tasks].sum() == 3

# 有边界的一维凸非线性示例。
fit = minimize(lambda z: (z[0]-3)**2, x0=[0.], bounds=[(0., 2.)])
assert fit.success
assert np.isclose(fit.x[0], 2.)
```

资源剩余为零代表该约束在当前解处紧约束；为正表示还有余量。求解器可能给出同利润的不同方案，因此断言应核对目标与可行性，而不总是强求逐项相同。

## 方法选择与常见误区

| 结构 | 可考虑的方法 | 判断重点 |
|---|---|---|
| 线性连续 | 线性规划 | 可行、有界、单位与约束 |
| 件数、开关、指派 | 整数或混合整数规划 | 整性、求解间隙与时间 |
| 平滑凸非线性 | 凸优化或相应数值方法 | 全局性质来自凸结构 |
| 一般非凸 | 多起点、全局方法或启发式 | 局部最优、预算与重复性 |
| 多阶段决策 | 动态规划 | 状态是否包含未来所需信息 |
| 多个冲突目标 | 约束法、加权法或 Pareto 分析 | 偏好、尺度与权衡 |

动态规划通过状态与转移复用子问题，例如背包状态 $F(i,c)$ 表示前 i 件、容量 c 下的最好价值。贪心只取当前最好动作，并非任何问题都最优。例如面额 1、3、4 凑 6，贪心取 $4+1+1$，而 $3+3$ 更少。

多目标加权前要处理单位与尺度；权重表示偏好，不能从某次排名倒推一个唯一“客观最优”偏好。非凸 Pareto 前沿也不一定能由简单加权穷尽。

## 练习

1. 容量限制“最多 10”与“至少 10”分别怎样写？
2. 为什么当前整数模型不能简单采用 $(2.5,2.5)$？
3. 连续最大利润 12.5、整数可行方案利润 12，能否说整数方案已经最优？还需要什么信息？
4. 目标“成本最低且等待最短”为什么不能直接相加原始数值？

## 参考答案

1. 最多写 $\le10$，至少写 $\ge10$；恰好写 $=10$。
2. 批数必须为整数，半批违反变量定义，即使其余资源约束满足也不可执行。
3. 单靠 12.5 上界还不总够；若已知目标在整数方案中只能取整数，则上界下取整为 12，可证明达到最优。本例也可直接完整枚举核对。
4. 金额与时间单位不同，直接相加隐含任意兑换率。需明确偏好、归一化或对其中一个目标设约束。

## 复习卡片与下一步

优化的检查顺序：变量能否控制 → 目标是否匹配 → 约束是否齐全 → 整性是否需要 → 解是否可行 → 最优性证据是什么。下一篇把节点、道路和容量关系组织成网络。

[上一篇 微分方程与动力系统](/notes/mathematical-modeling-differential-equations/) · [下一篇 图论与网络优化](/notes/mathematical-modeling-graphs/)

## 参考资料

- [Datawhale《数学建模导论》第3章](https://datawhalechina.github.io/intro-mathmodel/#/CH3/第三章-函数极值与规划模型)：用于建立知识目录，建议与本篇的手算和案例对照阅读。
- [SciPy linprog](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html)。
- [SciPy 指派问题](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html)。
