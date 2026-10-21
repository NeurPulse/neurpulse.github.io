---
title: "数学建模：微分方程与动力系统"
description: "从变化率建立模型，用解析解和数值解预测状态"
pubDate: 2026-08-07
updatedDate: 2026-10-21
order: 8
category: '数学建模'
tags: ["微分方程","数值方法","SciPy","SymPy","零基础"]
---

> [专题学习地图](/notes/mathematical-modeling-roadmap/) · 第 8 / 27 篇

## 学习目标与先修知识

目标是从“变化速度”建立微分方程，区分解析解、数值解和参数估计，并完成冷却过程的建模、求解与验证。先修为导数、指数、对数及基础数组。最后理解一阶、二阶模型与数值步长，不要求先会偏微分方程。

可回看：[必备数学与 Python 基础](/notes/mathematical-modeling-foundations/) · [几何模型与线性代数](/notes/mathematical-modeling-geometry/)。

## 问题背景与使用时机

一杯热饮逐渐接近室温，温度每分钟下降多少，往往与它比环境热多少有关。我们未必直接知道未来温度函数，但可以先写出变化率规律。这正是微分方程的用途。

微分方程通常描述连续时间；若只在每天或每代更新，可以使用差分方程。不能把所有时间序列都认为一定由一个已知物理方程控制。

## 核心概念与符号

### 从速率到状态

状态 $T(t)$ 表示温度，$dT/dt$ 表示温度变化率；环境温度为 $T_a$，冷却参数为 $k>0$：

$$
\frac{dT}{dt}=-k(T-T_a),\qquad T(0)=T_0.
$$

负号保证热饮高于环境时降温；k 的单位是时间的倒数。初始条件是确定具体轨迹所必需的，只有方程而没有初值，一般对应一族解。

令 $u=T-T_a$，得到 $u'=-ku$。积分或直接求导检验可得：

$$
T(t)=T_a+(T_0-T_a)e^{-kt}.
$$

### 一阶线性与二阶模型

一般一阶线性方程 $y'+P(t)y=Q(t)$ 用积分因子 $M(t)=e^{\int P(t)dt}$，因为 $(My)'=MQ$，所以 $y=M^{-1}(\int MQ\,dt+C)$。分离变量可处理一些特殊形式，不能直接套到任意非齐次方程。

二阶线性齐次方程 $y''+py'+qy=0$ 的特征根决定指数、重复根乘 t 或衰减振荡等形式。受迫方程的通解等于齐次通解加特解；若外力形式与特征根共振，待定特解需乘相应次数的 t。弹簧系统的惯性、阻尼、回复力分别对应 $my''$、$cy'$、$ky$。

### 数值近似

Euler 法为 $T_{j+1}=T_j+h f(t_j,T_j)$，用当前斜率走一小步。Runge–Kutta 在一步中组合多个斜率，提高近似精度。误差容限控制求解精度，不能修复错误的物理假设。

## 手算一个最小例子

设室温 $20^\circ C$、初温 $80^\circ C$、$k=0.1/\text{分钟}$。初始变化率为 $-0.1(80-20)=-6^\circ C/\text{分钟}$。

Euler 步长为 1 分钟时：

$$
T_1=80-6=74,\qquad
T_2=74-0.1(74-20)=68.6.
$$

解析解在 1 分钟为 $20+60e^{-0.1}\approx74.2902$，2 分钟约为 $69.1238$。Euler 用直线小步代替曲线，所以两者略有差异。

想知道何时低于 $30^\circ C$，解：

$$
30=20+60e^{-0.1t}
\Rightarrow t=\frac{\ln6}{0.1}\approx17.9176\text{分钟}.
$$

若不知道 k，但测得 5 分钟时为 $20+60e^{-0.5}$，则由 $k=-\ln[(T(5)-20)/60]/5$ 可恢复 k=0.1。这是“参数估计”，与已知 k 后“求解方程”不同。

## 建模步骤与完整案例

完整案例分为建立、校准、预测、检验四段：

1. 假设环境温度恒定，饮品内部近似均匀，不考虑蒸发等额外机制。
2. 根据温差与降温速度的关系提出方程，检查负号与量纲。
3. 用初温及测量温度估计参数；多次含噪测量时可用非线性最小二乘。
4. 计算未来温度及达到阈值的时间。
5. 用解析解核对数值程序，再用独立测量检验模型假设。
6. 改变 k 进行敏感性分析。若 k 从 0.1 增至 0.12，达到同一温差比例的时间缩短为原来的 $0.1/0.12$。

示范数据由模型生成，仅用于检查解法，真实应用还需要实测验证。

## Python 实现与结果解释

配套文件：[查看或下载本篇完整脚本](/downloads/math-modeling/mathematical-modeling-differential-equations.py)。

```python
import numpy as np
from scipy.integrate import solve_ivp
import sympy as sp

ambient, initial, rate = 20., 80., 0.1
def cooling(t, temperature):
    return -rate*(temperature-ambient)

time = np.arange(0., 21.)
solution = solve_ivp(
    cooling, (0., 20.), [initial],
    t_eval=time, rtol=1e-9, atol=1e-10
)
exact = ambient + (initial-ambient)*np.exp(-rate*time)
assert solution.success
assert np.max(np.abs(solution.y[0]-exact)) < 1e-6
print("20分钟数值与解析温度:", solution.y[0, -1], exact[-1])
print("到30度的时间:", np.log(6)/rate)

# 解析求解：符号函数未知，给定初始条件。
t = sp.symbols("t", real=True)
T = sp.Function("T")
symbolic = sp.dsolve(
    sp.Eq(sp.diff(T(t), t), -(T(t)-20)/10),
    ics={T(0): 80}
)
print("符号解:", symbolic)
assert sp.simplify(symbolic.rhs-(20+60*sp.exp(-t/10))) == 0

# 用测量值恢复参数；此处测量值是无噪声的教学值。
measured_at_5 = ambient + 60*np.exp(-0.5)
estimated = -np.log((measured_at_5-ambient)/60)/5
assert np.isclose(estimated, rate)
```

`solve_ivp` 的函数接收时间和状态数组，`solution.y` 的形状为“状态数 × 输出时刻数”。此例只有一个状态，因此取第 0 行。`t_eval` 指定输出时间，不等于要求求解器内部固定步长。

SymPy 返回解析表达式；SciPy 返回数值近似。两条路径相互核对，有助于发现初值、符号与索引错误。

## 方法选择与常见误区

- 对 $u'=-ku$，Euler 的误差传播因子为 $1-kh$，稳定性要求 $|1-kh|<1$；过大步长可能产生不真实振荡或发散。物理单调性要求还更严格。
- 许多动力系统有不同快慢尺度，称为刚性问题，可能需要 Radau、BDF 等方法；不能无限缩小显式步长而不看计算成本。
- 人口有限增长可用 $N'=rN(1-N/K)$，其中 K 是承载量假设；指数增长 $N'=rN$ 不包含资源限制。
- 若状态同时随空间与时间变化，如杆内温度 $u(x,t)$，会进入偏微分方程；边界条件与初值同样重要。常见热方程为 $u_t=a u_{xx}$。
- 最小残差、成功停止和物理正确是三个不同判断。应检查非负性、守恒或长期极限是否符合领域规律。

## 练习

1. 上例初温从 80 改为 60，初始降温速率是多少？
2. 在模型条件下，温度会在有限时间恰好降到 $20^\circ C$ 吗？
3. 为什么 Euler 步长越大可能越不可靠？
4. 把时间单位从分钟改为秒，k 的数值应是多少？
5. 在 $y''+2y'+y=0$ 中，特征根和通解是什么？

## 参考答案

1. $-0.1(60-20)=-4^\circ C/\text{分钟}$。
2. 不会。只要初温高于室温，指数温差在有限时间仍为正，长期极限才为 20。
3. 一步中的斜率变化被忽略，截断误差增加；某些步长还会违反数值稳定性。
4. $0.1/60=1/600$ 每秒。时间变大 60 倍，速率参数数值相应除以 60。
5. $(r+1)^2=0$，二重根为 -1，通解 $(C_1+C_2t)e^{-t}$，不能只写一个指数项。

## 复习卡片与下一步

先写变化率，再写初值，检查单位和极限，最后选择求解器。冷却案例已经连接“机制假设—估参—预测—验证”。下一篇学习在多个可行行动中选择最优方案。

[上一篇 几何模型与线性代数](/notes/mathematical-modeling-geometry/) · [下一篇 线性规划与整数优化](/notes/mathematical-modeling-optimization/)

## 参考资料

- [Datawhale《数学建模导论》第2章](https://datawhalechina.github.io/intro-mathmodel/#/CH2/第2章-微分方程与动力系统)：用于建立知识目录，建议与本篇的手算和案例对照阅读。
- [SciPy solve_ivp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)。
- [SymPy 微分方程](https://docs.sympy.org/latest/guides/solving/solve-ode.html)。
