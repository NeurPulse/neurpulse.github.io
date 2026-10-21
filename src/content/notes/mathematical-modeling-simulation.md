---
title: "数学建模：随机模拟与不确定性分析"
description: "用蒙特卡洛、马尔可夫和排队案例理解风险与误差"
pubDate: 2026-10-21
updatedDate: 2026-10-21
order: 24
category: '数学建模'
tags: ["蒙特卡洛","马尔可夫","排队论","零基础"]
---

> [专题学习地图](/notes/mathematical-modeling-roadmap/) · 第 24 / 27 篇

## 学习目标与先修知识

目标是理解随机采样为什么能估计期望、误差怎样随样本量变化，手算马尔可夫状态更新，并用一个排队模拟对照解析结果。先修为均值、方差、条件概率和循环。随机模拟是可检验的计算模型，不是随便生成随机数。

可回看：[必备数学与 Python 基础](/notes/mathematical-modeling-foundations/) · [统计模型与假设检验](/notes/mathematical-modeling-statistics/)。

## 问题背景与使用时机

需求、到达时间和设备状态都不完全确定。只用一个平均数可能隐藏缺货、拥堵或高损失尾部。模拟通过反复产生符合假设的情景，观察系统输出及风险。

要明确随机输入分布、依赖关系、时间单位与初始化。模拟再长，也只能反映给定模型，不能自动证明分布假设符合现实。

## 核心概念与符号

### Monte Carlo 估计

独立同分布样本 $X_1,\ldots,X_N$ 下，可用 $\bar g=\frac1N\sum g(X_i)$ 估计 $\mathbb E[g(X)]$，标准误约为 $s_g/\sqrt N$。适当大样本条件下可构造近似区间 $\bar g\pm1.96\,s_g/\sqrt N$。

误差常按 $1/\sqrt N$ 缩小，把标准误减半通常需要4倍样本；这不包含分布设错或系统偏差。相关样本不能直接按独立样本数量计算同样的标准误。

### 马尔可夫状态

行向量分布 $\boldsymbol\pi_t$ 与行随机转移矩阵 P 满足 $\boldsymbol\pi_{t+1}=\boldsymbol\pi_tP$，每行非负且和为1。马尔可夫假设是“给定当前状态后，下一状态不再依赖更久历史”，不是无条件独立。

### 排队

M/M/1 表示 Poisson 到达、指数服务、单服务台等设定。到达率 $\lambda$、服务率 $\mu$，稳定要求 $\rho=\lambda/\mu<1$。稳态平均系统停留时间 $W=1/(\mu-\lambda)$，平均等待 $W_q=W-1/\mu$。

Little 定律 $L=\lambda W$ 连接平均系统人数与平均停留时间，但应用时观察窗口、系统边界和稳定性需要一致。

## 手算一个最小例子

状态为“正常、故障”，转移矩阵：

$$
P=\begin{bmatrix}0.8&0.2\\0.3&0.7\end{bmatrix}.
$$

初始一定正常，$\pi_0=(1,0)$，则 $\pi_1=(0.8,0.2)$。再一步，正常概率 $0.8\times0.8+0.2\times0.3=0.7$，故 $\pi_2=(0.7,0.3)$。稳态方程给出 $0.2\pi_{\text{正常}}=0.3\pi_{\text{故障}}$，结合和为1，得 (0.6,0.4)。

排队设每小时到达2人、服务3人，$\rho=2/3$，$W=1$小时，$W_q=2/3$小时，$L=2$人。平均服务仅20分钟，并不意味着平均系统停留也是20分钟，等待会放大总时长。

## 建模步骤与完整案例

完整案例包含两个可独立核验的模拟：

- 用均匀随机数估计 $\int_0^1x^2dx=1/3$，计算抽样标准误。
- 用相邻到达间隔和服务时长，逐客更新单服务台完成时刻，对照 M/M/1 的理论平均停留时间。

排队递推为“开始服务时刻 = 到达时刻与上一位完成时刻的较大者”；完成时刻再加本次服务时长。初始空队会有过渡期，丢弃一段预热后再估计稳态量，并用不同种子检查。

## Python 实现与结果解释

配套文件：[查看或下载本篇完整脚本](/downloads/math-modeling/mathematical-modeling-simulation.py)。

```python
import numpy as np

rng = np.random.default_rng(42)
samples = rng.uniform(size=20000)**2
estimate = samples.mean()
se = samples.std(ddof=1)/np.sqrt(len(samples))
print("积分估计、标准误、近似95%区间:", estimate, se,
      (estimate-1.96*se, estimate+1.96*se))
print("解析值:", 1/3)
assert np.isfinite(estimate) and se > 0

P = np.array([[0.8, 0.2], [0.3, 0.7]])
distribution = np.array([1., 0.])
assert np.allclose(distribution@P@P, [0.7, 0.3])
A = np.vstack([P.T-np.eye(2), np.ones(2)])
stationary = np.linalg.lstsq(A, [0., 0., 1.], rcond=None)[0]
assert np.allclose(stationary, [0.6, 0.4])
print("状态稳态分布:", stationary)

arrival_rate, service_rate = 2., 3.
arrival = np.cumsum(rng.exponential(1/arrival_rate, size=30000))
service = rng.exponential(1/service_rate, size=len(arrival))
finish = np.empty(len(arrival))
previous_finish = 0.
for i in range(len(arrival)):
    start = max(arrival[i], previous_finish)
    finish[i] = start+service[i]
    previous_finish = finish[i]
system_time = finish-arrival
waiting = system_time-service
print("预热后模拟平均停留:", system_time[1000:].mean())
print("理论平均停留:", 1/(service_rate-arrival_rate))
assert np.all(waiting >= -1e-10)
```

积分区间有近似覆盖含义，不要求每次都覆盖真值。排队客户的等待时间彼此相关，不应直接把30000位客户视为30000个独立等待样本来算同一个 $s/\sqrt N$；可用独立重复模拟或合适的批均值方法评估输出不确定性。

## 方法选择与常见误区

当随机输入与状态更新清楚、解析求解困难时适合模拟。离散事件模拟适合到达与服务过程，元胞自动机适合空间网格局部规则，Monte Carlo 可用于期望、概率和风险评估。

元胞自动机还要说明邻域、同步或异步更新、边界与随机规则。森林传播等图形看起来逼真，不等于真实传播速度已经校准。

报告均值之外，可给高分位、失败概率与极端情景；罕见事件需要足够样本或专门抽样。固定种子保证可复现，多个种子检验抽样稳定性，真实数据检验分布假设，三者不能互相替代。

## 练习

1. 想让 Monte Carlo 标准误变为原来的1/3，样本量通常需要几倍？
2. 马尔可夫矩阵的一行和为何必须为1？
3. $\lambda=3,\mu=2$ 的 M/M/1 能直接使用稳态 $W=1/(\mu-\lambda)$ 吗？
4. 看到一个近似95%区间没有覆盖真值，是否一定算错？
5. 为什么模拟开始空队时要考虑预热？

## 参考答案

1. 约9倍，因为标准误按 $1/\sqrt N$ 缩小。
2. 当前状态给定后，下一状态必须落在某个允许状态中，总条件概率为1。
3. 不能，负数的“等待时间”提示稳定条件不满足，队列不会有该稳态。
4. 不一定，重复抽样中允许一部分区间不覆盖；还要检查近似条件及程序。
5. 空队是特定初态，可能使初期等待偏短，不能直接当作稳态典型观测。

## 复习卡片与下一步

模拟先定义机制，再做可解析小例子验证，随后增加规模和复杂性。接下来处理图像、文本和信号，再在综合案例中把预测与随机需求下的决策连接起来。

[上一篇 进化计算与群体智能](/notes/mathematical-modeling-evolutionary/) · [下一篇 图像文本与信号建模](/notes/mathematical-modeling-multimodal/)

## 参考资料

- [Datawhale《数学建模导论》第3章](https://datawhalechina.github.io/intro-mathmodel/#/CH3/第三章-函数极值与规划模型)：用于建立知识目录，建议与本篇的手算和案例对照阅读。
- [NumPy 随机数生成](https://numpy.org/doc/stable/reference/random/generator.html)。
- [Datawhale 动力系统与元胞自动机章节](https://datawhalechina.github.io/intro-mathmodel/#/CH2/第2章-微分方程与动力系统)。
