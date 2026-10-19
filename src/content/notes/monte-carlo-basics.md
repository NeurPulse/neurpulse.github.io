---
title: '蒙特卡洛模拟入门：从随机采样到期权定价'
description: '蒙特卡洛方法的核心思想、收敛速度分析，并用 Python 实现欧式看涨期权定价。'
pubDate: 2026-08-05
updatedDate: 2026-10-19
order: 1
featured: true
category: '蒙特卡洛模拟'
tags: ['蒙特卡洛', '期权定价', 'Python']
---

## 核心思想

蒙特卡洛方法用**随机采样**来近似数值结果。要计算期望

$$
I = \mathbb{E}[f(X)] = \int f(x) \, p(x) \, dx,
$$

只需从分布 $p(x)$ 中独立采样 $X_1, X_2, \dots, X_N$，用样本均值估计：

$$
\hat{I}_N = \frac{1}{N} \sum_{i=1}^{N} f(X_i).
$$

若 $\mathbb{E}[|f(X)|]<\infty$，大数定律给出 $\hat{I}_N\to I$（几乎处处收敛）。进一步，若 $v^2=\mathrm{Var}[f(X)]<\infty$，独立采样使估计量无偏，其均方根误差为：

$$
\mathrm{RMSE}(\hat{I}_N)
=\sqrt{\mathbb{E}[(\hat{I}_N-I)^2]}
=\frac{v}{\sqrt{N}}.
$$

当 $v>0$ 时，中心极限定理进一步说明：

$$
\frac{\sqrt{N}(\hat{I}_N-I)}{v}
\xrightarrow{d}\mathcal{N}(0,1).
$$

这描述的是随机误差的分布和统计尺度，**不是每次模拟都满足的确定性误差上界**。收敛指数 $N^{-1/2}$ 不显式依赖维度，但方差 $v^2$ 和单次采样成本可能随维度变化；蒙特卡洛并不保证在所有高维问题上都优于其他数值方法。

用样本标准差 $s$ 估计 $v$，在样本量足够大、正态近似合理时，可构造近似 95% 置信区间：

$$
\hat{I}_N\pm1.96\frac{s}{\sqrt{N}}.
$$

重复独立实验时，这种构造方式的覆盖率约为 95%，单次区间仍可能未覆盖真值。误差目标减半，通常需要将样本量增至约 4 倍。

## 应用：欧式期权定价

这里采用无分红、常数利率 $r$、常数波动率 $\sigma$ 的 Black–Scholes 模型，$T$ 以年为单位。风险中性测度 $\mathbb{Q}$ 下，标的价格服从几何布朗运动：

$$
dS_t = r S_t \, dt + \sigma S_t \, dW_t^{\mathbb{Q}},
$$

其精确解为

$$
S_T = S_0 \exp\left( \left( r - \frac{\sigma^2}{2} \right) T + \sigma\sqrt{T}Z \right),
\quad Z\sim\mathcal{N}(0,1).
$$

欧式看涨期权的价格即贴现期望 payoff：

$$
C = e^{-rT} \, \mathbb{E}^{\mathbb{Q}}\left[ \max(S_T - K, 0) \right].
$$

## Python 实现

下面返回价格估计和**近似 95% 置信区间的半宽**。这里只需要终值，使用精确终值分布即可，不引入时间离散误差；仍然存在有限样本的蒙特卡洛误差。

```python
import numpy as np


def validate_call_inputs(S0, K, T, r, sigma):
    if not np.isfinite([S0, K, T, r, sigma]).all():
        raise ValueError("参数必须为有限实数")
    if S0 <= 0 or K <= 0 or T < 0 or sigma < 0:
        raise ValueError("要求 S0、K > 0，T、sigma >= 0")


def mc_european_call(S0, K, T, r, sigma, n_paths=100_000, seed=42):
    validate_call_inputs(S0, K, T, r, sigma)
    if (isinstance(n_paths, (bool, np.bool_))
            or not isinstance(n_paths, (int, np.integer)) or n_paths < 2):
        raise ValueError("n_paths 必须是至少为 2 的整数")
    if T == 0:
        return float(max(S0 - K, 0)), 0.0
    if sigma == 0:
        return float(max(S0 - K * np.exp(-r * T), 0)), 0.0
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    discounted_payoff = np.exp(-r * T) * np.maximum(ST - K, 0)
    price = discounted_payoff.mean()
    standard_error = discounted_payoff.std(ddof=1) / np.sqrt(n_paths)
    ci_half_width = 1.96 * standard_error
    return float(price), float(ci_half_width)


price, half_width = mc_european_call(S0=100, K=105, T=1, r=0.03, sigma=0.2)
print(f"价格估计: {price:.4f}")
print(f"近似 95% 置信区间: [{price - half_width:.4f}, {price + half_width:.4f}]")
```

`ddof=1` 使用样本方差，因此要求至少两条样本。$T=0$ 时价格为到期收益，$\sigma=0$ 时终值是确定的，这两种情况不需要随机采样，区间半宽为 0。代码用于常规参数范围的标量教学示例，不处理极端参数造成的浮点溢出。

## 与解析解对比

同一模型下，当 $T>0$、$\sigma>0$ 时，Black–Scholes 公式给出解析解：

$$
C = S_0 \Phi(d_1) - K e^{-rT} \Phi(d_2),
$$

$$
d_1 = \frac{\ln(S_0/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}},
\qquad d_2=d_1-\sigma\sqrt{T}.
$$

$\Phi$ 是标准正态分布的累积分布函数。接着执行下面的代码，比较两种计算方式：

```python
from math import exp, log, sqrt
from statistics import NormalDist


def black_scholes_call(S0, K, T, r, sigma):
    validate_call_inputs(S0, K, T, r, sigma)
    if T == 0:
        return float(max(S0 - K, 0))
    if sigma == 0:
        return float(max(S0 - K * exp(-r * T), 0))
    d1 = (log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    normal = NormalDist()
    return S0 * normal.cdf(d1) - K * exp(-r * T) * normal.cdf(d2)


exact = black_scholes_call(100, 105, 1, 0.03, 0.2)
print(f"解析价格: {exact:.4f}；模拟误差: {price - exact:.4f}")
```

解析解可以作为数值对照，但不能把“每次都落入 95% 区间”作为程序正确的必要条件。还应检查不同随机种子、样本量和参数，并确认贴现与终值分布使用了相同的模型假设。

## 后续方向

- **方差缩减**：对偶变量法、控制变量法可降低估计量的方差，不是改变标的的波动率 $\sigma$。
- **路径依赖期权**（如亚式期权）：需要采样相关监测时刻的价格；连续路径泛函的离散近似还可能引入时间离散误差。
- **拟蒙特卡洛**（QMC）：使用低差异序列；改善程度依赖被积函数的光滑性、有效维度等条件，不能普遍保证 $O(N^{-1})$ 的速度，也不能直接套用独立样本的置信区间公式。

## 参考与推荐阅读

- [Mike Giles：Monte Carlo methods](https://people.maths.ox.ac.uk/gilesm/mc2013/ercoftac1-2x2.pdf)：均方根误差、中心极限定理与采样误差。
- [MIT：Risk Neutral Probability and Black-Scholes](https://math.mit.edu/~sheffield/440/Lecture36.pdf)：风险中性定价与欧式看涨期权公式。
