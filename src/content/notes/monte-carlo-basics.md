---
title: '蒙特卡洛模拟入门：从随机采样到期权定价'
description: '蒙特卡洛方法的核心思想、收敛速度分析，并用 Python 实现欧式看涨期权定价。'
pubDate: 2026-08-05
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

由大数定律，$\hat{I}_N \to I$（几乎处处收敛）；由中心极限定理，误差量级为

$$
\left| \hat{I}_N - I \right| = O\left( \frac{\sigma}{\sqrt{N}} \right),
$$

其中 $\sigma^2 = \mathrm{Var}[f(X)]$。**收敛速度 $O(N^{-1/2})$ 与维度无关**，这是蒙特卡洛在高维问题中胜过网格法的根本原因。

## 应用：欧式期权定价

风险中性测度下，标的价格服从几何布朗运动：

$$
dS_t = r S_t \, dt + \sigma S_t \, dW_t,
$$

其精确解为

$$
S_T = S_0 \exp\left( \left( r - \frac{\sigma^2}{2} \right) T + \sigma W_T \right), \quad W_T \sim \mathcal{N}(0, T).
$$

欧式看涨期权的价格即贴现期望 payoff：

$$
C = e^{-rT} \, \mathbb{E}\left[ \max(S_T - K, 0) \right].
$$

## Python 实现

```python
import numpy as np

def mc_european_call(S0, K, T, r, sigma, n_paths=100_000, seed=42):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)
    # 一步精确采样 S_T，无需逐步模拟路径
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoff = np.maximum(ST - K, 0)
    price = np.exp(-r * T) * payoff.mean()
    # 95% 置信区间
    ci = 1.96 * np.exp(-r * T) * payoff.std(ddof=1) / np.sqrt(n_paths)
    return price, ci

price, ci = mc_european_call(S0=100, K=105, T=1, r=0.03, sigma=0.2)
print(f"价格: {price:.4f} ± {ci:.4f}")
```

## 与解析解对比

Black-Scholes 公式给出解析解：

$$
C = S_0 \Phi(d_1) - K e^{-rT} \Phi(d_2), \quad
d_{1,2} = \frac{\ln(S_0/K) + (r \pm \sigma^2/2)T}{\sigma \sqrt{T}}.
$$

可以用它验证模拟结果的误差是否落在置信区间内。

## 后续方向

- **方差缩减**：对偶变量法、控制变量法，能显著降低 $\sigma$ 从而降低误差
- **路径依赖期权**（如亚式期权）：需要逐步模拟整条路径
- **拟蒙特卡洛**（QMC）：用低差异序列替代伪随机数，收敛速度可接近 $O(N^{-1})$
