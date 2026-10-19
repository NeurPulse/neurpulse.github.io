---
title: '分数布朗运动：定义、性质与模拟方法'
description: 'fBm 的协方差结构、Hurst 指数的含义，以及基于 Cholesky 分解与 FFT 的模拟方法。'
pubDate: 2026-07-20
updatedDate: 2026-10-19
order: 1
category: '分数布朗运动'
tags: ['分数布朗运动', 'Hurst 指数', '随机过程']
---

## 定义

分数布朗运动（fractional Brownian motion, fBm）$\{B_t^H\}_{t \geq 0}$ 是零均值连续高斯过程，$B_0^H = 0$，协方差函数为

$$
\mathbb{E}\left[ B_t^H B_s^H \right] = \frac{1}{2} \left( t^{2H} + s^{2H} - |t - s|^{2H} \right),
$$

其中 $H \in (0, 1)$ 称为 **Hurst 指数**。当 $H = 1/2$ 时退化为标准布朗运动。

## Hurst 指数的含义

由协方差结构可推出增量 $B_{t+h}^H - B_t^H$ 的方差：

$$
\mathbb{E}\left[ \left( B_{t+h}^H - B_t^H \right)^2 \right] = h^{2H},
$$

增量分布只依赖间隔，因而是平稳增量；高斯过程结合协方差的缩放性质还能得到 $B_{ct}^H\overset{d}{=}c^H B_t^H$，这是过程的自相似性。仅凭增量方差的形式不能对任意过程推出这两条性质。相隔 $k$ 步的两个增量的相关系数为

$$
\rho(k) = \frac{1}{2} \left( (k+1)^{2H} + |k-1|^{2H} - 2k^{2H} \right).
$$

| $H$ 的范围 | 增量相关性 | 直观含义 |
|---|---|---|
| $H > 1/2$ | 正相关（长记忆） | 增量有持续性，不等同于波动率聚集 |
| $H = 1/2$ | 不相关 | 标准布朗运动 |
| $H < 1/2$ | 负相关 | 增量反持续，路径更粗糙；过程本身不一定均值回归 |

金融波动率建模中常用的**粗糙波动率模型**（如 rough Heston）正是取 $H < 1/2$，因为实证发现对数波动率的 Hurst 指数约为 0.1 左右。

## 模拟方法

### 方法一：Cholesky 分解（精确）

在不含零时刻的网格 $t_i = i\Delta$（$i=1,\ldots,n$）上，协方差矩阵 $\Sigma_{ij} = \frac{1}{2}(t_i^{2H} + t_j^{2H} - |t_i - t_j|^{2H})$ 是对称正定的，分解 $\Sigma = LL^\top$，则 $B^H = LZ$，$Z \sim \mathcal{N}(0, I)$。

```python
import numpy as np

def fbm_cholesky(n, H, T=1.0, seed=42):
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n 必须是正整数")
    if not np.isfinite([H, T]).all() or not 0 < H < 1 or T <= 0:
        raise ValueError("要求 0 < H < 1 且 T > 0")
    rng = np.random.default_rng(seed)
    t = np.linspace(0, T, n + 1)
    # B_0 = 0 是确定值；含 t=0 的协方差矩阵有零行列，不能直接 Cholesky。
    s, u = np.meshgrid(t[1:], t[1:])
    cov = 0.5 * (s**(2*H) + u**(2*H) - np.abs(s - u)**(2*H))
    L = np.linalg.cholesky(cov)
    path = np.concatenate(([0.0], L @ rng.standard_normal(n)))
    return t, path
```

缺点：复杂度 $O(n^3)$，网格点上万就不可行。

### 方法二：循环嵌入 + FFT（快速精确法）

利用 fBm 增量的平稳性，把协方差矩阵嵌入为循环矩阵，用 FFT 求其"平方根"，复杂度降到 $O(n \log n)$。这是实践中模拟长路径的标准做法（Davies–Harte / Wood–Chan 方法）；需要检查循环嵌入的特征值非负，不能无条件把显著负值截为 0 后仍称为精确模拟。

## 待深入

- fBm 不是半鞅（$H \neq 1/2$ 时），经典 Itô 积分理论不适用 → Wick 积分、粗糙路径理论
- 分数 Black-Scholes 模型中的套利问题及其修正

## 参考资料

- [Dieker 与 Mandjes：On spectral simulation of fractional Brownian motion](https://www.columbia.edu/~ad3217/publications/specsim.pdf)：精确模拟、增量协方差和循环嵌入。
