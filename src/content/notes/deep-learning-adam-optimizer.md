---
title: '深度学习:Adam 优化器 — Momentum + RMSProp = 默认选择'
description: '理解 Adam 如何融合动量（一阶矩）与 RMSProp（二阶矩）、偏差修正的作用、各超参数的默认值，以及它在显存上的代价'
pubDate: 2026-08-12
category: '深度学习'
tags: ['Adam', '优化器', 'Momentum', 'RMSProp', '自适应学习率']
---

## 1. Adam = Momentum + RMSProp

| 组件 | 来源于 | 做什么 |
|------|--------|--------|
| **一阶矩 $V$**（梯度指数加权平均） | Momentum | 提供惯性，抑制振荡，冲过局部最优 |
| **二阶矩 $S$**（梯度平方指数加权平均） | RMSProp | 自适应学习率，梯度大的慢、梯度小的快 |

Adam 同时拥有两者的优点，是深度学习领域**默认的优化器**。

---

## 2. 计算过程

以参数 $w$ 为例：

### 2.1 计算梯度

$$
g_w = \frac{\partial \text{loss}}{\partial w}
$$

### 2.2 更新一阶矩和二阶矩

$$
V_w = \beta_1 V_w + (1-\beta_1) g_w
$$
$$
S_w = \beta_2 S_w + (1-\beta_2) g_w^2
$$

| 超参数 | 默认值 | 含义 |
|--------|--------|------|
| $\beta_1$ | 0.9 | 一阶矩的衰减系数（动量） |
| $\beta_2$ | 0.999 | 二阶矩的衰减系数（梯度平方） |

### 2.3 偏差修正

和指数加权平均一样，$V_w$ 和 $S_w$ 初始为 0，早期值偏小，需要修正：

$$
V_w^{\text{correct}} = \frac{V_w}{1 - \beta_1^t}
$$

$$
S_w^{\text{correct}} = \frac{S_w}{1 - \beta_2^t}
$$

$t$ 是当前迭代步数。注意 Adam 需要偏差修正（RMSProp 和 Momentum 单独使用时一般不做），这是因为二阶矩也受冷启动影响，不做修正训练初期会很不稳定。

### 2.4 更新参数

$$
w = w - lr \cdot \frac{V_w^{\text{correct}}}{\sqrt{S_w^{\text{correct}}} + \varepsilon}
$$

- 分子 $V$：动量的方向（往哪走）
- 分母 $\sqrt{S}$：自适应缩放（走多大步）
- $\varepsilon$：防止除零，默认 $10^{-8}$

---

## 3. 完整公式一览

$$
\begin{aligned} g_w &= \frac{\partial \text{loss}}{\partial w} \\[4pt] V_w &= \beta_1 V_w + (1-\beta_1) g_w \\[4pt] S_w &= \beta_2 S_w + (1-\beta_2) g_w^2 \\[4pt] \hat{V}_w &= \frac{V_w}{1 - \beta_1^t} \\[4pt] \hat{S}_w &= \frac{S_w}{1 - \beta_2^t} \\[4pt] w &= w - lr \cdot \frac{\hat{V}_w}{\sqrt{\hat{S}_w} + \varepsilon} \end{aligned}
$$

---

## 4. 三种优化器对比

| | Momentum | RMSProp | **Adam** |
|------|------|------|------|
| 一阶矩 $V$ | ✅ | — | ✅ |
| 二阶矩 $S$ | — | ✅ | ✅ |
| 偏差修正 | 一般不做 | 不做 | **做** |
| 自适应学习率 | — | ✅ | ✅ |
| 动量惯性 | ✅ | — | ✅ |
| 额外显存 | 1 倍参数量 | 1 倍参数量 | **2 倍参数量** |

---

## 5. 内存代价

Adam 需要为每个参数额外保存两个值：$V$ 和 $S$。

- 假设模型有 1 亿个参数（每个参数是 4 字节 float32）
- 参数本身：400 MB
- $V$ 和 $S$ 各占 400 MB → 额外 **800 MB**
- 总显存 ≈ 参数 × 3

这是 Adam 的主要缺点。但对于大多数场景，这个显存代价是可以接受的；当显存紧张时，可以选择 SGD+Momentum（只额外存 1 份）。

---

## 6. PyTorch 中使用

```python
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
# 绝大多数时候直接用默认参数即可
```

---

## 7. 总结

```
优化器演进路线：

SGD → + 指数加权平均 → Momentum        (有惯性，抑制振荡)
SGD → + 自适应学习率    → RMSProp         (梯度大→慢，梯度小→快)
Momentum + RMSProp     → Adam           (两者结合，默认选择)

Adam 的每一步：
  g → V(动量) + S(自适应) → 偏差修正 → 更新

代价：每个参数多存 V 和 S（2 倍显存）
收益：更稳定、更快速、几乎不需要调参
推荐：β₁=0.9, β₂=0.999, ε=1e-8, lr=1e-3 → 默认配置即可
```
