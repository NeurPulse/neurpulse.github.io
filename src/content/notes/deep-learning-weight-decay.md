---
title: '深度学习:权重衰减 — 直接缩小参数的防过拟合技术'
description: '理解权重衰减的原理（每次更新时直接缩小参数）、与 L2 正则化在 SGD 下等价但在 Adam 下不等价的数学推导，以及 PyTorch 中的一行实现'
pubDate: 2026-08-12
category: '深度学习'
tags: ['权重衰减', 'Weight Decay', 'L2正则化', '过拟合', 'Adam']
---

## 1. 权重衰减的原理

### 1.1 思想

与 L1/L2 正则化目标一致——防止参数绝对值过大，从而防止过拟合。但做法更直接：

> L1/L2 正则：在 loss 里加惩罚项，让梯度带上有衰减效果
> 权重衰减：**直接在更新时缩小参数**，不走 loss 这条路

### 1.2 更新公式

$$
w_t = w_t - lr \cdot g_w - lr \cdot \lambda \cdot w_t
$$

| 项 | 含义 |
|------|------|
| $lr \cdot g_w$ | 标准梯度下降的更新 |
| $lr \cdot \lambda \cdot w_t$ | 额外的衰减：把参数往 0 的方向拉 |

每次更新时，参数先被梯度更新，再被 $\lambda$ 按比例缩小。

### 1.3 $\lambda$ 怎么选

| $\lambda$ | 效果 |
|------------|------|
| $10^{-2}$ ~ $10^{-4}$ | 常用范围 |
| 偏大 | 过拟合严重时使用 |
| 偏小 | 数据量大、模型简单时使用 |

---

## 2. 权重衰减 vs L2 正则 —— 在 SGD 下等价

### 2.1 L2 正则的做法

在 loss 中加惩罚项：

$$
L = L_{\text{error}} + \frac{\lambda}{2} w^2
$$

参数更新（对 $w$ 求导）：

$$
w = w - lr \cdot \frac{\partial L}{\partial w}
$$

$$
= w - lr \cdot \frac{\partial L_{\text{error}}}{\partial w} - lr \cdot \lambda w
$$

### 2.2 权重衰减的做法

loss 不变，更新时直接减：

$$
w = w - lr \cdot \frac{\partial L_{\text{error}}}{\partial w} - lr \cdot \lambda w
$$

### 2.3 结果

**对于标准 SGD，两种方式推导出的最终公式完全一致。**

---

## 3. 但在 Adam 下不等价！

对于 Adam，更新公式中梯度经过了动量和二阶矩的调整：

| 方式 | 路径 |
|------|------|
| **L2 正则** | $\lambda w$ 先进入梯度 $g$ → $g$ 再经过 $V$ 和 $S$ 的变换 → 衰减效果被扭曲 |
| **权重衰减** | 梯度走 Adam 流程，衰减项另走一条路，直接减 $lr \cdot \lambda w$（不经过 V/S） |

**结论**：对于 Adam 优化器，必须用权重衰减而非 L2 正则。这就是 PyTorch 中 `AdamW` 的由来——它将权重衰减从梯度计算中解耦出来。

> `AdamW` = Adam + **解耦的** Weight Decay（Decoupled Weight Decay）

---

## 4. 权重衰减 vs L1/L2 正则

| | L1/L2 正则 | 权重衰减 |
|------|------|------|
| 实现方式 | 修改 loss 函数 | 修改参数更新步骤 |
| 经过 Adam 的 V/S | **会**（效果被扭曲） | **不会**（直接作用） |
| 适用于 SGD | ✅ | ✅（两者等价） |
| 适用于 Adam | ❌ 不推荐 | ✅ 用 AdamW |

---

## 5. PyTorch 实现

```python
# SGD + 权重衰减（等价于 L2 正则）
optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-4)

# Adam + 权重衰减（推荐用 AdamW，解耦版）
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
```

---

## 6. 总结

```
权重衰减 = 每次更新时直接缩小参数 → 防过拟合

公式：w = w - lr·g - lr·λ·w
                    └─ 这一项是额外加的衰减

与 L2 的关系：
  SGD 下：权重衰减 = L2 正则（公式完全一致）
  Adam 下：权重衰减 ≠ L2 正则（L2 的衰减被 V/S 扭曲）

实践：
  SGD   → weight_decay 即可（等价 L2）
  Adam  → 用 AdamW（解耦权重衰减）
```
