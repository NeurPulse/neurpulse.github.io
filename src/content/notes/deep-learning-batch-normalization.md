---
title: '深度学习:批量归一化 — 让深层网络每一层的输入都稳定'
description: '理解 Batch Normalization 的核心操作（标准化 → 缩放平移）、γ/β 可学习参数的意义、为什么 BN 前不加 bias、推理时的指数加权平均切换，以及 PyTorch 实现'
pubDate: 2026-08-12
category: '深度学习'
tags: ['Batch Normalization', '批量归一化', 'BN', '训练稳定性', '正则化']
---

## 1. 动机：深层网络的输入漂移

输入数据做了标准化 → 训练初始稳定。但随着网络前向传播，越深处的层，其输入分布越来越不稳定。

**在训练过程中，前面层参数不断更新 → 后面层接收到的输入分布在持续变化 → 后面层刚学到的模式可能因为输入变化而失效。**

> 输入归一化保证了第一层稳定，但谁保证后面各层的输入也稳定呢？

---

## 2. Batch Normalization 的核心操作

### 2.1 对什么做归一化

一般对激活**前**的 z 值做归一化（也可以用在对激活后的 a 值）。

### 2.2 四步操作

以某个神经元的 $z$ 值为例，batch size = $m$：

**第一步**：计算当前 batch 的均值
$$
\mu = \frac{1}{m} \sum_{i=1}^{m} z_i
$$

**第二步**：计算当前 batch 的标准差
$$
\sigma = \sqrt{\frac{1}{m} \sum_{i=1}^{m} (z_i - \mu)^2}
$$

**第三步**：标准化
$$
\hat{z}_i = \frac{z_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

**第四步**：缩放和平移（关键一步！）
$$
z_i^{\text{BN}} = \gamma \cdot \hat{z}_i + \beta
$$

### 2.3 $\gamma$ 和 $\beta$ 是什么

| 参数 | 初始化 | 说明 |
|------|--------|------|
| $\gamma$ | 1 | 可学习的缩放参数 |
| $\beta$ | 0 | 可学习的平移参数 |

有了 $\gamma$ 和 $\beta$，模型可以：
- 学回原始的分布（当 $\gamma=\sigma$, $\beta=\mu$ 时，$z^{\text{BN}} = z$）
- 或者学到任何更合适的分布

> BN 并不是把输出强行固定到 N(0,1)，而是给模型一个**稳定训练的起点**，然后让模型自己学最合适的变换。

---

## 3. 为什么 BN 前不加偏置 b

计算 z 时：$z = Wx + b$

BN 时：$\hat{z} = \frac{z - \mu}{\sigma}$

减均值后，$b$ 被抵消了——不论 $b$ 是多少，减去均值后效果相同。而 BN 里已经有了可学习的 $\beta$ 来控制平移。

**结论**：BN 前面的 Linear 层设 `bias=False`。

---

## 4. 直观理解

```
输入层（标准化） → 稳定
     ↓
第 1 层（BN）→ 输出分布稳定
     ↓
第 2 层（BN）→ 输出分布稳定  ← 深层在稳固的基础上学习
     ↓
    ...
```

每一层的输出都有稳定的均值和方差 → 深层网络不再需要反复适应前面层的变化 → 训练更稳定、收敛更快。

---

## 5. BN 的副作用：自带正则化

BN 最初不是为了正则化设计的，但实践中发现有防过拟合的效果：

- 每个 batch 的均值和标准差都不同（取决于具体抽了哪些数据）
- 相当于给数据注入了**随机噪声**
- 模型被迫学习更鲁棒的特征，不会过度依赖某个 batch 的特定数值

> 使用 BN 后，有时可以减小或去掉 Dropout。

---

## 6. 推理时的 BN

推理时可能只有 1 条数据，无法计算 batch 均值和标准差。

**解决方案**：训练时用指数加权平均维护一组全局的 running mean 和 running var：

```python
running_mean = β · running_mean + (1-β) · batch_mean
running_var  = β · running_var  + (1-β) · batch_var
```

推理时直接用 running mean 和 running var 做归一化。

---

## 7. 训练 vs 推理总结

| | 训练 | 推理 |
|------|------|------|
| $\mu$, $\sigma$ 来源 | 当前 batch 数据 | 训练的指数加权平均 |
| $\gamma$, $\beta$ | 学习更新 | **使用训练好的固定值** |
| 行为 | 带噪声（好处：正则化） | 无噪声（好处：稳定预测） |

和 Dropout 一样，需要正确切换 `model.train()` / `model.eval()`。

---

## 8. PyTorch 实现

```python
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(28*28, 128, bias=False),  # BN 前不加 bias
            nn.BatchNorm1d(128),                  # 128 是该层神经元数
            nn.ReLU(),

            nn.Linear(128, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 10)                     # 输出层不加 BN
        )
```

**层的顺序**：`Linear(bias=False) → BatchNorm → ReLU`

---

## 9. 总结

```
BN 的作用：让每一层输出的分布稳定，训练更深更快

一步 BN 做的事：
  z_norm = (z - μ) / √(σ² + ε)    ← 标准化（每个 batch 的统计值）
  z_BN = γ · z_norm + β            ← 缩放平移（可学习参数）

关键点：
  ├── γ=1, β=0 初始化，模型可学回原始分布
  ├── BN 前的 Linear 不用 bias（会被减均值抵消）
  ├── 训练用 batch 统计值，推理用 running 平均值
  ├── 自带轻微正则化效果（batch 噪声）
  └── 输出层一般不加 BN

PyTorch：Linear(bias=False) → BatchNorm1d → ReLU
```
