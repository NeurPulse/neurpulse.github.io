---
title: '深度学习：归一化与标准化— 为什么、怎么做、什么时候做'
description: '特征尺度不一致导致梯度下降不稳定的根本原因，以及归一化与标准化的原理、实现和预测时的注意事项'
pubDate: 2026-08-11
category: '深度学习'
tags: ['Normalization','标准化','梯度下降','MSE','特征工程']
---

# Normalization

##  问题的起点：一个不收敛的例子

**数据与真实规律**

 外卖送餐时间预测，两个特征：

$$
\text{time} = 2 \times \text{lights} + 0.01 \times \text{distance} + 5
$$

| time | lights | distance |
| ---- | ------ | -------- |
| 19   | 2      | 1000     |
| 31   | 3      | 2000     |
| 14   | 2      | 500      |
| 15   | 1      | 800      |
| 43   | 4      | 3000     |

  数据严格按照上述线性方程生成，没有噪声——理论上 loss 应该能降到接近 0。

梯度下降训练：

```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inputs = torch.tensor([[2, 1000], [3, 2000], [2, 500], [1, 800], [4, 3000]], dtype=torch.float, device=device)
labels = torch.tensor([[19], [31], [14], [15], [43]], dtype=torch.float, device=device)


w = torch.ones(2, 1, requires_grad=True, device=device)
b = torch.ones(1, requires_grad=True, device=device)

epoch = 200
lr = 0.0000001

for i in range(epoch):
    outputs = inputs @ w + b
    loss = torch.mean(torch.square(outputs - labels))
    print("loss", loss.item())
    loss.backward()
    print("w.grad", w.grad.tolist())
    with torch.no_grad():
        w -= w.grad * lr
        b -= b.grad * lr

    w.grad.zero_()
    b.grad.zero_()
```
---
  现象：
  - 学习率稍大一点，loss 立刻爆炸，直接超出 float 表示范围
  - 即使设得很小，loss 降到 7 左右就下不去了，无法接近 0

根因分析：**梯度尺度不一致**

看第一次迭代的梯度

```python
loss 2898583.75
w.grad [[8600.0], [5876040.0]]
```

  - lights 权重的梯度：8,600
  - distance 权重的梯度：5,876,040
  - distance 的梯度大约是 lights 的 ~680 倍

**为什么差这么多？**

> 两个特征的数值范围完全不同：

| 特征     | 取值范围 | 权重目标               | 对loss的影响 |
| -------- | -------- | ---------------------- | --------|
| lights   | 1~4      | $w_0 \rightarrow 2$    | 较小     |
| distance | 500~3000 | $w_1 \rightarrow 0.01$ | 非常大    |

distance 的值大约是 lights 的 1000 倍。同一个权重变化量，作用在 distance 上对 loss 的扰动，是作用在 lights 上的 1000
  倍。所以 loss 对 $w_1$ 的偏导数天然就大了两个数量级。

后果：共享学习率的困境

  $$w_1 := w_1 - lr \cdot \text{grad}_{w1}$$

  - $w_1$ 的梯度非常大，学习率必须极小，否则一步就跨过目标值 0.01，直接发散
  - 为了迁就 $w_1$，学习率被迫设得很小
  - 但 $w_0$ 的梯度本来就小，学习率再这么小，$w_0$ 几乎不动
  - 结果：$w_1$ 勉强收敛，$w_0$ 还没调到目标，loss 降不下去

  本质原因：不同特征的取值范围不同 → 对应权重的梯度量级不同 → 共用一个学习率时无法同时兼顾。

---

## 解决方案一：归一化(Normalization / Min-Max Scaling）

**思路**：

> 让所有特征的取值范围相同（都缩放到 [0, 1] 或 [-1, 1]），这样所有权重的梯度就在同一量级，可以用统一的学习率。
>
> bias 的"输入"永远是 1，所以把其他特征也缩放到 1 附近即可。

**实现**：

每个特征除以该特征在所有样本中的最大值：

```python
inputs = inputs / torch.tensor([4, 3000], device=device)
# lights:   [2, 3, 2, 1, 4] / 4  → [0.5, 0.75, 0.5, 0.25, 1.0]
# distance: [1000, 2000, 500, 800, 3000] / 3000 → [0.33, 0.67, 0.17, 0.27, 1.0]
```

```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inputs = torch.tensor([[2, 1000], [3, 2000], [2, 500], [1, 800], [4, 3000]], dtype=torch.float, device=device)
labels = torch.tensor([[19], [31], [14], [15], [43]], dtype=torch.float, device=device)

#进行归一化
inputs = inputs / torch.tensor([4, 3000], device=device)


w = torch.ones(2, 1, requires_grad=True, device=device)
b = torch.ones(1, requires_grad=True, device=device)

epoch = 1000
lr = 0.5

for i in range(epoch):
    outputs = inputs @ w + b
    loss = torch.mean(torch.square(outputs - labels))
    print("loss", loss.item())
    loss.backward()
    print("w.grad", w.grad.tolist())
    with torch.no_grad():
        w -= w.grad * lr
        b -= b.grad * lr

    w.grad.zero_()
    b.grad.zero_()
```

**效果**：

归一化后，学习率可以从 0.0000001 直接拉到 0.5：

```python
lr = 0.5      # 原来设 0.5 直接爆炸，现在正常训练
epoch = 1000  # loss 很快接近 0
```

训练变得稳定且快速。

**缺点：**

只用一个最大值决定缩放比例。如果这个最大值是异常值（outlier），整个缩放就会被带偏。

## 解决方案一：标准化（Standardization / Z-Score）

**思路**：

对每个特征，减去自己的均值，除以自己的标准差，使其变成均值为 0、标准差为 1 的分布。
$$
x' = \frac{x - \mu}{\sigma}
$$
**实现：**

```python
mean = inputs.mean(dim=0)   # 每个特征的均值
std  = inputs.std(dim=0)    # 每个特征的标准差
inputs_norm = (inputs - mean) / std
```

**对比：**

|                          | 归一化（Min-Max）                     | 标准化（Z-Score）                          |
| ------------------------ | ------------------------------------- | ----------------------------------------- |
| 被谁决定                 | 仅由最大值、最小值决定 | 由所有样本的均值和标准差决定              |
| 对异常值的敏感性         | 敏感（一个异常值毁全局）              | 鲁棒（考虑了整体分布）                    |
| 结果范围                 | $[0,\ 1]$                             | 均值 0，标准差 1（理论上无界）            |

标准化考虑了所有样本的分布，不会被单个异常值左右，训练更稳定。在实际深度学习中，标准化是默认选择。

**效果：**

```python
lr = 0.5       # 同样可以用大学习率
epoch = 1000   # loss 同样接近 0
```

**注：**

> **预测时也要归一化参数！**
>
> 训练时你对数据做了标准化——记录了 mean 和 std。预测新数据时，必须用同样的 mean 和 std
> 对新输入做变换，否则模型看到的数值尺度完全不对，预测结果毫无意义。
>
> 归一化参数是训练数据的"属性"，预测时直接用，不要拿新数据重新算。

## **归一化改变了数据，为什么不影响模型？**

> 归一化只是对数据做了一次可逆的线性变换，不改变数据之间的本质关系。模型可以通过调整权重来"补偿"这个变换——就像把单位从米换成千米，物理规律不变，只是数值变了。

  $$y = w \cdot x + b = (w \cdot k) \cdot \left(\frac{x}{k}\right) + b$$

## 总结

问题：特征取值范围不同 → 梯度量级不同 → 共享 lr 无法同时收敛

方案：把所有特征缩放到同一尺度（标准化 / 归一化）

关键：预测时用训练时的参数做同样的缩放

结论：所有深度学习模型默认做特征标准化，零成本，大收益

