---
title: 'LSTM 股价预测笔记：原理与 PyTorch 实现'
description: 'LSTM 的门控机制推导、股价预测中的数据构造细节，以及一个最小可复现的 PyTorch 实现。'
pubDate: 2026-08-01
category: '深度学习'
tags: ['LSTM', '时间序列', 'PyTorch']
---

## 为什么用 LSTM

普通 RNN 在长序列上有梯度消失问题。LSTM 通过**门控机制**和细胞状态 $c_t$ 缓解这一问题。其核心方程为：

$$
\begin{aligned}
f_t &= \sigma(W_f [h_{t-1}, x_t] + b_f) & \text{遗忘门} \\
i_t &= \sigma(W_i [h_{t-1}, x_t] + b_i) & \text{输入门} \\
\tilde{c}_t &= \tanh(W_c [h_{t-1}, x_t] + b_c) \\
c_t &= f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \\
o_t &= \sigma(W_o [h_{t-1}, x_t] + b_o) & \text{输出门} \\
h_t &= o_t \odot \tanh(c_t)
\end{aligned}
$$

其中 $\sigma$ 是 sigmoid 函数，$\odot$ 是逐元素乘法。

## 数据构造的两个坑

1. **归一化要在划分训练集之后做**，用训练集的均值方差变换测试集，否则就是数据泄漏。
2. **滑动窗口**：用过去 $L$ 天预测下一天，$(x_{t-L+1}, \dots, x_t) \to x_{t+1}$。

## 最小实现

```python
import torch
import torch.nn as nn

class StockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, features)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # 只取最后时刻的隐状态

model = StockLSTM()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

## 实验中的观察

> 对**价格本身**建模往往不如对**收益率**建模稳定——价格是非平稳序列，模型容易学到"明天约等于今天"的懒惰解。

评估时除了 MSE，还应该看方向准确率（涨跌预测对的比例），这更贴近交易场景。

## 待深入

- 加入成交量、技术指标等多特征输入
- 与 GRU、Transformer（如 Informer、PatchTST）对比
- 概率预测：输出分布而非点估计（分位数损失）
