---
title: 'PyTorch：基础配置'
description: '配置常用导入、训练超参数与计算设备，理解 CUDA 可见设备和逻辑编号。'
pubDate: 2026-08-08
updatedDate: 2026-10-19
order: 1
category: 'PyTorch'
tags: ['os', 'numpy', 'torch']
---

## 基础包

先确认当前解释器已经安装与硬件、驱动匹配的 PyTorch。不要仅因机器有 NVIDIA GPU，就假定当前 PyTorch 安装一定能使用 CUDA。

```python
import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
```

按项目实际需要导入包：表格处理常用 Pandas，图像处理可用 Pillow 或 torchvision，可视化可用 Matplotlib，指标计算可用 scikit-learn。它们不是每个项目都必须安装或导入的依赖。

## 超参数

```python
batch_size = 16
lr = 1e-4
max_epochs = 100
```

- `batch_size`：每次参数更新使用的样本批大小；启用梯度累积时，还要区分单批大小和有效批大小。
- `lr`：优化器的初始学习率，后续可由学习率调度器调整。
- `max_epochs`：完整遍历训练集的最大次数，不是单个 batch 的更新次数。

## GPU

### 选择计算设备

在未修改默认设备的常规设置下，新建张量和模块位于 CPU。可以显式指定设备；下面选择第一张**可见 CUDA GPU**，不可用时回退到 CPU。

```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = nn.Linear(4, 2).to(device)
features = torch.randn(3, 4).to(device)
labels = torch.tensor([0, 1, 0], dtype=torch.long, device=device)

logits = model(features)
loss = nn.CrossEntropyLoss()(logits, labels)
loss.backward()
assert logits.device == features.device == labels.device
```

`torch.cuda.is_available()` 只说明存在可用 CUDA 设备，不能证明 `cuda:1` 存在。需要第二张可见 GPU 时，应先检查 `torch.cuda.device_count() >= 2`。

张量的 `.to(device)` 返回转换后的张量，通常需要接收返回值；模块的 `.to(device)` 会迁移其已注册参数和缓冲区。输入、标签和参与运算的模型参数应位于兼容的设备，dtype 则按运算要求设置：例如分类索引标签通常为 `torch.long`，不应一律转为浮点数。

### 限制可见 GPU

`CUDA_VISIBLE_DEVICES` 控制 CUDA 进程可见的 GPU，必须在 CUDA 初始化之前设置。最清晰的方式是在启动 Python 前设置，例如在 Linux shell 中：

```bash
CUDA_VISIBLE_DEVICES=1 python train.py
```

此时被选择的物理设备会成为进程内的逻辑 `cuda:0`。类似地，`CUDA_VISIBLE_DEVICES=2,0` 会重新排列进程内的设备编号。具体物理对应关系还应结合系统的设备排序确认。

若在 Python 脚本内设置，应把设置放在导入 torch、创建 CUDA 张量或调用 CUDA 检查之前。修改已经初始化 CUDA 的 Notebook 会话中的变量，不能当作可靠的设备切换方式。

暴露多张 GPU **不会自动让模型并行训练**，多卡训练还需要 DistributedDataParallel 等机制。CUDA 配置也不等于昇腾 NPU 配置；后者需要匹配的 `torch_npu` 与 CANN 环境，见[Ascend C 接入 PyTorch](/notes/ascend-c-pytorch-kernel-integration/)。

## 参考

- [PyTorch：CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html)
- [PyTorch：CUDA 环境变量](https://docs.pytorch.org/docs/stable/cuda_environment_variables.html)
