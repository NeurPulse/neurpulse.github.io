---
title: '深度学习:迁移学习 — 让预训练模型为你所用'
description: '理解迁移学习的核心思想、神经网络中"冻结特征提取层 + 替换分类头"的标准流程，以及根据目标数据量选择不同微调策略的原则与 PyTorch 实现'
pubDate: 2026-08-17
category: '深度学习'
tags: ['迁移学习', '预训练模型', '微调', 'Fine-tuning', '分类头', 'ResNet']
---

## 1. 什么是迁移学习

### 1.1 一个生活类比

```
学会骑自行车 → 再学骑电动车就快得多
（平衡、转向能力直接迁移过去）
```

### 1.2 核心定义

> 把一个任务（源任务）学到的知识，应用到另一个相关但不同的任务（目标任务）。

好处：**大幅缩短学习新任务所需的时间和数据量。**

---

## 2. 神经网络中的迁移学习

### 2.1 基础认知

| 网络部分 | 学到什么 |
|----------|---------|
| 浅层 | 通用基础特征（线条、纹理、颜色） |
| 深层 | 抽象、与任务相关的高级特征 |
| 分类头 | 把特征映射到具体类别 |

**关键洞察**：浅中层学到的特征（边缘、纹理、物体部件）对大多数视觉任务通用。

### 2.2 标准四步流程（面包分类为例）

```
1. 加载预训练模型     → ImageNet 上训练好的 ResNet
2. 冻结特征提取层     → 锁定除分类层外的所有层参数
3. 替换分类头         → 1000 类 → 10 类面包
4. 微调新分类头       → 用面包数据只训练新分类层
```

结果：只需少量标注数据，就能快速得到不错的分类模型。

---

## 3. 策略选择：看数据量

| 目标数据量 | 策略 | 学习率 |
|-----------|------|--------|
| **很少**（几百张） | 只替换分类头，冻结其余全部 | 很小（1e-4, 1e-5） |
| **适中**（几千张） | 替换分类头 + 解冻最后几层（如 layer4） | 谨慎设置 |
| **较大**（几万张+） | 替换分类头 + 全网络微调 | 稍大，但小于从头训练 |

### 为什么主要替换最后一层？

| 原因 | 说明 |
|------|------|
| **分类需求不同** | 源任务 1000 类 vs 新任务 N 类，神经元数必须匹配 |
| **特征组合不同** | 即使类别数相同，语义也不同（"公/母" vs "金毛/哈士奇"），最后一层的组合关系无意义 |
| **特征通用性** | 浅中层特征是通用的，冻结它们可保留强大特征提取能力，避免小数据过拟合 |

### 任务相关性是关键

新任务与源任务领域越相近（如都是自然图像分类），迁移效果越好。

---

## 4. PyTorch 实现

### 4.1 只换分类头（数据量少）

```python
from torchvision import models

model = models.resnet18(pretrained=True)   # 加载 ImageNet 预训练权重
for param in model.parameters():
    param.requires_grad = False            # 冻结所有层

in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 1)       # 新二分类头
```

### 4.2 换分类头 + 解冻最后阶段（数据量适中）

```python
model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False            # 先冻结所有层

for param in model.layer4.parameters():    # 解冻最后一个残差模块
    param.requires_grad = True

in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 1)       # 二分类
```

### 4.3 猫狗分类完整流程

关键点：
- `pretrained=True` 加载 ImageNet 权重
- 数据增强（RandomCrop、RandomHorizontalFlip、ColorJitter）
- 归一化用 ImageNet 的均值方差 `[0.485, 0.456, 0.406]` 和 `[0.229, 0.224, 0.225]`（**必须与预训练一致**）
- 二分类：输出 1 个 logit + Sigmoid + BCELoss

---

## 5. 总结

```
迁移学习 = 借力打力

前提：源任务与目标任务领域相近

标准流程：
  预训练模型 → 冻结特征层 → 替换分类头 → 微调

策略随数据量变化：
  数据少    → 只训分类头（冻结其余）
  数据适中  → 解冻最后几层 + 分类头
  数据多    → 全网络微调

为什么换最后一层：
  类别数不同 + 语义不同 + 浅层特征通用

PyTorch 核心三行：
  models.resnet18(pretrained=True)
  param.requires_grad = False
  model.fc = nn.Linear(in_features, 新类别数)
```
