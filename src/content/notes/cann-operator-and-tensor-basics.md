---
title: '算子与张量基础：从模型节点到数据布局'
description: '梳理算子、Tensor、TensorDesc、shape、format 与 axis 的核心概念及其关系。'
pubDate: 2026-08-27
category: 'CANN / Ascend C'
tags: ['CANN', '算子', 'Tensor', 'Ascend C']
order: 3
---

## 学习目标

这篇笔记解决 Ascend C 入门前最基础的两个问题：算子是什么，以及算子如何理解和处理数据。学完后应能区分算子名称与算子类型、Tensor 与 TensorDesc，并能解释 shape、format、axis 三者分别描述什么。

## 先修知识

- Python 与 NumPy 基础
- 标量、向量、矩阵的基本概念
- 神经网络由多个计算层连接而成的基本认识

## 问题背景：为什么先学算子

AI 模型看起来很复杂，但执行时会被拆成许多基础计算单元，例如卷积、矩阵乘、激活和池化。只有理解这些计算单元以及它们接收的数据形式，才能进一步讨论算子如何映射到昇腾 NPU、如何优化性能。

人工智能从早期规则系统逐渐发展到深度学习和生成式 AI，但无论模型能力如何变化，底层计算仍然要落到一个个算子上。因此，算子可以看成模型与硬件之间的基本衔接点。

## 核心概念：算子是什么

**算子（Operator，OP）是完成某种确定计算逻辑的基础计算单元。** AI 模型通常可以表示为一张计算图，图中的节点是算子，边上传递的是 Tensor。

例如：

```text
输入 Tensor → Conv2D → ReLU → MaxPool → 输出 Tensor
```

在数学上，算子也可以泛指从一个空间到另一个空间的映射，例如微分、积分、`tanh` 和卷积。AI 工程中的“算子”更强调可被框架调度、可在硬件上执行的计算实现。

### 算子名称与算子类型

| 概念 | 回答的问题 | 特点 | 示例 |
|---|---|---|---|
| 算子名称（Name） | 这是网络中的哪一个节点？ | 同一网络中应唯一 | `conv1`、`conv2` |
| 算子类型（Type） | 这个节点执行什么计算？ | 同类型算子的核心计算逻辑一致 | `Conv2D`、`ReLU` |

因此，`conv1` 和 `conv2` 可以是两个不同的算子名称，但它们都属于 `Conv2D` 类型。

下图来自课程原 Notebook。图中 `Conv1`、`Pool1`、`Conv2` 是不同节点的名称；它们对应的类型需要结合模型定义判断，例如 `Conv1` 和 `Conv2` 通常都属于卷积算子类型。

![算子名称与算子类型示例](/images/cann/basic_concepts_of_operators_1.png)

## Tensor 与 TensorDesc

### Tensor：数据本体

Tensor 是算子输入和输出的标准数据容器，可以看作多维数组：

| 阶数 | 含义 | 示例 | shape |
|---|---|---|---|
| 0 维 | 标量 | `5` | `()` |
| 1 维 | 向量 | `[1, 2, 3]` | `(3,)` |
| 2 维 | 矩阵 | `[[1, 2], [3, 4]]` | `(2, 2)` |
| 3 维及以上 | 高维张量 | 批量图像、视频等 | 由各轴长度决定 |

### TensorDesc：数据说明书

TensorDesc 不保存实际数值，只描述 Tensor 的元信息。可以把 Tensor 比作“杯中的水”，TensorDesc 比作“杯子上的标签”。

常见元信息包括：

- `name`：Tensor 的标识；
- `shape`：每个维度的大小；
- `dtype`：元素的数据类型，如 `float16`、`float32`、`int8`；
- `format`：数据的物理排布方式，如 `NCHW`、`NHWC`、`ND`。

在 NumPy 和 PyTorch 中，这些描述能力通常内置在数组或 Tensor 对象里；在 CANN 的部分接口设计中，数据与描述信息会被显式区分。

## 最小例子

```python
import numpy as np

x = np.array([[1, 2], [3, 4]], dtype=np.float32)
y = np.array([[5, 6], [7, 8]], dtype=np.float32)
z = x + y

print(z)
print(z.shape)  # (2, 2)
print(z.dtype)  # float32
```

这里的 `x`、`y` 和 `z` 是 Tensor；加法是算子；`shape` 和 `dtype` 是描述 Tensor 的元信息。

## Shape：每个轴有多长

`shape=(D0, D1, ..., Dn-1)` 表示一个 n 维 Tensor，各元素分别给出对应轴的长度。

以 `shape=(4, 20, 20, 3)` 的图片数据为例：

```text
4  → 图片数量
20 → 图片高度
20 → 图片宽度
3  → RGB 通道数
```

Shape 不仅决定数据规模，也直接影响循环层级、索引方式、内存占用和算子的合法输入范围。

## Format：各轴按什么语义排列

同一批图像可以有相同的元素数量，但采用不同的维度顺序：

| 格式 | 轴顺序 | 常见 shape 示例 |
|---|---|---|
| `NCHW` | 批次、通道、高、宽 | `[1, 3, 224, 224]` |
| `NHWC` | 批次、高、宽、通道 | `[1, 224, 224, 3]` |

转换格式不是简单修改 shape，而是要重排数据轴：

下图对比了同一组图像数据采用 NCHW 与 NHWC 时的轴顺序。看图时重点关注通道 `C` 所在的位置。

![NCHW 与 NHWC 数据排布格式对比](/images/cann/data_layout_format_example.png)

```python
img_nchw = img_nhwc.transpose(0, 3, 1, 2)
img_nhwc_again = img_nchw.transpose(0, 2, 3, 1)
```

Format 会影响线性内存中的数据顺序，也会影响算子读取数据的方式与性能。

## Axis：当前操作沿哪个维度进行

Axis 是维度的下标。对于 `shape=(2, 3, 4)` 的 Tensor：

- `axis=0` 指第一维，长度为 2；
- `axis=1` 指第二维，长度为 3；
- `axis=2` 指第三维，长度为 4；
- `axis=-1` 等价于最后一维，即 `axis=2`。

对某个轴做聚合时，该轴通常会从输出 shape 中消失。例如：

```python
x = np.ones((2, 3, 4))
y = np.sum(x, axis=2)
print(y.shape)  # (2, 3)
```

## 三者的关系

```text
shape  → 每个轴有多少元素
format → 每个轴分别代表什么语义、按什么顺序排列
axis   → 本次操作选中哪一个维度
```

以 `NCHW` 的 `[1, 3, 224, 224]` 为例，`axis=1` 指通道；换成 `NHWC` 后，通道变成 `axis=3`。因此，脱离 format 只说 axis 的业务含义是不完整的。

## 实践经验

- 写算子前先列出所有输入、输出的 `shape`、`dtype` 和 `format`。
- 每一步计算后都检查输出 shape，维度对不上往往比数值错误更早暴露问题。
- 不要把“能广播”误认为“shape 必须相同”。NumPy/PyTorch 支持广播，但具体算子是否允许广播要看其定义。
- 对 RGB 通道直接求和只是通道聚合，不是严格的灰度化；标准灰度转换通常需要按感知权重加权。

## 常见误区

1. **名称相同就代表类型相同。** 名称用于唯一标识，类型才决定计算逻辑。
2. **TensorDesc 保存实际数据。** 实际数据在 Tensor 中，TensorDesc 保存元信息。
3. **Shape 相同，数据布局就一定相同。** Shape 只给出维度大小，format 才描述轴的语义顺序。
4. **Axis 只能为正数。** 多数框架允许负轴，`-1` 表示最后一维。
5. **转格式只需要改 shape。** 正确转换通常还要转置数据轴。

## 一句话总结

算子的本质是对 Tensor 执行确定计算逻辑，而 shape、format 和 axis 分别说明数据有多大、如何排列以及沿哪个维度操作。

## 复习卡片

- 核心问题：AI 模型的基本计算和数据单元是什么？
- 核心机制：算子接收 Tensor，按既定逻辑计算并输出新 Tensor。
- 关键区分：Name 标识节点，Type 决定逻辑；Tensor 存数据，TensorDesc 存元信息。
- Shape：各轴长度。
- Format：轴的语义与排布顺序。
- Axis：操作所针对的维度下标。

## 自测问题

1. `conv1` 与 `Conv2D` 分别更可能是算子名称还是算子类型？
2. Tensor 与 TensorDesc 的职责有什么区别？
3. `[1, 3, 224, 224]` 从 NCHW 转成 NHWC 后，shape 是什么？
4. 对 `shape=(2, 3, 4)` 的 Tensor 沿 `axis=2` 求和，输出 shape 是什么？
5. 为什么只知道 `axis=1`，还不能确定它是不是通道维？

<details>
<summary>参考答案</summary>

1. `conv1` 是名称，`Conv2D` 是类型。
2. Tensor 保存实际数值，TensorDesc 描述名称、shape、dtype、format 等元信息。
3. `[1, 224, 224, 3]`。
4. `(2, 3)`。
5. 轴的业务语义由 format 决定；在 NCHW 中 axis 1 是通道，在 NHWC 中 axis 3 才是通道。

</details>

## 图片来源

本文图片取自 [CANN Learning Hub：人工智能与算子基础](https://gitcode.com/cann/cann-learning-hub/blob/master/tutorials/ascendc_operator_development_light/01_basic_overview/01.02_ai_and_operator_basics.ipynb)，本地保存仅用于学习笔记。
