---
title: 'Ascend C 矢量编程范式：CopyIn、Compute、CopyOut 与 Double Buffer'
description: '梳理矢量算子的三级流水线、TPipe/TQue/TBuf 资源模型及 Double Buffer 的工作原理。'
pubDate: 2026-08-27
category: 'CANN / Ascend C'
tags: ['Ascend C', 'Vector', 'TPipe', 'Double Buffer']
order: 5
---

## 学习目标

这篇笔记解释 Ascend C 为什么采用流水线式编程，以及 `TPipe`、`TQue`、`TBuf`、`TPosition` 如何协同。学完后应能画出基本矢量算子的三级数据流，并判断 Double Buffer 何时可能带来收益。

## 问题背景：计算快，数据不一定跟得上

AI Core 中的数据搬运、Vector 计算和结果搬出由不同指令队列执行。若代码把它们完全串行化，Vector 单元会在等待数据时空闲。Ascend C 编程范式把一个算子拆成多个流水任务，并用队列传递数据和表达依赖。

## 三类典型范式

- 基本矢量范式：`CopyIn → Compute → CopyOut`；
- 基本矩阵范式：在搬入和搬出之间增加 Split、矩阵计算和 Aggregate；
- 复杂矢量/矩阵范式：把多个阶段首尾连接，形成更长的数据流。

入门时最重要的是基本矢量范式：

![Ascend C 基本矢量编程范式](/images/cann/vector_programming_paradigm.png)

## 三级流水任务

### CopyIn

1. 从输入队列申请 `LocalTensor`；
2. 使用 `DataCopy` 把数据从 Global Memory 搬到 Local Memory；
3. 使用 `EnQue` 告知下游“数据已就绪”。

### Compute

1. 使用 `DeQue` 取得输入 `LocalTensor`；
2. 为输出申请 `LocalTensor`；
3. 调用 `Add`、`Abs`、`Exp` 等 Vector API；
4. 释放输入并把输出 `EnQue`。

### CopyOut

1. 从输出队列 `DeQue` 计算结果；
2. 使用 `DataCopy` 写回 Global Memory；
3. 释放输出 `LocalTensor`。

课程原图把相关 API 和数据位置放在同一条数据流上：

![矢量编程范式中的 Tensor、队列与 API](/images/cann/vector_programming_paradigm_and_api.png)

## 核心组件

| 组件 | 职责 | 关键操作 |
|---|---|---|
| `GlobalTensor` | 表示 Global Memory 数据 | `SetGlobalBuffer`、索引偏移 |
| `LocalTensor` | 表示 Local Memory 数据 | 作为搬运与计算 API 的操作对象 |
| `TPipe` | 统一管理片上内存和同步资源 | `InitBuffer` |
| `TQue` | 任务间传递 Tensor 并同步 | `AllocTensor`、`EnQue`、`DeQue`、`FreeTensor` |
| `TBuf` | 管理任务内部临时内存 | `Get`；不能入队、出队 |
| `TPosition` | 抽象逻辑存储位置 | `GM`、`VECIN`、`VECOUT`、`VECCALC` |

一个 Kernel 中通常只初始化一个 `TPipe`。`TQue` 服务于任务之间的数据接力，`TBuf` 服务于单个任务内部的临时变量，二者用途不能混淆。

## 最小伪代码

```cpp
pipe.InitBuffer(inQueue, 2, tileBytes);
pipe.InitBuffer(outQueue, 2, tileBytes);

for (int i = 0; i < loopCount; ++i) {
    CopyIn(i);
    Compute(i);
    CopyOut(i);
}
```

虽然源码按这个顺序发出调用，不同硬件指令队列仍可在满足依赖的前提下重叠执行。`EnQue/DeQue` 不只是容器操作，也承担同步语义。

## Double Buffer

单缓冲时，一块数据依次经历搬入、计算和搬出。双缓冲准备两组片上缓冲，使不同数据块处在不同阶段：

![Double Buffer 让搬运与计算重叠](/images/cann/double_buffer_mechanism.png)

```text
时刻 1：块 0 CopyIn
时刻 2：块 0 Compute，同时块 1 CopyIn
时刻 3：块 0 CopyOut，同时块 1 Compute、块 2 CopyIn
```

队列内存块数通常体现为：

```cpp
pipe.InitBuffer(inQueue, 2, tileBytes);
```

第二个参数为 2 表示为该队列准备两块缓冲。但“双缓冲”不意味着数据只切成两块，而是流水线上可轮换使用两份缓冲资源。

## Double Buffer 何时有效

它主要隐藏数据搬运时间。若搬运和计算耗时接近，重叠带来的收益通常更明显。以下场景收益可能很小甚至为负：

- 数据量很小，一次即可算完；
- 计算远慢于搬运，隐藏搬运后的占比有限；
- 两份缓冲压缩了单块大小，导致循环次数明显增加；
- 同步或尾块处理不当，引入额外开销。

因此，是否启用 Double Buffer 应通过性能数据验证。

## API 层级

- 基础 API：计算、搬运、资源管理和同步，控制粒度更细；
- 高阶 API：封装 Matmul、Softmax、激活等常用算法，提高开发效率；
- Utils API：标准库、平台信息、运行时编译和日志等辅助能力。

选择原则是：能用成熟高阶 API 满足需求时优先复用；需要特殊布局或精细性能控制时，再组合基础 API。

## 常见误区

1. **`TPipe` 是数据队列。** 它是统一资源管理者，任务通信由 `TQue` 完成。
2. **`TBuf` 可以执行 `EnQue`。** `TBuf` 只管理任务内部临时内存。
3. **连续写出三个函数就一定形成有效流水。** 还需要正确的队列、缓冲数量、数据分块和同步关系。
4. **Double Buffer 总能提速。** 它增加片上内存占用，收益取决于搬运/计算比例和数据规模。
5. **释放 Tensor 可以省略。** 不及时 `FreeTensor` 会阻塞缓冲复用或耗尽队列资源。

## 一句话总结

Ascend C 矢量编程范式通过 TPipe 管资源、TQue 传数据，把 CopyIn、Compute、CopyOut 映射到可并行的硬件流水线。

## 复习卡片

- 三级任务：CopyIn、Compute、CopyOut。
- 数据路径：GM → Local → Vector → Local → GM。
- 资源管理：TPipe。
- 任务通信：TQue。
- 临时内存：TBuf。
- 性能优化：Double Buffer 隐藏搬运时间。

## 自测问题

1. `EnQue` 和 `DeQue` 为什么不仅是普通容器操作？
2. `TQue` 与 `TBuf` 的适用范围有什么区别？
3. Double Buffer 为什么需要更多 Local Memory？
4. 哪些情况下 Double Buffer 的收益可能很小？

<details>
<summary>参考答案</summary>

1. 它们还表达上下游任务之间的数据就绪与同步关系。
2. TQue 用于任务间传递数据，TBuf 用于单个任务内部临时计算。
3. 它需要两份可轮换的片上缓冲，使不同数据块同时处于不同流水阶段。
4. 小数据、计算远慢于搬运、切块后循环开销过大等场景。

</details>

## 图片来源

本文图片取自 [CANN Learning Hub：Ascend C 编程范式与 API](https://gitcode.com/cann/cann-learning-hub/blob/master/tutorials/ascendc_operator_development_light/02_AscendC_basic/02.03_ascendc_programming_paradigm_and_api_introduction.ipynb)，本地保存仅用于学习笔记。

