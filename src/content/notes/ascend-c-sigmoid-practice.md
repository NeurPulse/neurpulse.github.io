---
title: 'Ascend C Sigmoid 综合实践：从固定 Kernel 到泛化 Vector 算子'
description: '用 Sigmoid 串联矢量公式分解、三级流水、泛化 Tiling、多 dtype 与精度验证。'
pubDate: 2026-08-27
category: 'CANN / Ascend C'
tags: ['Ascend C', 'Sigmoid', 'Vector', 'Tiling']
order: 9
---

## 学习目标

这篇笔记总结 2day 最后的两组实践：先实现固定 shape、`float` 类型的 Sigmoid Kernel，再扩展为支持不同 shape 与 `float/half` 的泛化 Vector 算子。

## 算子定义

$$
\operatorname{sigmoid}(x)=\frac{1}{1+e^{-x}}
$$

输入与输出 shape 相同，每个元素独立计算，因此适合使用 Vector 单元并按数据块并行。

## 计算步骤分解

Sigmoid 可以拆成四步逐元素操作：

```text
t0 = -x
t1 = exp(t0)
t2 = 1 + t1
y  = 1 / t2
```

在 Ascend C 中可根据当前 CANN 版本选择对应的 Neg/Muls、Exp、Adds、Reciprocal/Div 等接口。具体 API 名称和支持 dtype 应以当前版本文档为准，不要只根据数学名称猜测。

## Local Memory 规划

与 Add 不同，Sigmoid 除输入和输出外还可能需要中间 Tensor。规划 UB 时要回答：

- 是否原地复用同一块临时内存；
- 每一步 API 是否允许输入输出别名；
- `float` 与 `half` 各占多少字节；
- Double Buffer 后需要几份输入/输出缓冲；
- 临时 Tensor 是否用 `TBuf` 管理。

如果 API 允许安全原地计算，可减少 UB 占用；否则必须为中间结果预留空间。

## 固定规格实现

第一阶段要求：

- 输入输出均为 `float`；
- 总长度固定为 `8×2048`；
- 采用两个缓冲块；
- 完成 Kernel 和 Host 侧直调；
- 与 CPU 公式结果比较。

代码骨架仍遵循：

```text
KernelSigmoid
├─ Init：设置 GlobalTensor、切块和队列
├─ Process：遍历所有数据块
├─ CopyIn：x GM → Local
├─ Compute：执行 Sigmoid 分解
└─ CopyOut：y Local → GM
```

Kernel 入口：

```cpp
__global__ __vector__ void sigmoid_custom(
    GM_ADDR x,
    GM_ADDR y,
    SigmoidCustomTilingData tiling)
{
    KernelSigmoid op;
    op.Init(x, y, tiling);
    op.Process();
}
```

## Host 侧直调

Host 的主要步骤与 Add 一致：

1. 初始化 AscendCL、Device 和 Stream；
2. 分配输入输出 Device 内存；
3. 输入数据拷到 Device；
4. 构造 TilingData；
5. 启动 `sigmoid_custom<<<...>>>`；
6. 同步并拷回输出；
7. 与 CPU Golden 比较；
8. 释放资源。

CPU Golden：

```cpp
y[i] = 1.0f / (1.0f + std::exp(-x[i]));
```

## 泛化版本

第二阶段要求支持：

| Case | Shape | dtype |
|---|---|---|
| 1 | `(7, 83)` | `float32` |
| 2 | `(1024, 1024)` | `float16` |
| 3 | `(999, 999)` | `float32` |

这三组数据故意包含小输入、大输入和无法整齐均分的输入，用来检查泛化 Tiling、尾核、尾块及 dtype 模板是否正确。

### 模板化 Kernel

```cpp
template <typename T>
__global__ __vector__ void sigmoid_custom(
    GM_ADDR x, GM_ADDR y,
    SigmoidCustomTilingData tiling);
```

Host 按 dtype 调用：

```cpp
RunCase<float>("float32", {7, 83}, 0);
RunCase<half>("float16", {1024, 1024}, 1);
```

### Tiling 必须随 dtype 改变

`float` 为 4B，`half` 为 2B。在相同 UB 字节容量下，单 Tile 可容纳的元素数量不同：

```text
tileElements = usableBytesPerBuffer / sizeof(T)
```

不能只把 Kernel 改成模板，却仍使用固定的元素切块参数。

## 精度与数值稳定性

Sigmoid 是非线性函数，不应使用逐位严格相等：

```text
abs(actual - expected) <= atol + rtol × abs(expected)
```

`half` 的精度低于 `float`，通常需要更宽松的容差。还要覆盖较大的正数和负数，因为 `exp(-x)` 可能发生上溢或下溢。对于极端输入，可考虑使用数值更稳定的等价形式或成熟高阶 API。

## 实践检查清单

- [ ] 每个核的 Global Memory 区间互不重叠；
- [ ] 小输入不会启动超过有效数据块数量的核；
- [ ] 大核、小核偏移连续且无空洞；
- [ ] 普通 Tile 与尾块使用正确的有效长度；
- [ ] UB 预算包含输入、输出、中间 Tensor 和双缓冲；
- [ ] `float/half` 的字节数参与所有 Tiling 计算；
- [ ] 输出采用合理的绝对/相对误差验证；
- [ ] 所有 Host/Device 资源均在异常路径和正常路径释放。

## 常见误区

1. **Sigmoid 是一条 Vector 指令。** 它可能由多条基础 API 组合，也可能使用版本提供的高阶 API。
2. **Add 的 UB 预算可以原样复用。** Sigmoid 通常需要额外中间结果。
3. **模板化数据类型就完成泛化。** Tiling、API 支持范围和误差阈值也必须随 dtype 调整。
4. **输出应该与 CPU 完全相等。** 非线性浮点运算应使用容差。
5. **只验证整齐 shape。** `(7,83)` 和 `(999,999)` 这类输入才能暴露尾块问题。

## 一句话总结

Sigmoid 实践的重点不只是写出公式，而是把多步 Vector 计算、临时内存、泛化 Tiling、多 dtype 和容差验证组合成一个可靠算子。

## 复习卡片

- 公式：`1 / (1 + exp(-x))`。
- 计算：取负 → Exp → 加一 → 倒数。
- 流水：CopyIn → 多步 Compute → CopyOut。
- 泛化：shape、dtype、核间余数、核内尾块。
- UB：输入、输出、中间量、双缓冲共同占用。
- 验证：`atol + rtol`，不能严格相等。

## 自测问题

1. Sigmoid 为什么可能比 Add 需要更多 UB 空间？
2. 支持 `float` 和 `half` 时，Tiling 哪个量会直接变化？
3. 为什么要测试 `(7,83)` 这样的 shape？
4. Sigmoid 输出为什么不适合使用严格相等校验？

<details>
<summary>参考答案</summary>

1. 它由多步运算组成，可能需要一个或多个中间 Tensor。
2. 单元素字节数变化会影响每个 Tile 可容纳的元素数量。
3. 元素总数通常不能被核数和 Tile 整齐切分，可暴露尾核、尾块错误。
4. Exp、倒数及不同精度会产生正常的浮点舍入差异。

</details>

## 参考来源

- [CANN Learning Hub：核函数实践](https://gitcode.com/cann/cann-learning-hub/blob/master/tutorials/ascendc_operator_development_light/02_AscendC_basic/02.07_kernel_function_practice.ipynb)
- [CANN Learning Hub：Vector 算子开发实践](https://gitcode.com/cann/cann-learning-hub/blob/master/tutorials/ascendc_operator_development_light/02_AscendC_basic/02.08_vector_operator_practice.ipynb)

