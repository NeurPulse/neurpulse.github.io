---
title: 'Ascend C Add 算子完整实现：从算子分析到 Host 验证'
description: '以逐元素 Add 为例，串联两级数据切分、Kernel 类、三级流水线、Host 调用与构建验证。'
pubDate: 2026-08-27
category: 'CANN / Ascend C'
tags: ['Ascend C', 'Add', 'Kernel', 'Double Buffer']
order: 6
featured: true
---

## 学习目标

这篇笔记用 Add 算子贯穿一次完整的 Ascend C 直调开发。学完后应能从算子规格推导所需 API，解释多核切分和核内切块，并理解 Kernel 类与 Host 调用代码的职责边界。

## 算子规格分析

逐元素 Add：

$$
z_i = x_i + y_i
$$

课程固定样例：两个输入 `x/y`，一个输出 `z`，shape 均为 `(8, 2048)`，dtype 为 `float`，format 为 `ND`。

![AddCustom 算子设计规格](/images/cann/ascend_c_add_operator_design_specification.png)

由计算过程可以直接得到 API 清单：

- `DataCopy`：Global 与 Local Memory 之间搬运；
- `Add`：Local Memory 中的矢量加法；
- `AllocTensor/FreeTensor`：管理 LocalTensor；
- `EnQue/DeQue`：传递数据并同步流水任务。

## 代码组织

```text
add_custom.asc
├─ TilingData：Host 传入切分参数
├─ KernelAdd：Device 侧算子实现类
│  ├─ Init
│  ├─ Process
│  ├─ CopyIn
│  ├─ Compute
│  └─ CopyOut
├─ add_custom：Device 侧核函数入口
├─ kernel_add：Host 侧调用封装
├─ VerifyResult：精度验证
└─ main：测试数据与程序入口
```

## TilingData 的作用

固定样例使用：

```cpp
struct AddCustomTilingData {
    uint32_t totalLength;
    uint32_t tileNum;
};
```

`totalLength` 与启动核数决定每核数据量，`tileNum` 决定每个核内部的切块数。它把“如何切数据”从 Kernel 计算逻辑中抽离出来。

## 两级数据切分

### 第一级：多核切分

若总长度为 `8×2048`，启动 8 个核：

```text
blockLength = totalLength / 8 = 2048
第 k 核起点 = k × blockLength
```

![Add 算子的多核数据切分](/images/cann/multi_core_parallel_processing_schematic.png)

```cpp
xGm.SetGlobalBuffer(
    reinterpret_cast<__gm__ float*>(x)
        + blockLength * AscendC::GetBlockIdx(),
    blockLength);
```

每个核必须指向不同的 Global Memory 区间，否则会重复计算或互相覆盖。

### 第二级：核内切块

单核的 2048 个元素再切成 8 个 Tile；开启 Double Buffer 后，每个 Tile 再对应两轮缓冲处理：

```text
tileLength = blockLength / tileNum / BUFFER_NUM
           = 2048 / 8 / 2
           = 128 个元素
loopCount  = tileNum × BUFFER_NUM = 16
```

![单核 Tiling 与 Double Buffer 切分](/images/cann/single_core_data_slicing_schematic.png)

这里的 `tileNum` 语义容易混淆：代码最终真正处理的块数是 `tileNum × BUFFER_NUM`。

## Kernel 类

核心成员：

```cpp
AscendC::TPipe pipe;
AscendC::TQue<AscendC::TPosition::VECIN, 2> inQueueX, inQueueY;
AscendC::TQue<AscendC::TPosition::VECOUT, 2> outQueueZ;
AscendC::GlobalTensor<float> xGm, yGm, zGm;
```

### Init

`Init` 完成三件事：

1. 计算 `blockLength`、`tileLength` 和循环次数；
2. 根据 `GetBlockIdx()` 设置本核的 GlobalTensor 起始地址；
3. 通过 `pipe.InitBuffer` 为三个队列分配片上缓冲。

### Process

```cpp
for (int i = 0; i < tileNum * BUFFER_NUM; ++i) {
    CopyIn(i);
    Compute(i);
    CopyOut(i);
}
```

### CopyIn

```cpp
auto xLocal = inQueueX.AllocTensor<float>();
auto yLocal = inQueueY.AllocTensor<float>();
AscendC::DataCopy(xLocal, xGm[i * tileLength], tileLength);
AscendC::DataCopy(yLocal, yGm[i * tileLength], tileLength);
inQueueX.EnQue(xLocal);
inQueueY.EnQue(yLocal);
```

### Compute

```cpp
auto xLocal = inQueueX.DeQue<float>();
auto yLocal = inQueueY.DeQue<float>();
auto zLocal = outQueueZ.AllocTensor<float>();
AscendC::Add(zLocal, xLocal, yLocal, tileLength);
outQueueZ.EnQue(zLocal);
inQueueX.FreeTensor(xLocal);
inQueueY.FreeTensor(yLocal);
```

### CopyOut

```cpp
auto zLocal = outQueueZ.DeQue<float>();
AscendC::DataCopy(zGm[i * tileLength], zLocal, tileLength);
outQueueZ.FreeTensor(zLocal);
```

## 核函数入口

```cpp
__global__ __vector__ void add_custom(
    GM_ADDR x, GM_ADDR y, GM_ADDR z,
    AddCustomTilingData tiling)
{
    KernelAdd op;
    op.Init(x, y, z, tiling.totalLength, tiling.tileNum);
    op.Process();
}
```

入口应保持轻量：创建算子对象、初始化、执行。复杂逻辑放在算子类中更容易复用和检查。

## Host 侧调用

带数据的直调流程：

1. 初始化 AscendCL、Device 和 Stream；
2. 计算总字节数；
3. 分配 Host 输出及 Device 输入输出内存；
4. 把 `x/y` 从 Host 拷到 Device；
5. 构造 TilingData 并启动 Kernel；
6. 同步 Stream；
7. 把 `z` 拷回 Host；
8. 与 CPU Golden 结果比较；
9. 释放所有资源。

浮点验证不宜长期使用严格的 `std::equal`。当算子包含非线性函数、不同精度或运算顺序时，应采用绝对/相对误差容忍。

## 构建交付

最小工程包含：

- `add_custom.asc`：Host 与 Device 混合源码；
- `CMakeLists.txt`：声明 ASC/CXX 双语言项目并设置 NPU 架构。

```cmake
find_package(ASC REQUIRED)
project(kernel_samples LANGUAGES ASC CXX)
add_executable(demo add_custom.asc)
target_compile_options(demo PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>)
```

## 常见误区

1. **只切核，不设置 GlobalTensor 偏移。** 多核会重复处理同一段数据。
2. **循环次数仍写 `tileNum`。** 课程实现启用双缓冲后，实际块数是 `tileNum × BUFFER_NUM`。
3. **先 Free 再 EnQue。** 输出 Tensor 必须先交给下游，再在合适阶段释放。
4. **固定样例等于泛化实现。** 当前代码要求长度能被核数、Tile 和缓冲数整除。
5. **精度校验总能严格相等。** 浮点计算应使用合理容差。

## 一句话总结

Add 样例把 Ascend C 开发归纳成“规格分析 → 两级切分 → 三级流水 → Host 调用 → 精度验证”的完整路径。

## 复习卡片

- 数学逻辑：逐元素 `z=x+y`。
- 多核：`blockLength` 与 `GetBlockIdx()`。
- 核内：`tileLength` 与循环次数。
- 流水：CopyIn、Compute、CopyOut。
- 资源：TPipe 与三组 TQue。
- 验证：Device 输出对比 CPU Golden。

## 自测问题

1. TilingData 为什么由 Host 构造并传给 Kernel？
2. `blockLength × GetBlockIdx()` 解决什么问题？
3. 为什么课程样例的循环次数是 16 而不是 8？
4. 固定 Add 样例距离泛化算子还缺哪些能力？

<details>
<summary>参考答案</summary>

1. Host 掌握全局 shape、硬件和调度信息，可把切分策略与计算逻辑解耦。
2. 它为每个核定位不同的 Global Memory 数据区间。
3. 8 个 Tile 配合两份 Double Buffer，真正处理的块数为 `8×2`。
4. 需要处理任意 shape、dtype、核间余数、尾块和平台差异。

</details>

## 图片来源

本文图片取自 [CANN Learning Hub：基于 Add 算子的核函数介绍](https://gitcode.com/cann/cann-learning-hub/blob/master/tutorials/ascendc_operator_development_light/02_AscendC_basic/02.04_introduction_to_kernel_functions_based_on_add_operator.ipynb)，本地保存仅用于学习笔记。

