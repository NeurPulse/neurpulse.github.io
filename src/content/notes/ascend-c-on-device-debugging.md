---
title: "Ascend C 上板调试：DumpTensor 与 printf"
description: "掌握在 NPU Kernel 中观察张量、标量和多核切分参数的方法，并建立低干扰的调试流程。"
pubDate: 2026-08-27
category: "CANN / Ascend C"
tags: ["Ascend C", "NPU", "调试", "DumpTensor", "printf"]
order: 14
---

## 学习目标

完成本节后，应当能够：

- 根据问题类型选择 `DumpTensor` 或 `printf`；
- 在指定 AI Core、指定循环轮次打印必要信息；
- 理解 `desc`、`dumpSize` 和 `ShapeInfo` 的作用；
- 避免调试输出干扰性能测量或淹没有效信息。

## 前置知识

需要了解 Ascend C Kernel 的 `Init → Process → CopyIn → Compute → CopyOut` 结构，以及 `GlobalTensor`、`LocalTensor`、多核切分和 Tile 的基本含义。

## 两种打印接口如何选择

| 需求 | 推荐接口 | 典型用途 |
| --- | --- | --- |
| 查看一段张量的数据 | `DumpTensor` | 检查搬入数据、计算结果、数据类型和存储位置 |
| 查看标量、索引或执行标记 | `printf` | 检查核号、循环轮次、Tile 长度和地址 |

两者都应写在 Kernel 侧。它们适合回答不同问题：

- “这一块数据到底是什么？”——使用 `DumpTensor`；
- “代码执行到哪里、当前参数是多少？”——使用 `printf`。

## DumpTensor：观察张量

常用接口形式如下：

```cpp
template <typename T>
__aicore__ inline void DumpTensor(
    const LocalTensor<T>& tensor,
    uint32_t desc,
    uint32_t dumpSize);
```

也可以传入 `GlobalTensor`，或增加 `ShapeInfo` 按指定形状展示。

| 参数 | 含义 | 使用建议 |
| --- | --- | --- |
| `tensor` | 待观察的张量 | 明确它位于 GM、UB、L1 还是 L0C |
| `desc` | 调试点标识 | 可传 `__LINE__`，便于反查源码位置 |
| `dumpSize` | 实际读取并打印的元素数 | 尽量小，只打印能证明问题的数据 |
| `shapeInfo` | 输出时使用的形状 | 只影响组织和显示，不会凭空增加有效数据 |

例如，在第一次 `Compute` 中分别查看两个输入和输出的前 8 个元素：

```cpp
if (progress == 0) {
    AscendC::DumpTensor(xLocal, __LINE__, 8);
    AscendC::DumpTensor(yLocal, __LINE__, 8);
}

AscendC::Add(zLocal, xLocal, yLocal, tileLength);

if (progress == 0) {
    AscendC::DumpTensor(zLocal, __LINE__, 8);
}
```

若 `ShapeInfo` 描述的是 `(2, 8)`，但 `dumpSize` 只有 8，第二行没有对应数据，输出会用 `-` 补齐。这里容易误解：形状是展示规则，`dumpSize` 才是实际打印数量。

## printf：观察控制信息

`AscendC::printf` 的用法接近 C/C++ `printf`：

```cpp
AscendC::printf(
    "totalLength: %d, tileNum: %d, blockNum: %lu\n",
    totalLength,
    tileNum,
    AscendC::GetBlockNum());
```

课程涉及的主要转换说明包括：

| 转换说明 | 用途 |
| --- | --- |
| `%d` / `%i` | 有符号整数 |
| `%u` | 无符号整数 |
| `%x` | 十六进制整数 |
| `%f` | `float`、`half`、`bfloat16_t` |
| `%s` | 字符串 |
| `%p` | 指针地址 |

格式说明必须与参数类型匹配。核数、核号、长度等变量可能是无符号或更宽的整数，不应机械套用 `%d`。

## 多核环境中的精准打印

如果每个核、每轮循环都打印，日志量会迅速膨胀。更有效的方式是先限定核，再限定轮次。例如，课程练习要求只在逻辑核 2 的最后一次 `Compute` 打印：

```cpp
const int32_t lastProgress = tileNum * BUFFER_NUM - 1;
if (AscendC::GetBlockIdx() == 2 && progress == lastProgress) {
    // 在这里调用带 ShapeInfo 的 DumpTensor
}
```

这里的关键不是背条件，而是建立两个坐标：

1. 空间坐标：由哪个 AI Core 处理；
2. 时间坐标：该核执行到了第几个 Tile。

## 推荐调试流程

1. 先确认 Host 侧输入、Tiling 参数和启动核数；
2. 在 `Init` 打印核号、单核长度和 Tile 长度；
3. 在 `CopyIn` 后检查输入张量；
4. 在 `Compute` 后检查局部结果；
5. 只打印一个核的一到两个 Tile；
6. 精度正确后关闭打印，再进行性能采集。

## 常见误区

- **打印越多越容易定位问题**：日志过多反而会隐藏第一个异常点。
- **带 Shape 的打印会读取整个 Shape**：实际读取数量仍受 `dumpSize` 限制。
- **调试输出不影响性能**：`printf` 和 `DumpTensor` 可能引入大量 Scalar 开销，不能带着它们测正式性能。
- **只看输出结果即可**：最终结果错误时，应沿 `CopyIn → Compute → CopyOut` 逐段定位，而不是只在末尾打印。
- **浮点结果总能严格相等**：一般场景应使用绝对或相对误差；课程 Add 常量样例的严格比较不宜直接推广。

正式性能构建可通过编译定义关闭调试输出：

```cmake
target_compile_definitions(demo PRIVATE ASCENDC_DUMP=0)
```

## 一句话总结

上板调试的核心不是“多打印”，而是在正确的核、正确的 Tile、正确的数据阶段打印最少且足够的信息。

## 复习卡片

| 问题 | 答案 |
| --- | --- |
| 看张量内容用什么？ | `DumpTensor` |
| 看核号和 Tile 参数用什么？ | `AscendC::printf` |
| `desc` 常用什么值？ | `__LINE__` 或自定义调试点编号 |
| 如何限制到某个逻辑核？ | 判断 `AscendC::GetBlockIdx()` |
| 性能采集前要做什么？ | 关闭或编译屏蔽调试打印 |

## 自测题

1. 为什么只判断 `progress == 0` 仍可能产生多组日志？
2. `ShapeInfo=(2,8)`、`dumpSize=8` 时，为什么只能看到一行有效数据？
3. 若输入正确但输出错误，下一步最适合在哪个位置打印？
4. 为什么不能用带 `printf` 的结果作为最终性能结论？

<details>
<summary>参考答案</summary>

1. 因为每个参与计算的 AI Core 都可能执行自己的 `progress == 0`。
2. `ShapeInfo` 只规定展示形状，实际打印数量仍为 8。
3. 先在计算接口之后、入输出队列之前检查局部结果。
4. 打印本身会产生额外指令和同步开销，尤其可能放大 Scalar 耗时。

</details>

## 内容来源

- 课程仓库：[CANN LearningHub](https://gitcode.com/cann/cann-learning-hub/tree/master)
- 参考 Notebook：`04.02_NPU_On-Board_Debugging.ipynb`
- 示例源码：课程内的 `src/04.02/add_custom.asc`

