---
title: 'Ascend C 核函数与 SPMD：从 Hello World 理解 Host—Device 调用'
description: '带着官方 HelloWorld 源码逐行走过选设备、启动 Block、等待与释放，读懂三尖括号里的每个参数。'
pubDate: 2026-08-27
updatedDate: 2026-11-09
category: 'CANN / Ascend C'
tags: ['Ascend C', 'Kernel', 'SPMD', 'AscendCL']
order: 4
---

## 学习目标

普通的 C++ 程序从 CPU 上的 `main` 开始。现在想让 NPU 打印一句话，CPU 要怎么把任务交出去，又怎样知道 NPU 已经做完？这篇用一个完整的程序回答这两个问题，下一篇再给同样的流程加上输入和输出。

## 先修知识

会读普通 C++ 函数即可。第一次看到 Host、Device、Block，可以先看 [异构系统与编程模型](/notes/cann-architecture-and-ascend-npu/)。准备动手时按 [环境准备](/notes/ascend-c-quick-start-environment/) 检查 NPU 和编译器。

## 先读官方完整代码

阅读时先找 `main`，然后顺着 `hello_world<<<...>>>()` 跳到上方的函数。前者在 Host（CPU）上执行，后者是 Device（NPU）的入口。

```cpp
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file hello_world.asc
 * \brief 快速入门示例，演示最基本的Ascend C内核启动和printf输出
 */

#include "utils/debug/asc_printf.h"
#include "acl/acl.h"

__global__ __vector__ void hello_world() { printf("Hello World!!!\n"); }

int main(int argc, char const* argv[])
{
    aclInit(nullptr);
    aclrtSetDevice(0); // Acquire runtime management resources.
    hello_world<<<8, 0, nullptr>>>();
    aclrtSynchronizeDevice();
    aclrtResetDevice(0); // Release runtime resources.
    aclFinalize();
    return 0;
}
```

源文件和 CMake 配置可一起下载。 [官方 HelloWorld 源码](https://gitcode.com/cann/asc-devkit/blob/e6dc42b66218e322567c26aa71b00bb39adbb79c/examples/01_simd_cpp_api/00_introduction/00_quickstart/hello_world/hello_world.asc) · [三份样例与许可证](/downloads/ascend-c-quick-start-source.zip)

## 核函数是什么

`__global__ __vector__ void hello_world()` 是一条函数声明。

| 关键字 | 读法 |
|---|---|
| `__global__` | 这是一个核函数入口，Host 能通过专用启动语法提交任务 |
| `__vector__` | 这个核函数安排到 Vector 核上执行 |
| `void` | 函数本身没有返回值 |
| `()` | 这个例子不需要传输入或输出参数 |

`printf` 写在核函数里，所以是由 NPU 执行的。源码开头的 `utils/debug/asc_printf.h` 提供设备侧打印接口，`acl/acl.h` 则为 Host 提供运行时接口。这也是一个 `.asc` 文件中可以同时看到两种代码的原因。 [官方 HelloWorld 教程](https://asc.gitcode.com/guide/getting_started/quick_start/simd_programming/hello_world.html)

## 核函数启动语法

完整源码的启动语句是 `hello_world<<<8, 0, nullptr>>>()`。三尖括号内是启动配置，小括号才用来传核函数的参数。

| 位置 | 本例 | 作用 |
|---|---|---|
| `numBlocks` | `8` | 提交 8 个逻辑 Block，让它们执行同一份核函数 |
| `dynUBufSize` | `0` | 动态 UB 空间的字节数。这个程序没有动态 UB 申请 |
| `stream` | `nullptr` | 把任务提交到默认 Stream。Stream 可以理解成设备任务的提交队列 |

沿用早期课程的笔记时，可能看到第二项写成 `l2ctrl`。本文使用这份官方快速入门所对应的 SIMD 启动语法，第二项读作 `dynUBufSize`。 [SIMD 启动语法](https://asc.gitcode.com/guide/programming_guide/language_extension/simd_builtin_keywords.html)

### 为什么会打印多次

每个 Block 都运行一次 `hello_world`，所以 8 个 Block 都会走到打印语句。把第一个配置值改成 `1`，就只会提交一个 Block。各 Block 的打印顺序由执行时机决定。真正做 Add 时，可以从内置的 `block_idx` 或 C++ 接口 `AscendC::GetBlockIdx()` 得到当前 Block 的编号，然后让 Block 0 处理下标 0～3、Block 1 处理 4～7。这个分工要在核函数里写明。 [SIMD 内置变量](https://asc.gitcode.com/guide/programming_guide/language_extension/simd_builtin_keywords.html)

![同一段核函数由多个 Block 执行，各自处理一段输入](/images/cann/spmd_parallel_computing_schematic.png)

## 提交以后为什么要等

顺着 `main` 往下读，可以把每个调用对应到一件事。

1. `aclInit(nullptr)` 初始化运行时。
2. `aclrtSetDevice(0)` 选 0 号设备，之后的核函数会提交给它。
3. `hello_world<<<8, 0, nullptr>>>()` 提交打印任务。CPU 可以在 NPU 完成前继续往下执行。
4. `aclrtSynchronizeDevice()` 让 CPU 等所选设备上已提交的任务完成。
5. `aclrtResetDevice(0)` 和 `aclFinalize()` 在任务结束后释放运行资源。

官方教程的另一个片段使用显式 Stream。下面是它在 Host 生命周期里的位置。把这一段放在 `aclrtSetDevice(0)` 之后、`aclrtResetDevice(0)` 之前，替换完整样例中的默认 Stream 启动和设备级等待即可。 [官方 HelloWorld 教程](https://asc.gitcode.com/guide/getting_started/quick_start/simd_programming/hello_world.html)

```cpp
aclrtStream stream = nullptr;
aclrtCreateStream(&stream);
constexpr uint32_t numBlocks = 8;
hello_world<<<numBlocks, 0, stream>>>();
aclrtSynchronizeStream(stream);
aclrtDestroyStream(stream);
```

这里 `aclrtSynchronizeStream(stream)` 等的是指定 Stream 上的任务。下一篇 C++ Add 会沿用它，确保 GM 中的结果准备好以后再拷回 Host。

## 编译和观察输出

加载 CANN 环境后，在源码所在目录运行官方命令。示例使用 `dav-2201`，目标值按实际 NPU 选。带 CMake 的编译方式见 [环境准备](/notes/ascend-c-quick-start-environment/)。

```bash
bisheng hello_world.asc --npu-arch=dav-2201 -o demo
./demo
```

留意设备输出中的核编号和 `Hello World!!!`。可以把启动的 Block 数改成 `1` 再观察。若能打印，说明 Host 下发、Device 执行、Host 等待这条路径已经跑通。

## 自测

1. 为什么在 `main` 里写一个普通的 `printf` 无法证明 NPU 执行过核函数？
2. `<<<8, 0, nullptr>>>` 里哪个值会改变核函数执行的实例数量？
3. 为什么等待应在释放运行资源之前？

<details>
<summary>参考答案</summary>

1. `main` 由 CPU 执行。把打印写在核函数里，才是在观察设备端执行。
2. 第一个值 `8`，它配置本次启动的 Block 数。
3. 提交是异步的，先等任务完成，后续释放才不会发生在设备仍然使用资源时。

</details>

图示来自 [CANN Learning Hub 的 HelloWorld 笔记](https://gitcode.com/cann/cann-learning-hub/blob/master/tutorials/ascendc_operator_development_light/02_AscendC_basic/02.02_HelloWorld.ipynb)。下一篇使用同一条启动路径来做逐元素加法。 [SIMD Add 快速入门](/notes/ascend-c-simd-add-quick-start/)

## 伪代码

```text
Host 初始化运行时并选择 NPU
Host 向默认 Stream 提交 hello_world，要求启动 8 个 Block
    每个 Block 执行同一个 hello_world，打印消息
Host 等待所选设备完成任务
Host 释放运行资源并结束
```
