---
title: 'Ascend C SIMT 入门 1：用 HelloWorld 读懂线程编号'
description: '从两组线程的打印结果入手，读懂 Grid、Block、Thread 和 SIMT 的四个启动参数。'
pubDate: 2026-11-09
category: 'CANN / Ascend C'
tags: ['Ascend C', 'SIMT', 'HelloWorld', 'Kernel']
order: 4
---

## 学习目标

Host 用 `hello_world<<<2, 32, 0, nullptr>>>()` 提交了一次任务。设备上会执行多少个线程？为什么程序只打印 6 行？先把线程的编号和实际输出对上，下一篇 Gather 就能用同一个编号给每个线程分配一个输出位置。

能读懂 `if` 和循环即可开始。想回顾 CPU 怎样调用 NPU，可以先看 [SIMD HelloWorld](/notes/ascend-c-kernel-and-spmd-basics/)。

## 1. 先把线程分成两组

SIMT 是 **Single Instruction, Multiple Threads（单指令多线程）**。写代码时，可以站在单个线程的视角想问题。每个线程从同一份核函数出发，有自己的编号，也可以根据编号决定走哪个分支。硬件把线程分组执行。

本例是一维任务，只看 `.x`。可以把 Grid 想成一份点名册，里面有两个小组，每个小组有 32 人。

| 层次 | 本例中的含义 |
|---|---|
| Grid（线程块网格） | 一次核函数启动所建立的全部线程块 |
| Thread Block（线程块） | 网格中的一组线程。块号是 0 或 1 |
| Thread（线程） | 一份核函数的执行实例。在每块内部从 0 编到 31 |

所以本次启动的**逻辑线程数是 64**。块 0 有一个局部编号为 2 的线程，块 1 也有一个。要指清其中一个，需要同时报块号和块内线程号。 [官方线程架构](https://asc.gitcode.com/guide/programming_guide/programming_model/ai_core_simt_programming/thread_architecture.html)

### 先猜哪几行会打印

规则是每块只让局部编号小于 3 的线程打印。下面的普通 Python 代码依次枚举两个线程块，方便在电脑上先算一遍。

```python
grid_size = 2
block_size = 32
printed = []

for block_id in range(grid_size):
    for thread_id in range(block_size):
        if thread_id < 3:
            printed.append((block_id, thread_id))

assert printed == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
print(printed)
```

Python 列表按循环的顺序排列。NPU 上看到的日志顺序取决于执行与输出的时机，读日志时按两个编号识别线程即可。

## 2. 读官方完整代码

下面是官方样例中的 `hello_world.asc`。先从 Host 上的 `main` 找到启动语句，再回到上方的 `hello_world` 看每个线程执行什么。 [官方教程](https://asc.gitcode.com/guide/getting_started/quick_start/simt_programming/hello_world.html) · [官方样例源码](https://gitcode.com/cann/asc-devkit/blob/e6dc42b66218e322567c26aa71b00bb39adbb79c/examples/03_simt_api/00_introduction/00_quickstart/hello_world_simt/hello_world.asc)

```cpp
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file hello_world.asc
 * \brief Hello World SIMT sample.
 */

#include "acl/acl.h"
#include "simt_api/asc_simt.h"
#include "utils/debug/asc_printf.h"

__global__ void hello_world()
{
    if (threadIdx.x < 3) {
        printf("[blockIdx (%u/%u)][threadIdx (%u/%u)]: Hello World!\n", blockIdx.x, gridDim.x, threadIdx.x, blockDim.x);
    }
}

int32_t main(int argc, char const* argv[])
{
    aclInit(nullptr);
    aclrtSetDevice(0);
    hello_world<<<2, 32, 0, nullptr>>>();
    aclrtSynchronizeDevice();
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
```

### Device 端怎样知道自己是哪个线程

`__global__` 把 `hello_world` 声明为 Host 可启动的设备核函数。这里由 SIMT 编译选项启用 SIMT 场景。核函数用到四个内置变量。它们在启动之后由运行环境提供，代码可以直接读取。

| 表达式 | 读法 | 本次启动的值 |
|---|---|---|
| `gridDim.x` | Grid 的 x 方向有多少个线程块 | 2 |
| `blockIdx.x` | 我在哪个线程块 | 0 或 1 |
| `blockDim.x` | 每块在 x 方向有多少个线程 | 32 |
| `threadIdx.x` | 我在这块中的编号 | 0 到 31 |

`if (threadIdx.x < 3)` 每个线程都判断一次。编号为 3 的线程也执行了核函数，只是条件为假，跳过了打印。以块 1 内的线程 2 为例，它打印 `[blockIdx (1/2)][threadIdx (2/32)]: Hello World!`。括号里的斜杠分隔的是**编号和总数**。 [官方内置变量说明](https://asc.gitcode.com/guide/programming_guide/programming_model/ai_core_simt_programming/kernel_function.html)

### Host 端怎样启动、等待和收尾

`hello_world<<<2, 32, 0, nullptr>>>()` 的小括号为空，因为核函数无需从 Host 接收数组。尖括号中的四项才是启动配置。

| 配置 | 本例 | 要解决的问题 |
|---|---|---|
| `blocks_per_grid` | `2` | 一共启动几块？ |
| `threads_per_block` | `32` | 每块有几个线程？ |
| `dyn_ubuf_size` | `0` | 为每个线程块动态申请多少字节的共享 UB？ |
| `stream` | `nullptr` | 把任务提交给哪个执行流？本例使用默认 Stream |

之前 SIMD HelloWorld 的三项配置是 `<<<numBlocks, dynUBufSize, stream>>>`。这一维 SIMT 样例在第 2 项显式指定**每块的线程数**，动态 UB 的字节数顺延到第 3 项。 [SIMD 启动配置](https://asc.gitcode.com/guide/programming_guide/language_extension/simd_builtin_keywords.html) · [SIMT 启动配置](https://asc.gitcode.com/guide/programming_guide/programming_model/ai_core_simt_programming/kernel_function.html)

从 `main` 往下读，`aclInit` 初始化运行时，`aclrtSetDevice(0)` 选择 0 号设备。核函数提交后，CPU 调用 `aclrtSynchronizeDevice()` 等设备任务结束，然后才通过 `aclrtResetDevice(0)` 和 `aclFinalize()` 释放运行资源。

## 3. 从一对编号算出全局编号

把每块的局部编号接到一起。块 0 贡献 0 到 31，块 1 接着贡献 32 到 63。对块 1 内的线程 2，前面有一整块 32 个线程，所以它的全局编号是 `1 × 32 + 2 = 34`。

```text
全局线程编号 = 块编号 × 每块线程数 + 块内线程编号
```

HelloWorld 把两个局部信息分开打印。下一篇会把它们合成一个编号，用来决定当前线程写 `output` 的哪个位置。 [SIMT Gather 快速入门](/notes/ascend-c-simt-gather-quick-start/)

## 4. 编译和观察输出

这两份官方 SIMT 快速入门样例面向 **Ascend 950PR／950DT（dav-3510）**，样例说明要求 CANN 9.1.0 或以上。先按 [环境准备](/notes/ascend-c-quick-start-environment/) 加载对应工具链。 [官方 HelloWorld 工程说明](https://gitcode.com/cann/asc-devkit/blob/e6dc42b66218e322567c26aa71b00bb39adbb79c/examples/03_simt_api/00_introduction/00_quickstart/hello_world_simt/README.md)

[下载 SIMT HelloWorld 和 Gather 的官方源码、CMake 配置及许可证](/downloads/ascend-c-simt-quick-start-source.zip)。解压后进入 `hello-world` 目录。官方教程用毕昇编译器直接编译，`--npu-arch` 选目标架构，`--enable-simt` 开启 SIMT。

```bash
bisheng hello_world.asc --npu-arch=dav-3510 -o demo --enable-simt
./demo
```

也可以用下载包里的 `CMakeLists.txt`。它已经设置了 `CMAKE_ASC_ENABLE_SIMT ON` 和默认架构。仍从 `hello-world` 目录运行下面的命令。

```bash
cmake -S . -B build -DCMAKE_ASC_ARCHITECTURES=dav-3510
cmake --build build -j
./build/demo
```

预期看到 6 条 Hello World 日志，块 0 和块 1 各有线程 0、1、2 的一条。修改打印条件为 `threadIdx.x < 1` 再重新编译，每个块就只剩线程 0 打印。

## 自测

1. `<<<2, 32, 0, nullptr>>>` 有几个线程块，每块几个线程？
2. `threadIdx.x == 2` 这个条件总共命中几个线程？块 1 内该线程的全局编号是多少？
3. 如果把打印条件换成 `blockIdx.x == 0`，会打印几行？

<details>
<summary>参考答案</summary>

1. 2 个线程块，每块 32 个线程，合计 64 个逻辑线程。
2. 每块各命中一个线程，一共两个。块 1 的该线程编号是 `1 × 32 + 2 = 34`。
3. 块 0 的全部 32 个线程都满足条件，所以会打印 32 行。

</details>

## 伪代码

```text
Host 选择 NPU，准备运行时
Host 提交核函数，指定 2 个线程块，每块 32 个线程

设备上的每个线程各自执行
    读取自己的块编号与块内线程编号
    如果块内编号小于 3
        打印块编号、块数、块内编号和每块线程数

Host 等设备任务完成
Host 释放运行资源
```
