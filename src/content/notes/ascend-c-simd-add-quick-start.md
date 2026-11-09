---
title: 'Ascend C SIMD Add 快速入门：从手算到两种官方实现'
description: '用八个数字读懂 Block 分工、GM 与 UB，再逐段阅读 C API、C++ Tensor 和 Host 源码，最后还原为伪代码。'
pubDate: 2026-11-09
category: 'CANN / Ascend C'
tags: ['Ascend C', 'SIMD', 'Add', 'Kernel', 'DataCopy']
order: 4
---

## 学习目标

`z[i] = x[i] + y[i]` 在 CPU 上一个循环就能写完。把它交给 NPU 时，需要回答三个新问题。哪个 Block 算第 `i` 个数？输入从哪里读，结果写到哪里？什么时刻可以读结果？本文先用手算回答这些问题，再在官方 C API 和 C++ Tensor 源码中找到对应的语句。

会读数组和循环就可以跟着做。如果还没看过 `<<<...>>>`，先读 [HelloWorld 与核函数启动](/notes/ascend-c-kernel-and-spmd-basics/)。要在设备上运行时，可按 [环境准备](/notes/ascend-c-quick-start-environment/) 配好工具链。

## 1. 先拿八个数字手算

设输入是 `x = [1, 2, 3, 4, 5, 6, 7, 8]`、`y = [10, 20, 30, 40, 50, 60, 70, 80]`。每个位置独立相加，结果是 `[11, 22, 33, 44, 55, 66, 77, 88]`。

让两个 Block 分工，每块处理 4 个元素。从下标 0 开始，分工如下。

| Block | 计算起点 | 负责的下标 | 算出的结果 |
|---|---|---|---|
| 0 | `0 × 4 = 0` | 0、1、2、3 | 11、22、33、44 |
| 1 | `1 × 4 = 4` | 4、5、6、7 | 55、66、77、88 |

以 Block 1 为例，输入片段是 `[5, 6, 7, 8]` 和 `[50, 60, 70, 80]`。块内下标 `1` 对应全局下标 `4 + 1 = 5`。这个位置算出 `6 + 60 = 66`，所以要写入完整结果的 `z[5]`。

### 先在普通电脑上复现分工

把每个 Block 对应的一段数据切出来，算完再放回原位置。这个 Python 例子串行模拟两块数据的读写，运行它只需要 Python。真正的 NPU 样例还要遵守搬运对齐、片上容量和同步要求。

```python
x = [1, 2, 3, 4, 5, 6, 7, 8]
y = [10, 20, 30, 40, 50, 60, 70, 80]
num_blocks = 2

assert len(x) == len(y) and len(x) % num_blocks == 0
block_length = len(x) // num_blocks
z = [0] * len(x)

for block_idx in range(num_blocks):
    begin = block_idx * block_length
    end = begin + block_length

    x_local = x[begin:end]
    y_local = y[begin:end]
    z_local = [x_local[j] + y_local[j] for j in range(block_length)]
    z[begin:end] = z_local
    print(f"Block {block_idx} [{begin}, {end}): {z_local}")

assert z == [11, 22, 33, 44, 55, 66, 77, 88]
print(z)
```

`[begin, end)` 包含开头，止于 `end - 1`。因此 Block 0 用完 `0～3`，Block 1 恰好从 `4` 接着算。同一份代码用不同的 Block 编号处理不同数据段，这是 **SPMD**。每个 Block 内调用向量加法处理一批元素，这是这里用到的 **SIMD**。 [编程模型概述](https://asc.gitcode.com/guide/programming_guide/programming_model/programming_model_overview.html)

## 2. 把手算的位置换成真实内存

在 Host 上，`x` 和 `y` 是 CPU 能读的数组。Host 先把它们复制到设备的 **GM（Global Memory，全局内存）**。每个 Block 在 GM 中找到自己的那一段，把数据搬到 **UB（Unified Buffer，向量核的片上缓冲区）**。向量加法读 UB 中的两段输入，也把输出放在 UB。最后由设备把输出写回 GM，再由 Host 拷回。

![Add 的两层搬运。Host 与 GM 之间复制，GM 与 UB 之间由核函数搬运](/images/cann/add-quick-start-flow.svg)

将视线放到一个 Block 上，可以数出三块局部缓冲区。两块装 `x` 和 `y`，第三块装 `z`。如果计算先于输入搬完，就可能读到尚未准备好的数据。如果搬出先于加法做完，写回 GM 的也可能是错误结果。因此官方代码在各阶段之间安排等待。 [官方 Add 算子快速入门](https://asc.gitcode.com/guide/getting_started/quick_start/simd_programming/add_op_quick_start.html)

### 官方样例有多少数据

这里的两份官方样例都用 8 个 Block，每块处理 2048 个 `float`，每个完整数组共 16384 个元素。前面 8 个元素的例子只用于追踪下标。下面的源码使用官方尺寸。

| 要计算的量 | 样例结果 | 在哪里用 |
|---|---|---|
| 一个 Block 的起点 | `block_idx * 2048` 个元素 | 对三个 GM 指针加同样的偏移 |
| 单块一个数组 | `2048 × 4 = 8192` 字节 | C API 的一次搬运，或一块局部缓冲的数据大小 |
| 单块三份局部缓冲 | `3 × 8192 = 24576` 字节 | 估算本例三块 UB 缓冲的数据量 |
| 一个完整数组 | `16384 × 4 = 65536` 字节 | Host 侧为一个设备数组分配的空间和整数组拷贝长度 |

例如 Block 1 执行 `x + 2048`。因为 `x` 是 `float*`，C++ 指针加法会自动跨过 2048 个 `float`，地址实际前进 8192 字节。这个偏移后面会同时出现在两种实现里。 [C API 官方源码](https://gitcode.com/cann/asc-devkit/blob/e6dc42b66218e322567c26aa71b00bb39adbb79c/examples/02_simd_c_api/00_introduction/01_add/c_api_async_add/c_api_add.asc) · [C++ 官方源码](https://gitcode.com/cann/asc-devkit/blob/e6dc42b66218e322567c26aa71b00bb39adbb79c/examples/01_simd_cpp_api/00_introduction/01_add/add/add.asc)

## 3. 读 C API 的官方核函数

先在代码里找到三个动词 `copy_gm2ub`、`add` 和 `copy_ub2gm`。它们对应搬入、计算、搬出。下面节选官方 `c_api_add.asc` 中的常量与完整核函数。 [源码和 CMake 工程下载](/downloads/ascend-c-quick-start-source.zip)

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

constexpr uint32_t TOTAL_LENGTH = 8 * 2048;
constexpr uint32_t NUM_BLOCKS = 8;

__vector__ __global__ void add_custom(__gm__ float* x, __gm__ float* y, __gm__ float* z)
{
    asc_init();

    constexpr uint32_t block_length = TOTAL_LENGTH / NUM_BLOCKS;

    __gm__ float* x_gm = x + block_idx * block_length;
    __gm__ float* y_gm = y + block_idx * block_length;
    __gm__ float* z_gm = z + block_idx * block_length;

    __ubuf__ float x_local[block_length];
    __ubuf__ float y_local[block_length];
    __ubuf__ float z_local[block_length];

    asc_copy_gm2ub(x_local, x_gm, block_length * sizeof(float));
    asc_copy_gm2ub(y_local, y_gm, block_length * sizeof(float));
    asc_sync();

    asc_add(z_local, x_local, y_local, block_length);
    asc_sync();

    asc_copy_ub2gm(z_gm, z_local, block_length * sizeof(float));
    asc_sync();
}
```

1. **定位 GM 片段。** `asc_init()` 初始化核状态。设当前是 Block 1，`x_gm` 就是 `x + 2048`，`y_gm`、`z_gm` 同理。它们是片段首地址。
2. **预留 UB 空间。** `__ubuf__` 说明这些数组放在片上 UB。每个 Block 的 `x_local`、`y_local` 和 `z_local` 长度都为 `block_length`。
3. **搬入和计算。** 两次 `asc_copy_gm2ub` 的第一个参数是目的地址，第二个是源地址，第三个是要搬的**字节数**。第一次 `asc_sync()` 后，`asc_add` 才能读两个局部数组。加法的最后一个参数是**元素数**，结果写入 `z_local`。
4. **搬出。** 第二次 `asc_sync()` 等计算完成。随后 `asc_copy_ub2gm(z_gm, z_local, ...)` 把结果写到 GM 的 `z_gm`，末尾再等待写回完成。

这一版把总长度和 Block 数写成常量，核内的 `block_length` 由它们相除得到。完整工程在 Host 侧用 `add_custom<<<num_blocks, 0>>>(x_device, y_device, z_device)` 提交到默认 Stream，再用 `aclrtSynchronizeDevice()` 等待。 [C API 官方源码](https://gitcode.com/cann/asc-devkit/blob/e6dc42b66218e322567c26aa71b00bb39adbb79c/examples/02_simd_c_api/00_introduction/01_add/c_api_async_add/c_api_add.asc)

## 4. 读 C++ Tensor 的官方核函数

同一段输入可以用一个对象来描述。`GlobalTensor<float>` 把 GM 指针和元素长度放在一起。分配出来的 `LocalTensor<float>` 描述片上的缓冲区。这样 `DataCopy` 看到源、目的的类型，就知道是在做 GM 到 UB，还是 UB 到 GM。 [C++ Tensor 编程概述](https://asc.gitcode.com/guide/programming_guide/programming_model/ai_core_simd_programming/cpp_tensor_programming/cpp_tensor_programming_overview.html)

下面是官方 `add.asc` 的完整核函数。

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

template <uint32_t blockLength>
__vector__ __global__ void add_custom(__gm__ float* x, __gm__ float* y, __gm__ float* z)
{
    AscendC::InitSocState();

    AscendC::GlobalTensor<float> xGm, yGm, zGm;
    xGm.SetGlobalBuffer(x + block_idx * blockLength, blockLength);
    yGm.SetGlobalBuffer(y + block_idx * blockLength, blockLength);
    zGm.SetGlobalBuffer(z + block_idx * blockLength, blockLength);

    AscendC::LocalMemAllocator<AscendC::Hardware::UB> ubAllocator;
    AscendC::LocalTensor<float> xLocal = ubAllocator.Alloc<float, blockLength>();
    AscendC::LocalTensor<float> yLocal = ubAllocator.Alloc<float, blockLength>();
    AscendC::LocalTensor<float> zLocal = ubAllocator.Alloc<float, blockLength>();

    AscendC::DataCopy(xLocal, xGm, blockLength);
    AscendC::DataCopy(yLocal, yGm, blockLength);
    AscendC::PipeBarrier<PIPE_ALL>();

    AscendC::Add(zLocal, xLocal, yLocal, blockLength);
    AscendC::PipeBarrier<PIPE_ALL>();

    AscendC::DataCopy(zGm, zLocal, blockLength);
    AscendC::PipeBarrier<PIPE_ALL>();
}
```

1. **从模板参数拿到块长度。** `template <uint32_t blockLength>` 接收一个编译期整数。Host 使用 `add_custom<2048>` 的效果是让核函数按每块 2048 个元素生成这一版代码。
2. **建立 GM 视图。** `xGm.SetGlobalBuffer(x + block_idx * blockLength, blockLength)` 记录片段的首地址和元素长度。设置视图后，数据还在原来的 GM 中。下一步的 `DataCopy` 才会移动它。
3. **分配局部 Tensor。** `ubAllocator.Alloc<float, blockLength>()` 为每个输入和输出各准备一块 UB。例中每个块的局部长度在编译期已知，启动语句里动态 UB 的值因此可以填 `0`。
4. **用 API 串起三阶段。** `DataCopy(xLocal, xGm, blockLength)` 是从 GM 搬入，`Add(zLocal, xLocal, yLocal, blockLength)` 是计算，`DataCopy(zGm, zLocal, blockLength)` 是从 UB 搬回 GM。三次 `PipeBarrier<PIPE_ALL>()` 对应本例的三处核内等待。

同样是 2048 个 `float`，C API 的连续搬运长度是 `8192` **字节**，这里的 C++ `DataCopy` 长度是 `2048` **个元素**。把 C++ 那行误写成 `8192`，接口会按 8192 个元素解释，已经超过样例为该块分配的空间。 [C API 搬运参数](https://asc.gitcode.com/api/SIMD-API/c_api/vector_datamove/asc_copy_gm2ub/asc_copy_gm2ub_arch_2201.html) · [C++ DataCopy 连续搬运](https://asc.gitcode.com/api/SIMD-API/basic_api/data_move_guide/overview/data_move_concept.html)

## 5. 读 Host 怎样把核函数跑起来

上面的两个核函数都从 GM 取输入。是谁把输入放到 GM 的？看官方 C++ 工程的 Host 函数 `kernel_add`。它收到 CPU 侧的 `x` 和 `y`，最后把 CPU 可以使用的 `std::vector<float>` 返回给 `main`。 [C++ 官方源码](https://gitcode.com/cann/asc-devkit/blob/e6dc42b66218e322567c26aa71b00bb39adbb79c/examples/01_simd_cpp_api/00_introduction/01_add/add/add.asc)

```cpp
std::vector<float> kernel_add(std::vector<float>& x, std::vector<float>& y)
{
    constexpr uint32_t numBlocks = 8;
    constexpr uint32_t blockLength = 2048;
    uint32_t totalLength = x.size();
    size_t totalByteSize = totalLength * sizeof(float);
    float* xDevice = nullptr;
    float* yDevice = nullptr;
    float* zDevice = nullptr;
    uint8_t* zHost = nullptr;

    aclInit(nullptr);
    int32_t deviceId = 0;
    aclrtSetDevice(deviceId);
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);

    aclrtMalloc((void**)&xDevice, totalByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&yDevice, totalByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&zDevice, totalByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMallocHost((void**)&zHost, totalByteSize);

    aclrtMemcpy(xDevice, totalByteSize, x.data(), totalByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(yDevice, totalByteSize, y.data(), totalByteSize, ACL_MEMCPY_HOST_TO_DEVICE);

    add_custom<blockLength><<<numBlocks, 0, stream>>>(xDevice, yDevice, zDevice);
    aclrtSynchronizeStream(stream);

    aclrtMemcpy(zHost, totalByteSize, zDevice, totalByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    std::vector<float> z((float*)zHost, (float*)(zHost + totalByteSize));

    aclrtFree(xDevice);
    aclrtFree(yDevice);
    aclrtFree(zDevice);
    aclrtFreeHost(zHost);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return z;
}
```

| 读到的代码 | 应该想到的数据位置或动作 |
|---|---|
| `x.data()`、`y.data()` | 取 Host 上两个输入数组的首地址 |
| `aclrtMalloc((void**)&xDevice, ...)` | 在设备 GM 上申请空间，把得到的设备地址写进变量 `xDevice`。`&xDevice` 是这个指针变量自身的地址 |
| `aclrtMallocHost((void**)&zHost, ...)` | 在主机内存申请接收结果的空间。此时 `zDevice` 和 `zHost` 位于两侧 |
| `aclrtMemcpy(dst, dstMax, src, count, kind)` | 复制 `count` 字节，从 `src` 写到 `dst`。第二项是目的缓冲区容量，最后一项表明方向 |
| `add_custom<blockLength><<<numBlocks, 0, stream>>>(...)` | 每块处理 2048 个元素，启动 8 个 Block，动态 UB 配置为 0，任务提交给 `stream` |
| `aclrtSynchronizeStream(stream)` | CPU 在这里等待指定 Stream。等核函数写完 GM，才能开始下一行设备到主机的复制 |

这段代码里，小括号里的三个 Device 指针在核函数中成为 `x`、`y`、`z`。核函数虽返回 `void`，却会通过 `z` 指针把每块结果写入 `zDevice`。随后 Host 把结果拷进 `zHost`，再复制到 `std::vector<float> z`。向量有自己的存储，返回之前可以释放原来的缓冲区。

**两层等待各自解决一个问题。** 核内屏障保证本 Block 的搬入、计算、搬出按依赖衔接。Host 的 Stream 等待保证整个核函数提交的计算结束，Host 才去取结果。只看最终的一次 Host 等待，无法替代核内的数据依赖。 [Add 快速入门中的同步位置](https://asc.gitcode.com/guide/getting_started/quick_start/simd_programming/add_op_quick_start.html)

## 6. 跑官方工程，看什么算通过

[下载 HelloWorld、C API Add、C++ Add 原始源码与许可证](/downloads/ascend-c-quick-start-source.zip)。加载 CANN 环境后，从解压后的目录运行下面的命令。这组命令以 Atlas A2/A3 的 `dav-2201` 为例。其他设备先按 [环境准备](/notes/ascend-c-quick-start-environment/) 检查样例支持范围。

```bash
cmake -S c-add -B c-add/build -DCMAKE_ASC_ARCHITECTURES=dav-2201
cmake --build c-add/build -j
./c-add/build/c_api_add_example

cmake -S cpp-add -B cpp-add/build -DCMAKE_ASC_ARCHITECTURES=dav-2201
cmake --build cpp-add/build -j
./cpp-add/build/demo
```

`CMakeLists.txt` 用 `find_package(ASC REQUIRED)` 找到 CANN 提供的工具链，为 `.asc` 文件启用 ASC 编译。Host 的 `main` 生成两组输入，用设备跑出 `output`，再用 CPU 逐个算出 `golden[i] = x[i] + y[i]` 来比对。C API 工程校验通过会输出 `[Success] Case accuracy is verification passed.`，C++ 工程输出 `test pass!`。 [C API 工程](https://gitcode.com/cann/asc-devkit/tree/e6dc42b66218e322567c26aa71b00bb39adbb79c/examples/02_simd_c_api/00_introduction/01_add/c_api_async_add) · [C++ 工程](https://gitcode.com/cann/asc-devkit/tree/e6dc42b66218e322567c26aa71b00bb39adbb79c/examples/01_simd_cpp_api/00_introduction/01_add/add)

## 7. 自测与下一步

1. 如果启动 4 个 Block，每块仍处理 2048 个元素，哪些下标得不到结果？
2. 要搬 2048 个 `float`，本文的 C API 搬运和 C++ `DataCopy` 各传什么长度？
3. 提交核函数后，Host 想从 `zHost` 读到新结果，要做哪两步？

<details>
<summary>参考答案</summary>

1. 只覆盖 `[0, 8192)`，所以漏掉 `[8192, 16384)`。
2. C API 传 `8192` 字节。C++ 这个重载传 `2048` 个元素。
3. 等待对应的设备任务完成，再用 Device-to-Host 方向的 `aclrtMemcpy` 把 `zDevice` 复制到 `zHost`。

</details>

手算例子每块只有 4 个 `float`，换成字节是 16。官方 C API 工程使用的 `dav-2201` 连续搬运接口要求长度是 32 字节的整数倍，样例的 8192 字节符合这个条件。实际输入有尾部时，还要给最后一块确定有效元素数，并选择符合地址和长度要求的搬运方式。数据放不进 UB 时，可以让一个 Block 分批处理，每一批叫一个 Tile。 [C API 连续搬运约束](https://asc.gitcode.com/api/SIMD-API/c_api/vector_datamove/asc_copy_gm2ub/asc_copy_gm2ub_arch_2201.html) · [核内切块与双缓冲](/notes/ascend-c-vector-programming-paradigm/) · [带 Tiling 的 Add 工程解析](/notes/ascend-c-add-operator-walkthrough/)

本篇的两个版本都以 UB 为输入和输出缓冲，属于 **Memory 矢量计算**。官方教程还列出了显式使用向量寄存器的 **Reg 矢量计算**样例。 [昇腾社区 Add 算子快速入门](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/programug/Ascendcopdevg/docs/zh/guide/getting_started/quick_start/simd_programming/add_op_quick_start.md) · [asc-devkit 同章文档](https://asc.gitcode.com/guide/getting_started/quick_start/simd_programming/add_op_quick_start.html)

沿官方入门路线继续往下，可以先用打印观察每个线程的编号，再让线程按索引取数。 [SIMT HelloWorld](/notes/ascend-c-simt-hello-world/) · [SIMT Gather](/notes/ascend-c-simt-gather-quick-start/)

## 伪代码

去掉 C API 和 C++ API 的拼写差异，两个实现都可以按下面的顺序重写。这里保留了关键的分工、数据所在位置和等待点。

```text
Host
    准备等长的 x 和 y，计算 CPU 参考结果 golden
    选 Block 数 B 和每块元素数 L，确认 B × L = 数组长度
    在 GM 申请 x_gm、y_gm、z_gm，把两份输入复制到设备
    提交 Add 核函数，启动 B 个 Block

每个 Block 并行执行
    begin = block_idx × L
    end = begin + L
    从 x_gm[begin:end] 和 y_gm[begin:end] 搬入两块 UB
    等输入搬完
    在 UB 中对两份输入逐元素相加，得到 z_ub
    等加法完成
    把 z_ub 写回 z_gm[begin:end]，等写回完成

Host
    等核函数完成，把 z_gm 复制到主机
    逐元素核对主机结果和 golden
    释放内存与运行资源
```
