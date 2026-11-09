---
title: 'Ascend C SIMT 入门 2：从手算到一维 Gather'
description: '用五个索引读懂 Gather，追踪每个线程怎样读索引、取输入和写输出，再对照官方 Kernel 与 Host。'
pubDate: 2026-11-09
category: 'CANN / Ascend C'
tags: ['Ascend C', 'SIMT', 'Gather', 'Kernel', '索引']
order: 4
---

## 学习目标

Gather 做的事像按清单取货。清单里每个整数告诉你去输入数组的哪个位置取数，取回来的值按清单顺序放入输出。把它写成 SIMT 核函数时，一个线程负责清单上的一项。读完本篇，你可以手算输出，解释线程编号和索引值的区别，并指出官方代码里哪一次检查保护了哪一块内存。

如果还不熟悉 `blockIdx.x`、`threadIdx.x` 或四项启动配置，先看 [SIMT HelloWorld](/notes/ascend-c-simt-hello-world/)。

## 1. 先按一张索引清单取数

输入是 `input = [10, 20, 30, 40]`，下标从 0 开始。索引清单是 `index = [2, 0, 2, 3, 1]`。用一行表示取数规则。

```text
output[i] = input[index[i]]
```

| 输出位置 `i` | 清单上的值 `index[i]` | 从输入取到的值 |
|---|---|---|
| 0 | 2 | `input[2] = 30` |
| 1 | 0 | `input[0] = 10` |
| 2 | 2 | `input[2] = 30` |
| 3 | 3 | `input[3] = 40` |
| 4 | 1 | `input[1] = 20` |

所以结果是 `[30, 10, 30, 40, 20]`。输出长 **5**，因为清单有 5 项。清单中的 2 出现两次，两个输出位置都取到 30。这里的 `i` 决定**写到哪里**，`index[i]` 决定**从哪里读**。 [官方 Gather 算子快速入门](https://asc.gitcode.com/guide/getting_started/quick_start/simt_programming/gather_op_quick_start.html)

![索引 2 出现两次时，两个线程从输入的同一个位置读取，分别写入输出位置 0 和 2](/images/cann/simt-gather-index.svg)

## 2. 给清单上的每一项分配一个线程

用一个可手算的配置演示。每块 4 个线程，要处理 5 项，向上取整需要 2 块，因此会启动 8 个逻辑线程。把块内编号换成全局编号后，分工如下。

```text
idx = 块编号 × 每块线程数 + 块内线程编号

块 0    idx = 0, 1, 2, 3      四个线程各写一个输出
块 1    idx = 4, 5, 6, 7      只有 idx = 4 有输出要写
```

块 1 的线程 0 算出 `idx = 1 × 4 + 0 = 4`。它先读 `index[4] = 1`，再读 `input[1] = 20`，最后写 `output[4] = 20`。块 1 的线程 1 算出 `idx = 5`，已经超过有效范围，应在读 `index[5]` 之前退出。

下面用 Python 串行模拟这些线程，4 是为了手算方便。实机仍按后面的官方配置运行。

```python
values = [10, 20, 30, 40]
indices = [2, 0, 2, 3, 1]
assert all(0 <= pos < len(values) for pos in indices)

count = len(indices)
threads_per_block = 4
blocks = (count + threads_per_block - 1) // threads_per_block
output = [None] * count
skipped = []

for block_id in range(blocks):
    for thread_id in range(threads_per_block):
        idx = block_id * threads_per_block + thread_id
        if idx >= count:
            skipped.append(idx)
            continue

        source_pos = indices[idx]
        output[idx] = values[source_pos]

assert skipped == [5, 6, 7]
assert output == [30, 10, 30, 40, 20]
print(output)
```

`(count + threads_per_block - 1) // threads_per_block` 是整数向上取整。最后一块哪怕只剩一项，也要为它启动一个线程块。核函数里的提前退出负责处理多出来的线程。 [官方线程索引与边界检查](https://asc.gitcode.com/guide/programming_guide/programming_model/ai_core_simt_programming/kernel_function.html)

## 3. 读官方 Device 核函数

下面是官方 `gather_1d.asc` 的完整核函数。源码和 Host 工程可以一起下载。 [官方源码](https://gitcode.com/cann/asc-devkit/blob/e6dc42b66218e322567c26aa71b00bb39adbb79c/examples/03_simt_api/00_introduction/01_gather/basic_gather/gather_1d/gather_1d.asc) · [SIMT 样例下载包](/downloads/ascend-c-simt-quick-start-source.zip)

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

__global__ void gather_1d_custom(float* input, int32_t* index, float* output, uint64_t index_total_length)
{
    // Calculate global thread ID
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Maps to the index of output tensor
    if (idx >= index_total_length) {
        return;
    }
    output[idx] = input[index[idx]];
}
```

1. **接收地址和长度。** `input`、`index`、`output` 是 Host 传入的三个设备地址，指向 GM（设备全局内存）。`index_total_length` 是清单的元素数，也就是要写的输出元素数。
2. **算自己负责的位置。** `blockIdx.x * blockDim.x` 跳过前面所有线程块，`threadIdx.x` 再在本块里向后移。两项相加得到 `idx`。
3. **确认这个线程有活要做。** `idx >= index_total_length` 时，`return` 结束当前线程的这次执行。其他线程各自继续。
4. **先读索引，再读输入，最后写输出。** `output[idx] = input[index[idx]]` 把这三步写在了一行。不同线程各写自己的 `output[idx]`，即使它们读同一个输入值也可以。

这里要分清**两种边界**。核函数检查的是 `idx < index_total_length`，保证清单和输出位置有效。取数还要求 `0 <= index[idx] < input_total_length`。官方样例在 Host 生成清单时用取模把值限制在合法范围。改成外部提供的索引时，需要另外校验这些值。C++ 指针下标 `input[-1]` 会访问数组起始地址之前的位置。

这一版核函数用指针下标直接读写 GM。回顾 [SIMD Add](/notes/ascend-c-simd-add-quick-start/) 时，可以把它与显式的 GM 到 UB 搬运相对应。Gather 这里的 Host 仍用 `aclrtMemcpy` 把输入和索引从主机送到设备，再把结果拷回。 [官方 Gather 数据访问说明](https://asc.gitcode.com/guide/getting_started/quick_start/simt_programming/gather_op_quick_start.html)

## 4. 读官方 Host 调用

先对照样例的三个数组。这个核函数为**每个输出**分配一个线程，所以启动规模根据索引清单的长度确定。

| 缓冲区 | 元素类型 | 元素数 | 申请的字节数 |
|---|---|---|---|
| `input_device` | `float` | 100000 | 400000 |
| `index_device` | `int32_t` | 12288 | 49152 |
| `output_device` | `float` | 12288 | 49152 |

下面是同一个官方文件里的 Host 函数 `gather_1d`。阅读时先找内存分配和复制，再跳到启动语句和等待。 [官方 Host 源码](https://gitcode.com/cann/asc-devkit/blob/e6dc42b66218e322567c26aa71b00bb39adbb79c/examples/03_simt_api/00_introduction/01_gather/basic_gather/gather_1d/gather_1d.asc)

```cpp
std::vector<float> gather_1d(std::vector<float>& input, std::vector<int32_t>& index)
{
    if (input.empty() || index.empty()) {
        std::cout << "[ERROR] Empty input tensors." << std::endl;
        return {};
    }

    uint32_t input_total_length = input.size();
    uint32_t index_total_length = index.size();

    size_t input_total_byte_size = input_total_length * sizeof(float);
    size_t index_total_byte_size = index_total_length * sizeof(int32_t);
    size_t output_total_byte_size = index_total_length * sizeof(float);

    int32_t device_id = 0;
    aclrtStream stream = nullptr;

    uint8_t* input_host = reinterpret_cast<uint8_t*>(input.data());
    uint8_t* index_host = reinterpret_cast<uint8_t*>(index.data());
    uint8_t* output_host = nullptr;
    float* input_device = nullptr;
    int32_t* index_device = nullptr;
    float* output_device = nullptr;
    // Init
    aclInit(nullptr);
    aclrtSetDevice(device_id);
    aclrtCreateStream(&stream);

    // Allocate host and device memory, and copy input data from host to device
    aclrtMallocHost((void**)(&output_host), output_total_byte_size);
    aclrtMalloc((void**)&input_device, input_total_byte_size, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&index_device, index_total_byte_size, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&output_device, output_total_byte_size, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(input_device, input_total_byte_size, input_host, input_total_byte_size, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(index_device, index_total_byte_size, index_host, index_total_byte_size, ACL_MEMCPY_HOST_TO_DEVICE);

    // Configure kernel launch parameters
    uint32_t blocks_per_grid = 48;    // Number of thread blocks (Grid size)
    uint32_t threads_per_block = 256; // Number of threads per block (Block size)
    uint32_t dyn_ubuf_size = 0;       // No dynamic memory required in this sample

    // Call kernel function with <<<...>>>
    gather_1d_custom<<<blocks_per_grid, threads_per_block, dyn_ubuf_size, stream>>>(
        input_device, index_device, output_device, index_total_length);

    // Wait for the gather_1d_custom kernel to complete
    aclrtSynchronizeStream(stream);

    // Copy the result from device memory to host memory
    aclrtMemcpy(output_host, output_total_byte_size, output_device, output_total_byte_size, ACL_MEMCPY_DEVICE_TO_HOST);
    std::vector<float> output((float*)output_host, (float*)(output_host + output_total_byte_size));

    // free memory
    aclrtFree(input_device);
    aclrtFree(index_device);
    aclrtFree(output_device);
    aclrtFreeHost(output_host);

    const char* err = aclGetRecentErrMsg();
    if (err != nullptr) {
        fprintf(stderr, "%s\n", err);
    }

    // DeInit
    aclrtDestroyStream(stream);
    aclrtResetDevice(device_id);
    aclFinalize();
    return output;
}
```

1. **空输入检查和大小计算。** 两个输入至少各有一个元素。分配和复制按**字节数**传参，核函数的 `index_total_length` 按**元素数**传参。输出有多少项由索引数组决定，元素类型则来自输入数组。
2. **初始化并准备三块设备内存。** `aclrtCreateStream(&stream)` 建立本次使用的 Stream。Host 将 `input` 和 `index` 拷到设备，先为 `output` 留好设备空间。它还用 `aclrtMallocHost` 为拷回的结果预留一块 Host 内存。
3. **启动核函数。** 尖括号里的 `48, 256, 0, stream` 对应 48 个线程块、每块 256 个线程、动态共享 UB 为 0 字节，以及指定的 Stream。小括号传入三个设备指针和清单长度。刚好有 `48 × 256 = 12288` 个逻辑线程，和本样例的输出数相同。
4. **等待、拷回和释放。** `aclrtSynchronizeStream(stream)` 等待结果写进设备内存。接着把结果拷回 `output_host`，复制进一个有独立存储的 C++ 向量，再释放原来的缓冲区、Stream 和设备运行资源。

仓库里的这份代码把块数固定为 48。若把清单改成 12289 项，线程编号最多只到 12287，最后一项会漏算。泛化时先用向上取整计算块数，再保留核函数里的 `idx` 检查。教程的 Host 片段把 `index.size()` 直接传给核函数，下载包中的完整工程先把同一个长度保存为 `index_total_length`，再传入它。 [官方教程 Host 片段](https://asc.gitcode.com/guide/getting_started/quick_start/simt_programming/gather_op_quick_start.html)

## 5. 读官方测试入口

官方 `main` 先造出输入与合法索引，再让 CPU 按同一条规则算出 `golden`（参考结果），最后拿设备结果逐元素对照。这样可以判断线程分工、数据拷贝和取值是否一起跑通。

```cpp
int32_t main(int32_t argc, char* argv[])
{
    constexpr uint32_t input_total_length = 100000;
    constexpr uint32_t index_total_length = 48 * 256;

    std::vector<float> input(input_total_length);
    for (uint32_t i = 0; i < input_total_length; i++) {
        input[i] = i * 1.2f;
    }

    std::vector<int32_t> index(index_total_length);
    for (uint32_t i = 0; i < index_total_length; i++) {
        index[i] = static_cast<int32_t>((i * 97 + 13) % input_total_length);
    }

    std::vector<float> golden(index_total_length);
    for (uint32_t i = 0; i < index_total_length; i++) {
        golden[i] = input[index[i]];
    }

    std::vector<float> output = gather_1d(input, index);
    if (output.empty()) {
        return 1;
    }
    return verify_result(output, golden);
}
```

`% input_total_length` 让清单值落在 `0～99999`。例如 `i = 0` 时，`index[0] = 13`，所以参考输出的第 0 项就是 `input[13]`。完整源码里的 `verify_result` 用 `std::equal` 比较两个向量在固定规格下对应的结果。 [官方验证代码](https://gitcode.com/cann/asc-devkit/blob/e6dc42b66218e322567c26aa71b00bb39adbb79c/examples/03_simt_api/00_introduction/01_gather/basic_gather/gather_1d/gather_1d.asc)

## 6. 编译并检查结果

这个官方样例面向 **Ascend 950PR／950DT（dav-3510）**，工程说明要求 CANN 9.1.0 或以上。按 [环境准备](/notes/ascend-c-quick-start-environment/) 加载相应 CANN 环境。 [官方 Gather 工程说明](https://gitcode.com/cann/asc-devkit/blob/e6dc42b66218e322567c26aa71b00bb39adbb79c/examples/03_simt_api/00_introduction/01_gather/basic_gather/gather_1d/README.md)

[下载官方 SIMT HelloWorld 和 Gather 工程](/downloads/ascend-c-simt-quick-start-source.zip)，解压后在根目录执行下面的命令。

```bash
cmake -S gather-1d -B gather-1d/build -DCMAKE_ASC_ARCHITECTURES=dav-3510
cmake --build gather-1d/build -j
./gather-1d/build/demo
```

包内原版 `CMakeLists.txt` 通过 `CMAKE_ASC_ENABLE_SIMT ON` 启用 SIMT，`find_package(ASC REQUIRED)` 查找编译工具链，并把 `gather_1d.asc` 编译进 `demo`。运行时会显示 `Output` 与 `Golden` 的开头，校验通过时会打印 `[Success] Case accuracy is verification passed.`。

## 自测

1. 手算例子里，块 1 的线程 0 读哪个输入元素，写哪个输出位置？
2. 如果把清单第 0 项改成 4，原来的 `idx >= index_total_length` 能拦住这次读取吗？
3. 一块 256 个线程，清单有 12289 项，需要几块？有几个线程会提前结束？

<details>
<summary>参考答案</summary>

1. `idx = 4`，从 `index[4]` 读出 1，于是读取 `input[1]`，写到 `output[4]`。
2. `idx = 0` 在清单范围内，会通过检查。接下来 `input[4]` 超出 4 元素输入的范围，因此还要检查清单里的索引值。
3. 需要 49 块，启动 `49 × 256 = 12544` 个逻辑线程，其中 `12544 - 12289 = 255` 个线程提前结束。

</details>

## 伪代码

把官方样例的固定块数抽象成按清单长度计算。这里在 Host 先验证索引值，作用和官方样例生成合法索引的逻辑相对应。

```text
Host
    读取输入数组 input 和索引清单 index
    如果清单为空，直接返回空输出
    确认清单中的每个值都在 input 的有效下标内
    N = 清单长度
    每块线程数 = T，块数 = 向上取整(N / T)
    在 GM 为 input、index、output 分配空间
    复制 input 和 index 到设备
    向 Stream 提交 Gather 核函数

设备上的每个线程各自执行
    idx = 块编号 × 每块线程数 + 块内线程编号
    如果 idx >= N，结束当前线程
    source_pos = index[idx]
    output[idx] = input[source_pos]

Host
    等待 Stream 上的计算完成，拷回 output
    按清单在 CPU 上重新取数，逐项核对
    释放资源，返回结果
```
