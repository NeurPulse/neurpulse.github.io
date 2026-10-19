---
title: 'Ascend C Kernel 接入 PyTorch：pybind11 与 torch.library'
description: '对比两条 PyTorch Kernel 直调路径，梳理 C++ 封装、NPU Stream、动态库构建和 Python 调用。'
pubDate: 2026-08-27
updatedDate: 2026-10-19
category: 'PyTorch'
tags: ['Ascend C', 'PyTorch', 'pybind11', 'torch.library']
order: 8
---

## 学习目标

这篇笔记说明如何把已经可运行的 Ascend C Kernel 接入 PyTorch。学完后应能区分 pybind11 模块函数和 torch.library 注册算子，并理解两条路径共同经过的 C++ 封装、NPU Stream 与 Device 指针。

以下是需要 CANN/NPU 环境的接口与调用片段，不是 CPU 上可直接运行的独立程序。为与[Add 算子实现解析](/notes/ascend-c-add-operator-walkthrough/)一致，本文统一使用固定 `[8,2048]`、**float32、ND 格式、8 核**的示例；其他课程工程可能使用 float16 或不同 Tiling ABI，不能只复制封装层就混用。示例仅用于无梯度推理，不宣称已经支持训练或 `torch.compile`。

## 先修与环境

- 已安装相互匹配的 PyTorch、`torch_npu`、CANN Toolkit；
- pybind11 路径还需要 `pybind11`；
- 编译时使用的 Python 必须与运行测试、安装 Torch 的 Python 一致；
- `--npu-arch` 必须匹配实际硬件。

## 公共 Kernel 层

两种调用方式可以复用同一个 `.asc` Kernel。下面的封装放在已经定义 `AddCustomTilingData` 和 `add_custom` 的 `.asc` 文件中，与 Add 文章的结构体参数保持一致：

```cpp
extern "C" void add_custom_impl(
    uint32_t numBlocks,
    void* aclStream,
    uint8_t* x,
    uint8_t* y,
    uint8_t* z,
    uint32_t totalLength)
{
    AddCustomTilingData tiling{totalLength, 8};
    add_custom<<<numBlocks, nullptr, aclStream>>>(x, y, z, tiling);
}
```

`extern "C"` 避免 C++ 名字改编，但不会自动解决所有 ABI 问题。两端仍需共享一致的函数声明、参数宽度和 Kernel/Tiling 定义；直接把 `totalLength` 传给期待结构体的 Kernel 是错误的。该入口约定只由完成下述检查的封装调用。

## pybind11 路径

调用链：

```text
Python import ascendc_ops
  → ascendc_ops.ascendc_add(x, y)
  → C++ at::Tensor 封装函数
  → add_custom_impl
  → Ascend C Kernel
```

### C++ 封装函数

```cpp
#include <ATen/ATen.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/FormatHelper.h"

extern "C" void add_custom_impl(uint32_t, void*, uint8_t*, uint8_t*, uint8_t*, uint32_t);

at::Tensor ascendc_add(const at::Tensor& x,
                       const at::Tensor& y)
{
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1,
                "NPU tensor required");
    TORCH_CHECK(x.device() == y.device(), "device mismatch");
    TORCH_CHECK(x.scalar_type() == at::kFloat && y.scalar_type() == at::kFloat,
                "float32 required");
    TORCH_CHECK(x.dim() == 2 && x.size(0) == 8 && x.size(1) == 2048,
                "fixed shape [8, 2048] required");
    TORCH_CHECK(x.sizes() == y.sizes(), "shape mismatch");
    TORCH_CHECK(x.is_contiguous() && y.is_contiguous(),
                "contiguous tensors required");
    TORCH_CHECK(!x.requires_grad() && !y.requires_grad(),
                "inference-only example: backward is not registered");

    c10_npu::NPUGuard guard(x.device());
    TORCH_CHECK(at_npu::native::FormatHelper::GetFormat(x) == ACL_FORMAT_ND &&
                at_npu::native::FormatHelper::GetFormat(y) == ACL_FORMAT_ND,
                "ND storage format required");
    auto stream = c10_npu::getCurrentNPUStream(x.get_device()).stream();
    at::Tensor z = at::empty(x.sizes(), x.options());
    TORCH_CHECK(at_npu::native::FormatHelper::GetFormat(z) == ACL_FORMAT_ND,
                "ND output required");
    const uint32_t totalLength = static_cast<uint32_t>(x.numel());

    add_custom_impl(
        8, stream,
        reinterpret_cast<uint8_t*>(x.data_ptr<float>()),
        reinterpret_cast<uint8_t*>(y.data_ptr<float>()),
        reinterpret_cast<uint8_t*>(z.data_ptr<float>()),
        totalLength);
    return z;
}
```

随后绑定为 Python 模块函数：

```cpp
#include <torch/extension.h>

PYBIND11_MODULE(ascendc_ops, m)
{
    m.def("ascendc_add", &ascendc_add);
}
```

Python 使用：

```python
import torch
import torch_npu
import ascendc_ops

x = torch.rand((8, 2048), dtype=torch.float32, device="npu:0")
y = torch.rand_like(x)
z = ascendc_ops.ascendc_add(x, y)
```

该方式直观，适合快速验证封装和实验性接口。

## torch.library 路径

调用链：

```text
torch.ops.load_library
  → torch.ops.ascendc_ops.ascendc_add
  → PyTorch Dispatcher
  → PrivateUse1 的 NPU 实现
  → add_custom_impl
  → Ascend C Kernel
```

先定义算子 schema：

```cpp
TORCH_LIBRARY(ascendc_ops, m)
{
    m.def("ascendc_add(Tensor x, Tensor y) -> Tensor");
}
```

再把 NPU 实现注册到 `PrivateUse1`：

```cpp
TORCH_LIBRARY_IMPL(ascendc_ops, PrivateUse1, m)
{
    m.impl("ascendc_add", TORCH_FN(ascendc_add));
}
```

Python 使用：

```python
import torch
import torch_npu

# 改为实际构建产物的绝对路径，不依赖当前工作目录。
torch.ops.load_library("/absolute/path/to/libascendc_ops.so")
x = torch.rand((8, 2048), dtype=torch.float32, device="npu:0")
y = torch.rand_like(x)
z = torch.ops.ascendc_ops.ascendc_add(x, y)
```

这种方式把自定义算子纳入 PyTorch 算子系统，更适合框架集成。注册 `PrivateUse1` 实现**不会自动提供 backward**；训练需另行注册自动求导规则，编译与导出等场景还需适当的 Fake/Meta 实现及验证。两条路径在独立构建/测试中选择其一，不要重复加载多个定义相同 schema 的注册库。

## 两种方式对比

| 项目 | pybind11 | torch.library |
|---|---|---|
| Python 入口 | 普通扩展模块函数 | `torch.ops` 注册算子 |
| 核心宏 | `PYBIND11_MODULE` | `TORCH_LIBRARY`、`TORCH_LIBRARY_IMPL` |
| 调度系统 | 绕过 Dispatcher 的普通函数调用 | 进入 PyTorch Dispatcher |
| 适合 | 快速验证、工具接口 | 正式算子适配、框架集成 |
| 公共部分 | `at::Tensor` 封装、当前 NPU Stream、Kernel C 入口 | 相同 |

## CMake 构建关系

通常生成两个动态库：

```text
add_custom.asc
    ↓ ASC 编译
libascendc_kernels.so
    ↑ 链接
C++ PyTorch 封装
    ↓
Python 扩展模块或 libascendc_ops.so
```

关键点：

- `find_package(ASC REQUIRED)`；
- 查询当前 Python 对应的 Torch、torch_npu、pybind11 路径；
- 封装库链接 Kernel 库与 `torch_npu`；
- RPATH 设为 `$ORIGIN`，使同目录动态库可被找到；
- CMake 的 `Python3_EXECUTABLE` 与测试解释器保持一致。

pybind11 构建还需匹配可导入扩展的模块名、Python 扩展后缀和 Torch 的 C++ ABI；单纯生成一个普通 `lib*.so` 并不保证可用 `import` 导入。本文不提供脱离具体版本即可通用的完整 CMake 文件。

## 参数检查不可省略

上面的检查位于取指针与启动 Kernel **之前**。它同时约束两个输入的设备、dtype、shape、连续性、存储格式以及当前不支持梯度的边界，而不是把检查留在未调用的独立代码块里。

NPU 的逻辑连续性不必然说明物理存储为 ND，因此还核对实际格式。`FormatHelper` 属于 torch_npu 内部 C++ 接口，头文件位置和可用性可能随版本变化；这里的写法依据文末源码，构建时必须核对已安装版本。不能为了让示例编译通过而直接删去格式检查。

这里读取 x/y 指针但不修改输入；输出写入新分配的 z。固定输入检查也保证 `numel()` 可以安全转换为 `uint32_t` 并被当前切分完整覆盖。若要放开 shape 或 dtype，必须同时修改 Kernel、Tiling 和边界测试。

## Stream 与异步语义

封装层使用**输入所在设备**的当前 NPU Stream；Device Guard 防止当前设备与输入设备不同而选错上下文。这里调用 `.stream()`，不应在不了解版本语义时照抄绕过任务队列的 `.stream(false)`。

同一 Stream 内按顺序执行，Kernel 启动通常异步，不必每次强制全局同步。若调用方使用多个 Stream，仍需按 PyTorch/torch_npu 的规则处理依赖与存储生命周期；使用“当前 Stream”本身不会替调用方建立所有跨 Stream 依赖。测试中的 `.cpu()` 会等待所需结果，但异步设备错误也可能在同步处才暴露。

## 精度测试

```python
import torch
import torch_npu
import ascendc_ops

custom_add = ascendc_ops.ascendc_add  # 或已加载的 torch.ops.ascendc_ops.ascendc_add
cpu_x = torch.arange(8 * 2048, dtype=torch.float32).reshape(8, 2048) / 1000
cpu_y = torch.flip(cpu_x, dims=[1]) * -0.25
x = cpu_x.to("npu:0")
y = cpu_y.to("npu:0")
actual = custom_add(x, y).cpu()
expected = cpu_x + cpu_y
assert actual.shape == expected.shape and actual.dtype == expected.dtype
assert torch.isfinite(actual).all()
torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
```

测试至少应覆盖正常输入、错误 dtype、错误 shape、非连续 Tensor、CPU 输入、不同 Device、内部格式与 `requires_grad=True` 输入。当前固定样例应拒绝不支持的输入，而不是让测试勉强通过。上述容差仅用于本 FP32 Add 示例，不是所有自定义算子的统一精度标准。

## 常见误区

1. **CMake 找到任意 Python 都可以。** 编译和运行解释器不一致会导致头文件、ABI 或模块加载问题。
2. **从 Tensor 取指针后可以忽略 Stream。** Stream 决定任务时序。
3. **torch.library 注册后无需加载动态库。** 只有加载库，静态注册代码才会执行。
4. **`PrivateUse1` 是普通 CPU 实现。** torch_npu 使用它作为 NPU 调度键。
5. **封装层只负责转发。** 它还应承担 shape、dtype、device、layout 等边界检查。

## 一句话总结

pybind11 把 Ascend C Kernel 包成普通 Python 扩展函数，torch.library 则把它注册进 PyTorch Dispatcher；二者共享同一个 Kernel、NPU Stream 和 Tensor 指针桥接层。

## 复习卡片

- 公共入口：`extern "C" ..._impl`。
- pybind11：`PYBIND11_MODULE`。
- torch.library：schema + `PrivateUse1` 实现。
- 执行顺序：当前 NPU Stream。
- 构建：Kernel so + 封装 so。
- 安全边界：shape、dtype、device、contiguous。

## 自测问题

1. 为什么两条路径可以复用同一个 `.asc` Kernel？
2. pybind11 与 torch.library 的 Python 调用入口分别是什么？
3. 为什么要获取 PyTorch 当前 NPU Stream？
4. `torch.ops.load_library` 起什么作用？

<details>
<summary>参考答案</summary>

1. 差异集中在框架封装层，底层 C ABI 入口和 Kernel 计算逻辑相同。
2. 前者是导入扩展模块后调用函数，后者通过 `torch.ops.<namespace>.<op>` 调用。
3. 保证自定义 Kernel 与前后 PyTorch NPU 操作处于正确的异步执行顺序。
4. 加载动态库并执行其中的 torch.library 注册代码。

</details>

## 参考来源

- [CANN Learning Hub：PyTorch 框架下 Kernel 直调](https://gitcode.com/cann/cann-learning-hub/blob/master/tutorials/ascendc_operator_development_light/02_AscendC_basic/02.06_kernel_pytorch_call.ipynb)
- [PyTorch：自定义 C++ 算子的注册、自动求导与 Fake 实现](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)
- [torch_npu：NPUStream 接口](https://github.com/Ascend/pytorch/blob/master/torch_npu/csrc/core/npu/NPUStream.h)
- [torch_npu：NPUGuard](https://github.com/Ascend/pytorch/blob/master/torch_npu/csrc/core/npu/NPUGuard.h)
- [torch_npu：FormatHelper](https://github.com/Ascend/pytorch/blob/master/torch_npu/csrc/framework/FormatHelper.h)

