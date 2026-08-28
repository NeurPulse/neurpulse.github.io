---
title: 'Ascend C Kernel 接入 PyTorch：pybind11 与 torch.library'
description: '对比两条 PyTorch Kernel 直调路径，梳理 C++ 封装、NPU Stream、动态库构建和 Python 调用。'
pubDate: 2026-08-27
category: 'PyTorch'
tags: ['Ascend C', 'PyTorch', 'pybind11', 'torch.library']
order: 8
---

## 学习目标

这篇笔记说明如何把已经可运行的 Ascend C Kernel 接入 PyTorch。学完后应能区分 pybind11 模块函数和 torch.library 注册算子，并理解两条路径共同经过的 C++ 封装、NPU Stream 与 Device 指针。

## 先修与环境

- 已安装相互匹配的 PyTorch、`torch_npu`、CANN Toolkit；
- pybind11 路径还需要 `pybind11`；
- 编译时使用的 Python 必须与运行测试、安装 Torch 的 Python 一致；
- `--npu-arch` 必须匹配实际硬件。

## 公共 Kernel 层

两种调用方式可以复用同一个 `.asc` Kernel，并提供稳定的 C ABI 入口：

```cpp
extern "C" void add_custom_impl(
    uint32_t numBlocks,
    void* aclStream,
    uint8_t* x,
    uint8_t* y,
    uint8_t* z,
    uint32_t totalLength)
{
    add_custom<<<numBlocks, 0, aclStream>>>(
        x, y, z, totalLength);
}
```

`extern "C"` 避免 C++ 名字改编，让封装动态库更稳定地链接这个入口。

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
at::Tensor ascendc_add(const at::Tensor& x,
                       const at::Tensor& y)
{
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    at::Tensor z = at::empty_like(x);
    uint32_t totalLength = x.numel();

    add_custom_impl(
        8, stream,
        reinterpret_cast<uint8_t*>(x.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(y.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(z.mutable_data_ptr()),
        totalLength);
    return z;
}
```

随后绑定为 Python 模块函数：

```cpp
PYBIND11_MODULE(ascendc_ops, m)
{
    m.def("ascendc_add", &ascendc_add);
}
```

Python 使用：

```python
import ascendc_ops

z = ascendc_ops.ascendc_add(x.npu(), y.npu())
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
torch.ops.load_library("libascendc_ops.so")
z = torch.ops.ascendc_ops.ascendc_add(x.npu(), y.npu())
```

这种方式把自定义算子纳入 PyTorch 算子系统，更适合框架集成。

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

## 参数检查不可省略

课程的基础 Add 示例假设 shape 为 `[8,2048]`、dtype 为 `float16`、连续存储且长度可被 8 核均分。更可靠的封装层应检查：

```cpp
TORCH_CHECK(x.device() == y.device(), "device mismatch");
TORCH_CHECK(x.scalar_type() == at::kHalf, "float16 required");
TORCH_CHECK(x.sizes() == y.sizes(), "shape mismatch");
TORCH_CHECK(x.is_contiguous() && y.is_contiguous(),
            "contiguous tensors required");
```

还应验证输入位于 NPU、输出 shape/dtype 正确，并保证 Kernel 的 Tiling 能覆盖真实 `numel()`。

## Stream 与异步语义

封装层必须使用 PyTorch 当前 NPU Stream，而不是私自创建无关 Stream。这样 Kernel 才能与前后 PyTorch 算子保持正确顺序。Kernel 启动通常是异步的，不应在每次调用中强制全局同步；测试读取 CPU 结果时会自然形成必要同步。

## 精度测试

```python
x = torch.rand((8, 2048), dtype=torch.float16)
y = torch.rand((8, 2048), dtype=torch.float16)

actual = custom_add(x.npu(), y.npu()).cpu()
expected = torch.add(x, y)
self.assertRtolEqual(actual, expected)
```

测试至少应覆盖正常输入、错误 dtype、错误 shape、非连续 Tensor、不同 Device 和边界长度。

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

