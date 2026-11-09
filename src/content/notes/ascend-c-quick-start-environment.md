---
title: 'Ascend C 环境准备：确认 NPU、CANN 与编译器'
description: '从设备到编译工具逐层检查，读懂架构参数，并用同一套官方样例验证 HelloWorld 和 Add。'
pubDate: 2026-11-09
category: 'CANN / Ascend C'
tags: ['Ascend C', 'CANN', '环境准备', 'CMake']
order: 1
---

## 学习目标

准备环境可以拆成三个问题。Linux 主机能否看到 NPU？当前终端能否找到 CANN 编译工具？这个样例是否支持手上的设备？分开检查这三件事，出问题时就知道从哪一层找原因。

## 先分清要准备什么

| 组件 | 在 HelloWorld 中的职责 |
|---|---|
| NPU 与驱动、固件 | 让系统识别并使用设备 |
| CANN 开发工具和运行时 | 提供头文件、毕昇编译器和执行核函数所需的库 |
| CMake | 根据样例的 `CMakeLists.txt` 生成构建规则，调用对应编译器 |
| 官方样例 | 提供要编译的 `.asc` 源码和该样例的配置 |

以下命令在实际开发用的 **Linux 主机或已配置的容器**中执行。按设备型号、操作系统和 CPU 架构选择匹配的驱动、固件与 CANN 版本。官方教程中的 CMake 样例要求 3.16 或以上。 [官方环境准备](https://asc.gitcode.com/guide/getting_started/environment_setup.html)

## 第一步，确认设备可见

```bash
npu-smi info
```

先记下设备型号和当前环境可用的设备。型号决定样例及编译目标的选择。样例用 `aclrtSetDevice(0)` 选逻辑编号为 0 的设备，使用容器时要核对容器里暴露的逻辑编号。若命令报找不到设备，先确认正在使用的机器或容器能看到 NPU，再检查驱动安装。 [官方驱动与固件说明](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/920beta2/softwareinst/instg/instg_0005.html)

## 第二步，确认当前终端能找到开发工具

```bash
# 这里用官方文档中的 root 安装路径作为示例。
source /usr/local/Ascend/cann/set_env.sh
command -v bisheng
bisheng --version
cmake --version
```

`source` 在**当前终端**加载 CANN 的环境变量，安装到其他位置时换成实际的 `set_env.sh` 路径。新开一个终端后也需要按你的环境加载。看到编译器路径说明当前终端已经能找到它。如果 CMake 报找不到 `ASC`，先核对环境变量是否加载、开发工具是否完整，以及样例版本是否和安装的 CANN 对得上。 [官方环境准备](https://asc.gitcode.com/guide/getting_started/environment_setup.html)

## 第三步，选样例支持的编译目标

`--npu-arch` 告诉毕昇编译器要为哪种 NPU 架构生成代码。下面的构建命令用 `dav-2201` 演示，官方这三份样例把它用于 Atlas A2/A3。使用其他型号时，需要同时检查**架构对应表和该样例的支持范围**。例如本系列讲解的 C API Add 工程在 CMake 中限定了 `dav-2201`。 [架构对应表](https://asc.gitcode.com/guide/programming_guide/language_extension/simd_builtin_keywords.html#npu-arch) · [C API Add 构建配置](https://gitcode.com/cann/asc-devkit/blob/e6dc42b66218e322567c26aa71b00bb39adbb79c/examples/02_simd_c_api/00_introduction/01_add/c_api_async_add/CMakeLists.txt)

## 先运行 HelloWorld

[下载三份官方样例源码、CMake 配置与许可证](/downloads/ascend-c-quick-start-source.zip)。解压后有 `hello-world`、`c-add` 和 `cpp-add` 三个目录。下面从解压后的目录开始，每个工程有自己的 `build` 目录。

```bash
cmake -S hello-world -B hello-world/build -DCMAKE_ASC_ARCHITECTURES=dav-2201
cmake --build hello-world/build -j
./hello-world/build/demo
```

能看到 `Hello World!!!`，说明当前配置跑通了编译、设备启动和等待。这个测试的任务很小，所以适合在读 Add 之前单独做。没有 NPU 时，可以在本系列中先读源码、手算分块并跑 Python 小例子。需要 CANN 的 CPU 域调试时，再按所用版本和样例的支持说明配置。 [HelloWorld 样例](https://gitcode.com/cann/asc-devkit/tree/e6dc42b66218e322567c26aa71b00bb39adbb79c/examples/01_simd_cpp_api/00_introduction/00_quickstart/hello_world) · [官方环境准备](https://asc.gitcode.com/guide/getting_started/environment_setup.html)

## 按报错判断查哪里

| 观察到的现象 | 首先检查 |
|---|---|
| `npu-smi info` 看不到预期设备 | 主机设备状态、容器设备权限、驱动和固件 |
| `bisheng` 找不到 | 当前终端的环境变量和 CANN 开发工具安装 |
| CMake 找不到 `ASC` | CANN 配置是否加载，工具链与样例是否匹配 |
| 编译通过，运行时报设备或二进制错误 | 设备编号、设备权限、编译架构和样例的产品支持 |
| HelloWorld 能运行，Add 校验失败 | 从分块范围、数据长度单位和同步点开始对照源码 |

下一篇会用一组数字说明数据为什么经过 Host 内存、Device 内存和片上缓冲。 [异构系统与编程模型](/notes/cann-architecture-and-ascend-npu/)

## 伪代码

```text
读取设备型号和可用设备编号
查找与设备匹配的 CANN 版本及样例
加载当前终端的 CANN 环境，确认编译器和 CMake 可用
用样例支持的目标架构编译 HelloWorld
运行程序并检查设备打印
用这套环境编译 Add，再检查数值输出
```
