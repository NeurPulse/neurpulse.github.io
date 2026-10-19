---
title: "Ascend C 仿真调优：从 Trace 定位到源码行"
description: "掌握 msProf simulator 的编译采集流程，并借助 Trace 时间线和源码周期定位性能热点。"
pubDate: 2026-08-27
category: "CANN / Ascend C"
tags: ["Ascend C", "msProf simulator", "Trace", "MindStudio Insight", "性能优化"]
order: 16
---

## 学习目标

- 正确编译可用于仿真和源码映射的 Ascend C 程序；
- 使用 `msprof op simulator` 采集指定产品、指定核的 Trace；
- 在时间线中识别 MTE、Vector、Scalar 和等待区段；
- 从性能热点回到源码行，并通过对照实验验证优化。

## 仿真分析解决什么问题

上板 Profiling 擅长回答“真实硬件上各类资源用了多少”；仿真 Trace 更适合回答“指令按什么顺序执行，哪一段时间对应哪行代码”。

它尤其适合：

- 在没有实体硬件时提前分析指令流水；
- 观察流水间的空洞、阻塞和先后关系；
- 将热点进一步定位到源码行。

但仿真结果不是最终硬件性能。正式结论仍应在目标芯片上复测。

## 三步采集流程

### 1. 保留源码行信息

在 Ascend C 编译选项中加入 `-g`：

```cmake
target_compile_options(demo PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
    $<$<COMPILE_LANGUAGE:ASC>:-g>
)
```

没有调试信息时，即使看到了耗时区段，也可能无法准确映射到源码行。

### 2. 使用仿真模式编译

```shell
cmake -B build \
  -DCMAKE_ASC_RUN_MODE=sim \
  -DCMAKE_ASC_ARCHITECTURES=dav-2201
cmake --build build
```

`CMAKE_ASC_RUN_MODE=sim` 开启仿真编译；架构参数必须与目标产品和课程环境相匹配。

### 3. 采集仿真 Trace

课程以 Atlas A2 训练产品为例：

```shell
msprof op simulator \
  --soc-version=Ascend910B1 \
  --output=./Sources/04.04/prof \
  --core-id=0 \
  ./Sources/04.04/build/demo
```

也可以设置对应 Simulator 库的 `LD_LIBRARY_PATH`，省略 `--soc-version`。两种方法的目的都是让工具知道使用哪一种 SoC 模型。

## `--core-id` 什么时候可以用

只采集一个核可以显著减少数据量，但隐含了“各核工作均匀”的假设。以下情况不能只看核 0：

- Tail Core 处理的长度不同；
- 不同核走了不同分支；
- Tiling 造成负载不均；
- 怀疑某个特定核存在等待或异常。

正确做法是先确认核间均匀性，再选代表性核心；有尾块时，应同时检查普通核和尾核。

## 输出文件

典型目录如下：

```text
OPPROF_xxx/
├── simulator/
│   ├── core0.veccore0/
│   ├── trace.json
│   └── visualize_data.bin
└── dump/
    ├── aicore_binary.o
    ├── object_dump.txt
    └── pc_start_addr.txt
```

- `trace.json`：可在时间线工具中观察指令流水；
- `visualize_data.bin`：供 MindStudio Insight 等工具读取；
- `core*.veccore*` / `core*.cubecore*`：按核和计算单元组织的数据；
- `dump/`：工具过程文件，一般无需从这里开始分析。

## 用时间线观察流水

课程使用 `chrome://tracing` 导入 `trace.json`。常用快捷键为：`W` 放大、`S` 缩小、`A` 左移、`D` 右移。

![Add 算子的仿真 Trace](/images/cann/trace_info.png)

> 图片来源：CANN 课程 `04.04_simulation_analysis.ipynb`，展示 Add 算子的 MTE2、Vector、MTE3 和 Scalar 时间线。

阅读顺序建议如下：

1. 先看整体时间跨度；
2. 找持续时间最长的泳道或区段；
3. 观察搬运、计算是否交叠，是否存在明显空洞；
4. 点击热点区段，查看对应代码行；
5. 回到源码确认这段工作是否必要。

课程样例中 MTE2、Vector、MTE3 都很短，而 Scalar 区段很长。继续查看源码映射后，可定位到 `printf` 调试输出。

## 使用 MindStudio Insight

MindStudio Insight 可以导入 `trace.json` 或 `visualize_data.bin`：

![在 MindStudio Insight 中选择性能文件](/images/cann/chose_file.png)

> 图片来源：CANN 课程 `04.04_simulation_analysis.ipynb`。

其时间线视图用于观察流水关系：

![MindStudio Insight 时间线](/images/cann/timeline.png)

> 图片来源：CANN 课程 `04.04_simulation_analysis.ipynb`。

源码视图可以按代码行显示时钟周期，更适合从热点回到实现：

![源码行周期信息](/images/cann/code_info.png)

> 图片来源：CANN 课程 `04.04_simulation_analysis.ipynb`。

如果本地浏览器不支持 `chrome://tracing`，优先使用课程推荐的 MindStudio Insight 或其他能够读取 Chrome Trace Event 格式的工具。

## 对照实验：关闭调试输出

课程通过编译定义关闭 `printf` 和 `DumpTensor`：

```cmake
target_compile_definitions(demo PRIVATE ASCENDC_DUMP=0)
```

重新编译和采集后，Scalar 长区段明显缩短：

![关闭打印后的 Trace](/images/cann/no_print_trace.png)

> 图片来源：CANN 课程 `04.04_simulation_analysis.ipynb`，展示关闭打印后的仿真 Trace。

这个实验同时验证了两件事：

- Trace 指出的热点确实对应打印代码；
- 删除该热点后，时间线变化与假设一致。

## 上板与仿真如何配合

| 维度 | 上板 Profiling | 仿真 Trace |
| --- | --- | --- |
| 运行环境 | 实体 NPU | Simulator |
| 优势 | 真实耗时、带宽和利用率 | 指令时序清晰、便于定位源码行 |
| 主要文件 | 多个性能 CSV | `trace.json`、`visualize_data.bin` |
| 适合回答 | “瓶颈属于哪类资源？” | “瓶颈发生在哪段流水、哪行代码？” |
| 局限 | 定位到源码细节可能较间接 | 性能不能完全替代真机结果 |

推荐顺序是：上板数据确定大方向，仿真 Trace 细化到指令和源码，再回到目标硬件验证收益。

## 常见误区

- **忘记 `-g`**：Trace 能生成，但源码映射能力受限。
- **SoC 型号随便填写**：仿真产品必须与目标架构对应。
- **默认所有核完全一致**：只采核 0 可能漏掉尾核或负载不均。
- **仿真更精确，所以无需上板**：仿真适合定位，不代表真实硬件最终耗时。
- **只看最长色块，不看源码**：相同流水类型可能来自多个不同代码位置。
- **优化后只看图变短**：还应重新做精度验证，并在真机比较端到端耗时。

## 一句话总结

仿真调优的价值是把抽象的性能统计展开成可点击的指令时间线，再把最长区段追溯到具体源码。

## 复习卡片

| 问题 | 答案 |
| --- | --- |
| 仿真编译模式？ | `CMAKE_ASC_RUN_MODE=sim` |
| 为什么加入 `-g`？ | 让 Trace 热点能够映射到源码行 |
| 仿真采集子命令？ | `msprof op simulator` |
| 主要时间线文件？ | `trace.json` |
| 只采一个核的前提？ | 已确认核间数据和执行路径具有代表性 |

## 自测题

1. 为什么 `--soc-version` 不能与目标产品不一致？
2. Trace 中 Scalar 很长时，应如何从现象定位到代码？
3. 哪些情况需要同时采集普通核和尾核？
4. 为什么仿真优化完成后仍需上板测试？

<details>
<summary>参考答案</summary>

1. 不同产品的架构和硬件行为可能不同，错误模型会使仿真无效或结论失真。
2. 放大热点区段，查看事件详情和源码行映射，再检查对应的 Scalar 代码是否必要。
3. 核间长度、分支或 Tiling 不一致，尤其存在尾块时。
4. Simulator 不能完整复现实体硬件的频率、带宽竞争和系统运行环境。

</details>

## 内容与图片来源

- 课程仓库：[CANN LearningHub](https://gitcode.com/cann/cann-learning-hub/tree/master)
- 参考 Notebook：`04.04_simulation_analysis.ipynb`
- 课程示例源自：`src/04.03/add_custom.asc`

