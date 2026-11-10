---
title: "数据科学导引第十五章：分布式计算"
description: "从 HDFS 数据块到 MapReduce、Spark RDD，复算词频、线性回归、SVM、K-means 与 PageRank 的分区计算，理解数据本地性、Shuffle、容错和迭代成本"
pubDate: 2026-11-10
updatedDate: 2026-11-10
category: '数据科学'
order: 16
tags: ['数据科学', '数据科学导引', '分布式计算', 'Hadoop', 'HDFS', 'MapReduce', 'Spark', 'RDD', 'PageRank', 'K-means', '读书笔记', '零基础']
---

> [《数据科学导引》学习地图](/notes/data-science-guide-roadmap/) · 第 15 / 15 章
>
> 上一章：[深度学习](/notes/data-science-guide-deep-learning/)

本文对应《数据科学导引》第十五章“分布式计算”，原书第 325—344 页，即所提供扫描版 PDF 第 334—353 页。笔记保留 HDFS、MapReduce、常见模型的 MapReduce 实现、Spark RDD 与 Hadoop 生态五条主线，并用同一组分区数据复算词频、线性回归、SVM 梯度、K-means 和 PageRank。教材以 Hadoop 1.x 和早期 Spark 为背景；明确标出的“版本提示”“学习补充”与“实践补充”用于区分历史架构和通用原理，并非教材原文。

## 学习目标

学完本章，应能回答十个问题：

1. 数据超过单机内存后，为什么还需要同时考虑磁盘、网络、计算与故障？
2. HDFS 的数据块、副本、NameNode 和 DataNode 分别承担什么职责？
3. MapReduce 的 Map、Shuffle、Reduce 怎样完成词频统计？
4. 为什么适合 MapReduce 的统计量通常能分解为局部求和与全局聚合？
5. 线性回归、SVM、K-means 和 PageRank 每轮分别广播什么、汇总什么？
6. 为什么不能直接平均各分区的均值？
7. 为什么多轮迭代会暴露经典 MapReduce 的作业与磁盘开销？
8. RDD 的分区、血缘、转换、行动和持久化分别是什么意思？
9. Spark 与 Hadoop 是替代关系还是可组合关系？
10. 一个分布式任务什么时候反而不值得上集群？

## 本章知识地图

| 层次 | 核心对象 | 关键机制 | 主要风险 |
|---|---|---|---|
| 分布式存储 | 文件、数据块、副本 | HDFS、NameNode、DataNode | 小文件、元数据压力、节点故障 |
| 批处理 | key-value 对、Map、Reduce | Shuffle、数据本地性、任务重试 | 磁盘中间结果、数据倾斜 |
| 模型并行化 | 局部统计量、模型参数 | 局部求和、全局归约、迭代广播 | 多轮作业、同步等待、通信成本 |
| 内存计算 | RDD、分区、血缘 DAG | 惰性转换、行动、持久化 | Shuffle、缓存压力、Driver 收集 |
| 生态与运维 | 仓库、数据库、协调、调度 | Hive、HBase、ZooKeeper 等 | 工具边界混淆、版本差异 |

一条通用链路是：

**数据分块与复制 → 计算靠近数据 → 局部转换与统计 → 按键 Shuffle → 全局聚合 → 必要时迭代 → 容错与结果评价。**

## 15.0 为什么需要分布式计算

前面章节常默认全部数据能放入一台计算机的内存或磁盘。互联网日志、金融交易、运营商记录、医学影像等数据可能超过单机存储和计算能力。分布式系统把数据与任务拆到多台普通计算机上，希望获得：

- **容量扩展**：数据分块后由多个节点共同保存；
- **并行计算**：多个分区同时处理，缩短可并行部分的耗时；
- **容错能力**：节点故障时利用副本、血缘或任务重试继续完成；
- **资源共享**：由集群统一管理 CPU、内存、磁盘与任务调度。

但“机器更多”不等于“任务一定更快”。总成本还包括：

$$
T_{\mathrm{total}}
=T_{\mathrm{local\ compute}}
+T_{\mathrm{shuffle}}
+T_{\mathrm{disk\ I/O}}
+T_{\mathrm{scheduling}}
+T_{\mathrm{straggler}}.
$$

如果任务很小、分区之间通信很多、少数键特别热，调度与网络成本可能超过并行收益。分布式计算首先是一种**数据与计算的分解方法**，不是免费的加速按钮。

### 15.0.1 单机装不下，不等于只有一种瓶颈

| 瓶颈 | 典型表现 | 优先检查 |
|---|---|---|
| 存储 | 原始数据超过单机磁盘 | 分块、压缩、副本与容量规划 |
| 内存 | 单机算法无法装入全部样本或中间状态 | 分区、流式扫描、溢写与缓存策略 |
| 计算 | 可独立处理的记录很多 | 增加有效并行度、减少单条处理成本 |
| 网络 | Shuffle 量大、跨节点传输慢 | 本地预聚合、减少字段、控制分区 |
| 调度 | 任务很多但每个任务只运行很短时间 | 合并小任务、减少作业轮数 |
| 倾斜 | 少数 Reduce 或 Executor 特别慢 | 热键拆分、均衡分区、局部聚合 |

### 15.0.2 先分清三个“分区”概念

1. **存储块**：HDFS 为文件划分的物理存储单位。
2. **计算分区**：MapReduce 或 Spark 为任务划分的数据处理单位。
3. **按键分区**：Shuffle 时根据 key 决定记录送往哪个 Reduce 或后续分区。

三者可能相关，却不等价。一个存储块不保证同 key 的记录都在一起；按 key 分组通常仍需要网络传输。

## 15.1 Hadoop：分布式存储与处理

教材回顾 Hadoop 从 Nutch、Lucene 项目发展而来，并给出 Hadoop 1.x 生态图。核心不是背诵组件名称，而是先分清两层：

- **HDFS**：负责分布式数据存储；
- **MapReduce**：负责分布式批处理计算。

图中其他项目围绕这两层提供数据仓库、数据库、挖掘、协调、采集、传输和工作流能力。

### 15.1.1 HDFS：把大文件切成有副本的数据块

HDFS（Hadoop Distributed File System）运行在普通计算机组成的集群上。教材示例把大文件切为 64 MB 数据块，并通常把每个块保存为 3 份，放在不同 DataNode 上。一个节点故障时，客户端仍可从其他副本读取数据。

> **版本提示**：64 MB 是教材所处 Hadoop 1.x 语境下的典型配置，不是所有集群永远固定的块大小。块大小和副本数应视为配置项；本文手算沿用教材的 64 MiB，仅为复算方便。

假设一个 192 MiB 文件按 64 MiB 分块，复制因子为 3：

$$
N_{\mathrm{block}}=\left\lceil\frac{192}{64}\right\rceil=3,
$$

$$
N_{\mathrm{replica}}=3\times3=9.
$$

不计校验、元数据和文件系统额外开销，三个逻辑块共占：

$$
192\ \mathrm{MiB}\times3=576\ \mathrm{MiB}
$$

的原始存储空间。副本提高了可用性，却不是免费备份：它需要额外容量，也要求故障后重新复制以恢复副本水平。

**副本也不等于异地灾备。** 如果整个集群因误删除、机房故障或权限事故不可用，同集群副本可能一起受影响。重要数据仍需独立的备份与恢复演练。

### 15.1.2 NameNode、DataNode 与客户端

教材中的基本职责是：

| 角色 | 主要职责 | 不应误解为 |
|---|---|---|
| NameNode | 保存文件到数据块、块到节点等元数据，协调命名空间 | 所有文件内容都经过它传输 |
| DataNode | 在本地磁盘保存实际数据块，执行读写与复制相关工作 | 负责全局文件目录 |
| HDFS 客户端 | 向 NameNode 查询元数据，再与 DataNode 读写数据 | 只和一个固定节点交互 |
| Secondary NameNode | 教材架构中的辅助名字节点，涉及元数据检查点 | 自动热备或立即接管的 NameNode |

读取时，客户端先获得块位置，再从相应 DataNode 获取数据；写入时，也由 NameNode 参与块分配与元数据管理。NameNode 因此十分关键，但文件内容不需要全部先流过 NameNode。

> **公式与架构辨析**：教材称 Secondary NameNode 可在 NameNode 故障时“马上代替”其提供读写服务，这容易被理解成热备。经典 Secondary NameNode 的主要作用是辅助合并编辑日志与文件系统镜像、形成检查点，**不能据此推断自动故障切换**。真正的高可用需要专门设计，不能把“辅助”直接等同于“热备”。

### 15.1.3 HDFS 适合与不适合什么

教材强调 HDFS 适合：

- GB、TB 乃至更大规模的大文件；
- 一次写入、多次读取的流式访问；
- 通过增加节点水平扩展；
- 用副本换取节点故障下的数据可用性。

不适合直接照搬的场景包括：

- **海量小文件**：每个文件和块都带来元数据开销，可能给 NameNode 造成压力；
- **大量并发写入或随意随机修改**：教材语境中的 HDFS 主要面向追加与顺序扫描，不是通用随机读写数据库；
- **低延迟单条记录查询**：文件扫描与块存储不等同于索引数据库；
- **只保存很少数据的小任务**：集群运维成本可能没有回报。

## 15.1.4 MapReduce：围绕 key-value 对组织计算

MapReduce 把数据抽象为 key-value 对，把一个作业组织为：

```text
输入 → 分片 → Map → Shuffle / 分组 → Reduce → 输出
```

抽象函数为：

$$
\operatorname{map}(k_1,v_1)
\longrightarrow \operatorname{list}(k_2,v_2),
$$

$$
\operatorname{reduce}(k_2,\operatorname{list}(v_2))
\longrightarrow \operatorname{newlist}(k_3,v_3).
$$

Map 可以把一条输入变成零条、一条或多条中间记录。Shuffle 按 key 分区、传输并分组，使相同 key 的值进入同一个 Reduce 调用。Reduce 再聚合这些值。

**Shuffle 不是简单的“整理顺序”。** 它往往涉及网络、排序、磁盘溢写和序列化，是许多作业的核心成本。

### 15.1.5 完整复算教材的词频统计

输入三行：

```text
Jane Amy Lisa
Ella Ella Lisa
Jane Ella Amy
```

Map 对每个词输出 $(w,1)$。例如第一行产生：

```text
(Jane, 1), (Amy, 1), (Lisa, 1)
```

Shuffle 后，同 key 的值聚在一起：

```text
Amy  → [1, 1]
Ella → [1, 1, 1]
Jane → [1, 1]
Lisa → [1, 1]
```

Reduce 求和，得到：

```text
Amy, 2
Ella, 3
Jane, 2
Lisa, 2
```

Map 不需要知道其他行有什么词；Reduce 不需要重新扫描原始文档。二者通过 key 和 Shuffle 衔接。

**学习补充：Combiner。** 如果局部聚合满足结合律，并且实现与最终 Reduce 的输入输出约定一致，可以在 Map 端先合并，例如把同一分区的三个 `Ella,1` 先变成 `Ella,3`，减少 Shuffle 量。但并非所有 Reduce 都能直接当 Combiner，求全局均值就是典型反例。

### 15.1.6 作业管理、心跳与任务重试

教材的 Hadoop 1.x 架构使用：

- **JobTracker**：管理与协调整个作业；
- **TaskTracker**：在工作节点上执行 Map 或 Reduce 任务并汇报状态。

TaskTracker 定期发送心跳。若节点失联，调度系统可停止向其分配新任务，并在其他节点重新执行未完成任务。容错依赖的不只是“发现故障”，还包括：

1. 输入数据仍然可读取；
2. 任务能够在别处重新运行；
3. 重试不会产生不可控的重复副作用；
4. 调度器能区分暂时缓慢与真正失败。

> **实践补充**：重试最适合可重入、幂等或有事务保护的任务。若任务每执行一次就无条件向外部系统追加一笔扣款或一行重复结果，简单重试可能放大错误。

## 15.2 常见模型的 MapReduce 实现

### 15.2.1 统计查询模型：局部和如何变成全局和

设数据分为 $D_1,\ldots,D_k$，查询函数为 $f(x,y)$。若目标依赖：

$$
F=\sum_{j=1}^k\sum_{(x_i,y_i)\in D_j}f(x_i,y_i),
$$

则 Map 在各分区计算局部和 $f_j$，Reduce 再计算 $\sum_jf_j$。模型训练若主要依赖这种可分解统计量，就能用相同模式并行化。

如果目标依赖两个和 $F=\sum f$ 与 $G=\sum g$，可以在同一轮扫描中同时收集二者，再计算所需比值或参数。关键是**合并统计量，而不是急于在每个分区合并最终答案**。

#### 分区均值不能直接平均

一个分区为 $D_1=\{1,3\}$，另一个为 $D_2=\{10\}$。全局均值为：

$$
\bar x=\frac{1+3+10}{3}=\frac{14}{3}\approx4.6667.
$$

若直接平均两个分区均值，则得到：

$$
\frac{2+10}{2}=6,
$$

结果不同。正确归约需要传递 $(\text{sum},\text{count})$：

$$
(4,2)+(10,1)=(14,3),
\qquad \bar x=\frac{14}{3}.
$$

只有各分区样本数相同，或按样本数加权时，才能由分区均值恢复全局均值。

### 15.2.2 线性回归：汇总 $X^{\mathsf T}X$ 与 $X^{\mathsf T}y$

线性回归的正规方程为：

$$
\hat w=(X^{\mathsf T}X)^{-1}X^{\mathsf T}y.
$$

令：

$$
A=X^{\mathsf T}X,\qquad b=X^{\mathsf T}y.
$$

它们都能分解到样本：

$$
A=\sum_i x_ix_i^{\mathsf T}
=\sum_{j=1}^k A_j,
\qquad A_j=\sum_{i\in D_j}x_ix_i^{\mathsf T},
$$

$$
b=\sum_i y_ix_i
=\sum_{j=1}^k b_j,
\qquad b_j=\sum_{i\in D_j}y_ix_i.
$$

Map 在每个分区计算 $(A_j,b_j)$，Reduce 得到 $(A,b)$，最后求解 $Aw=b$。矩阵求逆只需要作用在 $d\times d$ 的汇总矩阵上，而不是把全部 $n$ 行样本集中到一个 Reduce。

用三个带截距的样本：

$$
(x,y)=(0,1),(1,3),(2,5),
\qquad x_i=(1,x_i)^{\mathsf T}.
$$

则：

$$
A=\begin{bmatrix}3&3\\3&5\end{bmatrix},
\qquad
b=\begin{bmatrix}9\\13\end{bmatrix},
\qquad
\hat w=A^{-1}b=\begin{bmatrix}1\\2\end{bmatrix}.
$$

即 $\hat y=1+2x$。实际实现通常用稳定的线性方程求解或 QR 等方法，而不是显式计算逆矩阵；若特征共线，还需正则化或其他处理。

### 15.2.3 SVM：每轮并行汇总梯度

软间隔 SVM 的无约束目标可写为：

$$
L(w,b)=\frac12\lVert w\rVert_2^2
+C\sum_{i=1}^n\max\{0,1-y_i(w^{\mathsf T}x_i+b)\}.
$$

在当前参数下，满足：

$$
y_i(w^{\mathsf T}x_i+b)\le 1
$$

的样本对 hinge 损失的次梯度有贡献。记该集合为 $S$，一种次梯度写法为：

$$
\nabla_wL=w+C\sum_{i\in S}(-y_ix_i),
$$

$$
\nabla_bL=C\sum_{i\in S}(-y_i).
$$

每轮 MapReduce 的过程是：

1. 将当前 $(w^{(t)},b^{(t)})$ 提供给所有 Map；
2. Map 判断本分区样本是否进入 $S$，累计 $-y_ix_i$ 与 $-y_i$；
3. Reduce 汇总局部梯度项；
4. 驱动端加上正则项并更新：

$$
w^{(t+1)}=w^{(t)}-\eta\nabla_wL,
\qquad
b^{(t+1)}=b^{(t)}-\eta\nabla_bL.
$$

**边界提示**：$S$ 是当前参数下的梯度活跃集合，不应直接把每一轮集合都当作最终支持向量的严格定义。在 hinge 损失不可导的边界点，还需要明确次梯度约定。若训练 $T$ 轮，教材方案需要 $T$ 轮全局同步与聚合。

### 15.2.4 K-means：按簇汇总向量和与样本数

给定当前质心 $c_1,\ldots,c_K$，Map 对每个样本寻找最近簇：

$$
k_i=\arg\min_k\lVert x_i-c_k\rVert_2^2.
$$

教材写法是 Map 输出 $(k_i,x_i)$，Reduce 对同簇样本求均值。更高效且便于局部合并的形式是输出：

$$
(k_i,(x_i,1)).
$$

Reduce 汇总：

$$
(S_k,n_k)=\sum_{i:k_i=k}(x_i,1),
\qquad
c_k'=\frac{S_k}{n_k}.
$$

每轮都要广播当前质心，收集新质心，再判断是否收敛。空簇没有有效的 $S_k/n_k$，必须预先约定重新初始化、保留旧质心或其他策略。

### 15.2.5 PageRank：边贡献与图结构一起传递

设 $N^-(v)$ 为指向 $v$ 的节点集合，$N^+(u)$ 为 $u$ 指向的节点集合。教材更新式为：

$$
PR(v)=(1-\beta)\frac1n
+\beta\sum_{u\in N^-(v)}
\frac{PR(u)}{|N^+(u)|}.
$$

Map 对每个节点 $u$ 做两类输出：

1. 保留邻接表 $(u,N^+(u))$；
2. 对每个出边 $u\to v$，输出 $(v,PR(u)/|N^+(u)|)$。

Reduce 对相同 $v$ 的数值贡献求和，同时保留图结构，进入下一轮。单个作业只完成一次迭代，需要多轮作业直到满足收敛标准。

#### 教材公式需要补上的悬空节点处理

若某节点没有出边，$|N^+(u)|=0$，无法分配贡献。即使不把该节点作为 Map 的除零输入，其他节点的分数也会因没有接收到它的质量而逐轮流失。

设悬空节点总质量为：

$$
M_{\mathrm{dangling}}=\sum_{u:|N^+(u)|=0}PR(u).
$$

一种常见处理是把这部分质量均匀分给全部节点：

$$
PR_{\mathrm{new}}(v)
=\frac{1-\beta}{n}
+\beta\sum_{u\in N^-(v)}\frac{PR(u)}{|N^+(u)|}
+\beta\frac{M_{\mathrm{dangling}}}{n}.
$$

例如三节点链 $0\to1\to2$，节点 2 无出边，初始 $PR=(1/3,1/3,1/3)$，$\beta=0.85$。一轮后：

- 不处理悬空质量：$(0.05,1/3,1/3)$，总和仅 $0.716667$；
- 均匀回收悬空质量：约 $(0.144444,0.427778,0.427778)$，总和为 1。

这不是可有可无的小细节，而是保持 PageRank 概率质量的关键。后文代码逐项验证。

### 15.2.6 为什么模型训练会放大 MapReduce 的成本

四个模型例子的共同结构是：

```text
广播当前参数 → 扫描分区 → 汇总局部统计量 → 更新参数 → 再进入下一轮
```

每轮之间都存在全局同步，后续轮次依赖前一轮结果，不能把所有迭代简单摊成完全独立的任务。经典 MapReduce 还可能为每轮启动作业、读写中间数据并重新调度，因而多轮训练、图计算与交互查询的成本尤其明显。

这不表示 MapReduce “不能训练模型”，而是说：**能实现与适合高效实现是两回事。**

## 15.3 Spark：分布式数据分析

### 15.3.1 RDD：分区、不可变与血缘

弹性分布式数据集（Resilient Distributed Dataset，RDD）把物理上分布在多个节点的数据抽象成一个逻辑集合。理解 RDD 要抓住四个词：

1. **分区**：数据被拆成可并行处理的 partition；
2. **不可变**：转换生成新的 RDD，而不是原地修改旧 RDD；
3. **血缘**：记录新 RDD 如何从父 RDD 计算而来；
4. **弹性**：可持久化到内存，内存不足时可溢写磁盘，失败时可按血缘重算。

RDD 并不承诺“所有数据永远在内存”。持久化是明确的空间—时间权衡；如果缓存过大、反复溢写，性能也会下降。

创建 RDD 的三种教材方式：

- 并行化已有单机集合；
- 从 HDFS 等外部存储读取；
- 由其他 RDD 转换得到。

### 15.3.2 转换是惰性定义，行动才触发执行

常见转换（Transformation）包括：

- `map`：一条记录映射为一条记录；
- `flatMap`：一条记录映射为零到多条记录并展开；
- `filter`：保留满足条件的记录；
- `join`：按键连接；
- `reduceByKey`：按键聚合，通常涉及 Shuffle。

常见行动（Action）包括：

- `count`：返回记录数；
- `reduce`：归约并返回结果；
- `collect`：把结果收集到 Driver；
- `save...`：保存到外部存储；
- `take` / `top`：取部分结果到 Driver。

转换构造 RDD 之间的依赖关系，形成有向无环图（DAG）。只有行动触发时，系统才根据依赖图调度计算。

**`collect()` 不是“让分布式计算开始”的唯一行动，也不是默认的安全出口。** 如果把超出 Driver 内存的结果全部收回，集群计算仍可能因单机收集而失败。需要大结果时，应优先写回分布式存储或只取必要摘要。

### 15.3.3 窄依赖、宽依赖与 Shuffle

**学习补充**：理解执行成本时，可以进一步区分：

| 类型 | 父分区与子分区关系 | 常见操作 | 成本特征 |
|---|---|---|---|
| 窄依赖 | 每个父分区只被少数对应子分区使用 | `map`、`filter` | 通常可在同一流水线计算 |
| 宽依赖 | 多个父分区的数据需要重新分发 | `reduceByKey`、部分 `join` | 通常涉及 Shuffle 与新的阶段 |

连续 `map` 和 `filter` 不只是“语法更短”，还可能在同一分区内流水执行。`reduceByKey` 前若有可结合的本地预聚合，则通常比无约束地把全部单条记录 `groupByKey` 后再处理更有利。

### 15.3.4 Driver、Executor 与运行流程

Spark 程序可概括为：

- **Driver**：创建运行上下文、构建逻辑计划、申请资源、调度任务并收集必要结果；
- **Executor**：在工作节点执行分区任务、保存部分缓存数据；
- **集群管理器**：协调集群资源；
- **外部存储**：提供 HDFS、Hive、HBase、对象存储或本地文件等数据来源与结果去向。

单机本地模式可以模拟执行流程，便于学习语法和验证小规模逻辑，但不能用来证明集群性能、容错能力或网络开销。

教材的中文分词例子遵循：

```text
textFile → map(segment) → top(100)
```

其中 `map(segment)` 是转换，`top(100)` 是行动。真正运行还需要匹配 Spark、Python、分词库和集群地址等环境；笔记不把这个片段当作无需环境即可执行的代码。

### 15.3.5 Spark 与 Hadoop：不是简单替代关系

| 对比维度 | 经典 Hadoop MapReduce | Spark RDD |
|---|---|---|
| 主要定位 | 分布式批处理框架 | 支持多阶段与迭代的分布式计算框架 |
| 中间数据 | 作业间常借助磁盘共享 | 可按血缘组织并选择内存持久化 |
| 操作抽象 | 以 Map、Reduce 为核心 | 多种转换与行动 |
| 迭代与交互 | 多轮作业成本较突出 | 复用缓存和 DAG 执行更便利 |
| 存储职责 | Hadoop 生态含 HDFS | Spark 本身不替代分布式文件系统 |
| 共同成本 | Shuffle、倾斜、调度、故障 | 同样存在，不会因使用 Spark 消失 |

Spark 可以读取 HDFS，也可以把结果写回 HDFS。因而“用 Spark”不意味着“不需要 Hadoop 生态”，更不意味着所有作业都必然更快。性能取决于数据复用、缓存、Shuffle、任务大小、资源和实现方式。

教材还列出 Spark SQL、Spark Streaming、MLlib 和 GraphX，分别面向结构化数据、流数据、机器学习和图计算。这些是教材版本中的组件划分，学习时应把**数据类型与执行思想**放在组件名称之前。

## 15.4 其他分布式系统与生态工具

| 工具 | 教材中的定位 | 应理解的边界 |
|---|---|---|
| ZooKeeper | 分布式协调、命名与配置同步 | 不是大规模业务数据文件系统 |
| Hive | 面向结构化数据的数据仓库，提供类 SQL 查询 | 查询最终会转为底层执行任务，不等于单机数据库 |
| HBase | 分布式列式数据库，受 Bigtable 启发 | 与 HDFS 文件扫描和 Hive 仓库查询定位不同 |
| Mahout | 分布式机器学习与数据挖掘算法库 | 算法实现随版本与底层引擎演进 |
| Pig | 高层数据流处理工具 | 教材称其支持实时分析，不应据此推断现代低延迟流处理保证 |
| Sqoop | 在传统数据库与 Hadoop 之间传输数据 | 主要解决数据导入导出，不是计算模型 |
| Flume | 日志等数据采集 | 与数据库批量迁移工具场景不同 |
| Oozie | Hadoop 作业流调度 | 负责依赖与流程，不替代具体计算引擎 |
| Ambari | 集群部署、配置、维护与监控 | 管理工具不替代存储和计算组件 |

学习生态时，不必把历史组件都当成今天唯一选择，但应保留它们所代表的永久问题：**数据怎样进入、怎样组织、怎样查询、怎样调度、怎样协调、怎样监控。**

## 15.5 可运行综合示例

下面代码只依赖 Python 标准库与 NumPy，用单机分区模拟分布式计算语义，不需要 Hadoop 或 Spark 集群。它验证：

1. HDFS 块数与副本存储；
2. 教材词频统计及正确的全局均值；
3. 线性回归的局部矩阵汇总；
4. SVM 的一轮梯度；
5. K-means 的一轮质心更新；
6. PageRank 悬空质量修正；
7. 类似 RDD 转换的惰性执行与行动触发。

它是**逻辑验证，不是性能测试，也没有模拟真实网络、故障或调度器**。

```python
from collections import defaultdict
import math
import numpy as np

# 1. HDFS：192 MiB 文件、64 MiB 块、3 个副本。
file_mib = 192
block_mib = 64
replication = 3
block_count = math.ceil(file_mib / block_mib)
replica_count = block_count * replication
replicated_mib = file_mib * replication
print("HDFS 逻辑块、总副本、副本存储 MiB:", block_count, replica_count, replicated_mib)

# 2. MapReduce 词频：Map → 分区局部聚合 → Reduce。
documents = [
    "Jane Amy Lisa",
    "Ella Ella Lisa",
    "Jane Ella Amy",
]
map_output = [
    (word, 1)
    for document in documents
    for word in document.split()
]
grouped = defaultdict(list)
for word, value in map_output:
    grouped[word].append(value)
word_counts = {word: sum(values) for word, values in grouped.items()}
print("Map 输出:", map_output)
print("词频:", word_counts)

partitions = [[1.0, 3.0], [10.0]]
partial_stats = [(sum(part), len(part)) for part in partitions]
global_sum = sum(item[0] for item in partial_stats)
global_count = sum(item[1] for item in partial_stats)
global_mean = global_sum / global_count
naive_partition_mean = sum(sum(part) / len(part) for part in partitions) / len(partitions)
print("局部 (sum,count):", partial_stats)
print("正确全局均值／直接平均分区均值:", global_mean, naive_partition_mean)

# 3. 线性回归：每个分区返回 A_j=X_j^T X_j 和 b_j=X_j^T y_j。
regression_partitions = [
    [(0.0, 1.0), (1.0, 3.0)],
    [(2.0, 5.0)],
]
partial_matrices = []
for part in regression_partitions:
    X_part = np.array([[1.0, x] for x, _ in part])
    y_part = np.array([y for _, y in part])
    A_part = X_part.T @ X_part
    b_part = X_part.T @ y_part
    partial_matrices.append((A_part, b_part))
A = sum((item[0] for item in partial_matrices), start=np.zeros((2, 2)))
b = sum((item[1] for item in partial_matrices), start=np.zeros(2))
regression_w = np.linalg.solve(A, b)
print("线性回归局部矩阵:", partial_matrices)
print("全局 A、b、参数:", A, b, regression_w)

# 4. SVM：w=0.5,b=0,C=1，汇总 hinge 活跃样本的次梯度。
svm_samples = [(1.0, 1.0), (3.0, 1.0), (-1.0, -1.0), (2.0, -1.0)]
w_svm, b_svm, C = 0.5, 0.0, 1.0
active = [sample for sample in svm_samples if sample[1] * (w_svm * sample[0] + b_svm) <= 1]
grad_w = w_svm + C * sum(-y * x for x, y in active)
grad_b = C * sum(-y for _, y in active)
svm_loss = 0.5 * w_svm**2 + C * sum(max(0.0, 1 - y * (w_svm * x + b_svm)) for x, y in svm_samples)
print("SVM 活跃样本:", active)
print("SVM 损失与梯度:", svm_loss, grad_w, grad_b)

# 5. K-means：初始质心 0 和 10，按簇汇总 (向量和,样本数)。
points = [1.0, 2.0, 8.0, 9.0]
centers = [0.0, 10.0]
cluster_stats = defaultdict(lambda: [0.0, 0])
assignments = []
for point in points:
    cluster = min(range(len(centers)), key=lambda k: (point - centers[k])**2)
    assignments.append(cluster)
    cluster_stats[cluster][0] += point
    cluster_stats[cluster][1] += 1
new_centers = [cluster_stats[k][0] / cluster_stats[k][1] for k in range(len(centers))]
print("K-means 分配、局部统计与新质心:", assignments, dict(cluster_stats), new_centers)

# 6. PageRank：0→1→2，节点 2 悬空；只计算一轮。
edges = {0: [1], 1: [2], 2: []}
n = len(edges)
beta = 0.85
pr = {node: 1 / n for node in edges}
incoming = defaultdict(float)
dangling_mass = 0.0
for node, neighbors in edges.items():
    if neighbors:
        for target in neighbors:
            incoming[target] += pr[node] / len(neighbors)
    else:
        dangling_mass += pr[node]
base = (1 - beta) / n
pagerank_without_dangling = {node: base + beta * incoming[node] for node in edges}
pagerank_corrected = {
    node: base + beta * incoming[node] + beta * dangling_mass / n
    for node in edges
}
print("PageRank 入边贡献与悬空质量:", dict(incoming), dangling_mass)
print("未修正 PageRank:", pagerank_without_dangling, "总和:", sum(pagerank_without_dangling.values()))
print("修正后 PageRank:", pagerank_corrected, "总和:", sum(pagerank_corrected.values()))

# 7. TinyRDD 只用于演示惰性：转换只记录操作，collect 才逐分区执行。
execution_log = []

class TinyRDD:
    def __init__(self, data, operations=()):
        self.data = data
        self.operations = operations

    def map(self, function):
        return TinyRDD(self.data, self.operations + (("map", function),))

    def flat_map(self, function):
        return TinyRDD(self.data, self.operations + (("flatMap", function),))

    def collect(self):
        execution_log.append("collect")
        output = []
        for partition in self.data:
            for record in partition:
                values = [record]
                for operation, function in self.operations:
                    if operation == "map":
                        values = [function(value) for value in values]
                    else:
                        values = [item for value in values for item in function(value)]
                output.extend(values)
        return output

tiny_rdd = TinyRDD([["Jane Amy Lisa"], ["Ella Ella Lisa"]])
transformed = tiny_rdd.flat_map(str.split).map(lambda word: (word, 1))
log_before_action = list(execution_log)
lazy_result = transformed.collect()
log_after_action = list(execution_log)
print("行动前日志:", log_before_action)
print("行动后日志:", log_after_action)
print("惰性转换结果:", lazy_result)
```

关键输出为：

```text
HDFS：3 个逻辑块、9 个副本、576 MiB 原始副本存储
词频：Amy=2，Ella=3，Jane=2，Lisa=2
全局均值：14/3≈4.666667；直接平均分区均值：6
线性回归：A=[[3,3],[3,5]]，b=[9,13]，w=[1,2]
SVM：损失 3.125，梯度 grad_w=0.5，grad_b=1
K-means：分配 [0,0,1,1]，新质心 [1.5,8.5]
未修正 PageRank 总和：约 0.716667；修正后总和：1
行动前无执行日志；collect 后记录一次行动
```

## 实践补充：怎样判断一个任务该不该分布化

### 15.5.1 先写清单机基线

在引入集群前，先记录：

1. 数据总量、记录数和单条大小；
2. 单机扫描、训练或查询耗时；
3. 内存峰值和磁盘中间结果；
4. 算法需要扫描数据几遍、迭代几轮；
5. 结果需要返回到哪里、返回多大；
6. 失败重跑和数据恢复成本。

如果压缩、列裁剪、流式读取、更有效算法或单机能已经满足时限，就不必仅因“数据看起来很大”而增加分布式复杂度。

### 15.5.2 把计算改写为可归约统计量

适合并行归约的例子：

- 计数：$(n_j)\to\sum n_j$；
- 求和：$(s_j)\to\sum s_j$；
- 均值：$(s_j,n_j)\to\sum s_j/\sum n_j$；
- 平方和与矩阵：$(X_j^{\mathsf T}X_j)\to\sum X_j^{\mathsf T}X_j$；
- 梯度：$(g_j)\to\sum g_j$；
- K-means：$(S_{j,k},n_{j,k})\to(\sum_jS_{j,k})/(\sum_jn_{j,k})$。

不能直接照搬的例子包括未加权分区均值、分区中位数、分区模型指标和分区训练好的最终模型。它们并非永远无法分布式计算，但需要额外算法与误差分析，不能简单求平均。

### 15.5.3 把 Shuffle 当作一等成本

优化前先问：

- 是否真的需要按 key 全量分组？
- Map 端能否先做正确的局部聚合？
- 是否只传输必要字段？
- key 是否高度倾斜？
- 分区数是否太少导致并行不足，或太多导致小任务泛滥？
- 多轮迭代是否复用了可复用的数据？

“减少 Shuffle”不是绝对原则。如果局部聚合本身成本极高或改变数值语义，就需要在正确性和性能之间重新权衡。

### 15.5.4 容错还包括结果正确性

一个任务“重新跑完”不代表结果已经正确。还应检查：

- 输入副本或血缘是否足以重算；
- 输出写入是否支持覆盖、事务或去重；
- 非确定性随机种子是否受控；
- 重试是否产生重复记录或重复副作用；
- 全局聚合是否遗漏迟到、失败或重复分区的数据；
- 收敛判断是参数变化、目标变化、固定轮数还是验证指标。

## 常见误区与边界

1. **分布式计算就是增加机器。** 不对。还涉及数据分解、通信、调度、容错与结果一致性。
2. **机器数量翻倍，任何任务耗时都减半。** 不对。串行部分、Shuffle、倾斜和调度会限制加速。
3. **HDFS 的块大小永远是 64 MB。** 不对。这是教材历史配置示例，实际是可配置的。
4. **每个文件都恰好占满一个数据块。** 不对。最后一个块通常可以不足块大小。
5. **三个副本等于完成了异地灾备。** 不对。同集群副本不能覆盖所有误删与站点级风险。
6. **NameNode 负责传输所有文件内容。** 不对。它主要管理元数据，实际块读写由客户端与 DataNode 完成。
7. **Secondary NameNode 就是自动热备。** 不对。经典检查点辅助角色不能直接推断为故障自动接管。
8. **HDFS 很适合海量小文件和低延迟随机修改。** 不对。教材明确把小文件与随机修改列为弱项。
9. **存储块、计算分区和 key 分区完全相同。** 不对。它们属于不同层次。
10. **Map 的输出必须一条输入对应一条输出。** 不对。Map 或 flatMap 可以产生零到多条记录。
11. **Shuffle 只负责排序，不涉及多少成本。** 不对。它常包含网络、排序、溢写与序列化。
12. **所有 Reduce 都能直接作为 Combiner。** 不对。必须满足聚合语义与实现约定，均值不能直接套求和逻辑。
13. **任务重试一定让结果正确。** 不对。非幂等副作用可能因重试重复发生。
14. **分区均值再平均就是全局均值。** 仅在分区大小相等或正确加权时成立。
15. **线性回归必须把全部样本集中后才能求参数。** 不对。$X^{\mathsf T}X$ 与 $X^{\mathsf T}y$ 可分区求和。
16. **分布式线性回归要对完整 $n\times n$ 矩阵求逆。** 不对。汇总的是 $d\times d$ 矩阵，实际还宜用稳定求解方法。
17. **SVM 每轮只需传标签，不需要当前参数。** 不对。判断 hinge 活跃样本需要当前 $w,b$。
18. **当前梯度活跃集合与最终支持向量定义可以无条件混用。** 不对。二者语境不同，边界点还需次梯度约定。
19. **K-means 只需传每簇样本均值。** 不完整。还要传样本数，才能正确跨分区合并。
20. **K-means 空簇也能直接除以样本数。** 不对。空簇需要显式处理策略。
21. **PageRank 每轮只传分数，不需要图结构。** 不完整。下一轮仍需邻接关系，结构要保留或可重新获得。
22. **PageRank 可以忽略无出边节点。** 不对。会损失概率质量，需采用明确的悬空处理。
23. **所有迭代算法都能在经典 MapReduce 上高效运行。** 不对。多轮作业、同步与磁盘中间结果成本可能很高。
24. **RDD 的所有数据永远驻留内存。** 不对。持久化可选，内存不足可溢写或重算。
25. **调用 map 后 Spark 已立即扫描完所有数据。** 不对。转换通常是惰性的，行动才触发执行。
26. **collect 越大越能体现集群优势。** 不对。大结果可能压垮 Driver，应控制回传规模。
27. **Spark 不存在 Shuffle 和数据倾斜。** 不对。宽依赖仍会产生这些问题。
28. **Spark 是 HDFS 的替代品。** 不对。Spark 是计算框架，可读取并写回 HDFS。
29. **本地 Spark 运行成功就证明集群性能与容错合格。** 不对。本地测试未覆盖真实网络、资源和故障。
30. **任何历史生态组件名称都比问题本身重要。** 不对。应先识别存储、计算、协调、调度与查询需求，再选工具。

## 一句话总结

分布式计算把数据切成可存储、可并行、可重算的分区，以局部统计和全局归约完成计算；HDFS 解决块与副本，MapReduce 以 key-value 和 Shuffle 组织批处理，Spark 通过 RDD 血缘、惰性执行和可选持久化便利多阶段任务——**真正的难点是通信、迭代、倾斜与容错，而不是把单机函数机械地放到更多机器上。**

## 复习卡片

- **HDFS**：大文件分块、多副本，NameNode 管元数据，DataNode 管块。
- **Secondary NameNode**：经典检查点辅助角色，不等同热备。
- **MapReduce**：Map 转换记录，Shuffle 按 key 重分发，Reduce 聚合。
- **Combiner**：正确的局部预聚合可减少 Shuffle，但不能破坏语义。
- **全局均值**：传递 $(\text{sum},\text{count})$，不能直接平均分区均值。
- **线性回归**：$\sum x_ix_i^{\mathsf T}$ 与 $\sum y_ix_i$ 可分布式汇总。
- **SVM**：每轮广播 $w,b$，汇总 hinge 活跃样本的梯度。
- **K-means**：每轮按簇汇总 $(\text{向量和},\text{样本数})$。
- **PageRank**：传出边贡献并保留图结构，悬空质量要回收。
- **RDD**：分区、不可变、血缘、可持久化。
- **转换与行动**：转换惰性构 DAG，行动触发执行。
- **宽依赖**：通常带来 Shuffle，倾斜会让少数分区拖慢整体。
- **Spark 与 Hadoop**：计算框架与存储生态可组合，不是简单替代。
- **工程判断**：先单机基线，再看数据复用、通信、调度和故障成本。

## 自测问题

1. 192 MiB 文件按 64 MiB 分块并保存 3 个副本，有多少逻辑块、多少块副本？不计额外开销占多少原始存储？
2. 为什么 Secondary NameNode 不能直接理解为 NameNode 的热备？NameNode 与 DataNode 各管什么？
3. 对教材三行文档写出词频统计的 Map 输出、Shuffle 分组与最终结果。
4. 两个分区为 $\{1,3\}$ 与 $\{10\}$，全局均值是多少？直接平均分区均值为什么出错？
5. 用 $(0,1),(1,3),(2,5)$ 三个带截距样本计算 $A=X^{\mathsf T}X$、$b=X^{\mathsf T}y$ 与回归参数。
6. 在正文 SVM 参数与样本下，哪些样本进入梯度活跃集合？复算损失与两个梯度。
7. 对点 $1,2,8,9$ 和初始质心 $0,10$，用 $(\text{sum},\text{count})$ 完成一轮 K-means 更新。
8. 三节点链 $0\to1\to2$ 中，为什么未修正 PageRank 的总和不是 1？修正项是什么？
9. 比较 RDD 转换与行动，并说明为什么 `map` 后尚未必然执行，而大型 `collect` 可能失败。
10. 一个需要 100 轮迭代的模型，为什么可能比一次词频统计更不适合经典 MapReduce？Spark 能改善什么，又不能消除什么？

## 参考答案

1. 逻辑块为 $\lceil192/64\rceil=3$，块副本为 $3\times3=9$，原始副本存储为 $192\times3=576$ MiB。实际还要考虑最后一个块是否占满及系统额外开销。
2. 经典 Secondary NameNode 主要辅助元数据检查点，不应据此推断自动故障切换。NameNode 管命名空间和块位置等元数据，DataNode 保存并读写实际块。
3. Map 对每个词输出 $(w,1)$；Shuffle 后 Amy、Jane、Lisa 各有两个 1，Ella 有三个 1；Reduce 得 Amy=2、Ella=3、Jane=2、Lisa=2。
4. 正确均值是 $(1+3+10)/3=14/3\approx4.6667$。分区均值为 2 和 10，直接平均得 6，忽略了两个分区样本数分别为 2 和 1。
5. $A=\begin{bmatrix}3&3\\3&5\end{bmatrix}$，$b=(9,13)^{\mathsf T}$，解得 $w=(1,2)^{\mathsf T}$，即 $\hat y=1+2x$。MapReduce 只需汇总各分区的 $A_j,b_j$。
6. 活跃样本为 $(1,1),(-1,-1),(2,-1)$，$(3,1)$ 不活跃。损失为 $0.125+0.5+0.5+2=3.125$；$\nabla_wL=0.5+(-1-1+2)=0.5$，$\nabla_bL=-1+1+1=1$。
7. 分配为 $[0,0,1,1]$；簇 0 汇总 $(3,2)$，簇 1 汇总 $(17,2)$，新质心分别为 $1.5$ 和 $8.5$。
8. 节点 2 的 $1/3$ 质量没有出边可传，未修正总和为 $1-0.85/3\approx0.716667$。把 $\beta M_{\mathrm{dangling}}/n$ 加到每个节点后，总质量恢复为 1。
9. 转换只定义新 RDD 与依赖，行动才触发执行。`collect` 把结果集中到 Driver，若超过其内存或网络承载能力，就可能成为新的单机瓶颈。
10. 每轮都依赖上一轮参数，经典 MapReduce 需要重复作业调度、扫描和中间数据共享，成本随轮数累积。Spark 可通过 DAG 与持久化改善数据复用，但不能消除全局同步、Shuffle、倾斜和故障处理。

## 全书收束与附录建议

第十五章把前面模型放进集群环境：第三章回归变成矩阵统计量归约，第四章 SVM 变成梯度归约，第六章 K-means 变成簇统计量归约，第十三章 PageRank 变成图上的多轮消息聚合。至此，全书“数据—模型—算法—评价—计算系统”的主线已经闭合。

后续复习可按需要查阅附录：矩阵运算对应线性回归与 PageRank，概率论对应统计模型，优化对应迭代训练，距离对应 K-means，模型评估则始终约束分布式实现的最终目标。**系统只改变计算组织方式，不会自动修复错误的数据、模型或评价。**

## 参考资料

- 欧高炎、朱占星、董彬等：《数据科学导引》第十五章“分布式计算”，高等教育出版社，原书第 325—344 页。
