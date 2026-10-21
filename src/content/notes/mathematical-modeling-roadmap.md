---
title: "数学建模：学习地图与完整流程"
description: "从现实问题、假设和变量走到求解、验证与表达"
pubDate: 2026-10-21
updatedDate: 2026-10-21
order: 1
category: '数学建模'
tags: ["学习路线","建模流程","零基础"]
---

> [专题学习地图](/notes/mathematical-modeling-roadmap/) · 第 1 / 27 篇

## 学习目标与先修知识

学完本篇，你应能把一句生活问题改写为“已知什么、决定什么、优化什么、受什么限制、如何验证”。不需要先学完高等数学；遇到矩阵、导数和概率，回到下一篇基础课即可。

每篇按同样的 11 个部分组织。第一次阅读先看背景、手算和结果；第二次把代码中的变量逐一对应到公式；第三次遮住答案完成练习。看懂答案只是第一步，能换一组数字独立计算，才说明抓住了机制。

## 问题背景与使用时机

“明天应准备多少份午餐”包含至少三个不同任务：预测明天需求、决定备货数量、比较缺货与浪费损失。预测模型输出一个数，不会自动替你决定应该备多少。

数学模型是对现实的有目的简化。我们挑出与问题有关的因素，用方程、概率、规则或网络描述关系，再检查这些简化是否足以支持决策。算法负责求解模型；同一模型可以用不同算法求解，换算法不等于改好了模型。

## 核心概念与符号

| 名称 | 直观含义 | 面包生产例子 |
|---|---|---|
| 数据 | 已观察到的记录 | 上周销量、原料库存 |
| 参数 | 本次计算中给定的量 | 每批原料消耗、单位利润 |
| 决策变量 | 允许我们选择的量 | A、B 各做几批 |
| 目标函数 | 判断方案好坏的规则 | 总利润 |
| 约束 | 必须满足的条件 | 原料、烤箱容量、整数批次 |
| 假设 | 为了计算而采用的简化 | 每批利润固定，产品全部售出 |
| 输出 | 模型给出的结果 | 生产计划、预期收益、风险范围 |

模型必须同时交代单位。例如“2 千克/批 × 3 批 = 6 千克”，约束两边才能比较。预测值通常记为 $\hat y$，帽子提醒我们它是估计，不是真实观测 $y$。

![建模流程及验证反馈](/images/math-modeling/learning/modeling-cycle.svg)

验证步骤提醒我们：若结果不符合证据，需要回到数据、假设或模型重新检查，再开始下一轮计算。

## 手算一个最小例子

一家小店今天有 8 千克原料、最多烘烤 5 批。A 每批耗 2 千克，利润 3 元；B 每批耗 1 千克，利润 2 元。假设利润已扣除可变成本，批次必须为整数，且这些产品都能售出。

令 $x$、$y$ 为 A、B 批数：

$$
\max P=3x+2y,\qquad 2x+y\le8,\quad x+y\le5,\quad x,y\in\mathbb Z_{\ge0}.
$$

连续可行域的角点为 $(0,0),(4,0),(3,2),(0,5)$，对应利润为 $0,12,13,10$。这些角点恰好也是整数点，因此这里整数模型也在 $(3,2)$ 取得最大利润 13 元。

检查：原料 $2\times3+2=8$ 千克，批数 $3+2=5$，两个约束都满足。不要把这个特殊例子理解为“整数规划总能直接取连续解”。

## 建模步骤与完整案例

完整建模按七步进行：

1. **重述问题**：今天怎样安排两类面包的批次，使利润最大。
2. **整理数据**：资源、单耗、利润、售出条件，分别注明单位与来源。
3. **提出假设**：固定利润、没有换线成本、产品能够售出。若有销量上限，应加进约束。
4. **定义变量**：把“生产计划”写成可计算的 $x,y$。
5. **建立关系**：目标、资源约束和整数约束缺一不可。
6. **求解并验证**：先用枚举作小规模基准，再用优化求解器处理大规模。
7. **解释与敏感性分析**：减少原料或改变利润，重新求解，观察计划是否改变。

最终报告应同时给出计划、资源使用、收益、主要假设和不能覆盖的情况。不能只交一张求解器截图。

## Python 实现与结果解释

配套文件：[查看或下载本篇完整脚本](/downloads/math-modeling/mathematical-modeling-roadmap.py)。

下面只用 Python 标准库，直接枚举所有可能批次。每一行候选记录都是“一个具体方案”，过滤条件就是数学约束。

```python
def solve_plan(material):
    feasible = [
        (3*x + 2*y, x, y)
        for x in range(6)
        for y in range(6)
        if x + y <= 5 and 2*x + y <= material
    ]
    return max(feasible)

profit, x, y = solve_plan(8)
print("利润、A批数、B批数:", profit, x, y)
print("原料用量:", 2*x + y, "总批数:", x + y)
print("原料减少后:", solve_plan(7))
assert (profit, x, y) == (13, 3, 2)
assert 2*x + y <= 8 and x + y <= 5
```

原料为 7 时，最优方案变为 A 做 2 批、B 做 3 批，利润 12。这个变化说明生产建议依赖资源条件，而不是永远固定为“3 加 2”。枚举在这里可靠，是因为变量取值范围小且被完整遍历。

## 方法选择与常见误区

先按问题的输出选择方法：

| 问题在问什么 | 首先考虑 | 本专题入口 |
|---|---|---|
| 数据是否有问题 | 清洗、探索分析 | 数据预处理、可视化 |
| 差异和不确定性多大 | 统计区间与检验 | 统计模型 |
| 数值或类别是什么 | 回归、分类 | 回归、KNN、树模型 |
| 变量怎样随时间变化 | 动力系统或时序模型 | 微分方程、时间序列 |
| 资源怎样分配 | 目标与约束优化 | 规划优化 |
| 哪个已有方案更合适 | 指标、权重与评价 | 综合评价 |
| 节点如何连接或运输 | 图与网络模型 | 图论 |
| 随机风险如何量化 | 模拟与概率 | 随机模拟 |

“套一个复杂模型”不是建模步骤。先写出简单基线：预测可用历史均值或昨天数值，优化可用平均分配。复杂方法必须说明改进发生在哪里，以及增加了什么成本。

## 练习

1. 小店的单位利润与生产批数，哪个是参数，哪个是决策变量？
2. 若 B 今天最多卖 1 批，应增加什么条件？原来的方案还可行吗？
3. 为什么“模型算出利润最大”仍不能保证真实经营利润最大？
4. 不看答案，用纸列出原料为 7 时的最好方案。

## 参考答案

1. 单位利润是给定参数；生产批数是决策变量。实际项目中参数也可能需要先估计。
2. 增加 $y\le1$。原方案 $y=2$ 不再可行；重新求解可得 $(x,y)=(4,0)$，利润 12。
3. 结论只在模型与假设下成立。销量不足、利润估计偏差、设备故障或遗漏约束都可能改变结果。
4. $(2,3)$：耗料 $4+3=7$，共 5 批，利润 $6+6=12$。检查其他可行整数点，均不能超过 12。

## 复习卡片与下一步

**推荐阅读顺序**

第一阶段读基础、数据处理、可视化、统计与回归；第二阶段读几何、微分方程、规划、图论和评价；第三阶段读时间序列与机器学习；最后用综合案例把分析变成决策。

- [必备数学与 Python 基础](/notes/mathematical-modeling-foundations/)：用小例子理解向量矩阵、导数、概率与数组。
- [数据预处理与插值](/notes/mathematical-modeling-data-preprocessing/)：从缺失、重复和量纲走到可用数据，理解插值与信息泄漏。
- [数据可视化与探索分析](/notes/mathematical-modeling-data-visualization/)：围绕问题选择图表，读懂分布、关系和异常。
- [统计模型与假设检验](/notes/mathematical-modeling-statistics/)：从样本波动理解区间、p 值和检验选择。
- [回归与曲线拟合](/notes/mathematical-modeling-regression/)：从手算最小二乘到带验证的预测模型。
- [几何模型与线性代数](/notes/mathematical-modeling-geometry/)：从距离、投影和旋转走到定位、方程与分解。
- [微分方程与动力系统](/notes/mathematical-modeling-differential-equations/)：从变化率建立模型，用解析解和数值解预测状态。
- [线性规划与整数优化](/notes/mathematical-modeling-optimization/)：把资源分配写成目标与约束，再检查解的可行性。
- [图论与网络优化](/notes/mathematical-modeling-graphs/)：用节点和边建模，区分最短路、生成树、流与路线规划。
- [权重生成与综合评价](/notes/mathematical-modeling-evaluation/)：用 AHP、熵权和 TOPSIS 比较方案并检查排名稳健性。
- [主成分分析与降维](/notes/mathematical-modeling-pca/)：从相关特征找到低维表示，理解方差、投影与解释。
- [时间序列分析与验证](/notes/mathematical-modeling-time-series/)：识别趋势、季节和相关性，建立按时间验证的流程。
- [时间序列之移动平均法](/notes/mathematical-modeling-moving-average-method/)：手算平滑与一步预测，理解窗口、滞后和趋势修正。
- [时间序列之指数平滑法](/notes/mathematical-modeling-exponential-smoothing/)：从递推权重理解水平、趋势与季节模型。
- [ARIMA 与相关时间序列](/notes/mathematical-modeling-arima/)：从差分和自相关理解 AR、MA、ARIMA 的选择与诊断。
- [机器学习流程与 KNN](/notes/mathematical-modeling-ml-sklearn-knn/)：从距离投票理解监督学习、交叉验证和评估。
- [朴素贝叶斯](/notes/mathematical-modeling-ml-naive-bayes/)：从条件概率与平滑走到文本分类。
- [线性判别与支持向量机](/notes/mathematical-modeling-ml-lda-svm/)：从投影与间隔理解 LDA、软间隔和核函数。
- [树模型与集成学习](/notes/mathematical-modeling-ml-tree-ensemble/)：从手算分裂走到剪枝、随机森林和梯度提升。
- [聚类算法](/notes/mathematical-modeling-ml-clustering/)：通过手算分组理解 K-Means、DBSCAN、层次聚类和 GMM。
- [神经网络与反向传播](/notes/mathematical-modeling-ml-neural-network/)：从单个神经元的手算更新走到可验证的 MLP。
- [进化计算与群体智能](/notes/mathematical-modeling-evolutionary/)：理解 GA、PSO、ACO、SA 的更新与公平比较。
- [随机模拟与不确定性分析](/notes/mathematical-modeling-simulation/)：用蒙特卡洛、马尔可夫和排队案例理解风险与误差。
- [图像文本与信号建模](/notes/mathematical-modeling-multimodal/)：把非表格数据变成特征，理解分割、频谱与融合。
- [综合案例 从销量预测到备货决策](/notes/mathematical-modeling-integrated-case/)：完整连接数据检查、基线、验证、成本优化和报告。
- [研究生数学建模赛题的阅读与方法选择](/notes/mathematical-modeling-machine-learning-data-processing/)：结合 2021 至 2025 年题目学习如何拆任务和验证方案。

### 运行代码的准备

使用 Python 3.11 或兼容版本。先下载[完整示例代码包](/downloads/math-modeling/math-modeling-examples.zip)并解压，进入解压后的文件夹。包内每个英文文件对应一篇笔记，数据由代码生成或来自 scikit-learn 内置数据，不需要先准备 Bitcoin 或 Titanic 文件。

在终端执行：

```text
python -m venv .venv
Windows 命令提示符：.venv\Scripts\activate
macOS 或 Linux：source .venv/bin/activate
python -m pip install -r requirements.txt
python mathematical-modeling-roadmap.py
```

激活命令只运行与你的系统对应的一行。依赖安装需要网络；首次可先运行只用标准库的学习地图例子。若提示找不到模块，先确认安装依赖和运行程序使用的是同一个 Python 环境。绘图程序会打开图窗，关闭后继续运行即可。

### 与原参考教程怎样对照

| 原教程内容 | 本专题对应入口 |
|---|---|
| 绪论 | [学习地图与完整流程](/notes/mathematical-modeling-roadmap/) |
| 几何与方程 | [几何模型与线性代数](/notes/mathematical-modeling-geometry/)、[必备数学与 Python 基础](/notes/mathematical-modeling-foundations/) |
| 微分方程与动力系统 | [微分方程与动力系统](/notes/mathematical-modeling-differential-equations/)、[随机模拟与不确定性分析](/notes/mathematical-modeling-simulation/) |
| 函数极值与规划 | [线性规划与整数优化](/notes/mathematical-modeling-optimization/)、[综合案例 从销量预测到备货决策](/notes/mathematical-modeling-integrated-case/) |
| 网络与图论 | [图论与网络优化](/notes/mathematical-modeling-graphs/) |
| 进化计算与群体智能 | [进化计算与群体智能](/notes/mathematical-modeling-evolutionary/) |
| 数据处理与拟合 | [数据预处理与插值](/notes/mathematical-modeling-data-preprocessing/)、[数据可视化与探索分析](/notes/mathematical-modeling-data-visualization/)、[回归与曲线拟合](/notes/mathematical-modeling-regression/) |
| 权重与评价 | [权重生成与综合评价](/notes/mathematical-modeling-evaluation/)、[主成分分析与降维](/notes/mathematical-modeling-pca/) |
| 时间序列 | [时间序列分析与验证](/notes/mathematical-modeling-time-series/)、[时间序列之移动平均法](/notes/mathematical-modeling-moving-average-method/)、[时间序列之指数平滑法](/notes/mathematical-modeling-exponential-smoothing/)、[ARIMA 与相关时间序列](/notes/mathematical-modeling-arima/) |
| 统计与机器学习 | [统计模型与假设检验](/notes/mathematical-modeling-statistics/)、[机器学习流程与 KNN](/notes/mathematical-modeling-ml-sklearn-knn/)及后续模型 |
| 图像文本与信号 | [图像文本与信号建模](/notes/mathematical-modeling-multimodal/) |

每篇都应留下三样东西：一张能解释的手算纸、一段自己运行过的代码、一句话说明适用条件。下一篇从数组、矩阵、导数和概率补起。

[下一篇 必备数学与 Python 基础](/notes/mathematical-modeling-foundations/)

## 参考资料

- [Datawhale《数学建模导论》目录](https://datawhalechina.github.io/intro-mathmodel/#/)：用于建立知识目录，建议与本篇的手算和案例对照阅读。

- 本专题的主要学习参考为 Datawhale《数学建模导论》，项目负责人和主编为马世拓；章节作者及许可信息见[原项目](https://github.com/datawhalechina/intro-mathmodel)。下面的数字案例和代码按学习目标重新设计。
- 原教程覆盖的偏微分方程、DEA、GARCH 等进阶内容可在完成对应基础后继续阅读；本路线先讲清常用模型的建立、验证与选择。
