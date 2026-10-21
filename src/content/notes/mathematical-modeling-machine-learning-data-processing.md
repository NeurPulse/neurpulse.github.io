---
title: "数学建模：研究生数学建模赛题的阅读与方法选择"
description: "结合 2021 至 2025 年题目学习如何拆任务和验证方案"
pubDate: 2026-08-16
updatedDate: 2026-10-21
order: 27
category: '数学建模'
tags: ["数学建模","机器学习","深度学习","数据处理","真题分析","研究生数学建模竞赛","零基础"]
---

> [专题学习地图](/notes/mathematical-modeling-roadmap/) · 第 27 / 27 篇

## 学习目标与先修知识

目标是从长赛题提取任务、数据、约束、交付物和验证要求，区分“原题限制”与“自己的候选方法”，并把多个子问题组织成可检验的方案。先修为学习地图与综合案例。这里的年份索引用于复习选型，不能替代原题附件。

可回看：[学习地图与完整流程](/notes/mathematical-modeling-roadmap/) · [综合案例 从销量预测到备货决策](/notes/mathematical-modeling-integrated-case/)。

## 问题背景与使用时机

读到“数据驱动”“智能识别”并不意味着必须立刻选最复杂网络。题目可能同时包含特征筛选、预测、解释和优化，每个子问题的输入、标签和评价规则可能不同。

先把题目拆成“要输出什么、输出给谁、允许使用什么信息、怎样算完成”。题目背景中的算法举例、参考文献与明确规定，约束强度不同。尤其要区分“不得使用”和“不建议使用”，并确认适用的具体子问题。

## 核心概念与符号

### 一页任务表

| 要素 | 要写清的问题 |
|---|---|
| 输出 | 标签、数值、排名、路线、参数、概率还是方案 |
| 观察单位 | 患者、化合物、设备、日期、像素或网络 |
| 输入 | 实测、仿真、历史、未来可知还是未来未知 |
| 约束 | 方法、预算、运行时间、容量、格式与变量范围 |
| 验证 | 标签在哪里、按时间还是按组、误差指标是什么 |
| 交付 | 表格、图、代码、模型文件及文字解释 |
| 证据 | 每个任务和限制在原题的哪一页、哪一段 |

候选方法是对任务结构的判断，不是对原题的逐字转述。例如 K-Means 是无监督分群，不能把它列作已有标签的普通监督分类；pIC50 是连续结果，可做回归，ADMET 的各项标签可分别建分类器。特征选择只保留原变量，PCA 会构造新变量，不能在要求解释原始描述符时无条件替换。

多标签问题一条记录可同时有多个属性；多分类是一个属性有多个互斥类别。不同输出的评价不能混用。

## 手算一个最小例子

假设某子问题有100个样本，90个正常、10个异常。一个模型全部预测正常，准确率为90%，但异常召回率为0。若题目关心异常检出，这个高准确率不支持任务完成。

再看特征筛选：若先用全部100个样本与标签计算相关性，再只在其中70个上训练，即使模型没有直接拟合剩下30个，验证标签已影响入选特征，仍会泄漏。特征选择也必须进入训练内部 Pipeline。

## 建模步骤与完整案例

### 从多子问题到完整方案

以“化合物活性预测与性质筛选”为结构练习：先明确一行是一种化合物；描述符是输入；活性为连续目标；若干性质为多个二分类目标。拆分后分别建立回归与分类基线，再讨论怎样在性质约束下寻找候选方案。

新候选描述符组合可能在化学上不可实现，预测最优不等于能直接合成或获得真实效果。最终优化要把领域可行性和模型外推范围写进限制。

### 2021 至 2025 年主题复习索引

下表使用主题简称，表示本专题可以怎样练习方法选择；精确数据字段、样本量和方法限制须按官方原题与补充通知核对。方法列是候选路线，不是官方指定算法统计。

| 年份 | 主题简称 | 可练习的建模任务 |
|---|---|---|
| 2021 A | 相关矩阵组计算与存储 | 低秩、压缩、误差与计算成本 |
| 2021 B | 空气质量二次预报 | 时间对齐、预测、污染指标规则 |
| 2021 C | 脑深部电刺激 | 动力系统、参数和数值仿真 |
| 2021 D | 抗乳腺癌候选药物 | 特征选择、回归、性质分类与优化 |
| 2021 E | UWB 定位 | 干扰识别、测距与几何定位 |
| 2021 F | 航空机组排班 | 整数约束、覆盖与调度 |
| 2022 A | 移动场景定位 | 信号处理与空间参数估计 |
| 2022 B | 方形件组批 | 组合优化、排样与可行性 |
| 2022 C | 汽车生产调度 | 多工序约束与调度目标 |
| 2022 D | 芯片资源排布 | 图结构、容量和计算预算 |
| 2022 E | 草原放牧 | 回归、评价与情景分析 |
| 2022 F | 生活物资管理 | 需求模拟、选址与配送 |
| 2023 A | WLAN 接入机制 | 马尔可夫、排队与仿真 |
| 2023 B | DFT 类矩阵分解 | 线性代数与数值逼近 |
| 2023 C | 竞赛评审方案 | 标准化、公平性与稳健性 |
| 2023 D | 区域双碳路径 | 指标、预测与约束情景 |
| 2023 E | 出血性脑卒中 | 分组验证、有序结果与曲线拟合 |
| 2023 F | 强对流降水 | 时空外推、生成模型与降水估计 |
| 2024 A | 风电功率分配 | 疲劳量化、物理关系与优化 |
| 2024 B | WLAN 吞吐量 | 特征工程、分类和数值预测 |
| 2024 C | 磁芯损耗 | 波形特征、分类、回归与优化 |
| 2024 D | 地理综合问题 | 时空数据、区域差异与推断 |
| 2024 E | 应急车道 | 交通流、预测、触发规则与评价 |
| 2024 F | 脉冲星到达时间 | 时间系统、物理修正与随机过程 |
| 2025 A | NPU 核内调度 | DAG、缓存、约束与运行代价 |
| 2025 B | 无线链路速率 | 信道特征、物理抽象与拟合 |
| 2025 C | 围岩裂隙 | 分割、曲线、几何与重构 |
| 2025 D | 湍流监测与航路 | 多源融合、外推与路径规划 |
| 2025 E | 轴承故障迁移 | 跨域验证、特征与解释 |
| 2025 F | 园林美学 | 指标、空间结构、相似性与评价 |

这些主题之间可以复用流程，但不能复用未经核对的标签、指标或约束。不能从候选算法清单推算真实参赛使用频率，也不能据此声称某一年达到技术“高峰”。

## Python 实现与结果解释

配套文件：[查看或下载本篇完整脚本](/downloads/math-modeling/mathematical-modeling-machine-learning-data-processing.py)。

以下用完全模拟的描述符与标签演示“先选择再拟合”的合法边界，不把结果当作原赛题复现。

```python
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import mean_absolute_error

rng = np.random.default_rng(42)
X = rng.normal(size=(160, 60))
activity = 2*X[:, 0]-X[:, 1]+0.5*X[:, 2]+rng.normal(0, 0.5, len(X))
properties = (X[:, :5]+rng.normal(0, 0.5, (len(X), 5)) > 0).astype(int)
train_indices, test_indices = train_test_split(np.arange(len(X)), test_size=0.25, random_state=42)

regression = make_pipeline(
    StandardScaler(), SelectKBest(score_func=f_regression), Ridge()
)
search = GridSearchCV(
    regression,
    {"selectkbest__k": [5, 10, 20], "ridge__alpha": [1., 10.]},
    cv=KFold(3, shuffle=True, random_state=42),
    scoring="neg_mean_absolute_error"
)
search.fit(X[train_indices], activity[train_indices])
prediction = search.predict(X[test_indices])
print("选择参数:", search.best_params_)
print("回归留出MAE:", mean_absolute_error(activity[test_indices], prediction))

classifier = MultiOutputClassifier(
    make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
)
classifier.fit(X[train_indices], properties[train_indices])
predicted_properties = classifier.predict(X[test_indices])
print("各性质准确率:", (predicted_properties == properties[test_indices]).mean(axis=0))
assert predicted_properties.shape == (len(test_indices), 5)
```

模拟用了60个特征，目的是减小运行量并展示结构，而非伪造真实描述符。每折的 StandardScaler 与 SelectKBest 都只看训练折；多个性质分别输出，不能用其中一项分数代表全部任务。

## 方法选择与常见误区

### 怎样从候选方法作决定

先选简单基线，再按主要困难增加复杂度：高维且样本少先控制特征与正则化；时间相关先处理验证；跨域先研究来源与目标的分布差异；物理约束明确先写机制和可行性。

脑卒中 mRS 等有序结果有顺序，不应把等级间距离天然当相等；图像裂隙需要像素或几何标注，不能只报告整图分类正确率；雷达生成模型要检查空间、时间和强降水等任务相关指标；没有目标域标签时，不能直接声称已测得目标域准确率。

论文写作按“问题—假设—变量—模型—求解—验证—解释—局限”组织。每个重要结论应能回到数据、公式或实验，避免只有软件名和截图。

## 练习

1. 题目给出参考网络名称，是否一定强制使用？
2. 为什么原始描述符筛选与 PCA 降维不能不加区分？
3. 同一患者多次记录、同一设备多个窗口怎样划分更合适？
4. 只有预测文件、没有真实标签时，可以报告测试准确率吗？
5. 一个子问题含数值预测和资源分配，应怎样拆解？

## 参考答案

1. 需要看原文是示例、建议还是明确约束，并确认对应子问题；不能只凭名称判断。
2. 前者保留原变量，后者构造线性组合，解释对象不同；若要求对原始变量排序，要使用相应方法。
3. 通常按患者或设备分组，必要时再保留时间顺序，避免同一对象跨集合泄漏。
4. 不可以。只能提交预测，性能需要标签、可靠的外部验证或明确的不确定性说明。
5. 先建立并验证预测，再把预测及其不确定性输入目标和约束模型，最后评估实际决策指标。

## 复习卡片与下一步

拿到新题先完成一页任务表，再决定候选模型和验证。复习时可任选上表一个主题，写出观察单位、输出、约束、基线与交付物；若无法写清，先回到原题而不是继续堆算法。

[上一篇 综合案例 从销量预测到备货决策](/notes/mathematical-modeling-integrated-case/)

## 参考资料

- [Datawhale《数学建模导论》目录](https://datawhalechina.github.io/intro-mathmodel/#/)：用于建立知识目录，建议与本篇的手算和案例对照阅读。
- [赛事官方资料入口](https://cpipc.acge.org.cn/cw/contestPrevious/list/4/1)：核对原题与补充通知。
- [同济大学 2021 D 参赛记录](https://tops.tongji.edu.cn/info/1002/1283.htm)：题名为抗乳腺癌候选药物的优化建模。
