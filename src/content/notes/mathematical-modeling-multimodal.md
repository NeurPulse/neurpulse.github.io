---
title: "数学建模：图像文本与信号建模"
description: "把非表格数据变成特征，理解分割、频谱与融合"
pubDate: 2026-10-21
updatedDate: 2026-10-21
order: 25
category: '数学建模'
tags: ["多模态","图像","文本","信号","零基础"]
---

> [专题学习地图](/notes/mathematical-modeling-roadmap/) · 第 25 / 27 篇

## 学习目标与先修知识

目标是区分图像分类、检测和分割，理解词频与 TF-IDF、采样率与频谱，并把图像、文字和振动记录构造成设备级特征。先修为数组、平均、分类与训练边界。先掌握输入和输出，再决定是否需要深度模型。

可回看：[机器学习流程与 KNN](/notes/mathematical-modeling-ml-sklearn-knn/) · [时间序列分析与验证](/notes/mathematical-modeling-time-series/)。

## 问题背景与使用时机

一台设备可能同时留下巡检照片、文字描述与振动波形。它们的原始格式不同，但最终都要围绕同一个观察单位与预测时点组织。

把同一设备的照片切成多个块后随机分到训练与测试，可能只是识别出了设备本身；读取故障后的维修结论预测故障，也会使用未来信息。多模态模型同样必须先定义信息边界。

## 核心概念与符号

### 图像

灰度图是 $H\times W$ 数组，彩色图常为 $H\times W\times3$。分类输出整幅图标签，检测输出目标框与类别，分割输出每像素标签。去噪、对比度、边缘与形态学是常见预处理，但改变图像也可能改变任务所需细节。

卷积或局部滤波把核在邻域上移动。多数深度学习接口实际执行互相关，核是否翻转影响严格定义；对称均值核不受这个区别影响。缩放到0至1并不等于所有图像任务都应失去原始强度单位。

### 文本

词袋把文档转换成词次数向量，TF-IDF 降低在许多文档都出现的词的权重。一种平滑 IDF 为 $\log[(1+n)/(1+df_j)]+1$；是否使用归一化和词频变换应看接口设定。词序、否定与上下文可能被词袋丢失。

### 信号

采样率 $f_s$ 是每秒取样数，采样间隔 $1/f_s$；时长约为 $N/f_s$。FFT 把离散信号表示为不同频率分量；非负频率的间隔约为 $f_s/N$。在适当带限和滤波条件下，采样率通常需高于最高信号频率的两倍，避免混叠。

幅值谱、功率谱与功率谱密度不是同一个量。窗口、去趋势、泄漏与噪声会影响峰值，不能把最大峰直接认作故障原因。

### 融合

早期融合把各模态特征拼在一起；后期融合组合各模型分数。先按设备、时间对齐，再处理缺失与尺度；模态多不保证信息一定更有效。

## 手算一个最小例子

灰度小图为：

$$
I=\begin{bmatrix}0&0&0\\0&10&10\\0&10&10\end{bmatrix}.
$$

用2×2均值核、只保留完整窗口，输出为 $\begin{bmatrix}2.5&5\\5&10\end{bmatrix}$，例如左上窗口均值为 $(0+0+0+10)/4=2.5$。这会平滑边缘，不能假装所有细节保持不变。

两篇文档为“red blue”“red green”。red在两篇都出现，平滑IDF为1；blue只出现一次，IDF为 $\ln(3/2)+1\approx1.4055$。更稀有不等于一定更重要，但它对区分这两篇文档更有作用。

1秒内以8Hz采样1Hz正弦，得到8点，频谱分辨间隔为1Hz；理想信号峰值落在1Hz。若不知道采样率，仅有数组下标不能把峰的位置解释为“每秒多少次”。

## 建模步骤与完整案例

案例：构造80台独立教学设备，每台有一张8×8图、一秒128Hz振动和一段当时可用的巡检文字。用图像均值和波动、主峰频率和高频能量份额、是否记录“noisy”组成5维特征。

任务是预测二分类状态。文字必须是当时的观察，不能是后来形成的诊断。这里标签与特征均为模拟，用于确认对齐、提取与训练流程；实际故障诊断还需真实标注、跨设备划分和代价评价。

先用可解释的手工特征和逻辑回归，再判断更复杂图像或信号网络是否确实提高验证表现。

## Python 实现与结果解释

配套文件：[查看或下载本篇完整脚本](/downloads/math-modeling/mathematical-modeling-multimodal.py)。

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

image = np.array([[0., 0., 0.], [0., 10., 10.], [0., 10., 10.]])
filtered = np.array([[image[i:i+2, j:j+2].mean() for j in range(2)] for i in range(2)])
assert np.allclose(filtered, [[2.5, 5.], [5., 10.]])
vectorizer = TfidfVectorizer()
vectorizer.fit(["red blue", "red green"])
print("词表与IDF:", vectorizer.get_feature_names_out(), vectorizer.idf_)

rng = np.random.default_rng(42)
fs = 128
time = np.arange(fs)/fs
features, labels = [], []
for i in range(80):
    label = i % 2
    picture = rng.normal(0.3, 0.08, (8, 8))
    picture[2:5, 2:5] += 0.2*label
    signal = (np.sin(2*np.pi*8*time) + 0.6*label*np.sin(2*np.pi*32*time)
              + rng.normal(0, 0.2, len(time)))
    text = "inspection noisy" if rng.random() < (0.2+0.6*label) else "inspection routine"
    power = np.abs(np.fft.rfft(signal))**2
    frequency = np.fft.rfftfreq(len(signal), d=1/fs)
    peak = frequency[np.argmax(power)]
    high_share = power[frequency >= 20].sum()/power.sum()
    features.append([picture.mean(), picture.std(), peak, high_share, float("noisy" in text)])
    labels.append(label)
X, y = np.array(features), np.array(labels)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)
print("设备级特征形状:", X.shape)
print(classification_report(y_test, model.predict(X_test), zero_division=0))
assert X.shape == (80, 5)
assert np.isfinite(X).all()
```

高频能量份额只在本例生成机制中与状态有关；不能把对应频段直接迁移成真实设备的故障规则。检查单模态与融合模型的增益时，应使用相同设备划分，保持评价可比。

## 方法选择与常见误区

图像先看输出是标签、框还是像素；文本先看词序与语义是否关键；信号先看采样率、单位、时间段和工况。跨模态还要考虑时间不同步、缺失、尺度与样本数量。

分割常用 IoU 或 Dice，不可只报背景占多数时的像素准确率。信号可以先做统计、频谱与时频特征；文本可从 TF-IDF 开始，图像可从简单特征与预训练模型比较。更复杂结构应有任务和验证依据。

## 练习

1. 识别整张照片是否有裂隙，与逐像素标出裂隙分别属于什么任务？
2. 同一台设备的相邻窗口随机拆到训练测试，为什么有风险？
3. 每秒采100点、记录2秒，N是多少，频率间隔约是多少？
4. 多加一类模态后训练准确率升高，能否说明融合成功？

## 参考答案

1. 前者分类，后者分割；评价指标与标注需求不同。
2. 训练测试可能共享设备身份、工况与相近噪声，低估对新设备的误差。
3. N=200，频率间隔约 $f_s/N=0.5$Hz。
4. 不能，应在相同独立样本上比较效果、成本与缺失场景，检查是否引入泄漏。

## 复习卡片与下一步

输入格式可以不同，观察单位、信息时点和验证原则必须一致。下一篇把数据、预测、风险和实际决策完整串起来。

[上一篇 随机模拟与不确定性分析](/notes/mathematical-modeling-simulation/) · [下一篇 综合案例 从销量预测到备货决策](/notes/mathematical-modeling-integrated-case/)

## 参考资料

- [Datawhale《数学建模导论》第10章](https://datawhalechina.github.io/intro-mathmodel/#/CH10/第10章-图像、文本与信号数据)：用于建立知识目录，建议与本篇的手算和案例对照阅读。
- [NumPy FFT](https://numpy.org/doc/stable/reference/routines.fft.html)。
- [scikit-learn 文本特征](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)。
