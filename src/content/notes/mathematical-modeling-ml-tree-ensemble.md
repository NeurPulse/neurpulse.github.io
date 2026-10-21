---
title: "数学建模：树模型与集成学习"
description: "从手算分裂走到剪枝、随机森林和梯度提升"
pubDate: 2026-08-07
updatedDate: 2026-10-21
order: 20
category: '数学建模'
tags: ["决策树","随机森林","XGBoost","集成学习","机器学习","零基础"]
---

> [专题学习地图](/notes/mathematical-modeling-roadmap/) · 第 20 / 27 篇

## 学习目标与先修知识

目标是手算熵、信息增益、增益率、基尼与回归树平方误差，解释剪枝、Bagging、随机森林、AdaBoost 和梯度提升的区别，并完成训练内部选型。先修为条件概率、损失与交叉验证。

可回看：[机器学习流程与 KNN](/notes/mathematical-modeling-ml-sklearn-knn/)。

## 问题背景与使用时机

“如果尺寸小于某值，则预测某类”是一条可解释规则。决策树不断选择条件，把样本分到更相似的子组。深树容易把偶然噪声也拆成专门分支，所以需要限制复杂度。

集成把多棵树的结果组合起来，但不同组合方式的训练依赖、目标和代价并不相同。

## 核心概念与符号

### 分类分裂看纯度

节点 S 中第 i 类比例为 $p_i$，熵为 $H(S)=-\sum_i p_i\log_2p_i$，约定 $0\log0=0$。分支比例为 $q_j=|S_j|/|S|$：

$$
Gain=H(S)-\sum_jq_jH(S_j),\qquad
SplitInfo=-\sum_jq_j\log_2q_j,\qquad
GainRatio=\frac{Gain}{SplitInfo}.
$$

分母只看分支样本比例，不能用父节点类别熵或条件类别熵代替。常量属性 SplitInfo=0，不能用于该增益率划分。完整 C4.5 还涉及信息增益筛选、连续阈值、缺失与剪枝，不等于只最大化一条比值。

基尼不纯度 $G(S)=1-\sum_i p_i^2$，分类 CART 常最小化左右加权基尼；回归 CART 常最小化左右组的残差平方和，各叶预测组均值。连续特征直接比较阈值，不需要先把整列人为离散化。

### 控制与组合

预剪枝通过深度、叶样本数等限制生长；后剪枝在生成后用复杂度惩罚和验证选择简化程度，二者都是控制过拟合。

| 方法 | 怎样训练与组合 | 核心目的 |
|---|---|---|
| Bagging | 自助采样，多个模型可独立训练 | 降低不稳定模型的方差 |
| 随机森林 | 自助采样并随机限制分裂候选特征 | 减少树间相关性 |
| AdaBoost | 增加错分样本权重，顺序训练后加权投票 | 改善困难样本分类 |
| GBDT | 新树拟合当前损失的负梯度，顺序相加 | 逐步优化损失 |

随机森林回归通常平均预测，分类可平均概率再取最大。GBDT 的树依赖前一轮，不能像随机森林一样全部独立并行；同一轮内部计算可以并行。

## 手算一个最小例子

### 一次分类分裂

4个样本标签为0、0、1、1，父熵为1。分裂后左侧只有一个0，右侧是0、1、1。右熵为 $H(1/3,2/3)\approx0.9183$，故：

$$
Gain=1-\frac34\times0.9183\approx0.3113.
$$

分支比例1/4、3/4，对应 SplitInfo 约0.8113，所以增益率约0.3837。父熵恰为1并不表示分母应该用1。

### 回归分裂与提升

目标值1、2、8、9分成两组，均值为1.5和8.5，平方误差为 $0.25+0.25+0.25+0.25=1$；这里不使用分类基尼。

二分类 AdaBoost 加权错误率 $e=0.25$，系数 $\alpha=\frac12\ln[(1-e)/e]=\frac12\ln3$。错分权重乘 $\sqrt3$，正确乘 $1/\sqrt3$，相对比例增加3倍；这要求标签为±1且弱分类器优于随机猜测。

平方损失 $L=\frac12(y-F)^2$ 的负梯度为 $y-F$，所以梯度提升此时拟合残差；换成其他损失，负梯度不一定是普通残差。

## 建模步骤与完整案例

案例：Wine 内置数据分类。预先比较浅决策树、随机森林和梯度提升；每个模型的复杂度只在训练内部交叉验证选择，再按交叉验证表现选一个最终模型，对留出测试只评估一次。

小数据不需要先上大模型。报告需要同时说明选择依据、类别指标、模型复杂度与局限。特征重要性描述当前模型依赖哪些输入，不能直接解释为这些化学量导致类别差异。

## Python 实现与结果解释

配套文件：[查看或下载本篇完整脚本](/downloads/math-modeling/mathematical-modeling-ml-tree-ensemble.py)。

```python
import numpy as np
from scipy.stats import entropy
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report

gain = 1-0.75*entropy([1/3, 2/3], base=2)
split_info = entropy([0.25, 0.75], base=2)
print("增益、划分信息量、增益率:", gain, split_info, gain/split_info)
assert np.isclose(gain/split_info, 0.3836885465963443)

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)
candidates = [
    ("tree", DecisionTreeClassifier(random_state=42), {"max_depth": [2, 4, None]}),
    ("forest", RandomForestClassifier(n_estimators=60, random_state=42, n_jobs=2),
     {"max_depth": [3, None]}),
    ("boosting", GradientBoostingClassifier(n_estimators=60, random_state=42),
     {"max_depth": [1, 2]})
]
cv = StratifiedKFold(3, shuffle=True, random_state=42)
fitted = []
for name, model, grid in candidates:
    search = GridSearchCV(model, grid, cv=cv, scoring="balanced_accuracy")
    search.fit(X_train, y_train)
    fitted.append((search.best_score_, name, search))
    print(name, "训练内得分:", search.best_score_, "参数:", search.best_params_)
_, selected_name, selected = max(fitted, key=lambda item: item[0])
print("最终选择:", selected_name)
print(classification_report(y_test, selected.predict(X_test), zero_division=0))
```

本例使用预设的有限候选，测试不参与模型选择。树不依赖欧氏距离，因此统一缩放通常不改变相同特征内的阈值顺序，但数据编码、缺失处理和目标定义仍然重要。

## 方法选择与常见误区

ID3 主要按信息增益选择离散分类属性；C4.5 引入增益率等机制；CART 采用二分结构并支持分类与回归。scikit-learn 的树属于 CART 风格，不应把任意树输出统称 ID3。

XGBoost、LightGBM、CatBoost 是常见提升树框架。XGBoost 利用相应的梯度统计与正则化；LightGBM 强调高效分裂与直方图等设计；CatBoost 对类别特征等有专门机制。框架名称不替代数据划分、损失、参数与预算的说明。

不纯度重要性可能偏向候选切分多的特征；相关特征会分摊重要性。置换重要性也依赖模型、评价数据和特征相关结构。AdaBoost 基学习器还需支持相应的样本加权接口，不能无条件换成 KNN。

## 练习

1. 一个节点全部为同一类，熵和基尼各是多少？
2. 为什么增益率分母不能写成“右侧子节点的类别熵”？
3. 回归叶节点目标1、3、8，平方误差下应预测多少？
4. 随机森林与 GBDT 的树能否同样独立训练？
5. 加深树后训练准确率升高，下一步该看什么？

## 参考答案

1. 都为0，节点已经纯净。
2. 分母是划分的分支比例熵，与分支中类别分布是不同的对象。
3. 组均值 $(1+3+8)/3=4$，这是使组内平方误差最小的常数。
4. 通常不能。森林的树可独立生成，GBDT 后一轮依赖已有模型的预测。
5. 看训练内部验证误差、复杂度和稳定性；最终测试留待选型完成后。

## 复习卡片与下一步

分裂先辨分类还是回归，集成先辨独立还是逐轮依赖，重要性先辨预测关系还是因果结论。下一篇没有标签时，学习如何寻找和检验分组结构。

[上一篇 线性判别与支持向量机](/notes/mathematical-modeling-ml-lda-svm/) · [下一篇 聚类算法](/notes/mathematical-modeling-ml-clustering/)

## 参考资料

- [Datawhale《数学建模导论》第9章](https://datawhalechina.github.io/intro-mathmodel/#/CH9/第九章-机器学习与统计模型)：用于建立知识目录，建议与本篇的手算和案例对照阅读。
- [scikit-learn 决策树](https://scikit-learn.org/stable/modules/tree.html)。
- [集成学习](https://scikit-learn.org/stable/modules/ensemble.html)。
