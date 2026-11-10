---
title: "数据科学导引第八章：降维"
description: "用可复算的小例子理解 PCA、LDA、MDS 与局部线性嵌入，掌握特征值分解、散布矩阵、双重中心化和局部重构权重"
pubDate: 2026-11-10
updatedDate: 2026-11-10
category: '数据科学'
order: 9
tags: ['数据科学', '数据科学导引', '降维', 'PCA', 'LDA', 'MDS', '局部线性嵌入', '特征提取', '读书笔记', '零基础']
---

> [《数据科学导引》学习地图](/notes/data-science-guide-roadmap/) · 第 8 / 15 章
>
> 上一章：[关联规则挖掘](/notes/data-science-guide-association-rules/)

本文对应《数据科学导引》第八章“降维”，原书第 168—190 页，即所提供扫描版 PDF 第 177—199 页。笔记保留主成分分析、线性判别分析、多维尺度变换和局部线性嵌入四条主线，补充 PCA 重构误差、LDA 奇异性、经典 MDS 双重中心化和 LLE 局部权重的手算示例。明确标出的“学习补充”与“实践补充”用于帮助理解，并非教材原文。

## 学习目标

学完本章，应能回答七个问题：

- 降维与特征选择有什么区别，为什么高维数据的“内在维数”可能很低？
- PCA 为什么既等价于最大化投影方差，也等价于最小化线性重构误差？
- LDA 怎样利用类别标签，为什么它与 PCA 可能选出完全不同的方向？
- 类内散布矩阵奇异时，为什么收缩或正则化能够提供帮助？
- 经典 MDS 怎样只凭距离矩阵恢复坐标，为什么坐标可以整体旋转和平移？
- LLE 怎样先学习局部重构权重，再寻找保持这些关系的低维表示？
- 降维用于预测或可视化时，怎样避免数据泄漏和过度解释？

## 先看全局：不是简单删除列，而是构造新的低维表示

降维（dimensionality reduction）把数据从较高的特征维数转换到较低维数。原始数据虽然有很多列，但列之间可能高度相关，样本也可能集中在一个低维结构附近，因此数据的“内在维数”可能远低于表面维数。

教材用通信用户数据说明这一点：入网时间、套餐价格、话费、流量、通话时长、欠费金额和欠费月份等七个特征，可以进一步概括为消费能力、欠费指数和用户忠诚度等更抽象的三个维度。新维度不再只是某一列原始特征，而是原始信息的组合。

降维主要有三类价值：

- **减少计算与存储成本**，缓解部分高维估计和距离退化问题。
- **提取新的特征表示**，供后续回归、分类、聚类或检索使用。
- **降到二维或三维进行可视化**，帮助观察结构，但不代替严格评价。

它不同于特征选择：**特征选择保留原始特征的子集，降维通常构造新变量。** 新变量可能更易组合信息，也可能更难逐项解释。

本章四种方法优化的是不同目标：

| 方法 | 输入与标签 | 主要保留什么 | 是否有直接的 $XW$ 线性变换 | 关键限制 |
|---|---|---|---|---|
| PCA | 特征矩阵，不用标签 | 总体方差与线性重构信息 | 有 | 大不一定重要，受尺度与线性假设影响 |
| LDA | 特征矩阵与类别标签 | 类间分离相对于类内离散 | 有 | 标签泄漏、$S_w$ 奇异、维数受类别数限制 |
| MDS | 距离或相似关系，可不用原始特征 | 样本间距离或其次序 | 经典形式不是简单的原始 $XW$ | $n\times n$ 矩阵成本，非欧氏距离与嵌入误差 |
| LLE | 特征矩阵及近邻图，不用标签 | 局部线性重构关系 | 没有显式的统一 $XW$ 映射 | $k$、噪声、流形假设与新样本映射 |

## 8.1 主成分分析：用正交方向保留最大方差

### 从中心化数据到投影方差

设原始数据有 $n$ 个样本、$d$ 个连续特征。先计算每个特征的均值并中心化，把中心化后的数据记为 $X\in\mathbb R^{n\times d}$，每行是一个样本。若降到 $l$ 维，取 $W\in\mathbb R^{d\times l}$，则：

$$
Y=XW.
$$

其中 $W$ 的每一列 $\boldsymbol w_i$ 是一个投影方向。若各方向标准正交，即：

$$
W^TW=I,
$$

则不同主成分得分之间不相关。教材优化公式明确写出了单位长度约束，完整多方向形式还应同时包含正交约束。

中心化数据的协方差矩阵为：

$$
\Sigma=\frac1{n-1}X^TX.
$$

投影后的总方差（各维方差之和）为：

$$
\operatorname{tr}(W^T\Sigma W).
$$

PCA 的优化问题为：

$$
\max_W\operatorname{tr}(W^T\Sigma W),
\qquad \text{s.t. }W^TW=I.
$$

由拉格朗日条件可得：

$$
\Sigma\boldsymbol w_i=\lambda_i\boldsymbol w_i.
$$

因此，投影方向是协方差矩阵的特征向量。若特征值按：

$$
\lambda_1\ge\lambda_2\ge\cdots\ge\lambda_d\ge0
$$

排列，前 $l$ 个最大特征值对应的单位特征向量构成 $W$。保留的方差就是：

$$
\operatorname{tr}(W^T\Sigma W)=\sum_{i=1}^l\lambda_i.
$$

### 为什么最大方差等价于最小重构误差

使用 $W^T$ 把低维得分映射回原特征空间，并补回均值，就得到线性重构。中心化矩阵的重构为：

$$
\widehat X=XWW^T.
$$

重构误差为：

$$
\|X-XWW^T\|_F^2.
$$

在 $W^TW=I$ 且使用完整正交特征基时，样本的总平方长度可以拆成“保留部分”和“正交残差”：

$$
\|X\|_F^2
=\|XW\|_F^2+\|X-XWW^T\|_F^2.
$$

总平方长度固定，所以最大化保留部分的平方长度，就等价于最小化残差。除以 $n-1$ 后，这也正是“最大投影方差”和“最小均方重构误差”的等价性。

这个结论依赖**正交投影与平方欧氏误差**。换成非正交矩阵、其他重构目标或非线性映射，不能直接照搬。

### 算法步骤与保留方差比例

PCA 的基本流程为：

1. 用训练数据计算均值并中心化。
2. 计算协方差矩阵 $\Sigma$，或等价地对 $X$ 做奇异值分解。
3. 将特征值从大到小排序，取前 $l$ 个单位特征向量组成 $W$。
4. 计算 $Y=XW$；若用于新样本，必须复用训练阶段的均值、缩放参数和 $W$。

保留方差比例为：

$$
\operatorname{VarRatio}(l)
=\frac{\sum_{i=1}^l\lambda_i}
{\sum_{j=1}^d\lambda_j}.
$$

教材建议可设置例如 80% 的阈值来选择 $l$。这是一种工作准则，不是通用最优答案：若后续任务是分类，高方差方向未必最有类别区分力；若用于压缩，还需同时考虑重构误差和存储收益。

### 手算：协方差矩阵 $\begin{pmatrix}2&1\\1&2\end{pmatrix}$

使用学习补充的四个已中心化样本：

$$
(1.5,1.5),\quad(-1.5,-1.5),\quad
(b,-b),\quad(-b,b),
\qquad b=\frac{\sqrt3}{2}.
$$

直接计算得：

$$
X^TX=
\begin{pmatrix}
6&3\\
3&6
\end{pmatrix},
\qquad
\Sigma=\frac13X^TX=
\begin{pmatrix}
2&1\\
1&2
\end{pmatrix}.
$$

两个特征方向为：

$$
\boldsymbol w_1=\frac1{\sqrt2}(1,1)^T,
\qquad
\boldsymbol w_2=\frac1{\sqrt2}(1,-1)^T.
$$

对应特征值为：

$$
\lambda_1=3,\qquad\lambda_2=1.
$$

降到一维时选择 $\boldsymbol w_1$，保留方差比例为：

$$
\frac{3}{3+1}=0.75.
$$

前两个样本本来就位于第一主成分直线上，可以精确重构；后两个样本位于第二主成分方向，投影到第一方向后都变为原点。后两个样本的重构误差各为：

$$
\|(b,-b)\|_2^2=2b^2=1.5,
$$

总重构误差为 3。数据总平方长度为：

$$
(n-1)\operatorname{tr}(\Sigma)=3\times4=12,
$$

保留部分为 9，残差为 3，同样得到 $9/12=75\%$。舍弃特征值乘 $n-1$ 正好等于这部分平方重构误差。

### PCA 的适用边界

- **尺度**：金额、流量和次数若量纲差异很大，高数值范围特征可能主导协方差。可按任务先标准化，此时相当于分析相关矩阵，但必须在训练集拟合缩放。
- **信息目标**：最大方差不等于最强预测能力。噪声也可能方差很大，类别差异可能位于小方差方向。
- **线性结构**：PCA 找的是线性子空间，弯曲流形上的结构可能无法由一个低维平面良好表示。
- **解释性**：主成分是原始特征的线性组合，载荷可辅助解释，但不能仅凭第一主成分就命名为某种潜在因素。
- **解的表现形式**：特征向量整体乘以 $-1$，投影含义不变；重复特征值对应的子空间内，基也不唯一。

教材给出协方差特征分解的时间复杂度 $O(d^3)$、空间复杂度 $O(d^2)$。实际使用奇异值分解、截断算法或随机算法时，成本会随 $n,d,l$ 和实现而变化。

## 8.2 线性判别分析：让同类更紧凑、不同类更分离

### 类内散布与类间散布

LDA 是有监督的线性降维方法。设第 $c$ 类有 $n_c$ 个样本，类别均值为 $\boldsymbol m_c$，总体均值为 $\boldsymbol m$。教材先定义每类的散布矩阵：

$$
S_c=\sum_{\boldsymbol x_i\in D_c}
(\boldsymbol x_i-\boldsymbol m_c)
(\boldsymbol x_i-\boldsymbol m_c)^T.
$$

总类内散布矩阵为：

$$
S_w=\sum_{c=1}^C\frac{n_c}{n}S_c.
$$

类间散布矩阵为：

$$
S_b=\sum_{c=1}^C\frac{n_c}{n}
(\boldsymbol m_c-\boldsymbol m)
(\boldsymbol m_c-\boldsymbol m)^T.
$$

投影到方向 $\boldsymbol w$ 后，类内与类间离散程度分别为：

$$
\boldsymbol w^TS_w\boldsymbol w,
\qquad
\boldsymbol w^TS_b\boldsymbol w.
$$

LDA 希望前者小、后者大，因此最大化 Fisher 比率：

$$
J(\boldsymbol w)
=\frac{\boldsymbol w^TS_b\boldsymbol w}
{\boldsymbol w^TS_w\boldsymbol w}.
$$

这里使用的是教材的类别加权散布定义。其他资料可能使用未加权求和或除以自由度的协方差，比较数值前应先统一约定。

### 广义特征值问题

把 $\boldsymbol w$ 同乘一个非零常数，分子分母同比例变化，$J$ 不变，所以真正重要的是方向。可添加约束：

$$
\boldsymbol w^TS_w\boldsymbol w=1.
$$

拉格朗日条件给出：

$$
S_b\boldsymbol w=\lambda S_w\boldsymbol w.
$$

若 $S_w$ 可逆，则：

$$
S_w^{-1}S_b\boldsymbol w
=\lambda\boldsymbol w.
$$

取最大特征值对应的特征向量，就得到一维 Fisher 判别方向；多维时取前 $l$ 个广义特征向量。二分类且 $S_w$ 可逆时，也可写成：

$$
\boldsymbol w\propto S_w^{-1}(\boldsymbol m_1-\boldsymbol m_2).
$$

**学习补充**：$S_b$ 是 $C$ 个“类别均值减总体均值”的外积加权和，其秩至多为 $C-1$，所以通常最多有 $C-1$ 个非零判别方向。二分类只有一条非零 Fisher 方向，不能指望标准 LDA 再给出第二条同样意义的非零判别轴。

### 手算：奇异类内矩阵与收缩

使用学习补充的二维二分类数据：

$$
D_0=\{(0,0),(2,0)\},\qquad
D_1=\{(0,2),(2,2)\}.
$$

类别均值和总体均值为：

$$
\boldsymbol m_0=(1,0)^T,\qquad
\boldsymbol m_1=(1,2)^T,\qquad
\boldsymbol m=(1,1)^T.
$$

按教材的 $n_c/n=1/2$ 加权定义：

$$
S_w=
\begin{pmatrix}
2&0\\
0&0
\end{pmatrix},
\qquad
S_b=
\begin{pmatrix}
0&0\\
0&1
\end{pmatrix}.
$$

类别只沿 $x_2$ 分离，但每类在 $x_2$ 方向没有内部变化，所以 $S_w$ 的第二行、第二列全为 0，矩阵不可逆。不能直接计算 $S_w^{-1}S_b$。

教材给出的收缩调整为：

$$
\widetilde S_w=(1-\lambda)S_w+\lambda I.
$$

取学习补充参数 $\lambda=0.5$：

$$
\widetilde S_w=
\begin{pmatrix}
1.5&0\\
0&0.5
\end{pmatrix}.
$$

沿 $\boldsymbol w=(0,1)^T$ 投影时：

$$
\frac{\boldsymbol w^TS_b\boldsymbol w}
{\boldsymbol w^T\widetilde S_w\boldsymbol w}
=\frac1{0.5}=2.
$$

收缩让原本零方差方向具有可计算的分母，同时保留“类别均值沿 $x_2$ 分开”的结构。实际中 $\lambda$ 应通过训练流程选择，不能看完测试集再调。

### PCA 与 LDA 为什么可能选出不同方向

PCA 不知道类别标签，只寻找总体方差最大的方向。如果两类数据沿一个方向被拉开，但每类内部沿另一个方向散布更大，PCA 可能优先保留后者，投影后类别混在一起；LDA 则利用标签，选择类间差异相对于类内差异更大的方向。

教材用 26 个英文大写字母、20000 个样本、16 个特征的光学字符数据作对比，并选取 B、P、H 三个字母展示二维结果：PCA 图中字母较难区分，LDA 图中类别更分离。这说明的是该数据和目标下的表现，不代表任何任务中 LDA 都优于 PCA。

LDA 既可用于降维，也可在生成式概率假设下构造分类器。**作为分类器的 LDA 与本节的 Fisher 投影相关，但训练、预测和评价步骤不能省略。** 若在整个数据集上先用标签学习投影，再把同一批数据当测试集，会发生监督信息泄漏并高估效果。

## 8.3 多维尺度变换：从距离矩阵恢复样本坐标

### 优化目标与经典求解的区别

MDS 的输入可以是样本间距离或相似关系，而不必从原始特征矩阵开始。设距离矩阵为 $D=(d_{ij})\in\mathbb R^{n\times n}$，希望找到低维坐标 $\boldsymbol z_i$，使：

$$
\|\boldsymbol z_i-\boldsymbol z_j\|_2\approx d_{ij}.
$$

教材先用应力形式表达目标：

$$
\min_{z_1,\ldots,z_n}
\sum_{i\ne j}
\left(\|\boldsymbol z_i-\boldsymbol z_j\|_2-d_{ij}\right)^2.
$$

随后教材通过内积矩阵和特征分解给出**经典 MDS** 的求解。学习时需要区分：经典双重中心化直接优化的是与平方距离、内积相关的经典准则，不应把它当作上述任意应力形式在所有情况下的通用全局最优证明。一般应力最小化还可能需要数值迭代。

### 双重中心化：从平方距离恢复内积

先假设坐标已中心化，即 $\sum_i\boldsymbol x_i=\boldsymbol0$，令：

$$
B=XX^T,\qquad b_{ij}=\boldsymbol x_i^T\boldsymbol x_j.
$$

欧氏距离满足：

$$
d_{ij}^2=b_{ii}+b_{jj}-2b_{ij}.
$$

利用零均值条件，可以反过来只由距离计算 $B$。教材展开的行和公式为：

$$
b_{ij}=\frac12\left(
\frac1n\sum_{r=1}^n d_{rj}^2
+\frac1n\sum_{s=1}^n d_{is}^2
-\frac1{n^2}\sum_{r=1}^n\sum_{s=1}^n d_{rs}^2
-d_{ij}^2
\right).
$$

更简洁的矩阵写法是：

$$
B=-\frac12JD^{(2)}J,
$$

其中：

$$
J=I-\frac1n\boldsymbol1\boldsymbol1^T,
\qquad
D^{(2)}=(d_{ij}^2).
$$

$J$ 的作用是按行、列和总体进行双重中心化。注意输入是**平方距离矩阵**，不是直接把距离本身做特征分解。

对对称矩阵 $B$ 做特征分解：

$$
B=U\Lambda U^T.
$$

取前 $l$ 个正特征值及对应特征向量：

$$
Z=U_l\Lambda_l^{1/2}.
$$

于是 $ZZ^T\approx B$，$Z$ 的每一行就是一个样本的低维坐标。

### 手算：三个共线点的一维坐标

使用学习补充的三个点，真实位置为 $-1,0,1$。距离矩阵为：

$$
D=
\begin{pmatrix}
0&1&2\\
1&0&1\\
2&1&0
\end{pmatrix}.
$$

双重中心化得到：

$$
B=
\begin{pmatrix}
1&0&-1\\
0&0&0\\
-1&0&1
\end{pmatrix}.
$$

$B$ 的非零特征值为 2，对应单位特征向量可取：

$$
\boldsymbol u_1=\frac1{\sqrt2}(1,0,-1)^T.
$$

因此：

$$
Z=\boldsymbol u_1\sqrt2=(1,0,-1)^T.
$$

这与真实位置 $(-1,0,1)$ 只差整体符号，三个距离都精确恢复。即使真实坐标整体平移或旋转，距离矩阵也完全相同，所以 MDS 不能仅凭距离决定绝对位置、方向和朝向。

### 距离不一定能精确欧氏嵌入

若 $B$ 有负特征值，说明给定距离不能全部由某个欧氏坐标系统精确产生。此时不能对负数直接开平方，需要检查距离质量、降维近似和算法选择。

教材还区分：

- **度量 MDS**：尽量保留距离数值。若输入是欧氏距离，经典 MDS 与 PCA 在相应设定下可得到一致结果。
- **非度量 MDS**：重点保留距离次序。若 $d_{ik}<d_{jk}$，希望低维中也尽量有 $\|\boldsymbol z_i-\boldsymbol z_k\|_2<\|\boldsymbol z_j-\boldsymbol z_k\|_2$。

教材案例把北京、上海、哈尔滨、乌鲁木齐等 10 个中国城市的距离降到二维，并进行旋转和平移后展示相对位置。MDS 也可用于图布局与分子构象分析。

存储完整距离矩阵通常需要 $O(n^2)$ 空间，稠密特征分解也有较高计算成本。城市地图看起来合理，不等于所有距离都被无损保留，仍应检查应力或距离误差。

## 8.4 局部线性嵌入：保持每个样本与近邻的局部关系

### 三个步骤

高维空间中的直线距离未必反映弯曲流形上的结构。流形可以理解为“局部近似具有欧氏空间性质”的空间，例如三维空间中的曲线或曲面。LLE 假设数据位于或靠近一个低维光滑流形，并通过局部关系降维。

算法分为三步：

1. 对每个样本 $\boldsymbol x_i$ 找出 $k$ 个近邻 $N(i)$。
2. 用近邻的线性组合重构 $\boldsymbol x_i$，学习权重 $w_{ij}$。
3. 固定这些权重，寻找低维坐标 $\boldsymbol y_i$，让局部重构关系尽量保持。

高维重构目标为：

$$
\min_W\sum_{i=1}^n
\left\|\boldsymbol x_i-
\sum_jw_{ij}\boldsymbol x_j\right\|_2^2.
$$

每个样本只使用自己的近邻，并满足：

$$
\sum_jw_{ij}=1,
\qquad w_{ij}=0\quad(j\notin N(i)).
$$

权重和为 1 使重构关系对整体平移保持一致。教材文字将近邻系数写成 $w_{ij}>0$，但标准 LLE 的二次最小二乘形式通常只要求“非近邻为 0、近邻权重和为 1”，并不要求所有近邻权重非负；若加入非负约束，就是不同的优化问题。

### 局部格拉姆矩阵与权重求解

对固定的 $\boldsymbol x_i$，利用 $\sum_jw_{ij}=1$：

$$
\boldsymbol x_i-\sum_jw_{ij}\boldsymbol x_j
=\sum_jw_{ij}(\boldsymbol x_i-\boldsymbol x_j).
$$

把 $\boldsymbol x_i-\boldsymbol x_j$ 作为第 $j$ 列组成 $Z_i\in\mathbb R^{d\times k}$，局部格拉姆矩阵为：

$$
G_i=Z_i^TZ_i.
$$

则单个样本的权重问题是：

$$
\min_{\boldsymbol w_i}
\boldsymbol w_i^TG_i\boldsymbol w_i,
\qquad
\text{s.t. }\boldsymbol w_i^T\boldsymbol1=1.
$$

若 $G_i$ 可逆，先求：

$$
\widetilde{\boldsymbol w}_i=G_i^{-1}\boldsymbol1,
$$

再归一化：

$$
\boldsymbol w_i
=\frac{\widetilde{\boldsymbol w}_i}
{\boldsymbol1^T\widetilde{\boldsymbol w}_i}.
$$

当 $k>d$ 或局部差分向量共线时，$G_i$ 可能不可逆。教材加入正则项：

$$
\boldsymbol w_i^TG_i\boldsymbol w_i
+\lambda\boldsymbol w_i^T\boldsymbol w_i,
$$

相当于用 $G_i+\lambda I$ 求解后再归一化。实际计算通常解线性方程组，而不是显式求逆。

### 手算：三个近邻精确重构一个样本

设学习补充样本和近邻为：

$$
\boldsymbol x_i=(1,0.5)^T,
$$

$$
\boldsymbol x_1=(0,0)^T,\quad
\boldsymbol x_2=(2,0)^T,\quad
\boldsymbol x_3=(0,2)^T.
$$

取权重：

$$
w_{i1}=0.25,\qquad w_{i2}=0.5,\qquad w_{i3}=0.25.
$$

权重和为 1，并且：

$$
0.25(0,0)+0.5(2,0)+0.25(0,2)=(1,0.5),
$$

所以重构误差为 0。局部格拉姆矩阵为：

$$
G_i=
\begin{pmatrix}
1.25&-0.75&0.25\\
-0.75&1.25&-1.75\\
0.25&-1.75&3.25
\end{pmatrix}.
$$

代入可得 $G_i\boldsymbol w_i=\boldsymbol0$。由于存在精确零误差解，该矩阵本身奇异，也说明“直接求 $G_i^{-1}$”并非永远可行。零特征值对应的方向在归一化后给出精确权重；实际数据通常需要处理噪声、多个解和正则化。

### 固定权重，寻找低维坐标

低维空间的目标为：

$$
\Phi(Y)=
\sum_{i=1}^n
\left\|\boldsymbol y_i-
\sum_jw_{ij}\boldsymbol y_j\right\|_2^2.
$$

令：

$$
M=(I-W)^T(I-W),
$$

可写成：

$$
\Phi(Y)=\operatorname{tr}(Y^TMY).
$$

若不加约束，所有 $\boldsymbol y_i$ 都取 0 就得到无意义的最小值。因此加入：

$$
\sum_i\boldsymbol y_i=\boldsymbol0,
\qquad
\frac1nY^TY=I.
$$

第一式去除平移自由度，第二式避免尺度塌缩并令各维不相关。$M$ 的最小特征值通常为 0，对应全 1 的常量坐标；实际取最小的 $l+1$ 个特征向量，去掉最小的零特征方向，用其余 $l$ 个方向组成坐标，并按约束调整尺度。

### LLE 的优势与边界

- **非线性来自局部结构**：每一步使用线性重构，但整体可以表示弯曲流形，所以 LLE 属于非线性降维。
- **参数会改变邻域图**：$k$ 太小可能不稳定或形成断裂，太大则可能把不应相连的远处结构混在一起。
- **对噪声和异常值敏感**：错误近邻会同时影响权重和低维坐标。
- **全局特征解不等于无条件可靠**：固定 $W$ 后的约束特征问题有全局解，但近邻选择、噪声和参数仍会改变结果。
- **没有直接的 $Y=XW$ 新样本映射**：标准 LLE 不像 PCA、LDA 那样给出一个可立即用于任意新样本的统一线性矩阵，需要额外的 out-of-sample 扩展或重新处理。
- **依赖流形假设**：样本太稀疏、邻域跨越不同分支或数据并非单一流形时，局部线性假设可能失效。

## 8.5 其他降维方法与实践选择

教材还简要介绍：

- **核主成分分析（KPCA）**：通过核函数在隐式高维特征空间做 PCA，以处理非线性结构。
- **等距离映射（Isomap）**：近邻处使用欧氏距离，远处通过近邻图最短路径估计测地距离，再做 MDS；对图连通性和噪声敏感。
- **扩散映射（diffusion maps）**：利用扩散过程和多路径关系度量样本联系。
- **t-SNE**：用概率表示高维与低维相似性并尽量保持，常用于可视化；不能把图中簇大小和远距离直接当作原空间的定量结论。
- **其他方法**：自编码器、Sammon 映射和拉普拉斯特征映射等。

“方法更非线性”不等于“结果更真实”。可视化方法可能强调局部分离而扭曲全局尺度，二维图上的空白、簇形和方向都需要回到原数据与任务指标验证。

## 实践补充：把降维放进完整训练流程

### 防止预处理与标签泄漏

若降维用于预测，正确顺序是：

1. 先划分训练集、验证集和测试集。
2. 只在训练集上拟合缺失处理、缩放、PCA 或 LDA。
3. 用同一套已拟合参数转换验证集和测试集。
4. 维数 $l$、收缩参数、近邻数 $k$ 等通过训练流程或验证集选择。
5. 测试集只做最后评价，不参与方向、阈值或参数选择。

PCA 虽不使用标签，在全数据上拟合仍会把测试数据的分布信息带进训练表示；LDA 还使用类别标签，泄漏会更直接。

### 按任务评价，而不是只看图

- **压缩与重构**：检查保留方差、重构误差和存储成本。
- **后续预测**：比较降维前后在未参与训练数据上的指标、训练时间和稳定性。
- **探索与可视化**：检查不同参数、抽样和初始化下结构是否稳定，并回查原始样本。
- **类别分离**：不能把训练集 LDA 图上的分离直接当成测试集分类性能。
- **距离保持**：报告低维距离与原距离的误差，不能只展示一张坐标图。

教材章末任务包括：从恶性间皮瘤的 34 个相关特征出发，用 PCA 降维后建立诊断分类器；对 1288 张政要照片提取特征脸并用 SVM 分类；将 1797 张 $8\times8$、64 维的手写数字图片降到二维展示。这些任务都应把“降维表示”和“最终评价”连接起来。

## 可运行示例：核对 PCA、LDA、MDS 与 LLE 的关键计算

下面只用 Python 标准库，复算正文中的四个小例子。代码针对教学矩阵直接计算，不替代成熟库的奇异值分解、广义特征求解、稀疏特征分解和近邻索引。

```python
import math

# 通用小矩阵工具
def dot(left, right):
    return sum(a * b for a, b in zip(left, right))

def mat_vec(matrix, vector):
    return [dot(row, vector) for row in matrix]

def quadratic(matrix, vector):
    return dot(vector, mat_vec(matrix, vector))

def squared_distance(left, right):
    return sum((a - b)**2 for a, b in zip(left, right))

def rounded_matrix(matrix):
    return [[round(value, 6) for value in row] for row in matrix]

# 1. PCA：协方差、特征方向、保留方差和重构误差
b = math.sqrt(3) / 2
X = [
    [1.5, 1.5],
    [-1.5, -1.5],
    [b, -b],
    [-b, b],
]
means = [sum(row[j] for row in X) / len(X) for j in range(2)]
assert all(math.isclose(value, 0.0, abs_tol=1e-12) for value in means)
pca_covariance = [
    [
        sum(row[i] * row[j] for row in X) / (len(X) - 1)
        for j in range(2)
    ]
    for i in range(2)
]
w1 = [1 / math.sqrt(2), 1 / math.sqrt(2)]
w2 = [1 / math.sqrt(2), -1 / math.sqrt(2)]
pca_eigenvalues = [quadratic(pca_covariance, w1), quadratic(pca_covariance, w2)]
variance_ratio = pca_eigenvalues[0] / sum(pca_eigenvalues)
reconstruction_sse = 0.0
for row in X:
    score = dot(row, w1)
    reconstructed = [score * value for value in w1]
    reconstruction_sse += squared_distance(row, reconstructed)
total_sse = sum(dot(row, row) for row in X)
assert pca_covariance == [[2.0, 1.0], [1.0, 2.0]]
assert math.isclose(pca_eigenvalues[0], 3.0)
assert math.isclose(pca_eigenvalues[1], 1.0)
assert math.isclose(variance_ratio, 0.75)
assert math.isclose(reconstruction_sse, 3.0)
assert math.isclose(total_sse, 12.0)
assert math.isclose(total_sse - reconstruction_sse, 9.0)
print("PCA 协方差:", pca_covariance)
print("PCA 特征值、保留比例、总平方长度、重构误差:")
print(pca_eigenvalues, variance_ratio, total_sse, reconstruction_sse)

# 2. LDA：教材加权散布定义与收缩
class_0 = [[0.0, 0.0], [2.0, 0.0]]
class_1 = [[0.0, 2.0], [2.0, 2.0]]
lda_classes = [class_0, class_1]
class_means = [
    [sum(row[j] for row in samples) / len(samples) for j in range(2)]
    for samples in lda_classes
]
all_samples = class_0 + class_1
overall_mean = [sum(row[j] for row in all_samples) / len(all_samples) for j in range(2)]

def outer(vector):
    return [[a * b for b in vector] for a in vector]

def matrix_add(left, right):
    return [[a + b for a, b in zip(row_a, row_b)] for row_a, row_b in zip(left, right)]

def matrix_scale(scalar, matrix):
    return [[scalar * value for value in row] for row in matrix]

def subtract(left, right):
    return [a - b for a, b in zip(left, right)]

zero = [[0.0, 0.0], [0.0, 0.0]]
within_scatter = zero
for samples, mean in zip(lda_classes, class_means):
    class_scatter = zero
    for row in samples:
        class_scatter = matrix_add(class_scatter, outer(subtract(row, mean)))
    within_scatter = matrix_add(within_scatter, matrix_scale(0.5, class_scatter))
between_scatter = zero
for mean in class_means:
    between_scatter = matrix_add(
        between_scatter,
        matrix_scale(0.5, outer(subtract(mean, overall_mean))),
    )
shrinkage = 0.5
identity = [[1.0, 0.0], [0.0, 1.0]]
regularized_within = matrix_add(
    matrix_scale(1 - shrinkage, within_scatter),
    matrix_scale(shrinkage, identity),
)
lda_w = [0.0, 1.0]
lda_ratio = quadratic(between_scatter, lda_w) / quadratic(regularized_within, lda_w)
assert within_scatter == [[2.0, 0.0], [0.0, 0.0]]
assert between_scatter == [[0.0, 0.0], [0.0, 1.0]]
assert regularized_within == [[1.5, 0.0], [0.0, 0.5]]
assert lda_ratio == 2.0
print("LDA 类内、类间与收缩后的类内矩阵:")
print(within_scatter, between_scatter, regularized_within)
print("LDA 投影方向与 Fisher 比率:", lda_w, lda_ratio)

# 3. 经典 MDS：距离矩阵双重中心化，恢复三个共线点
D = [
    [0.0, 1.0, 2.0],
    [1.0, 0.0, 1.0],
    [2.0, 1.0, 0.0],
]

def double_center_gram(distances):
    n = len(distances)
    squared = [[value**2 for value in row] for row in distances]
    row_means = [sum(row) / n for row in squared]
    grand_mean = sum(row_means) / n
    return [
        [
            -0.5 * (
                squared[i][j]
                - row_means[i]
                - row_means[j]
                + grand_mean
            )
            for j in range(n)
        ]
        for i in range(n)
    ]

mds_gram = double_center_gram(D)
mds_eigenvalue = 2.0
mds_u = [1 / math.sqrt(2), 0.0, -1 / math.sqrt(2)]
mds_z = [[value * math.sqrt(mds_eigenvalue)] for value in mds_u]
recovered_gram = [[dot(left, right) for right in mds_z] for left in mds_z]
recovered_distances = [
    [math.sqrt(squared_distance(left, right)) for right in mds_z]
    for left in mds_z
]
expected_mds_gram = [[1.0, 0.0, -1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 1.0]]
assert all(
    math.isclose(mds_gram[i][j], expected_mds_gram[i][j], abs_tol=1e-12)
    for i in range(3)
    for j in range(3)
)
assert all(
    math.isclose(recovered_gram[i][j], mds_gram[i][j], abs_tol=1e-12)
    for i in range(3)
    for j in range(3)
)
assert all(
    math.isclose(recovered_distances[i][j], D[i][j], abs_tol=1e-12)
    for i in range(3)
    for j in range(3)
)
print("MDS 内积矩阵、坐标与恢复距离:")
print(rounded_matrix(mds_gram), mds_z, recovered_distances)

# 4. LLE：权重和为 1，三个近邻精确重构目标样本
x_i = [1.0, 0.5]
neighbors = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
lle_w = [0.25, 0.5, 0.25]
difference_columns = [subtract(x_i, neighbor) for neighbor in neighbors]
lle_gram = [
    [dot(left, right) for right in difference_columns]
    for left in difference_columns
]
reconstructed_x = [
    sum(weight * neighbor[j] for weight, neighbor in zip(lle_w, neighbors))
    for j in range(2)
]
lle_gradient = mat_vec(lle_gram, lle_w)
assert math.isclose(sum(lle_w), 1.0)
assert reconstructed_x == x_i
assert all(math.isclose(value, 0.0, abs_tol=1e-12) for value in lle_gradient)
print("LLE 权重、格拉姆矩阵、重构结果与 Gw:")
print(lle_w, rounded_matrix(lle_gram), reconstructed_x, lle_gradient)
```

关键输出为：

```text
PCA 协方差: [[2.0, 1.0], [1.0, 2.0]]
PCA 特征值、保留比例、总平方长度、重构误差:
[2.9999999999999996, 0.9999999999999998] 0.7500000000000001 12.0 2.9999999999999996
LDA 投影方向与 Fisher 比率: [0.0, 1.0] 2.0
MDS 内积矩阵、坐标与恢复距离:
[[1.0, 0.0, -1.0], [0.0, -0.0, 0.0], [-1.0, 0.0, 1.0]] [[1.0], [0.0], [-1.0]] [[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]]
LLE 权重、格拉姆矩阵、重构结果与 Gw:
[0.25, 0.5, 0.25] [[1.25, -0.75, 0.25], [-0.75, 1.25, -1.75], [0.25, -1.75, 3.25]] [1.0, 0.5] [0.0, 0.0, 0.0]
```

示例中的浮点尾差来自二进制浮点运算，不代表数学公式有额外误差。真实项目应使用经过验证的数值库，并检查特征值、矩阵秩、近邻图和结果稳定性。

## 常见误区

1. **降维就是删除不重要的原始列。** 不对。删除列属于特征选择；PCA、LDA 等会构造原始特征的新组合。
2. **原始维数很高，数据的真实结构也一定有同样高的维数。** 不对。相关性和流形结构可能让数据集中在低维区域。
3. **PCA 不中心化也没关系。** 通常不对。直接对未中心化二阶矩分解会把均值方向混入结果，和标准协方差 PCA 不同。
4. **任何任务做 PCA 前都必须标准化。** 不对。是否标准化取决于量纲和任务；但不同单位、尺度悬殊时需要特别检查方差是否被数值范围主导。
5. **保留方差越高，后续分类一定越好。** 不对。类别信息可能位于小方差方向，高方差也可能是噪声。
6. **主成分方向乘以 $-1$ 就改变了模型。** 不对。特征向量符号和得分同时翻转，重构与方差不变。
7. **LDA 和 PCA 一样不需要类别标签。** 不对。LDA 的类内、类间散布依赖标签，必须按监督流程防止泄漏。
8. **LDA 降到任意 $l$ 维都有同样多的非零判别方向。** 不对。$S_b$ 的秩通常至多为 $C-1$，二分类只有一条非零判别方向。
9. **$S_w$ 奇异说明数据完全不能分析。** 不对。它意味着不能直接求逆，可考虑收缩、正则化、广义求解或合适的数值方法。
10. **MDS 能恢复样本唯一的真实坐标和朝向。** 不对。距离不决定平移、旋转和反射，MDS 只能确定相对结构。
11. **经典 MDS 直接对距离矩阵 $D$ 做特征分解。** 不对。它先由平方距离双重中心化得到 $B=-\frac12JD^{(2)}J$，再分解内积矩阵。
12. **任意距离矩阵都能无损嵌入指定维数的欧氏空间。** 不对。可能出现负特征值和不可忽略的嵌入误差，非度量 MDS 还可只关注次序。
13. **LLE 中每一步是线性的，所以它属于线性降维。** 不对。局部线性关系组合起来可描述整体非线性流形。
14. **LLE 的无约束最优低维坐标就是全部样本重合。** 这是无意义退化解，因此必须加入零均值和协方差约束，并去除常量零特征方向。
15. **LLE 使用全局特征分解，所以结果不依赖近邻数。** 不对。固定 $W$ 后的特征问题有全局解，但 $W$ 本身由 $k$、距离和噪声决定。
16. **先在全数据上做 LDA，再划分测试集可以节省时间。** 不对。标签和数据分布已经泄漏，测试指标不可信。
17. **二维降维图上分成几团，就证明原数据有同样清晰的类别。** 不对。参数、距离扭曲和算法目标都会影响图，需结合稳定性和独立评价。

## 一句话总结

降维的核心，是根据任务选择要保留的信息——总体方差、类别分离、样本距离或局部关系，**并用未泄漏的数据、误差指标和稳定性分析确认低维表示确实有用。**

## 复习卡片

- **PCA**：中心化 → $\Sigma=X^TX/(n-1)$ → 最大特征向量 → $Y=XW$。
- **PCA 等价目标**：正交投影下，最大投影方差等价于最小平方重构误差。
- **保留方差**：前 $l$ 个特征值之和除以全部特征值之和，不等于预测性能。
- **LDA**：最大化 $\boldsymbol w^TS_b\boldsymbol w/(\boldsymbol w^TS_w\boldsymbol w)$，利用类别标签。
- **LDA 求解**：$S_b\boldsymbol w=\lambda S_w\boldsymbol w$；$S_w$ 奇异时考虑收缩或正则化。
- **维数上限**：标准 LDA 非零判别方向通常至多 $C-1$。
- **经典 MDS**：平方距离 → 双重中心化 $B$ → 特征分解 → $Z=U_l\Lambda_l^{1/2}$。
- **MDS 不唯一性**：整体平移、旋转和反射不改变样本间距离。
- **LLE 高维权重**：近邻重构、权重和为 1，局部 $G_i=Z_i^TZ_i$，必要时正则化。
- **LLE 低维坐标**：固定 $W$，最小化 $\operatorname{tr}(Y^TMY)$，加约束并去掉常量零方向。
- **工程流程**：先划分数据，再在训练集拟合预处理和降维；按最终任务评价。
- **可视化边界**：二维图用于提出假设，不单独证明类别、因果或全局距离关系。

## 自测问题

1. 为什么 PCA 的最大投影方差与最小重构误差等价？这一等价依赖哪些条件？
2. 对协方差矩阵 $\begin{pmatrix}2&1\\1&2\end{pmatrix}$，两个特征方向、特征值和一维保留方差比例是什么？正文示例的总重构误差是多少？
3. 为什么 PCA 可能保留不利于分类的方向？是否意味着它总是不如 LDA？
4. 根据正文二分类数据计算 $S_w,S_b$，解释为什么不能直接求 $S_w^{-1}S_b$，收缩后 Fisher 比率是多少？
5. 为什么标准 LDA 的非零判别方向通常不超过 $C-1$？这对二分类可视化有什么影响？
6. 对距离矩阵 $D=\begin{pmatrix}0&1&2\\1&0&1\\2&1&0\end{pmatrix}$，经典 MDS 的 $B$ 和一维坐标是什么？为什么坐标符号不唯一？
7. 经典 MDS 的双重中心化与一般应力最小化为什么不能简单视为同一个求解结论？负特征值提示什么？
8. 对正文 LLE 示例，验证权重和、重构结果以及 $G_i\boldsymbol w_i=0$，并解释为什么此时不能直接求逆。
9. 为什么 LLE 寻找低维坐标时必须加约束？最小特征值对应的常量方向为什么要删除？
10. 若要比较“原始特征 + 分类器”与“LDA + 分类器”的测试性能，应怎样安排数据划分、参数选择和评价？

## 参考答案

1. 完整正交基下，总平方长度等于保留部分与正交残差之和；总量固定，因此保留越多、残差越少。结论依赖 $W^TW=I$、线性正交投影和平方欧氏误差。
2. 方向为 $(1,1)^T/\sqrt2$ 和 $(1,-1)^T/\sqrt2$，特征值为 3 和 1，一维保留比例为 $3/4$；四个样本的总平方长度为 12，降到第一方向后的总重构误差为 3。
3. PCA 只看总体方差，不知道类别，类内大散布可能盖过类间方向。它并非总不如 LDA：无标签、压缩重构或类别假设不合适时，PCA 仍可能更合适，应按最终任务比较。
4. $S_w=\operatorname{diag}(2,0)$、$S_b=\operatorname{diag}(0,1)$。$S_w$ 第二方向为 0、矩阵奇异，不能直接求逆；取 $\lambda=0.5$ 后 $\widetilde S_w=\operatorname{diag}(1.5,0.5)$，沿 $(0,1)^T$ 的比率为 $1/0.5=2$。
5. $S_b$ 由 $C$ 个类别均值偏差的外积组成，这些偏差加权后满足一个线性约束，所以秩至多 $C-1$。二分类只有一条非零判别轴，第二个正交绘图轴不能解释成另一条非零 Fisher 判别方向。
6. $B=\begin{pmatrix}1&0&-1\\0&0&0\\-1&0&1\end{pmatrix}$，非零特征值 2，对应 $u=(1,0,-1)^T/\sqrt2$，所以 $Z=(1,0,-1)^T$。整体乘 $-1$、平移或旋转不改变距离。
7. 双重中心化的特征分解对应经典内积和平方距离准则，而直接最小化距离应力通常需要另一类优化分析。$B$ 的负特征值说明距离无法由欧氏坐标完全实现，不能无处理地开平方。
8. 权重为 $(0.25,0.5,0.25)$，和为 1，重构为 $(1,0.5)$，误差为 0，代入也有 $G_iw_i=0$。存在非零向量使 $G_iw_i=0$，所以 $G_i$ 奇异，不能直接用普通逆矩阵公式。
9. 无约束时所有低维点重合可得到零误差，却没有信息。零均值和协方差约束去除平移、尺度塌缩；全 1 常量方向只表示整体平移，不包含有用的样本差异，应删除。
10. 先固定训练、验证、测试划分，只用训练特征与标签拟合缩放和 LDA，在训练流程内选择维数和其他参数，再转换验证、测试数据；最后比较同一测试集上的指标，并同时报告训练成本和稳定性。

## 下一章预告

下一章：[第九章：特征选择——过滤式、封装式与嵌入式方法](/notes/data-science-guide-feature-selection/)。

第九章将从“构造新表示”转向“保留原始特征子集”，介绍特征选择的一般过程、常用评价与搜索方法，以及无监督特征选择。需要特别区分：减少特征数量并不总能提升性能，选择过程本身也必须放在训练流程中，避免用测试数据挑特征。

## 参考资料

- 欧高炎、朱占星、董彬等：《数据科学导引》第八章“降维”，高等教育出版社，原书第 168—190 页。
