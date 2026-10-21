---
title: "数学建模：几何模型与线性代数"
description: "从距离、投影和旋转走到定位、方程与分解"
pubDate: 2026-08-07
updatedDate: 2026-10-21
order: 7
category: '数学建模'
tags: ["几何模型","向量","NumPy","线性代数","零基础"]
---

> [专题学习地图](/notes/mathematical-modeling-roadmap/) · 第 7 / 27 篇

## 学习目标与先修知识

目标是把位置、方向和距离写成向量，手算一个旋转与定位问题，理解方程组、最小二乘、特征值和 SVD 分别在做什么。先修为数组形状、矩阵乘法与平方根；不要求先掌握抽象线性空间理论。

可回看：[必备数学与 Python 基础](/notes/mathematical-modeling-foundations/)。

## 问题背景与使用时机

几何建模常见于定位、路线、设施选址、姿态变化和三维重构。例如，知道一个设备到几个基站的距离，能否恢复设备坐标？第一步不是寻找机器学习模型，而是把距离关系写成方程。

选择坐标系是建模的一部分。坐标原点、轴方向、长度单位不同，输入的数字就不同，但同一物理关系应保持一致。

## 核心概念与符号

点 $P=(x,y)$ 的位置可写成列向量。两点距离为：

$$
d(P,Q)=\|\boldsymbol p-\boldsymbol q\|_2
=\sqrt{(x_P-x_Q)^2+(y_P-y_Q)^2}.
$$

内积为 $\boldsymbol a^T\boldsymbol b=\|\boldsymbol a\|\|\boldsymbol b\|\cos\theta$。内积为零时两个非零向量垂直；向量在单位方向 $\boldsymbol u$ 上的有符号投影长度为 $\boldsymbol u^T\boldsymbol x$。

采用右手坐标系、主动旋转和列向量，二维逆时针旋转为：

$$
R(\theta)=
\begin{bmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{bmatrix},
\qquad\boldsymbol p_{\mathrm{new}}=R\boldsymbol p.
$$

旋转保持长度，$R^TR=I$、$\det R=1$。三维旋转一般不可交换；$R_zR_yR_x$ 作用于列向量时，右侧 $R_x$ 先作用。固定轴旋转与随物体运动的轴旋转应明确区分。

方程组 $A\boldsymbol x=\boldsymbol b$ 用系数矩阵描述多个关系。可逆方阵有唯一解；方程太多、带噪声或秩不足时，需要看可解性与最小二乘，不能默认矩阵都有逆。

## 手算一个最小例子

### 旋转

点 $(3,4)$ 绕原点逆时针转 $90^\circ$，余弦为 0、正弦为 1：

$$
\begin{bmatrix}0&-1\\1&0\end{bmatrix}
\begin{bmatrix}3\\4\end{bmatrix}
=\begin{bmatrix}-4\\3\end{bmatrix}.
$$

旋转前后长度都为 5，这是一项很好的自检。

### 三基站定位

三个基站为 $A=(0,0),B=(10,0),C=(0,10)$，测得距离平方为 $25,65,45$。未知点满足：

$$
x^2+y^2=25,\quad(x-10)^2+y^2=65,\quad x^2+(y-10)^2=45.
$$

第二式减第一式得 $-20x+100=40$，故 $x=3$；第三式减第一式得 $-20y+100=20$，故 $y=4$。再代回第一式验证 $9+16=25$。只解差分线性式而不检查原距离，可能漏掉不一致。

## 建模步骤与完整案例

案例：设备在平面内定位。假设基站坐标准确、距离使用相同单位、忽略高度差。先用无噪声距离确认方程，再加入小测量误差。

有误差时，三个圆通常没有共同交点。可最小化各测量距离残差：

$$
\min_{\boldsymbol p}\sum_i\left(\|\boldsymbol p-\boldsymbol a_i\|-r_i\right)^2.
$$

这里变量为两个坐标，目标是残差平方和，参数是已知基站与测距。输出除位置外还应包含距离残差及对噪声的敏感程度。若基站几乎共线，某些方向的定位会不稳定，增加数值精度不能替代更好的几何布局。

## Python 实现与结果解释

配套文件：[查看或下载本篇完整脚本](/downloads/math-modeling/mathematical-modeling-geometry.py)。

```python
import numpy as np
from scipy.optimize import least_squares

point = np.array([3., 4.])
R = np.array([[0., -1.], [1., 0.]])
rotated = R @ point
print("旋转后:", rotated)
assert np.allclose(rotated, [-4., 3.])
assert np.isclose(np.linalg.norm(rotated), np.linalg.norm(point))

anchors = np.array([[0., 0.], [10., 0.], [0., 10.]])
ranges = np.linalg.norm(anchors-point, axis=1)
A = 2*(anchors[1:]-anchors[0])
b = (ranges[0]**2-ranges[1:]**2
     + np.sum(anchors[1:]**2, axis=1)-np.sum(anchors[0]**2))
linear_position = np.linalg.solve(A, b)
assert np.allclose(linear_position, point)

observed = ranges + np.array([0.1, -0.1, 0.05])
def residual(position):
    return np.linalg.norm(position-anchors, axis=1)-observed

fit = least_squares(residual, x0=linear_position)
print("带噪定位:", fit.x, "距离残差:", residual(fit.x))
assert fit.success

# SVD：把矩阵分解后再重建，是比逐项比较奇异向量更可靠的检查。
matrix = np.array([[1., 2.], [2., 4.], [3., 6.]])
U, singular_values, Vh = np.linalg.svd(matrix, full_matrices=False)
assert np.allclose(U @ np.diag(singular_values) @ Vh, matrix)
print("奇异值:", singular_values)
```

无噪声解必须重现 $(3,4)$；带噪声时通常略有偏离。`fit.success` 表示求解器满足停止条件，不表示真实位置必然准确，也不是全局最优的普遍保证。

## 方法选择与常见误区

### 特征值、SVD 与伪逆

特征关系 $A\boldsymbol v=\lambda\boldsymbol v$ 表示某些方向只被伸缩而不改变方向，适用于方阵；特征向量可整体变号，重复特征值的基也不唯一。

SVD 为 $A=U\Sigma V^T$，可用于任意实矩阵。NumPy 第三个返回值是 `Vh`，即 $V$ 的共轭转置。上例两列成比例，只有一个显著非零奇异值，说明两列信息冗余；下一阶段的 PCA 会利用这种结构降维。

伪逆 $A^+$ 不等于奇异矩阵的普通逆。$A^+\boldsymbol b$ 给出最小范数的最小二乘解，不能保证任意 b 都满足原方程。

位置可由明确几何关系描述时，先尝试方程或优化；传感器有偏差时还需校准；若关系复杂且有大量标注，可再引入学习模型。地理经纬度也不能不加处理就当成普通平面米制坐标。

## 练习

1. 点 $(1,0)$ 逆时针旋转 $90^\circ$ 后是什么？
2. 向量 $(1,2)$ 与 $(2,-1)$ 是否垂直？
3. 三个带噪声圆没有共同交点，是否说明不能定位？
4. SVD 重建中把 Vh 再转置一次会怎样？

## 参考答案

1. $(0,1)$；注意主动旋转方向和列向量约定。
2. 内积 $1\times2+2(-1)=0$，两个非零向量垂直。
3. 可以按误差模型做最小二乘或稳健估计，并报告残差；不能要求噪声数据严格满足全部等式。
4. 对一般矩阵会破坏正确的乘法与重建，因为 NumPy 已经返回转置后的形式；偶然对称时数值可能迷惑人，应按定义写代码。

## 复习卡片与下一步

复习卡：距离看差向量；投影看内积；旋转先约定坐标与顺序；求解先检查秩和残差；分解用重建验证。下一篇把静态位置关系扩展为随时间变化的动力系统。

[上一篇 回归与曲线拟合](/notes/mathematical-modeling-regression/) · [下一篇 微分方程与动力系统](/notes/mathematical-modeling-differential-equations/)

## 参考资料

- [Datawhale《数学建模导论》第1章](https://datawhalechina.github.io/intro-mathmodel/#/CH1/第1章-解析方法与几何模型)：用于建立知识目录，建议与本篇的手算和案例对照阅读。
- [NumPy 线性代数](https://numpy.org/doc/stable/reference/routines.linalg.html)。
- [SciPy 非线性最小二乘](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)。
