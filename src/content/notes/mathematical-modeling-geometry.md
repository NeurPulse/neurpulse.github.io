---
title: '数学建模：几何模型'
description: '向量表示法、旋转矩阵与NumPy线性代数'
pubDate: 2026-08-07
category: '数学建模'
tags: ['几何模型','向量','NumPy','线性代数']
---

## 几何模型

### 向量表示法与几何建模

$$
x = [x_1,x_2,\dots,x_n]^T
$$

```python
import numpy as np
x = np.array([1, 2, 3, 5, 8])
# array([1, 2, 3, 5, 8])
```

> 如果需要求一个点旋转后的点的坐标，需要用这个点的坐标乘旋转矩阵

二维坐标变换：旋转矩阵
$$
\begin{bmatrix} cos\theta & -sin\theta \\ sin\theta & cos\theta \end{bmatrix}
$$

```python
import numpy as np

# 设定旋转角度，这里我们以30度为例
theta = np.radians(30)  # 将30度转换为弧度

# 创建旋转矩阵
rotation_matrix = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
])
a,b = 5,3
# 假设我们有一个点 (a, b)
point = np.array([a, b])

# 通过旋转矩阵变换这个点的坐标
rotated_point = rotation_matrix.dot(point)

print("原坐标为:", point)
print("旋转后的坐标为:", rotated_point)
# 原坐标为: [5 3]
# 旋转后的坐标为: [2.83012702 5.09807621]

```

三维旋转矩阵

z轴不变
$$
$R_{z}(\alpha)=\begin{bmatrix} cos\alpha & -sin\alpha & 0 \\ sin\alpha & cos\alpha & 0 \\ 0 & 0 & 1\end{bmatrix}
$$
y轴不变
$$
R_{y}(\beta)=\begin{bmatrix} cos\beta & 0 & sin\beta \\ 0 & 1 & 0 \\ -sin\beta & 0 & cos\beta\end{bmatrix}
$$
x轴不变
$$
R_{x}(\gamma)=\begin{bmatrix} 1 & 0 & 0 \\ 0 & cos\gamma & -sin\gamma \\ 0 & sin\gamma & cos\gamma \end{bmatrix}
$$

> 如果一点需要绕三个旋转轴旋转，最终的旋转矩阵就为上面三个旋转矩阵的乘积 $ R = R_{z}(\alpha)R_{y}(\beta)R_{x}(\gamma)$

```python
import numpy as np

# 定义旋转角度（以弧度为单位）
alpha = np.radians(30)  # 绕 Z 轴旋转
beta = np.radians(45)   # 绕 Y 轴旋转
gamma = np.radians(60)  # 绕 X 轴旋转

# 定义旋转矩阵
R_z = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                [np.sin(alpha), np.cos(alpha), 0],
                [0, 0, 1]])

R_y = np.array([[np.cos(beta), 0, np.sin(beta)],
                [0, 1, 0],
                [-np.sin(beta), 0, np.cos(beta)]])

R_x = np.array([[1, 0, 0],
                [0, np.cos(gamma), -np.sin(gamma)],
                [0, np.sin(gamma), np.cos(gamma)]])

# 总旋转矩阵
R = R_z @ R_y @ R_x

# 定义点P的坐标
P = np.array([1, 2, 3])

# 计算旋转后的坐标
P_rotated = R @ P

print("旋转后P点的坐标为:", P_rotated)
# 旋转后P点的坐标为: [3.39062937 0.11228132 1.57829826]

```

欧拉角旋转顺序

- **Z-Y-X（Roll-Pitch-Yaw）**：首先绕z轴旋转*α*（Yaw），然后绕*新位置的*yy轴旋转*β*（Pitch），最后绕*新位置的*x轴旋转γ*（Roll）。
- **Z-X-Y（Yaw-Pitch-Roll）**：首先绕z轴旋转α*（Yaw），然后绕*新位置的*xx轴旋转*β*（Pitch），最后再次绕y轴旋转γ*（Roll）。

```python


# 定义欧拉角（以弧度为单位）
alpha = np.radians(30)  # 绕 z 轴的 Yaw 角度
beta = np.radians(45)   # 绕 y 轴的 Pitch 角度
gamma = np.radians(60)  # 绕 x 轴的 Roll 角度

# 构建对应的旋转矩阵
R_z = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                [np.sin(alpha), np.cos(alpha), 0],
                [0, 0, 1]])
R_y = np.array([[np.cos(beta), 0, np.sin(beta)],
                [0, 1, 0],
                [-np.sin(beta), 0, np.cos(beta)]])
R_x = np.array([[1, 0, 0],
                [0, np.cos(gamma), -np.sin(gamma)],
                [0, np.sin(gamma), np.cos(gamma)]])

# 总旋转矩阵，注意乘法的顺序
R = np.dot(R_x, np.dot(R_y, R_z))

print("组合旋转矩阵为:")
print(R)
# 组合旋转矩阵为:
# [[ 0.61237244 -0.35355339  0.70710678]
#  [ 0.78033009  0.12682648 -0.61237244]
#  [ 0.12682648  0.9267767   0.35355339]]
```

### Numpy 与线性代数

**创建向量和矩阵**

```python
import numpy as np
# 创建向量
vector = np.array([1, 2, 3])
# 创建矩阵
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
```

**向量和矩阵的基本属性**

```python
# 向量的维度
print(vector.shape) # (3, )
# 矩阵的维度
print(matrix.shape) # (3, 3)
# 矩阵的行数和列数
print(matrix.shape[0])  # 行数, 3
print(matrix.shape[1])  # 列数, 3
```

**索引和切片**

```python
# 索引
print(vector[0])  # 输出第一个元素, 1
print(matrix[1, 1])  # 输出第二行第二列的元素, 5

# 切片
print(vector[0:2])  # 输出前两个元素, [1, 2]
print(matrix[0:2, 0:2])  # 输出左上角的2x2子矩阵, [[1, 2], [4, 5]]
```

**向量和矩阵的运算**

```python
# 向量加法
vector1 = np.array([1, 2, 3])
vector2 = np.array([4, 5, 6])
print(np.add(vector1, vector2))
# [5, 7, 9]

# 矩阵乘法
matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6], [7, 8]])
print(np.dot(matrix1, matrix2))  # 或使用 matrix1 @ matrix2
# [[19, 22],
#  [43, 50]]
```

**线性代数基本运算**

```python
import numpy as np

# 数量乘法示例
scalar = 5
scaled_vector = scalar * vector
print("Scaled vector:", scaled_vector)
# Scaled vector: [ 5 10 15]

# 矩阵的转置示例
transposed_matrix = matrix.T
print("Transposed matrix:\n", transposed_matrix)
# Transposed matrix:
# [[1, 4, 7]
#  [2, 5, 8]
#  [3, 6, 9]]

# 计算行列式示例
matrix_determinant = np.linalg.det(matrix)
print("Matrix determinant:", matrix_determinant)
# Matrix determinant: 0.0

# 求解线性方程组示例
A = np.array([[3, 1], [1, 2]])
b = np.array([9, 8])
solution = np.linalg.solve(A, b)
print("Solution of the linear system:", solution)
# Solution of the linear system: [2. 3.]
```

**numpy.linalg 的使用**

> NumPy中的 `linalg` 子模块包含了一系列关于线性代数的函数。
>
> 求解方程组,计算特征值、特征向量,执行奇异值分解

**计算逆矩阵**

```python
import numpy as np
# If the matrix is singular, use the pseudo-inverse
pseudo_inverse_matrix = np.linalg.pinv(matrix)
print("Pseudo-inverse of the matrix:")
print(pseudo_inverse_matrix)
# Pseudo-inverse of the matrix:
# [[-6.38888889e-01 -1.66666667e-01  3.05555556e-01]
#  [-5.55555556e-02  4.20756436e-17  5.55555556e-02]
#  [ 5.27777778e-01  1.66666667e-01 -1.94444444e-01]]
```

**特征值和特征向量**

```python
eigenvalues, eigenvectors = np.linalg.eig(matrix)
print(eigenvalues)
# [ 1.61168440e+01 -1.11684397e+00 -1.30367773e-15]

print(eigenvectors)
# [[-0.23197069 -0.78583024  0.40824829]
#  [-0.52532209 -0.08675134 -0.81649658]
#  [-0.8186735   0.61232756  0.40824829]]
```

**奇异值分解**

```python
U, S, V = np.linalg.svd(matrix)
print(U)
# [[-0.21483724  0.88723069  0.40824829]
#  [-0.52058739  0.24964395 -0.81649658]
#  [-0.82633754 -0.38794278  0.40824829]]

print(S)
# [1.68481034e+01 1.06836951e+00 4.41842475e-16]

print(V)
# [[-0.47967118 -0.57236779 -0.66506441]
#  [-0.77669099 -0.07568647  0.62531805]
#  [-0.40824829  0.81649658 -0.40824829]]
```

**范数计算**

```python
norm = np.linalg.norm(vector)
print(norm)
# 3.7416573867739413
```

