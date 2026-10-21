# -*- coding: utf-8 -*-
# 数学建模：几何模型与线性代数
# Source: /notes/mathematical-modeling-geometry/
# Generated from the Markdown examples; edit the note and export again.

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
