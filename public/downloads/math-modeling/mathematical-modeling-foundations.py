# -*- coding: utf-8 -*-
# 数学建模：必备数学与 Python 基础
# Source: /notes/mathematical-modeling-foundations/
# Generated from the Markdown examples; edit the note and export again.

import numpy as np

X = np.array([[50., 2.], [80., 3.], [100., 3.]])
w = np.array([0.4, 2.0])
prediction = X @ w + 5.0
print("形状:", X.shape, w.shape, prediction.shape)
print("预测租金（百元/月）:", prediction)
assert np.allclose(prediction, [29., 43., 51.])

values = np.array([2., 4., 6.])
print("均值、样本方差:", values.mean(), values.var(ddof=1))
weight = 1.0
gradient = 4 * (2*weight - 6)
weight -= 0.1 * gradient
print("更新后的参数:", weight)
assert np.isclose(weight, 2.6)

# 条件概率：总共100人，20人带伞；其中12人遇到下雨。
print("已知带伞时下雨的样本比例:", 12/20)
