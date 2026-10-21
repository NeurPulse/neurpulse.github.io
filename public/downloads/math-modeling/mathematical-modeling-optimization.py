# -*- coding: utf-8 -*-
# 数学建模：线性规划与整数优化
# Source: /notes/mathematical-modeling-optimization/
# Generated from the Markdown examples; edit the note and export again.

import numpy as np
from scipy.optimize import linprog, linear_sum_assignment, minimize

M = 7.5
cost = np.array([-3., -2.])  # 最大利润转为最小负利润
A = np.array([[2., 1.], [1., 1.]])
b = np.array([M, 5.])
relaxed = linprog(cost, A_ub=A, b_ub=b, bounds=[(0, None)]*2, method="highs")
integer = linprog(cost, A_ub=A, b_ub=b, bounds=[(0, None)]*2,
                  integrality=[1, 1], method="highs")
assert relaxed.success and integer.success
print("连续解:", relaxed.x, "利润:", -relaxed.fun)
print("整数解:", integer.x, "利润:", -integer.fun)
print("整数资源剩余:", b-A@integer.x)
oracle = max(3*x+2*y for x in range(6) for y in range(6)
             if 2*x+y <= M and x+y <= 5)
assert np.isclose(-integer.fun, oracle)
assert np.all(A@integer.x <= b+1e-8)
assert np.allclose(integer.x, np.round(integer.x))

# 指派：每个工人、每个任务各使用一次，使总成本最小。
assignment_cost = np.array([[4., 1.], [2., 3.]])
workers, tasks = linear_sum_assignment(assignment_cost)
print("指派总成本:", assignment_cost[workers, tasks].sum())
assert assignment_cost[workers, tasks].sum() == 3

# 有边界的一维凸非线性示例。
fit = minimize(lambda z: (z[0]-3)**2, x0=[0.], bounds=[(0., 2.)])
assert fit.success
assert np.isclose(fit.x[0], 2.)
