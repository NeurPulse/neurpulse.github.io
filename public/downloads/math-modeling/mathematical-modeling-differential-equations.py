# -*- coding: utf-8 -*-
# 数学建模：微分方程与动力系统
# Source: /notes/mathematical-modeling-differential-equations/
# Generated from the Markdown examples; edit the note and export again.

import numpy as np
from scipy.integrate import solve_ivp
import sympy as sp

ambient, initial, rate = 20., 80., 0.1
def cooling(t, temperature):
    return -rate*(temperature-ambient)

time = np.arange(0., 21.)
solution = solve_ivp(
    cooling, (0., 20.), [initial],
    t_eval=time, rtol=1e-9, atol=1e-10
)
exact = ambient + (initial-ambient)*np.exp(-rate*time)
assert solution.success
assert np.max(np.abs(solution.y[0]-exact)) < 1e-6
print("20分钟数值与解析温度:", solution.y[0, -1], exact[-1])
print("到30度的时间:", np.log(6)/rate)

# 解析求解：符号函数未知，给定初始条件。
t = sp.symbols("t", real=True)
T = sp.Function("T")
symbolic = sp.dsolve(
    sp.Eq(sp.diff(T(t), t), -(T(t)-20)/10),
    ics={T(0): 80}
)
print("符号解:", symbolic)
assert sp.simplify(symbolic.rhs-(20+60*sp.exp(-t/10))) == 0

# 用测量值恢复参数；此处测量值是无噪声的教学值。
measured_at_5 = ambient + 60*np.exp(-0.5)
estimated = -np.log((measured_at_5-ambient)/60)/5
assert np.isclose(estimated, rate)
