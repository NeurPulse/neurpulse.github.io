# -*- coding: utf-8 -*-
# 数学建模：时间序列之移动平均法
# Source: /notes/mathematical-modeling-moving-average-method/
# Generated from the Markdown examples; edit the note and export again.

import numpy as np

def moving_average(values, window):
    values = np.asarray(values, dtype=float)
    if (values.ndim != 1 or not np.isfinite(values).all()
            or isinstance(window, bool) or not isinstance(window, (int, np.integer))
            or not 1 <= window <= len(values)):
        raise ValueError("需要有限一维序列，窗口为有效整数")
    out = np.full(len(values), np.nan)
    out[window-1:] = np.convolve(values, np.ones(window)/window, mode="valid")
    return out

small = np.arange(10., 22., 2.)
first = moving_average(small, 3)
second = moving_average(first[2:], 3)
a = 2*first[-1]-second[-1]
b = 2*(first[-1]-second[-1])/(3-1)
assert np.isclose(a+b, 22.)

rng = np.random.default_rng(42)
t = np.arange(140)
y = 30 + 0.05*t + 3*np.sin(2*np.pi*t/7) + rng.normal(0, 1, len(t))
def predict_at(values, time, window):
    return np.mean(values[time-window:time])

validation = {}
for window in [3, 7, 14]:
    prediction = np.array([predict_at(y, i, window) for i in range(84, 112)])
    validation[window] = np.mean(np.abs(y[84:112]-prediction))
best_window = min(validation, key=validation.get)
test_prediction = np.array([predict_at(y, i, best_window) for i in range(112, 140)])
print("验证MAE:", validation, "选中窗口:", best_window)
print("最终测试MAE:", np.mean(np.abs(y[112:]-test_prediction)))
assert np.isnan(first[:2]).all()
assert np.allclose(first[2:], [12., 14., 16., 18.])
