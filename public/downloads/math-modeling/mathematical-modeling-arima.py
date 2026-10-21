# -*- coding: utf-8 -*-
# 数学建模：ARIMA 与相关时间序列
# Source: /notes/mathematical-modeling-arima/
# Generated from the Markdown examples; edit the note and export again.

import numpy as np
from statsmodels.tsa.arima.model import ARIMA

rng = np.random.default_rng(42)
values = np.empty(280)
values[0] = 10.
for i in range(1, len(values)):
    values[i] = 10 + 0.6*(values[i-1]-10) + rng.normal()
y = values[100:]
train, valid, test = y[:108], y[108:144], y[144:]

orders = [(1, 0, 0), (2, 0, 0)]
errors = {"mean": float(np.mean(np.abs(valid-train.mean())))}
for order in orders:
    fit = ARIMA(train, order=order, trend="c").fit()
    prediction = np.asarray(fit.forecast(len(valid)))
    errors[order] = float(np.mean(np.abs(valid-prediction)))
choice = min(errors, key=errors.get)
history = y[:144]
if choice == "mean":
    prediction = np.full(len(test), history.mean())
else:
    fit = ARIMA(history, order=choice, trend="c").fit()
    prediction = np.asarray(fit.forecast(len(test)))
print("验证误差:", errors, "选择:", choice)
print("最终多步MAE:", np.mean(np.abs(test-prediction)))
print("已知参数的两步预测:", 10+0.6**2*(15-10))
assert np.isclose(10+0.6**2*5, 11.8)
assert np.isfinite(prediction).all()
