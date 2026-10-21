# -*- coding: utf-8 -*-
# 数学建模：时间序列之指数平滑法
# Source: /notes/mathematical-modeling-exponential-smoothing/
# Generated from the Markdown examples; edit the note and export again.

import numpy as np
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing

level = 10.
one_step_errors = []
for observation in [12., 14.]:
    one_step_errors.append(observation-level)
    level = 0.5*observation + 0.5*level
assert np.isclose(level, 12.5)
assert np.allclose(one_step_errors, [2., 3.])

rng = np.random.default_rng(42)
t = np.arange(140)
y = 30 + 0.05*t + 3*np.sin(2*np.pi*t/7) + rng.normal(0, 1, len(t))
def fit_kind(kind, data):
    if kind == "level":
        return SimpleExpSmoothing(data, initialization_method="estimated").fit()
    if kind == "trend":
        return Holt(data, initialization_method="estimated").fit()
    return ExponentialSmoothing(
        data, trend="add", seasonal="add", seasonal_periods=7,
        initialization_method="estimated"
    ).fit()

validation = {}
for kind in ["level", "trend", "seasonal"]:
    forecast = np.asarray(fit_kind(kind, y[:84]).forecast(28))
    validation[kind] = np.mean(np.abs(y[84:112]-forecast))
selected = min(validation, key=validation.get)
final_forecast = np.asarray(fit_kind(selected, y[:112]).forecast(28))
print("验证MAE:", validation, "选择:", selected)
print("28步最终测试MAE:", np.mean(np.abs(y[112:]-final_forecast)))
assert final_forecast.shape == (28,)
assert np.isfinite(final_forecast).all()
