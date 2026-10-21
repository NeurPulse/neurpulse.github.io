# -*- coding: utf-8 -*-
# 数学建模：时间序列分析与验证
# Source: /notes/mathematical-modeling-time-series/
# Generated from the Markdown examples; edit the note and export again.

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

rng = np.random.default_rng(42)
t = np.arange(140)
y = 30 + 0.05*t + 3*np.sin(2*np.pi*t/7) + rng.normal(0, 1, len(t))
series = pd.Series(y, index=pd.date_range("2026-01-01", periods=len(y), freq="D"))
cut = 112
train = series.iloc[:cut]
# 分解只在训练段做描述，不把居中趋势作为预测输入。
decomposition = seasonal_decompose(train, model="additive", period=7)
reconstructed = decomposition.trend+decomposition.seasonal+decomposition.resid
valid = reconstructed.notna()
assert np.allclose(reconstructed[valid], train[valid])

truth = y[cut:]
predictions = {
    "training mean": np.full(len(truth), train.mean()),
    "yesterday": y[cut-1:-1],
    "last week": y[cut-7:-7],
}
for name, prediction in predictions.items():
    mae = np.mean(np.abs(truth-prediction))
    rmse = np.sqrt(np.mean((truth-prediction)**2))
    print(name, "MAE", round(mae, 3), "RMSE", round(rmse, 3))
print("一阶差分均值:", np.diff(train).mean())
assert len(predictions["last week"]) == len(truth)
