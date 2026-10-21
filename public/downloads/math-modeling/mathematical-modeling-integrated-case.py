# -*- coding: utf-8 -*-
# 数学建模：综合案例 从销量预测到备货决策
# Source: /notes/mathematical-modeling-integrated-case/
# Generated from the Markdown examples; edit the note and export again.

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

rng = np.random.default_rng(42)
t = np.arange(210)
promotion = rng.binomial(1, 0.15, len(t))  # 预先计划，预测时可知
demand = np.maximum(np.rint(
    50 + 0.08*t + 5*np.sin(2*np.pi*t/7) + 6*promotion + rng.normal(0, 3, len(t))
), 0).astype(int)
table = pd.DataFrame({
    "date": pd.date_range("2026-01-01", periods=len(t), freq="D"),
    "day": t, "promotion": promotion, "demand": demand
})
assert table["date"].is_unique and table["date"].is_monotonic_increasing
assert table["demand"].ge(0).all()
X = np.column_stack([t, np.sin(2*np.pi*t/7), np.cos(2*np.pi*t/7), promotion])

def fit_predict(kind, end, query_indices):
    if kind == "mean":
        return np.full(len(query_indices), demand[:end].mean()), None
    if kind == "weekday":
        means = np.array([demand[:end][t[:end] % 7 == d].mean() for d in range(7)])
        return means[query_indices % 7], None
    model = make_pipeline(StandardScaler(), Ridge(alpha=float(kind)))
    model.fit(X[:end], demand[:end])
    return model.predict(X[query_indices]), model

candidates = ["mean", "weekday", 0.1, 1., 10.]
valid_indices = np.arange(112, 140)
errors = {}
for candidate in candidates:
    predicted, _ = fit_predict(candidate, 112, valid_indices)
    errors[candidate] = np.mean(np.abs(demand[valid_indices]-predicted))
selected = min(errors, key=errors.get)
print("验证MAE与选择:", errors, selected)

all_indices = np.arange(140, 210)
forecast, fitted_model = fit_predict(selected, 140, all_indices)
calibration_residuals = demand[140:168]-forecast[:28]
test_forecast = forecast[28:]
test_demand = demand[168:]
assert len(test_forecast) == len(test_demand) == 42

def loss(quantity, scenarios, under=6., over=3.):
    return under*np.maximum(scenarios-quantity, 0) + over*np.maximum(quantity-scenarios, 0)

def choose_stock(prediction, residuals, under=6., over=3.):
    scenarios = np.maximum(np.rint(prediction+residuals), 0).astype(int)
    choices = np.arange(scenarios.max()+1)
    expected = np.array([loss(q, scenarios, under, over).mean() for q in choices])
    return int(choices[expected.argmin()])

stock = np.array([choose_stock(p, calibration_residuals) for p in test_forecast])
point_stock = np.maximum(np.rint(test_forecast), 0).astype(int)
def summarize(quantity):
    profit = 10*np.minimum(quantity, test_demand) + np.maximum(quantity-test_demand, 0)-4*quantity
    return {
        "mean_profit": float(profit.mean()),
        "mean_shortage": float(np.maximum(test_demand-quantity, 0).mean()),
        "mean_leftover": float(np.maximum(quantity-test_demand, 0).mean()),
        "mean_loss": float(loss(quantity, test_demand).mean())
    }
print("最终预测MAE:", np.mean(np.abs(test_forecast-test_demand)))
print("点预测备货:", summarize(point_stock))
print("情景优化备货:", summarize(stock))
print("缺货代价18时平均备货:",
      np.mean([choose_stock(p, calibration_residuals, under=18.) for p in test_forecast]))

# 独立核验手算例子。
scenarios = np.array([10, 12, 12, 14])  # 以重复次数表达概率
best_q = min(range(16), key=lambda q: loss(q, scenarios).mean())
assert best_q == 12 and np.isclose(loss(12, scenarios).mean(), 4.5)
assert np.all(stock >= 0) and np.issubdtype(stock.dtype, np.integer)
