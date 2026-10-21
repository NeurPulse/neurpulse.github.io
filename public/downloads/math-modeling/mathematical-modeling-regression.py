# -*- coding: utf-8 -*-
# 数学建模：回归与曲线拟合
# Source: /notes/mathematical-modeling-regression/
# Generated from the Markdown examples; edit the note and export again.

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error

x = np.arange(1., 13.).reshape(-1, 1)
y = 40 + 3*x.ravel() + np.array([0., 1., -1., 2., 0., -2., 1., 0., -1., 1., 0., -1.])
X_train, X_test = x[:10], x[10:]
y_train, y_test = y[:10], y[10:]
pipeline = make_pipeline(
    PolynomialFeatures(include_bias=False),
    StandardScaler(),
    Ridge()
)
search = GridSearchCV(
    pipeline,
    {"polynomialfeatures__degree": [1, 2], "ridge__alpha": [0.1, 1., 10.]},
    cv=KFold(5, shuffle=True, random_state=42),
    scoring="neg_mean_absolute_error"
)
search.fit(X_train, y_train)
prediction = search.predict(X_test)
baseline = np.full_like(y_test, y_train.mean())
print("选择的参数:", search.best_params_)
print("测试预测:", prediction)
print("模型MAE、基线MAE:",
      mean_absolute_error(y_test, prediction),
      mean_absolute_error(y_test, baseline))
assert np.isfinite(prediction).all()

# 重现手算：加入一列1表示截距。
A = np.column_stack([np.ones(3), [1., 2., 3.]])
coefficient = np.linalg.lstsq(A, [2., 3., 5.], rcond=None)[0]
print("手算的截距、斜率:", coefficient)
assert np.allclose(coefficient, [1/3, 1.5])
