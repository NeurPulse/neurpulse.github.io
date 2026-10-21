# -*- coding: utf-8 -*-
# 数学建模：数据预处理与插值
# Source: /notes/mathematical-modeling-data-preprocessing/
# Generated from the Markdown examples; edit the note and export again.

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

X_train = np.array([[1.], [2.], [np.nan], [4.], [5.]])
y_train = np.array([52., 54., 56., 58., 60.])
X_test = np.array([[3.], [6.]])
y_test = np.array([56., 62.])
model = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    LinearRegression()
)
model.fit(X_train, y_train)
prediction = model.predict(X_test)
imputer = model.named_steps["simpleimputer"]
print("训练填补值:", imputer.statistics_)
print("预测值:", prediction)
print("测试MAE:", np.mean(np.abs(prediction-y_test)))
assert np.allclose(imputer.statistics_, [3.])
assert np.allclose(prediction, [56., 62.])

# 独立的插值案例：只在已知区间内查询。
from scipy.interpolate import interp1d, CubicSpline, BarycentricInterpolator
x = np.array([1., 2., 3., 4.])
y = np.array([1., 4., 9., 16.])
linear = interp1d(x, y, kind="linear", bounds_error=True)
natural_spline = CubicSpline(x, y, bc_type="natural", extrapolate=False)
polynomial = BarycentricInterpolator(x, y)
print("2.5处三种插值:", linear(2.5), natural_spline(2.5), polynomial(2.5))
assert np.isclose(linear(2.5), 6.5)
assert np.isclose(polynomial(2.5), 6.25)
