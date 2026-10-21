# -*- coding: utf-8 -*-
# 数学建模：研究生数学建模赛题的阅读与方法选择
# Source: /notes/mathematical-modeling-machine-learning-data-processing/
# Generated from the Markdown examples; edit the note and export again.

import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import mean_absolute_error

rng = np.random.default_rng(42)
X = rng.normal(size=(160, 60))
activity = 2*X[:, 0]-X[:, 1]+0.5*X[:, 2]+rng.normal(0, 0.5, len(X))
properties = (X[:, :5]+rng.normal(0, 0.5, (len(X), 5)) > 0).astype(int)
train_indices, test_indices = train_test_split(np.arange(len(X)), test_size=0.25, random_state=42)

regression = make_pipeline(
    StandardScaler(), SelectKBest(score_func=f_regression), Ridge()
)
search = GridSearchCV(
    regression,
    {"selectkbest__k": [5, 10, 20], "ridge__alpha": [1., 10.]},
    cv=KFold(3, shuffle=True, random_state=42),
    scoring="neg_mean_absolute_error"
)
search.fit(X[train_indices], activity[train_indices])
prediction = search.predict(X[test_indices])
print("选择参数:", search.best_params_)
print("回归留出MAE:", mean_absolute_error(activity[test_indices], prediction))

classifier = MultiOutputClassifier(
    make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
)
classifier.fit(X[train_indices], properties[train_indices])
predicted_properties = classifier.predict(X[test_indices])
print("各性质准确率:", (predicted_properties == properties[test_indices]).mean(axis=0))
assert predicted_properties.shape == (len(test_indices), 5)
