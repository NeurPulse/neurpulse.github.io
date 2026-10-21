# -*- coding: utf-8 -*-
# 数学建模：树模型与集成学习
# Source: /notes/mathematical-modeling-ml-tree-ensemble/
# Generated from the Markdown examples; edit the note and export again.

import numpy as np
from scipy.stats import entropy
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report

gain = 1-0.75*entropy([1/3, 2/3], base=2)
split_info = entropy([0.25, 0.75], base=2)
print("增益、划分信息量、增益率:", gain, split_info, gain/split_info)
assert np.isclose(gain/split_info, 0.3836885465963443)

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)
candidates = [
    ("tree", DecisionTreeClassifier(random_state=42), {"max_depth": [2, 4, None]}),
    ("forest", RandomForestClassifier(n_estimators=60, random_state=42, n_jobs=2),
     {"max_depth": [3, None]}),
    ("boosting", GradientBoostingClassifier(n_estimators=60, random_state=42),
     {"max_depth": [1, 2]})
]
cv = StratifiedKFold(3, shuffle=True, random_state=42)
fitted = []
for name, model, grid in candidates:
    search = GridSearchCV(model, grid, cv=cv, scoring="balanced_accuracy")
    search.fit(X_train, y_train)
    fitted.append((search.best_score_, name, search))
    print(name, "训练内得分:", search.best_score_, "参数:", search.best_params_)
_, selected_name, selected = max(fitted, key=lambda item: item[0])
print("最终选择:", selected_name)
print(classification_report(y_test, selected.predict(X_test), zero_division=0))
