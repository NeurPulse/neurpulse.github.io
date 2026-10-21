# -*- coding: utf-8 -*-
# 数学建模：机器学习流程与 KNN
# Source: /notes/mathematical-modeling-ml-sklearn-knn/
# Generated from the Markdown examples; edit the note and export again.

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def hand_knn(train, labels, query, k):
    train, query = np.asarray(train, float), np.asarray(query, float)
    labels = np.asarray(labels)
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool) or not 1 <= k <= len(train):
        raise ValueError("k必须是有效整数")
    index = np.argsort(np.sum((train-query)**2, axis=1), kind="stable")[:k]
    classes, counts = np.unique(labels[index], return_counts=True)
    return classes[counts.argmax()]  # 平票时按类别排序取前者

assert hand_knn([[0.], [1.], [2.]], ["A", "B", "B"], [0.1], 3) == "B"
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)
search = GridSearchCV(
    make_pipeline(StandardScaler(), KNeighborsClassifier()),
    {"kneighborsclassifier__n_neighbors": [1, 3, 5, 7, 9, 11, 15]},
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring="accuracy"
)
search.fit(X_train, y_train)
prediction = search.predict(X_test)
baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
print("选择的k:", search.best_params_)
print("模型与基线准确率:", accuracy_score(y_test, prediction), baseline.score(X_test, y_test))
print("混淆矩阵:", confusion_matrix(y_test, prediction))
print(classification_report(y_test, prediction, zero_division=0))
