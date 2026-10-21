# -*- coding: utf-8 -*-
# 数学建模：线性判别与支持向量机
# Source: /notes/mathematical-modeling-ml-lda-svm/
# Generated from the Markdown examples; edit the note and export again.

import numpy as np
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

direction = np.linalg.solve([[2., 0.], [0., 1.]], [2., 0.])
assert np.allclose(direction, [1., 0.])
hand = SVC(kernel="linear", C=1e4).fit([[0.], [1.], [3.], [4.]], [-1, -1, 1, 1])
print("手算SVM的w、b:", hand.coef_, hand.intercept_)
assert np.allclose(hand.coef_, [[1.]], atol=1e-3)
assert np.allclose(hand.intercept_, [-2.], atol=1e-3)

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)
lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
lda.fit(X_train, y_train)
search = GridSearchCV(
    make_pipeline(StandardScaler(), SVC(kernel="rbf")),
    {"svc__C": [0.1, 1., 10.], "svc__gamma": ["scale", 0.1, 1.]},
    cv=StratifiedKFold(5, shuffle=True, random_state=42)
)
search.fit(X_train, y_train)
print("SVM参数:", search.best_params_)
print("LDA测试准确率:", lda.score(X_test, y_test))
print("SVM测试准确率:", search.score(X_test, y_test))
