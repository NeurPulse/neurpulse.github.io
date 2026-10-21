# -*- coding: utf-8 -*-
# 数学建模：主成分分析与降维
# Source: /notes/mathematical-modeling-pca/
# Generated from the Markdown examples; edit the note and export again.

import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

small = np.array([[-1., -1.], [0., 0.], [1., 1.]])
hand = PCA(n_components=1).fit(small)
reconstruction = hand.inverse_transform(hand.transform(small))
print("手算案例方差比:", hand.explained_variance_ratio_)
assert np.allclose(reconstruction, small)

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42)
pipeline = make_pipeline(
    StandardScaler(), PCA(n_components=0.95),
    LogisticRegression(max_iter=2000)
)
pipeline.fit(X_train, y_train)
pca = pipeline.named_steps["pca"]
print("原特征数、主成分数:", X.shape[1], pca.n_components_)
print("累计解释方差:", pca.explained_variance_ratio_.sum())
print("留出准确率:", pipeline.score(X_test, y_test))
assert pca.explained_variance_ratio_.sum() >= 0.95
assert pca.n_components_ <= X.shape[1]
