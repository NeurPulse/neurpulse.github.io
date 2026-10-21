# -*- coding: utf-8 -*-
# 数学建模：神经网络与反向传播
# Source: /notes/mathematical-modeling-ml-neural-network/
# Generated from the Markdown examples; edit the note and export again.

import numpy as np
from copy import deepcopy
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import log_loss, accuracy_score

x, y, w, b = 2., 1., 0.5, 0.
sigmoid = lambda z: 1/(1+np.exp(-z))
p = sigmoid(w*x+b)
old_loss = -np.log(p)
gradient_w, gradient_b = (p-y)*x, p-y
w -= 0.1*gradient_w
b -= 0.1*gradient_b
new_loss = -np.log(sigmoid(w*x+b))
print("更新后的w、b、损失:", w, b, new_loss)
assert new_loss < old_loss

X, labels = load_wine(return_X_y=True)
X_development, X_test, y_development, y_test = train_test_split(
    X, labels, test_size=0.3, stratify=labels, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_development, y_development, test_size=0.2,
    stratify=y_development, random_state=42)
scaler = StandardScaler().fit(X_train)
z_train, z_valid = scaler.transform(X_train), scaler.transform(X_valid)
network = MLPClassifier(
    hidden_layer_sizes=(16,), activation="relu", solver="adam",
    alpha=0.01, learning_rate_init=0.01, early_stopping=False, random_state=42)
classes = np.unique(y_train)
best_loss, stale, best_network, best_epoch = np.inf, 0, None, 0
for epoch in range(500):
    network.partial_fit(z_train, y_train, classes=classes)
    validation_loss = log_loss(y_valid, network.predict_proba(z_valid), labels=classes)
    if validation_loss < best_loss-1e-4:
        best_loss, stale = validation_loss, 0
        best_network, best_epoch = deepcopy(network), epoch+1
    else:
        stale += 1
    if stale >= 30:
        break
probability = best_network.predict_proba(scaler.transform(X_test))
prediction = best_network.classes_[probability.argmax(axis=1)]
baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
print("网络与基线测试准确率:", accuracy_score(y_test, prediction), baseline.score(X_test, y_test))
print("停止轮数、保存轮数:", epoch+1, best_epoch)
assert probability.shape == (len(X_test), 3)
assert np.allclose(probability.sum(axis=1), 1.)
