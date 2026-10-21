# -*- coding: utf-8 -*-
# 数学建模：图像文本与信号建模
# Source: /notes/mathematical-modeling-multimodal/
# Generated from the Markdown examples; edit the note and export again.

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

image = np.array([[0., 0., 0.], [0., 10., 10.], [0., 10., 10.]])
filtered = np.array([[image[i:i+2, j:j+2].mean() for j in range(2)] for i in range(2)])
assert np.allclose(filtered, [[2.5, 5.], [5., 10.]])
vectorizer = TfidfVectorizer()
vectorizer.fit(["red blue", "red green"])
print("词表与IDF:", vectorizer.get_feature_names_out(), vectorizer.idf_)

rng = np.random.default_rng(42)
fs = 128
time = np.arange(fs)/fs
features, labels = [], []
for i in range(80):
    label = i % 2
    picture = rng.normal(0.3, 0.08, (8, 8))
    picture[2:5, 2:5] += 0.2*label
    signal = (np.sin(2*np.pi*8*time) + 0.6*label*np.sin(2*np.pi*32*time)
              + rng.normal(0, 0.2, len(time)))
    text = "inspection noisy" if rng.random() < (0.2+0.6*label) else "inspection routine"
    power = np.abs(np.fft.rfft(signal))**2
    frequency = np.fft.rfftfreq(len(signal), d=1/fs)
    peak = frequency[np.argmax(power)]
    high_share = power[frequency >= 20].sum()/power.sum()
    features.append([picture.mean(), picture.std(), peak, high_share, float("noisy" in text)])
    labels.append(label)
X, y = np.array(features), np.array(labels)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)
print("设备级特征形状:", X.shape)
print(classification_report(y_test, model.predict(X_test), zero_division=0))
assert X.shape == (80, 5)
assert np.isfinite(X).all()
