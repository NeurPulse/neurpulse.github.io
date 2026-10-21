# -*- coding: utf-8 -*-
# 数学建模：朴素贝叶斯
# Source: /notes/mathematical-modeling-ml-naive-bayes/
# Generated from the Markdown examples; edit the note and export again.

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report

toy = ["win free", "win deal", "meeting project", "meeting free"]
labels = [1, 1, 0, 0]
hand = make_pipeline(CountVectorizer(), MultinomialNB(alpha=1.0)).fit(toy, labels)
probability = hand.predict_proba(["win free"])[0]
print("正常、垃圾概率:", probability)
assert np.allclose(probability, [0.25, 0.75])

texts = [
    "win free prize", "claim cash prize", "win cash now", "free offer now",
    "cheap deal prize", "claim free cash", "win exclusive offer", "free bonus deal",
    "meeting project today", "project report attached", "team meeting tomorrow",
    "review project plan", "team lunch today", "report meeting notes",
    "please review report", "tomorrow project discussion"
]
y = np.array([1]*8 + [0]*8)
train, test, y_train, y_test = train_test_split(
    texts, y, test_size=0.25, stratify=y, random_state=42)
search = GridSearchCV(
    make_pipeline(CountVectorizer(), MultinomialNB()),
    {"multinomialnb__alpha": [0.5, 1., 2.]},
    cv=StratifiedKFold(3, shuffle=True, random_state=42),
    scoring="f1"
)
search.fit(train, y_train)
print("选择:", search.best_params_)
print(classification_report(y_test, search.predict(test), zero_division=0))
