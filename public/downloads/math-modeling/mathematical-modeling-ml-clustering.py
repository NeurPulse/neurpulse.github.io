# -*- coding: utf-8 -*-
# 数学建模：聚类算法
# Source: /notes/mathematical-modeling-ml-clustering/
# Generated from the Markdown examples; edit the note and export again.

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

small = np.array([[0.], [2.], [8.], [10.]])
hand = KMeans(n_clusters=2, init=np.array([[0.], [10.]]), n_init=1, random_state=42)
hand.fit(small)
assert np.allclose(np.sort(hand.cluster_centers_.ravel()), [1., 9.])
assert np.isclose(hand.inertia_, 4.)

X, _ = make_blobs(n_samples=180, centers=3, cluster_std=0.7, random_state=42)
z = StandardScaler().fit_transform(X)
scores = {}
for k in range(2, 6):
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(z)
    scores[k] = silhouette_score(z, labels)
print("各K轮廓系数:", scores)

density_labels = DBSCAN(eps=0.35, min_samples=5).fit_predict(z)
hierarchy_labels = AgglomerativeClustering(n_clusters=3, linkage="ward").fit_predict(z)
mixture = GaussianMixture(n_components=3, random_state=42).fit(z)
probability = mixture.predict_proba(z)
print("DBSCAN簇标签及数量:", np.unique(density_labels, return_counts=True))
print("层次聚类标签:", np.unique(hierarchy_labels))
print("第一个点的GMM归属概率:", probability[0])
assert np.allclose(probability.sum(axis=1), 1.)
