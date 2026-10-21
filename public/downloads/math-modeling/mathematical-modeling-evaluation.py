# -*- coding: utf-8 -*-
# 数学建模：权重生成与综合评价
# Source: /notes/mathematical-modeling-evaluation/
# Generated from the Markdown examples; edit the note and export again.

import numpy as np

raw = np.array([[100., 8., 90.], [80., 6., 85.], [120., 9., 80.]])
benefit = np.array([False, True, True])
span = np.ptp(raw, axis=0)
assert np.all(span > 0)
positive = np.where(benefit,
                    (raw-raw.min(axis=0))/span,
                    (raw.max(axis=0)-raw)/span)

def topsis(z, weights):
    weighted = z*np.asarray(weights)
    best, worst = weighted.max(axis=0), weighted.min(axis=0)
    d_best = np.linalg.norm(weighted-best, axis=1)
    d_worst = np.linalg.norm(weighted-worst, axis=1)
    denominator = d_best+d_worst
    if np.any(denominator == 0):
        raise ValueError("没有区分能力，不能计算接近度")
    return d_worst/denominator

weights = np.array([0.5, 0.3, 0.2])
score = topsis(positive, weights)
print("正向化数据:", positive)
print("A/B/C接近度:", score, "排序:", np.argsort(-score))

# 熵权作为另一种权重来源，不能把它直接称为真实偏好。
probability = positive/positive.sum(axis=0)
terms = np.zeros_like(probability)
mask = probability > 0
terms[mask] = probability[mask]*np.log(probability[mask])
entropy = -terms.sum(axis=0)/np.log(len(raw))
diversity = 1-entropy
entropy_weights = diversity/diversity.sum()
print("熵权及其接近度:", entropy_weights, topsis(positive, entropy_weights))
assert np.isclose(entropy_weights.sum(), 1)

pairwise = weights[:, None]/weights[None, :]
eigenvalues = np.linalg.eigvals(pairwise)
lambda_max = eigenvalues.real.max()
CR = ((lambda_max-3)/2)/0.58
print("一致比较矩阵的CR:", CR)
assert abs(CR) < 1e-10

for cost_weight in [0.2, 0.5, 0.8]:
    w = np.array([cost_weight, (1-cost_weight)*0.6, (1-cost_weight)*0.4])
    print("价格权重:", cost_weight, "排名:", np.argsort(-topsis(positive, w)))
