# -*- coding: utf-8 -*-
# 数学建模：数据可视化与探索分析
# Source: /notes/mathematical-modeling-data-visualization/
# Generated from the Markdown examples; edit the note and export again.

import numpy as np
import matplotlib.pyplot as plt

hours = np.array([1., 2., 3., 4., 5., 6., 7., 8.])
scores = np.array([51., 55., 56., 60., 61., 65., 66., 60.])
slope, intercept = np.polyfit(hours, scores, 1)
fitted = slope*hours + intercept
residual = scores-fitted

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
axes[0].hist(hours, bins=[0.5, 2.5, 4.5, 6.5, 8.5], edgecolor="black")
axes[0].set(xlabel="practice hours", ylabel="count", title="sample distribution")
axes[1].scatter(hours, scores)
axes[1].plot(hours, fitted, color="tomato")
axes[1].set(xlabel="practice hours", ylabel="score", title="relationship")
axes[2].scatter(hours, residual)
axes[2].axhline(0, color="gray", linestyle="--")
axes[2].set(xlabel="practice hours", ylabel="observed - fitted", title="residuals")
plt.tight_layout()
plt.show()
print("最后一个样本残差:", residual[-1])
assert abs(residual.mean()) < 1e-10
