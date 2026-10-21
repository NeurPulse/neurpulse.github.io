# -*- coding: utf-8 -*-
# 数学建模：统计模型与假设检验
# Source: /notes/mathematical-modeling-statistics/
# Generated from the Markdown examples; edit the note and export again.

import numpy as np
from scipy import stats

weight = np.array([8., 9., 10., 11., 12.])
n = len(weight)
se = stats.sem(weight)
result = stats.ttest_1samp(weight, popmean=9)
interval = stats.t.interval(0.95, n-1, loc=weight.mean(), scale=se)
print("均值、样本标准差、标准误:", weight.mean(), weight.std(ddof=1), se)
print("t与p:", result.statistic, result.pvalue)
print("95%双侧区间:", interval)
assert np.isclose(result.statistic, np.sqrt(2))
assert interval[0] < 9 < interval[1]

before = np.array([10., 12., 11., 14., 13.])
after = before + np.array([1., 2., 1., 3., 2.])
paired = stats.ttest_rel(after, before)
difference_test = stats.ttest_1samp(after-before, 0)
assert np.isclose(paired.statistic, difference_test.statistic)
print("配对变化:", paired)
print("两条独立生产线的Welch检验:",
      stats.ttest_ind(weight, [10., 11., 14., 13., 12., 15.], equal_var=False))
print("分类列联表:", stats.chi2_contingency([[20, 10], [10, 20]], correction=False))
