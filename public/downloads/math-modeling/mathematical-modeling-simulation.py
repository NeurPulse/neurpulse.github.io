# -*- coding: utf-8 -*-
# 数学建模：随机模拟与不确定性分析
# Source: /notes/mathematical-modeling-simulation/
# Generated from the Markdown examples; edit the note and export again.

import numpy as np

rng = np.random.default_rng(42)
samples = rng.uniform(size=20000)**2
estimate = samples.mean()
se = samples.std(ddof=1)/np.sqrt(len(samples))
print("积分估计、标准误、近似95%区间:", estimate, se,
      (estimate-1.96*se, estimate+1.96*se))
print("解析值:", 1/3)
assert np.isfinite(estimate) and se > 0

P = np.array([[0.8, 0.2], [0.3, 0.7]])
distribution = np.array([1., 0.])
assert np.allclose(distribution@P@P, [0.7, 0.3])
A = np.vstack([P.T-np.eye(2), np.ones(2)])
stationary = np.linalg.lstsq(A, [0., 0., 1.], rcond=None)[0]
assert np.allclose(stationary, [0.6, 0.4])
print("状态稳态分布:", stationary)

arrival_rate, service_rate = 2., 3.
arrival = np.cumsum(rng.exponential(1/arrival_rate, size=30000))
service = rng.exponential(1/service_rate, size=len(arrival))
finish = np.empty(len(arrival))
previous_finish = 0.
for i in range(len(arrival)):
    start = max(arrival[i], previous_finish)
    finish[i] = start+service[i]
    previous_finish = finish[i]
system_time = finish-arrival
waiting = system_time-service
print("预热后模拟平均停留:", system_time[1000:].mean())
print("理论平均停留:", 1/(service_rate-arrival_rate))
assert np.all(waiting >= -1e-10)
