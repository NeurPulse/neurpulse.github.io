# -*- coding: utf-8 -*-
# 数学建模：进化计算与群体智能
# Source: /notes/mathematical-modeling-evolutionary/
# Generated from the Markdown examples; edit the note and export again.

import numpy as np

class Objective:
    def __init__(self, limit=2000):
        self.limit, self.calls = limit, 0
        self.best, self.position = np.inf, None
    def __call__(self, x):
        if self.calls >= self.limit:
            raise RuntimeError("超出评估预算")
        self.calls += 1
        value = 100*(x[1]-x[0]**2)**2 + (1-x[0])**2
        if value < self.best:
            self.best, self.position = float(value), x.copy()
        return value

def genetic(seed):
    rng, f = np.random.default_rng(seed), Objective()
    population = rng.uniform(-2, 2, (20, 2))
    values = np.array([f(x) for x in population])
    while f.calls < f.limit:
        children = []
        for _ in range(20):
            i, j = rng.integers(0, 20, 2)
            father = population[i if values[i] < values[j] else j]
            mother = population[rng.integers(20)]
            ratio = rng.random(2)
            child = ratio*father+(1-ratio)*mother
            child += rng.normal(0, 0.15, 2)*(rng.random(2) < 0.3)
            children.append(np.clip(child, -2, 2))
        population = np.array(children)
        values = np.array([f(x) for x in population])
    return f

def particle_swarm(seed):
    rng, f = np.random.default_rng(seed), Objective()
    x = rng.uniform(-2, 2, (20, 2))
    v = rng.uniform(-0.2, 0.2, x.shape)
    personal = x.copy()
    personal_value = np.array([f(row) for row in x])
    while f.calls < f.limit:
        global_best = f.position.copy()
        v = (0.7*v + 1.5*rng.random(x.shape)*(personal-x)
             + 1.5*rng.random(x.shape)*(global_best-x))
        v = np.clip(v, -0.4, 0.4)
        x = np.clip(x+v, -2, 2)
        values = np.array([f(row) for row in x])
        improved = values < personal_value
        personal[improved] = x[improved]
        personal_value[improved] = values[improved]
    return f

def annealing(seed):
    rng, f = np.random.default_rng(seed), Objective()
    x = rng.uniform(-2, 2, 2)
    value = f(x)
    while f.calls < f.limit:
        proposal = x+rng.uniform(-0.35, 0.35, 2)
        if np.any(np.abs(proposal) > 2):
            continue  # 拒绝越界，且不消耗目标函数调用预算
        temperature = 2*(0.005/2)**(f.calls/(f.limit-1))
        new_value = f(proposal)
        difference = new_value-value
        if difference <= 0 or rng.random() < np.exp(-difference/temperature):
            x, value = proposal, new_value
    return f

for name, method in [("GA", genetic), ("PSO", particle_swarm), ("SA", annealing)]:
    runs = [method(seed) for seed in range(3)]
    print(name, "最好值:", [round(r.best, 6) for r in runs],
          "中位数:", np.median([r.best for r in runs]))
    assert all(r.calls == 2000 and r.best >= 0 for r in runs)
    assert all(np.all(np.abs(r.position) <= 2) for r in runs)

import numpy as np

rng = np.random.default_rng(42)
points = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
distance = np.linalg.norm(points[:, None]-points[None, :], axis=2)
pheromone = np.ones((4, 4))
best_length, best_route = np.inf, None
for generation in range(40):
    paths = []
    for ant in range(10):
        route = [int(rng.integers(4))]
        while len(route) < 4:
            allowed = [j for j in range(4) if j not in route]
            current = route[-1]
            weights = pheromone[current, allowed]/distance[current, allowed]**2
            route.append(int(rng.choice(allowed, p=weights/weights.sum())))
        route.append(route[0])
        length = sum(distance[a, b] for a, b in zip(route, route[1:]))
        paths.append((route, length))
        if length < best_length:
            best_length, best_route = length, route.copy()
    pheromone *= 0.8
    for route, length in paths:
        for a, b in zip(route, route[1:]):
            pheromone[a, b] += 1/length
            pheromone[b, a] += 1/length
print("最好路线与长度:", best_route, best_length)
assert len(set(best_route[:-1])) == 4 and best_route[0] == best_route[-1]
assert best_length >= 4-1e-10
