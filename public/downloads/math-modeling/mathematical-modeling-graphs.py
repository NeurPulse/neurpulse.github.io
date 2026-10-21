# -*- coding: utf-8 -*-
# 数学建模：图论与网络优化
# Source: /notes/mathematical-modeling-graphs/
# Generated from the Markdown examples; edit the note and export again.

from heapq import heappush, heappop
from itertools import permutations
from math import inf

edges = [
    ("A", "B", 2), ("A", "C", 5), ("B", "C", 1), ("B", "D", 4),
    ("C", "D", 1), ("C", "E", 5), ("D", "E", 3), ("A", "E", 6)
]
graph = {v: {} for v in "ABCDE"}
for a, b, w in edges:
    assert w >= 0
    graph[a][b] = graph[b][a] = w

distance = {v: inf for v in graph}
distance["A"] = 0
queue = [(0, "A")]
while queue:
    current, node = heappop(queue)
    if current != distance[node]:
        continue
    for neighbor, weight in graph[node].items():
        candidate = current + weight
        if candidate < distance[neighbor]:
            distance[neighbor] = candidate
            heappush(queue, (candidate, neighbor))
print("A到各点最短距离:", distance)
assert distance["E"] == 6

parent = {v: v for v in graph}
def find(v):
    while parent[v] != v:
        parent[v] = parent[parent[v]]
        v = parent[v]
    return v

tree = []
for a, b, weight in sorted(edges, key=lambda e: e[2]):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[ra] = rb
        tree.append((a, b, weight))
assert len(tree) == len(graph)-1
print("生成树及总权:", tree, sum(e[2] for e in tree))
assert sum(e[2] for e in tree) == 7

# 小图穷举巡回，缺失边记为不可行。
routes = []
for middle in permutations("BCDE"):
    route = ("A",) + middle + ("A",)
    cost = sum(graph[a].get(b, inf) for a, b in zip(route, route[1:]))
    routes.append((cost, route))
best = min(routes)
print("最短巡回:", best)
assert best[0] == 13
