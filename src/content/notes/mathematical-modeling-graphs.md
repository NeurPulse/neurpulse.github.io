---
title: "数学建模：图论与网络优化"
description: "用节点和边建模，区分最短路、生成树、流与路线规划"
pubDate: 2026-10-21
updatedDate: 2026-10-21
order: 10
category: '数学建模'
tags: ["图论","最短路","网络优化","零基础"]
---

> [专题学习地图](/notes/mathematical-modeling-roadmap/) · 第 10 / 27 篇

## 学习目标与先修知识

目标是把地点、联系、道路和容量表示成图，区分最短路径、最小生成树、最大流、TSP 与 VRP，并手算一个小网络。先修为集合、距离与最小值。代码用 Python 标准库即可运行，便于看清算法与模型的对应。

可回看：[线性规划与整数优化](/notes/mathematical-modeling-optimization/)。

## 问题背景与使用时机

救援物资从仓库到站点，首先要知道道路连接关系及通行时间。几何直线距离可能最短，但并没有直达道路；把问题画成图能保留这种结构。

“每个地点都连接”“从 A 到 E 最快”“让所有地点形成闭合巡回”“单位时间运多少物资”是不同问题，目标不同，不能都套一个最短路函数。

## 核心概念与符号

图 $G=(V,E)$ 由节点集合 V 和边集合 E 构成。节点表示对象，边表示允许的关系；有向边区分方向，无向边允许双向。边权可表示时间、距离或成本；容量表示可通过的上限，两者不是同一个量。

| 模型 | 要做什么 | 必要区分 |
|---|---|---|
| BFS / DFS | 遍历可达节点 | 不等于带权最短路 |
| 最短路径 | 两点之间总边权最小 | 不要求访问其他节点 |
| 最小生成树 | 连通所有节点且总边权最小 | 不含环，不负责两两最快 |
| 最大流 | 在容量限制下从源到汇输送最多 | 中间节点满足流量守恒 |
| TSP | 每点访问一次并返回起点 | 与最短路的目标不同 |
| VRP | 多辆车完成配送 | 还可能有容量、时间窗和车辆数 |

Dijkstra 依次确定当前距离最小的未处理节点，再尝试经它到邻居是否更便宜。非负边权是关键条件；存在负权时不能直接沿用其保证。

生成树算法如 Kruskal 从短边开始，加入不会成环的边，直到连通。若图不连通，结果是生成森林，不存在覆盖全图的生成树。

## 手算一个最小例子

无向网络边为 AB=2、AC=5、BC=1、BD=4、CD=1、CE=5、DE=3、AE=6，单位为分钟。

![五节点配送网络及各边耗时](/images/math-modeling/learning/network-example.svg)

从 A 出发：初始 B=2、C=5、E=6；处理 B 后，C 改为 3，D 暂为 6；处理 C 后，D 改为 4；处理 D 后，经 D 到 E 需 7，比直达 6 差。因此到 E 最短路是 A→E，耗时 6。

最小生成树可取 BC、CD、AB、DE，总权 $1+1+2+3=7$。它连接全部 5 点，但沿这棵树从 A 到 E 要 7 分钟，说明“总连接成本最低”不等于“每条路都最快”。

另看流网络 S→A 容量3、S→B容量2、A→T容量2、A→B容量1、B→T容量3。可以发送两条直接流 2、2，再经 S→A→B→T 发送1，总流5；源点出边总容量也是5，故已经达到上界。

## 建模步骤与完整案例

完整案例：比较一个配送网络中的三种任务。

先记录节点和边的单位，确认道路双向且时间非负。第一项问单次急送到 E 的时间，解最短路；第二项问铺设低成本连通骨干，解生成树；第三项问一辆车依次检查全部站点并回仓库，解 TSP。

这些例子把时间同时用作连接代价，只是教学简化；实际铺设成本和行驶时间通常应建立不同边权字段。缺失边表示不允许直接走，不能填成 0。若允许途经节点重复，则巡回模型还需另外定义距离闭包和路线规则。

## Python 实现与结果解释

配套文件：[查看或下载本篇完整脚本](/downloads/math-modeling/mathematical-modeling-graphs.py)。

```python
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
```

最短巡回可为 A→B→C→D→E→A，总耗时 13；反向路线也可能同价。只在 5 点网络上枚举容易，节点增加后排列数量迅速增长，才需要整数规划、分支定界或启发式。

## 方法选择与常见误区

- 道路时长依出发时间变化时，静态边权只是近似，不能直接解释为实时导航。
- 边权可以是距离，也可以是费用；选错字段比选错算法更常见。
- 图论中的度是相连边的数量；度中心性描述连接多寡，不直接等于因果影响。
- PageRank 通过随机游走及跳转分配稳定分数，刻画特定链接规则下的访问倾向，不保证内容质量。
- 最大流需检查容量和守恒；最短路需检查连通与边权；VRP 需检查每单服务、载重和时间窗。

## 练习

1. 若只问从仓库到医院最快，是否需要 TSP？
2. 最小生成树有 n 个节点，通常有多少条边？
3. 上例关闭 AE 后，A 到 E 最快需多久？
4. 最大流例子中，若 S→B 容量降为1，总流最大还能是5吗？

## 参考答案

1. 不需要，首先是最短路径。若有道路时变、禁行等条件，再扩展模型。
2. 连通无向图的一棵生成树有 $n-1$ 条边。
3. A→B→C→D→E，耗时 $2+1+1+3=7$。
4. 不能，源点出容量上界变为 $3+1=4$；也可以构造4单位可行流，故新最大流为4。

## 复习卡片与下一步

建图先明确节点、方向、边权与容量，再按输出目标选算法。下一篇从“怎样连接”转向“怎样比较多个已有方案”。

[上一篇 线性规划与整数优化](/notes/mathematical-modeling-optimization/) · [下一篇 权重生成与综合评价](/notes/mathematical-modeling-evaluation/)

## 参考资料

- [Datawhale《数学建模导论》第4章](https://datawhalechina.github.io/intro-mathmodel/#/CH4/第4章-复杂网络与图论模型)：用于建立知识目录，建议与本篇的手算和案例对照阅读。
- [NetworkX 算法目录](https://networkx.org/documentation/stable/reference/algorithms/index.html)：更大网络可使用成熟实现。
