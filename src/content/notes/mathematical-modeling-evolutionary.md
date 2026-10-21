---
title: "数学建模：进化计算与群体智能"
description: "理解 GA、PSO、ACO、SA 的更新与公平比较"
pubDate: 2026-08-07
updatedDate: 2026-10-21
order: 23
category: '数学建模'
tags: ["优化算法","遗传算法","粒子群算法","蚁群算法","模拟退火","零基础"]
---

> [专题学习地图](/notes/mathematical-modeling-roadmap/) · 第 23 / 27 篇

## 学习目标与先修知识

目标是解释 GA、PSO、ACO、SA 分别怎样产生新方案，手算一次更新，保存历史最好可行解，并按相同目标函数调用预算比较随机算法。先修为优化目标、可行域、概率和数组。先有清楚的目标与约束，再选择启发式。

可回看：[线性规划与整数优化](/notes/mathematical-modeling-optimization/)。

## 问题背景与使用时机

变量不可导、是排列、存在多个局部极值或求精确解代价过高时，随机搜索能提供候选方案。但若问题本来是规模可控的线性规划或凸优化，应先比较成熟的结构化方法。

“迭代曲线下降”通常只说明算法保存的最好值改善，不是全局最优证明。比较时需要统一问题、预算、停止条件与随机种子组。

## 核心概念与符号

### 四种机制

| 方法 | 保存什么 | 怎样探索 |
|---|---|---|
| GA 遗传算法 | 一组编码后的候选解 | 选择、交叉、变异 |
| PSO 粒子群 | 位置、速度、个人最好、群体最好 | 向经验位置移动 |
| ACO 蚁群 | 边上的信息素与每条已走路径 | 按信息素和启发式构造组合解 |
| SA 模拟退火 | 当前状态和历史最好状态 | 随温度接受部分变差提议 |

GA 编码的是决策变量或方案，不是把目标函数本身编码。实数、二进制和排列编码适用于不同约束，交叉后必须检查可行性。

PSO 的速度与位置为：

$$
v_i^{t+1}=\omega v_i^t+c_1r_1(p_i-x_i^t)+c_2r_2(g-x_i^t),\quad
x_i^{t+1}=x_i^t+v_i^{t+1}.
$$

其中 $p_i,g$ 是历史最好位置，必须独立保存，不能让数组引用随当前粒子一起改变。

TSP 蚁群在可行、尚未访问的邻居集合 $\mathcal A_i$ 上选下一点：

$$
P(i\to j)=\frac{\tau_{ij}^{\alpha}\eta_{ij}^{\beta}}
{\sum_{k\in\mathcal A_i}\tau_{ik}^{\alpha}\eta_{ik}^{\beta}},
\quad j\in\mathcal A_i,
\qquad \tau\leftarrow(1-\rho)\tau+\Delta\tau.
$$

其他点概率为0，常取 $\eta=1/d$。禁忌表记录已访问点，信息素矩阵记录边状态，两者不同。

SA 对称提议下，最小化时接受概率为 $\min(1,e^{-\Delta f/T})$。较优解直接接受；温度高时更愿意接受较差解，温度日程控制探索与收敛权衡。

## 手算一个最小例子

GA 两个二进制父代0011、1100，在中间交叉可产生0000、1111；再把0000最后一位翻转为0001。每一步改变的是候选编码，适应度还需解码后计算。

PSO 一维 $x=2,v=0.5,p=1,g=3$，取 $\omega=0.8,c_1=c_2=1,r_1=r_2=0.5$。新速度 $0.4-0.5+0.5=0.4$，新位置2.4。

ACO 两条可行边距离1、2，信息素相同，$\alpha=1,\beta=2$，权重为1、1/4，选近边概率 $1/1.25=0.8$。

SA 提议使目标变差2，T=1，接受概率 $e^{-2}\approx0.1353$。若抽到均匀随机数0.1则接受，抽到0.5则拒绝。接受较差解不意味着要把历史最好记录覆盖掉。

## 建模步骤与完整案例

完整案例比较 GA、PSO 和 SA 最小化二维 Rosenbrock 函数：

$$
f(x,y)=100(y-x^2)^2+(1-x)^2,\qquad (x,y)\in[-2,2]^2.
$$

解析上 $f\ge0$，且 (1,1) 取0，因此有独立的最优值参照。每个算法统一调用2000次目标函数，比较多个种子，并保存所有实际评估过的最好解。耗时仍受实现影响，不能仅凭迭代代数比较。

ACO 另用四点方形 TSP 演示标准边信息素机制，因为排列路线与连续位置不是同一个搜索空间；不把它与连续函数算法的数值结果混作一张速度排名。

## Python 实现与结果解释

配套文件：[查看或下载本篇完整脚本](/downloads/math-modeling/mathematical-modeling-evolutionary.py)。

### 同预算的连续优化

```python
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
```

中位数和最差种子能帮助判断稳定性。这里3个种子只是最小演示，正式比较应使用更多重复、预设统计和公平调参预算。

### 四点 TSP 的标准蚁群

```python
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
```

正方形沿边巡回长度为4，是可独立核查的最优参照。返回路线必须覆盖各点并回到起点，只有目标值小还不足以确认路线合法。

## 方法选择与常见误区

- GA 适合能设计合理编码和算子的离散或混合空间；排列问题不能随意用普通二进制交叉。
- PSO 常用于连续空间，对尺度、边界处理与参数敏感。
- ACO 常用于路线与组合构造，信息素正反馈也可能过早集中。
- SA 可用于具有邻域操作的问题，提议分布与降温日程影响很大。
- 约束不能只靠目标罚一个大数后就忽略；应核查最终方案是否可行。罚系数过小或过大会改变搜索行为。
- 每种方法都应保存历史最好与当前状态，二者可能不同；任何有限次随机搜索都不自动获得全局最优保证。

## 练习

1. 为什么用相同迭代次数比较 GA 和 SA 可能不公平？
2. PSO 的个人最好位置为什么要 copy？
3. SA 接受较差解后，历史最好值是否应变差？
4. ACO 下一节点概率为什么不能在全部节点上直接归一化？

## 参考答案

1. GA一代可能评估一整个种群，SA一次可能只评估一个提议，目标函数调用成本不同。
2. 若与当前位置共享可变数组，位置更新会把历史记录一起改掉，失去“过去最好”的意义。
3. 不应该。当前状态可以变差，历史最好应独立保存且不变差。
4. 已访问或不允许的节点必须排除，否则可能生成重复访问或非法路线。

## 复习卡片与下一步

复习卡：先检查问题结构，再统一预算；算子保证或修复可行性；保存历史最好；重复多个种子；用小规模精确解或理论下界核查。下一篇把随机性用于估计风险与不确定性。

[上一篇 神经网络与反向传播](/notes/mathematical-modeling-ml-neural-network/) · [下一篇 随机模拟与不确定性分析](/notes/mathematical-modeling-simulation/)

## 参考资料

- [Datawhale《数学建模导论》第5章](https://datawhalechina.github.io/intro-mathmodel/#/CH5/第五章-进化计算与群体智能)：用于建立知识目录，建议与本篇的手算和案例对照阅读。
- [SciPy 全局优化指南](https://docs.scipy.org/doc/scipy/tutorial/optimize.html#global-optimization)。
