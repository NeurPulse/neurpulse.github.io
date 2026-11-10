---
title: "数据科学导引第七章：关联规则挖掘"
description: "用可手算的小例子理解支持度、置信度、提升度、Apriori 与 FP-Growth，完整推导候选项集、FP-tree 和条件模式基"
pubDate: 2026-11-10
updatedDate: 2026-11-10
category: '数据科学'
order: 8
tags: ['数据科学', '数据科学导引', '关联规则', '频繁项集', 'Apriori', 'FP-Growth', '数据挖掘', '读书笔记', '零基础']
---

> [《数据科学导引》学习地图](/notes/data-science-guide-roadmap/) · 第 7 / 15 章
>
> 上一章：[聚类模型](/notes/data-science-guide-clustering/)

本文对应《数据科学导引》第七章“关联规则挖掘”，原书第 150—167 页，即所提供扫描版 PDF 第 159—176 页。笔记保留支持度、置信度、Apriori 和 FP-Growth 四条主线，完整演算教材中的手机评论、候选项集与 FP-tree 示例，并补充提升度、结果验证和可运行的标准库代码。明确标出的“学习补充”与“实践补充”用于帮助理解，并非教材原文。

## 学习目标

学完本章，应能回答七个问题：

- 关联规则与分类规则有什么区别，为什么共现不等于因果？
- 支持度、置信度和提升度分别衡量什么，为什么高置信度仍可能没有正向关联？
- 支持度的反单调性怎样让 Apriori 安全地剪掉候选项集？
- Apriori 的连接、剪枝、计数和规则生成分别怎样进行？
- FP-tree 怎样压缩事务，条件模式基与条件 FP-tree 又分别是什么？
- 为什么 FP-Growth 避免了显式候选生成，却不代表挖掘过程没有成本？
- 面对评分、问卷或购物数据，怎样构造事务、筛选规则并检查结果是否可靠？

## 先看全局：从事务共现到可解释的关联规则

关联规则挖掘希望从大量事务中发现项目之间的共现模式。一个事务可以是一份订单、一位用户的一次会话，或经过明确整理的一条评论记录；事务中的项目可以是商品、行为、标签或离散化后的特征取值。

规则写成：

$$
A\Rightarrow B,
$$

其中 $A$ 和 $B$ 是非空项集，通常要求 $A\cap B=\varnothing$。它表示：在包含 $A$ 的事务中，$B$ 也常以某种程度出现。这里的箭头不是“导致”，也不是训练一个从 $A$ 预测固定类别 $B$ 的分类器。

完整流程可以压缩为：

1. **构造事务**：明确一条事务代表什么，并把原始数据转换为项目集合。
2. **挖掘频繁项集**：找出出现次数或支持度达到阈值的项集。
3. **生成候选规则**：把频繁项集拆成前件与后件，计算置信度等指标。
4. **筛选与验证**：结合提升度、冗余关系、时间稳定性和领域意义检查规则。

聚类关注“哪些样本相似”，关联规则关注“哪些项目经常共同出现”。两者通常都不依赖预先给定的类别标签，但任务、输入表示和输出含义不同。

| 方法 | 核心对象 | 主要思路 | 典型优势 | 主要限制 |
|---|---|---|---|---|
| Apriori | 逐层候选项集 | 利用反单调性连接、剪枝并扫描计数 | 原理清楚，容易实现和解释 | 候选数量与重复扫描成本可能很高 |
| FP-Growth | FP-tree 与条件模式基 | 压缩共享前缀，在条件树上递归挖掘 | 避免显式生成全部候选项集 | 树压缩依赖数据，递归与输出仍可能很大 |

## 7.1 关联规则的度量：常见、可靠与正向提升不是一回事

### 从事务数据库到项集

教材使用手机评论示例，五个项目可理解为评论中出现的五个特征或主题，五条事务为：

| 事务 | 项目集合 |
|---|---|
| $T_1$ | $\{X_1,X_2,X_3,X_4\}$ |
| $T_2$ | $\{X_2,X_5\}$ |
| $T_3$ | $\{X_2,X_3\}$ |
| $T_4$ | $\{X_1,X_3,X_4\}$ |
| $T_5$ | $\{X_3,X_4,X_5\}$ |

项集是项目的集合，因此同一项目在同一事务中重复出现，通常仍只计为一个事务。若有 $d$ 个不同项目，非空项集最多有：

$$
2^d-1
$$

个。即使单项数量不大，直接枚举所有项集也可能迅速变得昂贵。

连续数值或多值属性通常不能直接当作购物篮项目，需要先按任务定义离散化，再转换成“属性 = 区间”或 One-Hot 项目。例如“评分 = 高”与“评分 = 低”应是两个不同项目。离散化边界会改变支持度和最终规则，因此不是无关紧要的技术细节。

### 支持度：整个项集有多常见

项集 $A$ 的支持度是包含 $A$ 的事务所占比例。规则 $A\Rightarrow B$ 的支持度定义为前后件并集的支持度：

$$
S(A\Rightarrow B)=S(A\cup B)
=\frac{\text{包含 }A\cup B\text{ 的事务数}}
{\text{总事务数}}.
$$

支持度衡量规则覆盖了多少数据。支持度很低的规则即使置信度为 1，也可能只来自极少数事务，不能据此断言它稳定或重要。

手机数据中：

$$
S(X_2)=\frac35=0.6,\qquad
S(X_3)=\frac45=0.8,\qquad
S(X_2,X_3)=\frac25=0.4.
$$

这里 $\{X_2,X_3\}$ 出现在 $T_1,T_3$ 中，因此支持度计数为 2。

### 置信度：已知前件时后件有多常见

规则置信度定义为：

$$
C(A\Rightarrow B)
=\frac{S(A\cup B)}{S(A)}
=\frac{\text{包含 }A\cup B\text{ 的事务数}}
{\text{包含 }A\text{ 的事务数}}.
$$

它是在包含前件 $A$ 的事务中，同时也包含后件 $B$ 的比例。若最小支持度为 0.4、最小置信度为 0.6，同时达到两个阈值的规则称为**强关联规则**。

正向规则为：

$$
C(X_2\Rightarrow X_3)=\frac{0.4}{0.6}
=\frac23\approx0.6667,
$$

其支持度为 0.4，因此满足上述两个阈值。反向规则则是：

$$
C(X_3\Rightarrow X_2)=\frac{0.4}{0.8}=0.5,
$$

不满足最小置信度 0.6。两条规则支持度相同，置信度却不同，说明**箭头方向会影响条件比例**。

### 学习补充：提升度为什么还要比较后件自身的基础频率

置信度只回答“有 $A$ 时，$B$ 多常见”，没有单独比较“不管有没有 $A$，$B$ 本来有多常见”。提升度定义为：

$$
\operatorname{Lift}(A\Rightarrow B)
=\frac{C(A\Rightarrow B)}{S(B)}
=\frac{S(A\cup B)}{S(A)S(B)}.
$$

在事务独立性的参照下，可作如下解释：

- 提升度大于 1：$A$ 出现时，$B$ 的条件比例高于 $B$ 的总体比例，呈正向共现。
- 提升度等于 1：条件比例与总体比例相同，符合该独立性参照。
- 提升度小于 1：条件比例低于总体比例，呈负向共现。

手机示例中：

$$
\operatorname{Lift}(X_2\Rightarrow X_3)
=\frac{2/3}{0.8}
=\frac56\approx0.8333.
$$

这是一条很有提醒意义的规则：它的置信度约为 66.67%，已经超过 60% 的阈值，但 $X_3$ 本身就在 80% 的事务中出现；知道 $X_2$ 出现后，$X_3$ 的比例反而下降。**高置信度不自动等于正向关联，更不等于有业务价值。**

提升度对前后件交换是对称的：

$$
\operatorname{Lift}(A\Rightarrow B)
=\operatorname{Lift}(B\Rightarrow A),
$$

但置信度通常不对称。评价规则时，应同时报告支持度、置信度和提升度，而不是只挑一个看起来最高的数字。

### 从频繁项集生成规则

若项集 $I$ 频繁，对其每个非空真子集 $A\subset I$，令 $B=I\setminus A$，就得到候选规则 $A\Rightarrow B$。它的支持度等于 $S(I)$，再计算：

$$
C(A\Rightarrow I\setminus A)=\frac{S(I)}{S(A)}.
$$

因此，频繁项集挖掘和规则筛选是两个相连但不同的步骤。FP-Growth 输出的也是频繁项集，并不会因为算法名字不同就自动得到最终可用规则。

## 7.2 Apriori：用支持度反单调性逐层缩小搜索空间

### 反单调性为什么能安全剪枝

若 $A\subseteq B$，包含 $B$ 的事务一定也包含 $A$，所以：

$$
S(A)\ge S(B).
$$

这就是支持度的**反单调性**。由此得到两个等价而重要的结论：

1. 如果 $A$ 非频繁，那么它的任何超集都非频繁。
2. 如果某个项集频繁，那么它的所有非空子集都频繁。

因此，只要发现一个 $k-1$ 项子集不频繁，就可以剪掉包含它的 $k$ 项候选，而不必扫描事务计数。这是“安全剪枝”，不会因为被剪项集本身可能频繁而误删它。

### 连接、剪枝与计数

Apriori 按项集长度逐层推进。设 $L_{k-1}$ 为全部频繁 $k-1$ 项集，$C_k$ 为候选 $k$ 项集：

1. **连接**：由 $L_{k-1}$ 中的项集生成可能的 $k$ 项并集。若项目已排序，通常连接共享前 $k-2$ 项、只有最后一项不同的两个项集。
2. **子集剪枝**：候选的每个 $k-1$ 项子集都必须在 $L_{k-1}$ 中，否则删除该候选。
3. **扫描计数**：统计剩余候选在事务数据库中的出现次数。
4. **支持度过滤**：保留达到最小计数的候选，形成 $L_k$。
5. 无法生成下一层候选时停止，合并各层频繁项集。

连接条件只是高效生成候选的一种方式，核心正确性仍来自“所有子集必须频繁”。实现时不能只连接而不检查子集，否则可能保留本可提前排除的候选。

### 手算：手机数据的完整 Apriori 过程

教材这一处使用最小支持度 0.6，即五条事务中至少出现三次；前面讲规则时使用的是 0.4，两个示例阈值不要混淆。

第一层单项计数为：

| 项目 | 出现事务 | 计数 | 是否进入 $L_1$ |
|---|---|---|---:|---|
| $X_1$ | $T_1,T_4$ | 2 | 否 |
| $X_2$ | $T_1,T_2,T_3$ | 3 | 是 |
| $X_3$ | $T_1,T_3,T_4,T_5$ | 4 | 是 |
| $X_4$ | $T_1,T_4,T_5$ | 3 | 是 |
| $X_5$ | $T_2,T_5$ | 2 | 否 |

所以：

$$
L_1=\{\{X_2\},\{X_3\},\{X_4\}\}.
$$

由这三个频繁单项连接得到三个二项候选：

| 候选 | 出现事务 | 计数 | 是否进入 $L_2$ |
|---|---|---|---:|
| $\{X_2,X_3\}$ | $T_1,T_3$ | 2 | 否 |
| $\{X_2,X_4\}$ | $T_1$ | 1 | 否 |
| $\{X_3,X_4\}$ | $T_1,T_4,T_5$ | 3 | 是 |

于是：

$$
L_2=\{\{X_3,X_4\}\}.
$$

只有一个频繁二项集，无法再通过标准连接生成三项候选；即使考虑 $\{X_2,X_3,X_4\}$，它的子集 $\{X_2,X_3\}$ 和 $\{X_2,X_4\}$ 已非频繁，也必须剪掉。最终频繁项集只有：

$$
\{X_2\},\quad\{X_3\},\quad\{X_4\},\quad\{X_3,X_4\}.
$$

### 从 $\{X_3,X_4\}$ 生成两条规则

若最小置信度为 0.5：

$$
C(X_3\Rightarrow X_4)=\frac{3/5}{4/5}=0.75,
$$

$$
C(X_4\Rightarrow X_3)=\frac{3/5}{3/5}=1.
$$

两条规则都满足阈值，但含义仍不相同：前者针对含 $X_3$ 的事务，后者针对含 $X_4$ 的事务。它们的提升度相同：

$$
\operatorname{Lift}(X_3\Rightarrow X_4)
=\frac{0.75}{0.6}=1.25,
$$

$$
\operatorname{Lift}(X_4\Rightarrow X_3)
=\frac{1}{0.8}=1.25.
$$

这次提升度大于 1，说明两条方向都呈正向共现；但仍不能仅凭这个统计量推断一个主题导致另一个主题。

### Apriori 的成本在哪里

反单调性减少了需要检查的项集，却没有消除组合增长。教材指出，仅 1000 个频繁单项，就可能连接出：

$$
\binom{1000}{2}=\frac{1000\times999}{2}=499500
$$

个二项候选。若这些候选大量频繁，下一层还会继续增长；每一层又需要统计支持度。实际成本取决于数据稀疏程度、最小支持度、事务宽度、候选实现和频繁模式数量，不能只说“剪枝后一定很快”。

## 7.3 FP-Growth：把事务压缩成树，再按后缀递归挖掘

### 为什么不再显式生成候选项集

FP-Growth 的核心不是逐层列出 $C_k$，而是：

1. 第一次扫描原始事务，统计单项支持度并确定频繁项目顺序。
2. 第二次扫描原始事务，删除非频繁项目，按统一顺序排列，再插入 FP-tree。
3. 对目标项目收集条件模式基，即在树中通向该项目的前缀路径及其计数。
4. 从条件模式基构造条件 FP-tree，递归挖掘以该项目为后缀的频繁项集。

“避免显式候选生成”不等于“无需计数”或“无需递归”。它只是改变了组织事务与搜索频繁模式的方式。

### 第一次扫描：确定频繁项目与统一顺序

教材 FP-tree 示例的五条原始事务为：

| 事务编号 | 原始项目 |
|---:|---|
| 100 | $c,f,a,m,p$ |
| 200 | $c,f,a,b,m$ |
| 300 | $f,b$ |
| 400 | $c,b,p$ |
| 500 | $c,f,a,m,p$ |

最小支持度仍为 0.6，即至少出现三次。单项统计与排序为：

```text
c:4, f:4, a:3, b:3, m:3, p:3
```

同计数项目按教材顺序排列。这个统一顺序很重要：较常见项目靠近根节点，不同事务才更可能共享前缀。若各条事务使用不同顺序，同一组项目就无法稳定地复用路径。

### 第二次扫描：过滤、排序并插入路径

五条事务过滤非频繁项并按上述顺序整理为：

```text
100: c f a m p
200: c f a b m
300: f b
400: c b p
500: c f a m p
```

插入时，从根节点开始逐项走路径：

- 子节点已经存在，就把该节点计数加上当前事务的权重；普通未加权事务权重为 1。
- 子节点不存在，就创建节点并赋予相应计数。
- 共享前缀只增加计数，不同后缀产生新分支。

完整 FP-tree 为：

```text
null
├─ c:4
│  ├─ f:3
│  │  └─ a:3
│  │     ├─ m:2
│  │     │  └─ p:2
│  │     └─ b:1
│  │        └─ m:1
│  └─ b:1
│     └─ p:1
└─ f:1
   └─ b:1
```

例如，事务 100 和 500 完全相同，所以路径 $c\to f\to a\to m\to p$ 的各节点计数增加 2；事务 200 共享 $c\to f\to a$，随后在 $b$ 处分支。根节点另一支的 $f:1\to b:1$ 来自事务 300，不能漏掉。

FP-tree 还配有频繁项表（header table），记录各项目并链接树中同名节点，以便收集该项目的全部前缀。这些横向节点链接不是父子边；真正用于还原前缀的是树中从根到节点的路径。

### 条件模式基：只看通向目标后缀的前缀

要挖掘以某项目为后缀的频繁项集，先沿频繁项表找到它的所有树节点。每个节点贡献一条“从根到该节点、但不包含该节点自身”的前缀路径，路径权重等于该节点的计数。

教材主要后缀的条件模式基可整理为：

| 后缀 | 条件模式基：前缀路径及计数 |
|---|---|
| $f$ | $\{c:3\}$ |
| $a$ | $\{cf:3\}$ |
| $m$ | $\{cfa:2,\ cfab:1\}$ |
| $p$ | $\{cfam:2,\ cb:1\}$ |
| $b$ | $\{cfa:1,\ c:1,\ f:1\}$ |
| $c$ | 空前缀，不需要继续构造条件树 |

其中 $b$ 的第三条路径 $f:1$ 来自根节点另一分支。扫描版表 7.6 只列出 $\{cfa:1,c:1\}$，但按图 7.2 的完整 FP-tree，应补上 $f:1$；它不影响最小支持度为 3 时的最终频繁项集。

注意，条件模式基不是简单列出“与后缀同时出现过的所有项目”。它必须保留各条前缀路径及节点计数，才能在后续递归中还原更高阶组合。

### 后缀 $f$ 与 $a$

$f$ 的条件模式基只有 $c:3$，所以除自身 $f:4$ 外，还得到：

$$
cf:3.
$$

$a$ 的条件模式基只有 $cf:3$，除自身 $a:3$ 外，可沿单路径枚举所有非空前缀组合：

$$
ca:3,\qquad fa:3,\qquad cfa:3.
$$

这里没有 $b$ 参与，因为 $a$ 节点都位于 $c\to f\to a$ 路径上；处于其下方的 $b$ 是与 $a$ 共现的后缀，不属于通向 $a$ 的条件前缀。

### 后缀 $m$：汇总、过滤，再枚举单路径组合

$m$ 的条件模式基为：

$$
\{cfa:2,\ cfab:1\}.
$$

按路径权重汇总项目支持度：

| 条件项目 | 汇总计数 |
|---|---:|
| $c$ | $2+1=3$ |
| $f$ | $2+1=3$ |
| $a$ | $2+1=3$ |
| $b$ | $1$ |

删除非频繁的 $b$ 后，条件 FP-tree 只剩：

```text
c:3 → f:3 → a:3
```

在这条单路径上，每个非空子路径组合与 $m$ 拼接，支持度计数都为 3。因此除 $m:3$ 外，得到七个项集：

```text
cm:3, fm:3, am:3, cfm:3, cam:3, fam:3, cfam:3
```

这里不能把已删除的 $b$ 再加入 $bm$ 或 $abm$；它在 $m$ 的条件数据库中只出现一次，不满足最小支持度。

### 后缀 $p$：原始支持度不能代替条件支持度

$p$ 的条件模式基为：

$$
\{cfam:2,\ cb:1\}.
$$

汇总得到：

| 条件项目 | 汇总计数 |
|---|---:|
| $c$ | $2+1=3$ |
| $f$ | $2$ |
| $a$ | $2$ |
| $m$ | $2$ |
| $b$ | $1$ |

条件树只剩 $c:3$，所以得到：

$$
p:3,\qquad cp:3.
$$

虽然 $f,a,m$ 在原始数据库中各出现三次，但它们在“通向 $p$ 的条件数据库”中各只出现两次，因此不能生成 $fp,ap,mp$。这说明每次递归都要在**当前条件模式基**上重新计数，不能沿用原始单项支持度。

### 后缀 $b$ 与 $c$

$b$ 的完整条件模式基为 $\{cfa:1,c:1,f:1\}$。各条件项目汇总计数都小于 3，所以无法得到含其他项目的频繁二项集，只保留 $b:3$。

$c$ 位于所有相关路径的最前面，没有非空条件前缀，因此只保留 $c:4$。

### 汇总：18 个频繁项集

把各后缀及其条件递归结果合并，共得到 18 个频繁项集：

| 项数 | 频繁项集及计数 | 数量 |
|---|---|---:|
| 1 | $c:4,\ f:4,\ a:3,\ b:3,\ m:3,\ p:3$ | 6 |
| 2 | $cf:3,\ ca:3,\ fa:3,\ cm:3,\ fm:3,\ am:3,\ cp:3$ | 7 |
| 3 | $cfa:3,\ cfm:3,\ cam:3,\ fam:3$ | 4 |
| 4 | $cfam:3$ | 1 |

这些项集还需按“非空真子集作为前件”的方式生成规则，再计算置信度和其他筛选指标。频繁项集本身还不是 $A\Rightarrow B$ 形式的关联规则。

### FP-Growth 也不保证低成本

教材所说的两次扫描，是指对**原始事务数据库**的初始扫描。后续仍需遍历节点、构造条件模式基、建立条件树并递归挖掘。需要记住四个边界：

- **输出可能指数增长**：若大量项目共同频繁，满足条件的项集本身就可能有极多，任何精确算法都无法避免输出成本。
- **长单路径仍有组合枚举**：长度为 $k$ 的频繁单路径可对应 $2^k-1$ 个非空组合。
- **共享前缀少时压缩有限**：若事务之间很难复用路径，FP-tree 可能接近原始事务的存储规模。
- **递归条件树有额外成本**：原始数据只扫描两次，不代表整个算法只做两遍线性处理。

因此，比较 Apriori 与 FP-Growth 时，应结合实际数据、阈值、实现和输出规模，而不是把“无候选”理解成“无条件更快”。

## 实践补充：怎样把关联规则用于真实数据

### 先明确事务，而不是先运行算法

同一个原始表可以有多种事务定义，选择不同就会回答不同问题：

- 每行订单商品明细不能直接当成一个购物篮，通常要按订单编号聚合。
- 用户长期历史与单次会话是不同观察窗口，前者更容易混入时间跨度很大的行为。
- 每条评分明细不能自动当成独立购物篮；若研究用户共同喜欢的电影，应先按用户聚合，再定义什么评分算“喜欢”。
- 问卷数值和字符答案要转换为含义明确的项目，不能把缺失值静默填成普通类别。

还要决定是否合并同义项目、是否保留否定行为、怎样处理重复事件，以及时间窗口是否会让同一事务过度膨胀。

### 阈值与规则筛选

1. **先看支持度计数**：只有“0.4”还不够，要同时知道它对应多少条事务。
2. **联合看置信度与提升度**：避免把高基础频率的后件误当成强关联。
3. **检查冗余规则**：若 $A\Rightarrow B$ 与增加无关条件后的 $A\cup C\Rightarrow B$ 表达同样信息，后者未必值得单独报告。
4. **做时间或抽样稳定性检查**：在不同时间段、用户群或重采样中重新估计指标，避免只报告一次性共现。
5. **回到原始记录抽查**：确认规则不是编码错误、重复数据、缺失值处理或项目定义造成的假象。
6. **把规则当作行动假设**：若要做推荐、捆绑或干预，仍需通过后续实验或严谨因果设计验证效果。

最小支持度越低，通常保留的稀有模式越多，但计算量和偶然模式也会增加；最小支持度过高，又可能只留下人人都知道的常见项目。阈值应结合任务和成本做敏感性分析，而不是寻找一个通用固定值。

### 章末案例的数据意识

教材章末练习覆盖三类数据：

- **GroupLens 电影评分**：671 名用户、9125 部电影、100004 条评分。需要先确定用户事务与评分离散化规则。
- **青少年问卷**：1010 份问卷、150 个问题，含缺失值以及数值、字符混合特征。编码和缺失处理会直接改变项目含义。
- **商场购物数据**：9835 条事务，更接近标准购物篮形式，但仍需检查项目粒度、支持度阈值与规则解释。

这三类任务不能共用一套不加思考的“直接转项集”操作。数据表示清楚之后，Apriori 或 FP-Growth 的选择才有意义。

## 可运行示例：核对规则指标、Apriori 与 FP-tree 条件模式基

下面只用 Python 标准库。代码完成四件事：计算手机数据的正反向规则；运行一个教学用 Apriori；构造教材 FP-tree 并提取条件模式基；最后在小数据上用 Apriori 交叉核对 18 个频繁项集。最后的交叉核对只是结果验证，**不是用 Apriori 冒充 FP-Growth 实现**。

```python
from collections import Counter, defaultdict
from itertools import combinations
import math

# 通用辅助函数与教学用 Apriori
def support_count(itemset, transactions):
    itemset = frozenset(itemset)
    return sum(itemset <= transaction for transaction in transactions)

def support(itemset, transactions):
    return support_count(itemset, transactions) / len(transactions)

def confidence(antecedent, consequent, transactions):
    antecedent = frozenset(antecedent)
    union = antecedent | frozenset(consequent)
    return support(union, transactions) / support(antecedent, transactions)

def lift(antecedent, consequent, transactions):
    return confidence(antecedent, consequent, transactions) / support(
        consequent, transactions
    )

def apriori(transactions, min_count):
    transactions = [frozenset(transaction) for transaction in transactions]
    items = sorted(set().union(*transactions))
    current = {}
    for item in items:
        candidate = frozenset([item])
        count = support_count(candidate, transactions)
        if count >= min_count:
            current[candidate] = count
    frequent = dict(current)
    k = 2
    while current:
        previous = sorted(current, key=lambda itemset: tuple(sorted(itemset)))
        candidates = set()
        for index, left in enumerate(previous):
            for right in previous[index + 1:]:
                candidate = left | right
                if len(candidate) != k:
                    continue
                if all(
                    frozenset(subset) in current
                    for subset in combinations(candidate, k - 1)
                ):
                    candidates.add(candidate)
        current = {}
        for candidate in candidates:
            count = support_count(candidate, transactions)
            if count >= min_count:
                current[candidate] = count
        frequent.update(current)
        k += 1
    return frequent

# 1. 手机评论数据：支持度、置信度、提升度
phone_transactions = [
    {"X1", "X2", "X3", "X4"},
    {"X2", "X5"},
    {"X2", "X3"},
    {"X1", "X3", "X4"},
    {"X3", "X4", "X5"},
]
phone_transactions = [frozenset(transaction) for transaction in phone_transactions]
phone_support = support({"X2", "X3"}, phone_transactions)
forward_confidence = confidence({"X2"}, {"X3"}, phone_transactions)
reverse_confidence = confidence({"X3"}, {"X2"}, phone_transactions)
forward_lift = lift({"X2"}, {"X3"}, phone_transactions)
print("手机示例支持度、正反向置信度与提升度:")
print(phone_support, forward_confidence, reverse_confidence, forward_lift)
assert math.isclose(phone_support, 0.4)
assert math.isclose(forward_confidence, 2 / 3)
assert math.isclose(reverse_confidence, 0.5)
assert math.isclose(forward_lift, 5 / 6)

# 2. Apriori：最小支持度 0.6，即至少三次
phone_frequent = apriori(phone_transactions, min_count=3)
expected_phone_frequent = {
    frozenset({"X2"}): 3,
    frozenset({"X3"}): 4,
    frozenset({"X4"}): 3,
    frozenset({"X3", "X4"}): 3,
}
assert phone_frequent == expected_phone_frequent
print("手机数据频繁项集:")
print(sorted((tuple(sorted(items)), count) for items, count in phone_frequent.items()))

confidence_34 = confidence({"X3"}, {"X4"}, phone_transactions)
confidence_43 = confidence({"X4"}, {"X3"}, phone_transactions)
lift_34 = lift({"X3"}, {"X4"}, phone_transactions)
lift_43 = lift({"X4"}, {"X3"}, phone_transactions)
assert math.isclose(confidence_34, 0.75)
assert math.isclose(confidence_43, 1.0)
assert math.isclose(lift_34, 1.25)
assert math.isclose(lift_43, 1.25)
print("X3 与 X4 两条规则的置信度、提升度:")
print(confidence_34, confidence_43, lift_34, lift_43)
assert math.comb(1000, 2) == 499500

# 3. FP-tree：过滤、统一排序、插入五条路径
fp_transactions = [
    ["c", "f", "a", "m", "p"],
    ["c", "f", "a", "b", "m"],
    ["f", "b"],
    ["c", "b", "p"],
    ["c", "f", "a", "m", "p"],
]
single_counts = Counter(item for transaction in fp_transactions for item in transaction)
frequent_order = sorted(single_counts, key=lambda item: (-single_counts[item], item))
order_index = {item: index for index, item in enumerate(frequent_order)}
filtered_paths = [
    sorted(
        (item for item in transaction if single_counts[item] >= 3),
        key=order_index.__getitem__,
    )
    for transaction in fp_transactions
]
expected_paths = [
    ["c", "f", "a", "m", "p"],
    ["c", "f", "a", "b", "m"],
    ["f", "b"],
    ["c", "b", "p"],
    ["c", "f", "a", "m", "p"],
]
assert filtered_paths == expected_paths
print("FP-tree 的过滤排序路径:", filtered_paths)

class Node:
    def __init__(self, item):
        self.item = item
        self.count = 0
        self.children = {}

def insert_path(node, path):
    for item in path:
        if item not in node.children:
            node.children[item] = Node(item)
        node = node.children[item]
        node.count += 1

root = Node("null")
for path in filtered_paths:
    insert_path(root, path)

# 对每个同名节点，保存不包含后缀自身的前缀及节点计数
def collect_conditional_bases(root):
    bases = defaultdict(Counter)
    def visit(node, prefix):
        for item, child in node.children.items():
            if prefix:
                bases[item][tuple(prefix)] += child.count
            visit(child, prefix + [item])
    visit(root, [])
    return bases

def conditional_support(base):
    counts = Counter()
    for path, count in base.items():
        for item in path:
            counts[item] += count
    return counts

bases = collect_conditional_bases(root)
assert bases["m"] == Counter({("c", "f", "a"): 2, ("c", "f", "a", "b"): 1})
assert bases["p"] == Counter({("c", "f", "a", "m"): 2, ("c", "b"): 1})
assert bases["b"] == Counter({("c", "f", "a"): 1, ("c",): 1, ("f",): 1})
print("m 的条件模式基:", dict(bases["m"]))
print("p 的条件模式基:", dict(bases["p"]))
print("b 的完整条件模式基:", dict(bases["b"]))

m_conditional_counts = conditional_support(bases["m"])
p_conditional_counts = conditional_support(bases["p"])
assert m_conditional_counts == Counter({"c": 3, "f": 3, "a": 3, "b": 1})
assert p_conditional_counts == Counter({"c": 3, "f": 2, "a": 2, "m": 2, "b": 1})

# m 的条件树只剩 c:3 → f:3 → a:3，枚举其七个非空组合
m_suffix_frequent = {}
for size in range(1, 4):
    for prefix in combinations(("c", "f", "a"), size):
        m_suffix_frequent[frozenset(prefix + ("m",))] = 3
assert len(m_suffix_frequent) == 7

# 4. 小数据交叉核对：全部频繁项集共 18 个
fp_frequent_by_apriori = apriori(fp_transactions, min_count=3)
expected_fp_frequent = {
    frozenset({item}): count
    for item, count in (("c", 4), ("f", 4), ("a", 3), ("b", 3), ("m", 3), ("p", 3))
}
for itemset in (
    "cf", "ca", "fa", "cm", "fm", "am", "cp",
    "cfa", "cfm", "cam", "fam", "cfam",
):
    expected_fp_frequent[frozenset(itemset)] = 3
assert fp_frequent_by_apriori == expected_fp_frequent
assert len(fp_frequent_by_apriori) == 18
assert all(
    m_suffix_frequent[itemset] == fp_frequent_by_apriori[itemset]
    for itemset in m_suffix_frequent
)
print("条件支持度汇总:", m_conditional_counts, p_conditional_counts)
print("m 后缀的条件频繁项集数:", len(m_suffix_frequent))
print("FP 示例全部频繁项集数:", len(fp_frequent_by_apriori))
```

关键输出为：

```text
手机示例支持度、正反向置信度与提升度:
0.4 0.6666666666666667 0.5 0.8333333333333334
手机数据频繁项集:
[(('X2',), 3), (('X3',), 4), (('X3', 'X4'), 3), (('X4',), 3)]
X3 与 X4 两条规则的置信度、提升度:
0.7499999999999999 1.0 1.2499999999999998 1.25
b 的完整条件模式基: {('c', 'f', 'a'): 1, ('c',): 1, ('f',): 1}
m 后缀的条件频繁项集数: 7
FP 示例全部频繁项集数: 18
```

这段 Apriori 适合理解算法和核对小例子，不替代成熟实现的压缩事务表示、高效索引、并行计算和规则后处理。若把真实数据转换为数百万个 Python 集合，还需先评估内存与运行成本。

## 常见误区

1. **$A\Rightarrow B$ 表示 $A$ 导致 $B$。** 不对。它首先描述事务中的条件共现，因果结论需要额外设计与证据。
2. **关联规则就是分类模型中的 if-then 规则。** 不对。分类围绕预定标签学习决策边界，关联规则挖掘的是项目共现；输出形式和训练目标不同。
3. **规则支持度分别等于前件支持度乘后件支持度。** 不对。$S(A\Rightarrow B)=S(A\cup B)$；只有独立性参照下才有 $S(A\cup B)=S(A)S(B)$。
4. **交换前后件不会改变置信度。** 不对。两条方向共享并集支持度，但分母分别是 $S(A)$ 和 $S(B)$。
5. **置信度超过阈值就说明有正向关联。** 不对。后件本身可能非常常见；手机示例的正向规则置信度为 $2/3$，提升度却只有 $5/6$。
6. **支持度很低的规则一定没有价值。** 不一定。稀有模式可能有特定意义，但需要更多证据，不能把极少数事务的偶然共现直接当成稳定规律。
7. **非频繁项集的某个超集仍可能频繁。** 不对。根据支持度反单调性，超集支持度不可能更高。
8. **Apriori 只要连接频繁项集，不必检查候选的全部子集。** 不对。完整子集检查是重要剪枝步骤，也保证候选满足频繁项集的必要条件。
9. **频繁项集挖完后就已经得到关联规则。** 不完整。还要拆分前后件，计算置信度、提升度等并筛选规则。
10. **FP-tree 中同名节点的横向链接就是父子路径。** 不对。频繁项表链接用于定位节点，条件前缀仍由树中根到节点的路径确定。
11. **条件模式基中的项目沿用原始支持度即可。** 不对。例如 $f,a,m$ 原始支持度都是 3，但在 $p$ 的条件模式基中都只有 2，必须重新计数。
12. **FP-Growth 只扫描两次原始数据，所以总成本就是两遍线性扫描。** 不对。条件树构造、递归、长路径组合枚举和大量输出仍可能很昂贵。
13. **只要把每行原始数据转成集合，就能直接挖掘。** 不对。观察单位、聚合键、时间窗口、离散化与缺失值处理都会改变问题本身。
14. **提升度大于 1 且规则稳定，就可以直接采取干预并获得收益。** 不对。统计关联稳定仍不等于干预有效，行动价值还需领域分析和实验验证。

## 一句话总结

关联规则挖掘的核心，是在明确事务定义后用支持度发现频繁共现，再用置信度、提升度与稳定性筛选规则，**并始终区分统计关联、算法输出和可以采取行动的因果结论。**

## 复习卡片

- **支持度**：$S(A\Rightarrow B)=S(A\cup B)$，衡量规则覆盖多少事务。
- **置信度**：$C(A\Rightarrow B)=S(A\cup B)/S(A)$，方向通常不可交换。
- **提升度**：$C(A\Rightarrow B)/S(B)$；大于、等于、小于 1 分别对应独立参照下的正向、无、负向共现。
- **强规则**：同时达到最小支持度和最小置信度，但不自动有因果或业务价值。
- **反单调性**：非频繁项集的所有超集非频繁，频繁项集的所有非空子集频繁。
- **Apriori**：$L_{k-1}$ 连接生成 $C_k$ → 子集剪枝 → 扫描计数 → 支持度过滤。
- **手机 Apriori**：阈值 0.6 时仅 $\{X_2\},\{X_3\},\{X_4\},\{X_3,X_4\}$ 频繁。
- **FP-tree**：第一次统计单项，第二次过滤排序并插入；共享前缀累计节点计数。
- **条件模式基**：目标后缀各节点的前缀路径及其节点计数，条件支持度必须重新汇总。
- **$m$ 后缀**：条件树为 $c:3\to f:3\to a:3$，生成七个含 $m$ 的条件频繁组合。
- **$p$ 后缀**：条件树只剩 $c:3$，所以只有 $p$ 和 $cp$，没有 $fp,ap,mp$。
- **成本边界**：Apriori 有候选与扫描成本；FP-Growth 有树、递归、组合和输出成本。

## 自测问题

1. 手机数据中，$S(X_2,X_3)$、$C(X_2\Rightarrow X_3)$、$C(X_3\Rightarrow X_2)$ 和提升度分别是多少？为什么高置信度方向仍不是正向提升？
2. 若 $A\subseteq B$，为什么必有 $S(A)\ge S(B)$？这一关系怎样支持 Apriori 剪枝？
3. 在手机数据最小支持度为 0.6 时，$L_1$、三个二项候选计数和最终频繁项集分别是什么？1000 个频繁单项最多连接出多少个二项候选？
4. 对 $\{X_3,X_4\}$，两个方向的置信度与提升度各是多少？为什么提升度相同而置信度不同？
5. FP-Growth 的两次原始数据扫描分别完成什么？为什么所有事务必须使用统一项目顺序？
6. $m$ 的条件模式基是什么？过滤 $b$ 后得到怎样的条件树，能生成哪些含 $m$ 的频繁项集？
7. $p$ 的条件模式基中，$f,a,m$ 的原始支持度都是 3，为什么不能生成 $fp,ap,mp$？
8. 完整 FP-tree 中，$b$ 的条件模式基为什么应包含 $f:1$？补上它会不会在最小支持度 3 下产生含 $b$ 的频繁二项集？
9. 为什么不能把“FP-Growth 避免显式候选生成”理解为“总能在线性时间内输出全部频繁项集”？
10. 对 GroupLens 的“用户—电影—评分”明细，若希望发现共同高分电影，应怎样构造事务？得到稳定规则后为什么仍不能直接声称推荐干预有效？

## 参考答案

1. 支持度为 $2/5=0.4$，正反向置信度分别为 $2/3$ 和 $0.5$，提升度均为 $5/6$。$X_3$ 的总体支持度为 0.8，高于条件于 $X_2$ 时的 $2/3$，所以即使置信度超过 0.6，提升度仍小于 1。
2. 包含较大项集 $B$ 的事务必然包含其所有子集，所以包含 $B$ 的事务集合是包含 $A$ 的事务集合的子集。若 $A$ 非频繁，任何超集支持度也不可能达到阈值，可安全剪掉。
3. $L_1$ 为 $X_2,X_3,X_4$；$X_2X_3,X_2X_4,X_3X_4$ 的计数分别为 2、1、3，最终只有三个频繁单项和 $\{X_3,X_4\}$。1000 个单项可形成 $\binom{1000}{2}=499500$ 个二项候选。
4. $C(X_3\Rightarrow X_4)=0.75$，$C(X_4\Rightarrow X_3)=1$，两者提升度均为 1.25。置信度使用不同前件支持度作分母，提升度则都化简为 $S(X_3,X_4)/(S(X_3)S(X_4))$。
5. 第一次统计单项支持度并确定频繁项目顺序，第二次过滤非频繁项、统一排序并插入树。统一顺序让相同或相似前缀稳定复用同一路径，否则树会失去相应压缩效果。
6. 条件模式基为 $\{cfa:2,cfab:1\}$，汇总后 $c,f,a$ 各为 3、$b$ 为 1。过滤后只剩 $c:3\to f:3\to a:3$，得到 $cm,fm,am,cfm,cam,fam,cfam$，计数均为 3；另有 $m:3$ 自身。
7. 条件挖掘只使用通向 $p$ 的前缀。$p$ 的条件模式基为 $\{cfam:2,cb:1\}$，其中 $f,a,m$ 各只出现 2 次，小于阈值 3，只剩 $c:3$，因此只有 $cp$。
8. 事务 300 在根节点下形成独立的 $f:1\to b:1$ 分支，该 $b$ 节点贡献前缀 $f$、计数 1。补上后所有条件项目仍不足 3，所以只保留 $b:3$，不会产生含 $b$ 的频繁二项集。
9. 条件树构造和递归仍有成本；长单路径要枚举大量组合，共享前缀少时压缩有限，而且频繁项集输出本身可能指数增长。初始两次扫描不代表整个算法只有两次线性工作。
10. 先按用户聚合评分记录，并明确例如评分达到某阈值才把“电影 = 高分”加入该用户事务，再挖掘项集和规则。规则只反映观察数据中的共现，推荐干预是否改善点击、满意或留存，还需时间外验证、实验或其他因果评估。

## 下一章预告

下一章：[第八章：降维——PCA、LDA、MDS 与局部线性嵌入](/notes/data-science-guide-dimensionality-reduction/)。

第八章将讨论怎样用较低维表示保留数据中的主要信息，重点包括主成分分析、线性判别分析、多维尺度变换与局部线性嵌入。需要特别区分：降维通常构造新变量，特征选择则保留原始特征的子集；线性投影能保留什么，也取决于目标是否使用标签以及所优化的准则。

## 参考资料

- 欧高炎、朱占星、董彬等：《数据科学导引》第七章“关联规则挖掘”，高等教育出版社，原书第 150—167 页。
