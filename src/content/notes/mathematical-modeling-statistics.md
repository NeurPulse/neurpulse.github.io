---
title: '数学建模：统计模型与假设检验'
description: '统计分布、假设检验、t检验与回归分析'
pubDate: 2026-08-07
category: '数学建模'
tags: ['统计','假设检验','t检验','回归','SciPy']
---

## 统计模型





### 统计分布与假设检验


#### **假设检验**

为何需要假设检验？

> 考虑这样一种情况：现在突然爆发了一种传染病，得了这种传染病的人会腹泻。医院里面有一批患者，医生将这群人分成了两组，一群人通过营养液补充体力和水分；一群人除了注射营养液以外还需要服用由中药成分A和成分B制成的胶囊，发现实验组患者的腹泻频率比对照组低。（表象统计得到数据）
>
> 那这个低是偶然导致的，还是两味药材在一起作用真的有用呢？如果有用，究竟是A在起作用，还是B在起作用，还是二者配方以后一同起作用呢

定义：

> 针对实现现象做出假设，然后根据样本的观测值去构建统计指标，分析指标的统计分布并计算接受概率，进而决定要不要接受假设。

流程
$$
\text{做出假设} \to  \text{计算检验统计量}\to  \text{根据分布计算概率值} \to \text{根据概率值对假设进行评价得到统计结论}
$$
注：

> 假设检验适用于小体量数据，目的探究现象背后某个猜想是否正确是否真实的统计的统计测度。以概率的形式对一个假设是否成立给予数量上的支持。



#### 一些经典的分布：

* 标准正态分布N(0,1)

* 若对n个服从标准正态分布的随机变量$(x_1,x_2,\cdots,x_n)$,定义卡方分布：
  $$
  \chi^2(n) = X^2_1 + X^2_2 + \cdots + X^2_n
  $$

* 若对X服从标准正态分布，$Y \sim \chi^2(n)$,定义t分布：
  $$
  t(n) = \frac{X}{\sqrt{Y/n}}
  $$

* 若$X1 \sim \chi^2(n_1) \quad ,\quad X2 \sim \chi^2(n_2)$,定义F分布：
  $$
  F(n_1,n_2) = \frac{X_1 / n_1}{X_2 / n_2}
  $$

详解：

1. 正态分布：

   正态分布是一种概率分布，其特征为钟形曲线，且曲线关于均值对称。在统计学中，许多随机变量都服从或近似服从正态分布，如人的身高、考试分数等。

   正态分布具有三个主要性质：

   * 集中性，即曲线的峰值位于均值处
   * 对称性，即曲线关于均值对称；
   * 均匀变动性，即正态分布曲线以均值为中心，向两侧均匀展开。

   正态分布的概率密度曲线解析式：
   $$
   f = \frac{1}{\sqrt{2\pi \sigma}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}
   $$


2. 卡方分布：

   假设有n个独立的随机变量$X_1,X_2,\cdots,X_n$​，每个随机变量都来自标准正态分布（均值为0，标准差为1），那么这n个随机变量的平方和就服从自由度为n的卡方分布。形如：
   $$
   \chi^2(n) = X^2_1 + X^2_2 + \cdots + X^2_n
   $$
   卡方分布主要性质：

   * 随机变量取值范围为非负实数
   * 随着自由度的增加，卡方分布趋近于正态分布
   * 卡方分布具有可加性，即若随机变量相互独立，则它们的平方和服从卡方分布

   常见统计量例如样本方差等都服从卡方分布。


3. t-分布：

   *t*-分布是由一个服从标准正态分布的随机变量*X*和一个服从自由度为*n*的卡方分布的随机变量*Y*组合而来的。它的表达式形如：
   $$
   t(n) = \frac{X}{\sqrt{\frac{Y}{n}}}
   $$
   t分布主要性质：

   * 随着自由度的增加，*t*分布趋近于正态分布
   * *t*分布具有可加性，即若随机变量相互独立，则它们的*t*值之和仍服从*t*分布
   * 对于不同的自由度，*t*分布的形状会发生变化，但总是关于其均值对称

   *t*分布在统计学中有着广泛的应用，尤其是在小样本数据分析、方差分析、回归分析等领域。

   由于*t*分布对样本大小和方差的变化较为稳健，因此在实践中常常用来进行假设检验和置信区间的计算。同时，*t*分布也是构建其他统计量的基础，如Z分布、F分布等。

   ![屏幕截图 2026-07-28 163827](C:/Users/lichi/Pictures/Screenshots/屏幕截图 2026-07-28 163827.png)

4.  F-分布：

   F分布是通过将两个正态分布的随机变量的比值进行标准化而得到的。具体来说，假设有两个正态分布的随机变量X和Y，它们的方差分别为$σ^2x$和$σ^2y$​，且X和Y相互独立，那么随机变量X²/Y²就服从自由度为m和n的F分布，其中m和n分别为该F分布的第一个和第二个自由度。
   $$
   F(n) = \frac{X_1 / n_1}{X_2 / n_2}
   $$
   F分布主要性质：

   * 随着自由度的增加，F分布趋近于正态分布
   * F分布具有可加性，即若两个随机变量相互独立，则它们的F值之和仍服从F分布
   * 对于不同的自由度，F分布的形状会发生变化，但总是关于其均值对称

   F分布在统计学中主要用于方差分析和回归分析等领域。

   在方差分析中，通过比较组间方差和组内方差，可以检验不同组之间的差异是否显著。

   在回归分析中，通过计算决定系数R²，可以评估模型对数据的拟合程度。

![屏幕截图 2026-07-28 163836](C:/Users/lichi/Pictures/Screenshots/屏幕截图 2026-07-28 163836.png)

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, t, f, norm

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ========== 卡方分布 ==========
x_chi = np.linspace(0, 30, 1000)
plt.figure(figsize=(10,4))
for df in [2,5,10,20]:
    plt.plot(x_chi, chi2.pdf(x_chi, df), label=f'df={df}')
plt.title("卡方分布 $\chi^2(df)$")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# ========== t分布 ==========
x_t = np.linspace(-5,5,1000)
plt.figure(figsize=(10,4))
plt.plot(x_t, norm.pdf(x_t), 'k--', label="标准正态 N(0,1)")
for df in [2,5,10]:
    plt.plot(x_t, t.pdf(x_t, df), label=f't(df={df})')
plt.title("t分布 $t(df)$")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# ========== F分布 ==========
x_f = np.linspace(0,6,1000)
plt.figure(figsize=(10,4))
plt.plot(x_f, f.pdf(x_f, 5,10), label='$F(5,10)$')
plt.plot(x_f, f.pdf(x_f,10,10), label='$F(10,10)$')
plt.plot(x_f, f.pdf(x_f,10,5), label='$F(10,5)$')
plt.title("F分布 $F(n_1,n_2)$")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# ====================== 正态分布 ======================
x_norm = np.linspace(-5, 5, 1000)
plt.figure(figsize=(10,4))
plt.plot(x_norm, norm.pdf(x_norm, loc=0, scale=1), label='$N(0,1)$ 标准正态', lw=2)
plt.plot(x_norm, norm.pdf(x_norm, loc=0, scale=2), label='$N(0,2^2)$', lw=2)
plt.plot(x_norm, norm.pdf(x_norm, loc=1, scale=1), label='$N(1,1^2)$', lw=2)
plt.title('正态分布 $N(\mu,\sigma^2)$ 概率密度曲线')
plt.xlabel('x')
plt.ylabel('$f(x)$')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```



#### 正态性检验：

> 正态性检验的目的是为了检测一组数据是否服从正态分布，是否表现出正态分布的特性（保证异常数据样本尽可能少）。
>
> 正态性检验的方法有很多，包括**QQ图、KS检验、SW检验、JB检验等**等。

* Shapiro-Wilk检验是一种常用的正态性检验方法。首先我们需要对原数据升序排序，然后构造统计量：
  $$
  W = \frac{(\sum ^n _{i=1}a_i x_i)^2}{\sum ^n _{i=1}(x_i - \bar x)^2}
  $$
  其中 a 表示显著水平，v为方差矩阵
  $$
  a = \begin{bmatrix}a_1 , a_2 , \cdots,a_n\end{bmatrix}^T = 
  \frac{\mu ^ T V^{-1}}{\sqrt{\mu^T (V^{-1})^T V^{-1} \mu}}
  $$
  注：

  > 统计量W最大值是1，最小值是$\frac{na^2_1}{n-1}$。可以把W看作是顺序排列样本值$y_i$和系数$a_i$​之间相关系数的平方或者是线性回归的确定性系数，他的值越高，越表示样本与正态分布匹配。之后为根据统计量W进行假设检验，但是W没有确定的分布形式。通过近似拟合来求解

  参考解释：

  > Shapiro-Wilk检验是一种用于验证数据集是否符合正态分布的统计方法。该方法通过计算样本数据的顺序统计量，并比较这些观察值与理论正态分布的期望值之间的差异来进行评估。Shapiro-Wilk检验的核心理念在于，它假设数据集遵循正态分布。
  >
  > 为了验证这一假设，该方法首先计算Shapiro-Wilk统计量W。统计量是一个衡量数据与正态分布拟合程度的指标，其基于实际观察值与理论正态分布期望值之间的差异。如果W值越接近1，则表明数据更符合正态分布。随后，Shapiro-Wilk统计量W与临界值进行比较。临界值是根据特定的显著性水平（通常为5%）和数据集的大小计算得出的。这一比较过程是判断数据是否服从正态分布的关键步骤。最终，根据统计量W与临界值的比较结果，可以得出结论。如果W值显著低于临界值，则可以拒绝零假设，这意味着数据不服从正态分布。相反，如果W值不低于临界值，则不能拒绝零假设，这表明数据可能服从正态分布。

  ```python
  import scipy.stats as st
  
  # 执行Shapiro-Wilk正态性检验
  
  statistic, p_value = st.shapiro(data)
  
  # 输出检验结果
  
  print("Shapiro-Wilk统计量:", statistic)
  
  print("p-value:", p_value)
  ```
  
* KS检验：

  首先定义一个经验分布概率 $F_n(x) = \frac{1}{n} \sum ^n _{i = 1} I(x_i < x)$。在进行K-S检验的时候我们首先会构造统计量$D_n = \underset{x}{sup}\vert F_n(x)-F(x) \vert$,并计算统计量。拒绝域由概率值或分布给出

  计算过程：

  * 计算实际均值方差，以及各水平对应的频数。
  * 对水平大小进行排序得到频数序列，并且计算累积频数
  * 计算累积频率，以及对应水平经过Z-score标准化以后的取值
  * 利用x查表得到理论分布F(x)。 $\to$​   计算统计量。 统计量越小越接近一个正态分布。
  
  ```python
  statistic_1, p_value_1 = st.kstest(data,'norm')
  
  # 输出检验结果
  
  print("K-S统计量:", statistic_1)
  
  print("p-value:", p_value_1)
  ```

* QQ图：

  QQ图是一种直观观察数据是否服从正态性的方法。QQ图可以用于检验一组数据是否服从某一分布，或者检验两个分布是否服从同一个分布。如果QQ图呈现出直线趋势，且数据点大致分布在直线的周围，则说明数据比较接近正态分布。如果数据点呈现出弯曲趋势或分散分布，则说明数据可能偏离正态分布。在画QQ图时，应注意数据的样本量大小、异常值情况等因素，这些因素可能会影响QQ图的准确度。需要注意的是，QQ图是一种直观的图形工具，可以辅助判断数据的正态性，但不能完全准确地判断数据的分布情况。

  ```python
  import numpy as np  
  
  import matplotlib.pyplot as plt  
  
  # 生成标准正态分布的数据  
  
  data = np.random.normal(0, 1, 1000)  
  
  #Python绘制QQ图的方法集成在statsmodels当中，通过如下方式调用：
  
  import statsmodels.api as sm
  
  import matplotlib.pyplot as plt
  
  # 创建 Q-Q 图，并增加 45度线
  # 点分布在直线附近，说明服从正态分布
  fig = sm.qqplot(data, line='45')
  
  plt.show()
  ```



* J-B检验（Jarque-Bera检验）：

  是一种用于检验数据是否服从正态分布的统计检验方法。它基于数据的偏度和峰度两个统计量，通过计算统计量的标准化值来判断数据是否符合正态分布。计算过程包括：首先，计算偏度S和峰度K以衡量数据分布的不对称性和尖锐程度。然后，根据这些值计算J-B统计量，它是偏度和峰度的标准化值之和。接下来，查找临界值表或使用软件计算临界值，将J-B统计量与临界值进行比较。如果J-B统计量大于临界值，则拒绝原假设（数据服从正态分布），认为数据不符合正态分布。如果J-B统计量小于临界值，则不能拒绝原假设，认为数据可能服从正态分布。需要注意的是，J-B检验是一种非参数检验方法，对数据分布的假设较少，因此在某些情况下可能比其他参数检验方法更为稳健。

  ```python
  statistic_2, p_value_2 = st.jarque_bera(data)
  
  # 输出检验结果
  
  print("J-B统计量:", statistic_2)
  
  print("p-value:", p_value_2)
  ```

#### **独立性检验**

> 抽不抽烟和得不得肺癌的关系。二者不能说存在因果关系，因果关系是指牛顿第二定律一样：物体有合外力就会产生一个加速度，力是产生加速度的原因。但一个人即使抽烟也可能永远不得肺癌，有的人抽烟抽到了一百多岁活得非常健康，有人不抽烟不喝酒却年纪轻轻就得了癌症。因此二者存在的是==相关关系而非因果关系==。
>
> 卡方独立性检验统计的是离散的相关关系，因为得不得肺癌只有两类离散取值：得或者不得，抽不抽烟也只有两类取值：抽或者不抽。两两组合就有四类人群。统计不同的人群可以列出一个列联表，构造的统计量也是一个服从==卡方分布==的统计量，查表可得结果
>

给出原假设为H0：抽烟和得肺癌是独立的。

现在在医院某科室里面调查发现，抽烟的患者有556人，其中得肺癌的有324个人；不抽烟的患者有260人，其中得肺癌的有98人。

|            | 抽烟者 | 不抽烟者 | 总计 |
| :--------- | :----- | :------- | ---- |
| 得肺癌者   | 324    | 98       | 422  |
| 不得肺癌者 | 232    | 162      | 394  |
| 总计       | 556    | 260      | 816  |

```python
import numpy as np  

from scipy.stats import chi2_contingency  

data=np.array([[324,98],[232,162]])

# 执行卡方独立性检验  

stat, p, dof, expected = chi2_contingency(data)   

# 输出结果  

print('卡方统计量:', stat)  

print('自由度:', dof)  

print('期望频数:', expected)   

print('p值:', p)
```



#### **两组样本的差异性检验**

> 两组样本的差异性检验可以通过*t*-检验实现。
>
> T-检验分为三种不同的类型：**单样本t检验**、**配对样本t检验**和**独立样本t检验**。其中，单样本t检验解决的是检验正态性的问题，这里主要讨论配对样本t检验和独立样本t检验。

* 配对样本t检验适合检验同一组样本在进行某一操作前后的状态差异。例如，想探究一笼健康的小白鼠在注射某神经亢奋药物前后的神经活跃性差异，这种情况就适合使用配对样本t检验。因为在注射药物前后，小白鼠始终是同一批小白鼠，没有新的老鼠混进来也没有老鼠逃走，它们只是需要被检测注射药物前后两种不同的状态。

* 独立样本t检验适合检验两组不同的样本在某一方面的表现差异。例如，想探究高一学生2000米跑成绩和高三学生2000米跑的成绩差异，这种情况就适合使用独立样本t检验。因为高一学生和高三学生是两批不同的人，它们的男女比例不同、年龄不同、平均身高体重不同……甚至连人数都是不一样的！

区别独立样本和配对样本一个最根本的特征：样本是同一批还是不同的两批。

最直观的特征就是两组样本的数量是否相同

差异性检验的原假设H0认为：**两类样本之间没有差异。**

```python
from scipy.stats import ttest_rel,ttest_ind

import numpy as np

# 假设有三组样本的数据  

data1 = np.random.normal(10,5,100)

data2 = np.random.normal(12,6,100)

data3 = np.random.normal(10,5,55)

# 执行配对样本t检验  

t_statistic_1, p_value_1 = ttest_rel(data1, data2)   

# 输出结果  

print('t统计量:', t_statistic_1)  

print('p值:', p_value_1)

# 执行独立样本t检验  

t_statistic_2, p_value_2 = ttest_ind(data1, data3)   

# 输出结果  

print('t统计量:', t_statistic_2)  

print('p值:', p_value_2)
```

注：

> t检验之前需要分析数据是否满足方差齐性，通过莱文检验实现

补：莱文检验（Levene's test）

> 是一种用于检验两组数据方差是否相等的统计检验方法。它的基本思想是比较两组数据的变异程度，如果两组数据的方差相等，那么它们的变异程度应该相似。如果两组数据的方差不相等，则它们的变异程度可能会有显著差异。

> 在进行t检验之前进行莱文检验的原因是，t检验的前提假设是两个样本的方差相等。如果这个假设不成立，t检验的结果可能会受到方差不等的影响，导致错误的结论。因此，在进行t检验之前，需要进行莱文检验来检验两个样本的方差是否相等。

```python
from scipy.stats import levene

# 执行莱文检验  

w, p = levene(data1, data2)  

# 输出结果  

print('W统计量:', w)  

print('p值:', p)
```

stats.levene()函数用于执行莱文检验。该函数返回两个值：W统计量和p值。W统计量越小，说明两组数据的方差越接近相等；p值越接近0，说明拒绝原假设（即两组数据的方差相等）的证据越强。通常情况下，如果p值小于设定的显著性水平（例如0.05），则认为两组数据的方差不相等，需要进一步分析或处理。

#### **方差分析**

如果出现多个要素对一件事起作用造成的差异，就使用方差分析

方差分析（ANOVA）可以用于两个样本及以上样本之间的比较，并可以用于分离各有关因素并估计其对总变异的作用，以及分析因素间的交互作用（也就是分析哪些因素有用）。方差分析可以用于均数差别的显著性检验、分离各有关因素并估计其对总变异的作用、分析因素间的交互作用和方差齐性检验等

方差分析的基本思想：**用于两个及以上的样本均数差别的显著性检验。通过分析研究不同变量的差异对总变异的贡献大小，确定控制变量对结果的影响力大小。**方差可以分解成三个部分：*Q*=*Q*1+*Q*2+*Q*3。其中，*Q*1是指多个控制变量单独作用引起的平方和，可以用来描述每个变量单独是否存在影响；*Q*2是指多个控制变量交互作用引起的离差平方和，可以用来描述变量之间是否存在协同效应或交互；*Q*3则是随机扰动，用于反映结果受随机影响的程度。

> 注：对于传统的方差分析而言，自变量是离散型的分类变量，目的是比较自变量的不同水平上因变量的均值差异。而对于回归分析而言，自变量是连续性的数值变量，用以估计自变量发生变动的时因变量的平均改变。对于一个一般的线性模型，我们把自变量的数据便准放宽，**方差分析和回归分析是可以统一的**

```python
import numpy as np  

from scipy.stats import f_oneway

# 创建数据  

np.random.seed(0)  # 设置随机种子以保证结果可复现  

group1 = np.random.normal(loc=5, scale=1, size=10)  # 只接受营养液  

group2 = np.random.normal(loc=4, scale=1, size=10)  # 接受营养液并服用成分A  

group3 = np.random.normal(loc=3, scale=1, size=10)  # 接受营养液并服用成分B  

group4 = np.random.normal(loc=2, scale=1, size=10)  # 接受营养液并服用成分A和B  

groups = [group1, group2, group3, group4]  

group_names = ['只接受营养液', '接受营养液并服用成分A', '接受营养液并服用成分B', '接受营养液并服用成分A和B']  

# 执行ANOVA  

F_stat, p_value = f_oneway(*groups)   

print('F统计量:', F_stat)  

print('p值:', p_value)
```

发现p值是小于0.05的，说明存在显著性差异。

但是究竟是怎样的显著性差异，作用机理是什么？A和B谁更有效？它们是否存在协同作用？

```python
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
# 记录是否服用A
a=[0]*10+[1]*10+[0]*10+[1]*10
# 记录是否服用B
b=[0]*20+[1]*20
groups=np.array(groups).flatten()
data={'A':a,'B':b,'groups':groups}
data=pd.DataFrame(data)


# 创建方差分析模型
model = ols('groups ~ A + B + A*B', data=data).fit()
# 分析方差分析模型
anova_results = sm.stats.anova_lm(model, typ=2)
print(anova_results)

```

#### **事后多重比较**

> 事后多重比较是基于方差分析进行的，用于分析定类数据和定量数据之间的关系情况。
>
> 例如研究人员想知道三组学生（本科以下，本科，本科以上）的智商平均值是否有显著差异。比如分析显示三组学生智商有着明显的差异，那具体是本科以下与本科这两组之间，还是本科以下与本科以上两组之间的差异；即具体两两组别之间的差异对比，则称为事后多重比较。常用的方法包括LSD、Turkey法等。通常会跟在方差检验后面分析合理
>
> 事后多重比较是指在方差分析之后，对各组之间的差异进行两两比较的方法。在方差分析中，我们只能判断各组均值是否存在显著差异，但无法确定具体是哪些组之间存在差异。通过事后多重比较，我们可以进一步确定哪些组之间的差异是显著的，从而更准确地了解数据之间的具体差异。

```python
import numpy as np  
from scipy.stats import f_oneway
from scipy.stats import tukey_hsd
# 创建数据  
group1 = np.random.normal(loc=5, scale=1, size=10)  # 只接受营养液  
group2 = np.random.normal(loc=4, scale=1, size=10)  # 接受营养液并服用成分A  
group3 = np.random.normal(loc=3, scale=1, size=10)  # 接受营养液并服用成分B  
group4 = np.random.normal(loc=2, scale=1, size=10)  # 接受营养液并服用成分A和B  
groups = [group1, group2, group3, group4]  
group_names = ['只接受营养液', '接受营养液并服用成分A', '接受营养液并服用成分B', '接受营养液并服用成分A和B']  
# 执行ANOVA  
F_stat, p_value = f_oneway(*groups)   
# TurkeyHSD法进行事后多重比较
# 进行事后多重比较  
mc_result = tukey_hsd(group1,group2,group3,group4)  
# 输出结果  
print(mc_result)
```







#### **相关系数**

> 相关系数的计算其实并不能称作一种检验，它的本质是针对两组连续值样本之间相关性做出计算和分析。但中学接触到的相关系数是有条件的，数据必须是正态或近似正态并且有一定程度的线性关系，不然不能用。相关系数其实常见的有三种：皮尔逊相关系数，斯皮尔曼相关系数和肯德尔相关系数。

* 皮尔逊相关系数:
  $$
  \rho = \frac{\text{Cov}(X,Y)}{\sqrt{D(X)} \sqrt{D(Y)}}
  $$

* 斯皮尔曼相关系数:不要求数据必须正态，是可以有偏的。在X和Y序列中得到每个元素的排名并作差得到新序列d
  $$
  \rho = 1 - \frac{6\sum_{i=1}^n d_i^2}{n(n^2-1)}  
  $$

> 一般认为相关系数大于0.7时就具备比较强的相关性了，0.9以上相关性非常强。但是否真的存在相关关系仍然可以通过假设检验的手法去证明。


### 回归

####  **从线性回归到拟合模型**

一元线性回归的最小二乘公式：对多元函数求极值
$$
\begin{cases}
y = wx + b \newline \\ 
w = \frac{ \sum ^n _{i = 1}x_iy_i - n \bar{x} \bar{y} } {\sum^n _{i = 1}x_i ^2 - n \bar{x}^2 } \newline\\
b = \bar{y} - w \bar{x}
\end{cases}
$$

均方误差——损失函数 ：实际值和预测值的偏差方差：
$$
J(w,b) = \frac{1}{n} \sum ^n _{t=1} (y_t - w x_i -b)^2
$$
为了拟合的效果最好，就需要让损失函数最小，这样就转化为多元函数求极值的问题了

因拟合的一系列的数据点都是已知的，所以此函数就是n个有关的二次式求和而来，是二元二次函数。求误差最小，因多元函数求极值，需要二者的偏导为0 ，公式如下：
$$
\begin{cases}
\frac{ \partial J }{ \partial w } = 0 \\ 
\frac{ \partial J }{ \partial b } = 0 \\
\end{cases}
$$

> 对b的偏导为0   $\to$   回归方程通过样本中心点。  将其带入对w的偏导，就有如下公式
> $$
> \begin{cases}
> \hat w = \frac{ \sum ^n _{i = 1} x_i y_i - n \bar x \bar y}{\sum ^n _{i = 1} x_i^2 - n \bar x ^2} \\
> \hat b = \bar y - \hat w \bar x \\
> \end{cases}
> $$
> 对多元线性回归，矩阵形式 $ y = W^TX $,用上述同样方法，写出均方误差（损失函数），对每个待估计参数分别求偏导   $\to$​  得到结果   。
> $$
> W = (X ^T X)^{-1}(X^Ty)
> $$
> 同样的方法也可以用于  指数拟合    对数拟合   三角拟合     
>
> 上述所述为 最小二乘法来历

``` python
import numpy as np

x = np.arange(-1.5,1.6,0.5)
y = [-4.45,-0.45,0.55,0.05,-0.44,0.54,4.55]
an = np.polyfit(x,y,3)   # 给出横坐标 纵坐标  3阶的式子(3次函数做拟合) 如果是1，就是1次函数做拟合  返回系数
print(an) # 系数
p1 = np.poly1d(an) # 将系数转化回函数
print(p1)
```

> 线性回归
>
> ```python
> X = np.arange(1,11,1)
> Y = np.array([1.1,2.5,3.6,4.9,6.2,9.0,9.5,11.0,15.6,14.1])
> p = np.polyfit(X,Y,1)
> print(p)
> ```
>
> 

#### Python 曲线拟合

> 指数函数与对数函数可以通过 对数  或  指数变换  $\to$  线性拟合  或  多项式拟合
>
> 除了 `numpy `  ,  `statsmodel`当中也提供了OLS方法可进行线性回归

```python
import pandas as pd  # 数据的基础处理
import numpy as np
import statsmodels.api as sm   # 实现类似二元中的统计模型，比如ols普通最小二乘法
import statsmodels.stats.api as sms # 实现统计工具 ，比如 t检验，F检验。。。
import statsmodels.formula.api as smf
import scipy

np.random.seed(991)  # 随机数种子
x1 = np.random.normal(0,0.4,100) # 生成符合正态分布的随机数（均值，标准差，所生成随机数的个数）
x2 = np.random.normal(0,0.6,100)
x3 = np.random.normal(0,0.2,100)
eps = np.random.normal(0,0.05,100) # 生成噪声数据，保证后面模拟所生成的因变量的数据比较接近实际的环境
X = np.c_[x1,x2,x3] # 调用c_函数来生成自变量的数据的矩阵，按照列进行生成的: 100 × 3 的矩阵
beta = [0.1,0.2,0.7] # 生成模拟数据时候的系数的值
y = np.dot(X,beta) + eps # 点积 + 噪声
X_model = sm.add_constant(X) # add_constant给矩阵加上了一列常量1，主要目的：便于估计多元线性回归模型的截距，便于后面进行参数估计时的计算 否则会从一次函数退化为正比例函数
model = sm.OLS(y,X_model) # 创建线性回归的对象 调用OLS普通最小二乘
results = model.fit() # fit拟合，主要功能就是进行参数估计，参数估计的主要目是估计出回归系数，根据参数估计结果来计算统计量，这些统计量主要的目的就是对我们模型的有效性 或者是 显著性水平来体现 
results.summary() # summary 方法主要是为了显示拟合的结果
```

```python
import statsmodels.api as sm  # statsmodels为统计而生  统计回归 
import numpy as np
x1 = np.random.normal(0,1,1000)
x2 = np.random.normal(0,1.5,1000)
x3 = np.random.normal(0,2,1000)
eps = np.random.normal(0,0.3,1000)
c = np.array([1,2,3])
x = np.c_[x1,x2,x3]
y = x.dot(c) + eps
# 理论上的方程应该是 y = x1 + 2x2 + 3x3
# 实际上的方程： y = w1x1 + w2x2 + w3x3
x_model = sm.add_constant(x)# 添加常数项
model = sm.OLS(y,x_model)
results = model.fit()
print(results.summary())
# 回归的效果 看 R^2的系数   越接近1  拟合效果越好
#             p的值越小越好
```



```python
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt  # 绘图

X = np.arange(1,11,1)
Y = np.array([1.1,2.5,3.6,4.9,6.2,9.0,9.5,11.0,15.6,14.1])
p = np.polyfit(X,Y,1)
print(p)

def Pfun(X,a,b):
    return 1/(a + b*X)
popt,pcov = curve_fit(Pfun,X,Y)# 可以通过定义自定义的一个曲线函数  可以输入自变量  还可以输入参数  传入函数名  横坐标  纵坐标  返回 参数估计值 popt 误差值(拟合效果程度) pcov
print(popt)

plt.plot(X,Y,'*',X,np.polyval(p,X),'r-')
plt.plot(X,Pfun(X,*popt),'b-')
plt.show()
```


### **scipy&statsmodels**

Scipy.stats模块是用于统计计算的模块。

```python
# 统计相关模块stats
sp.stats.norm.rvs()#标准正态分布
sp.stats.norm.fit()#估算正态分布的参数
sp.stats.norm.pdf()#计算对应位置的概率密度
sp.stats.norm.ppf()#找到标准正态分布中概率恰好为一半的点
sp.stats.expon()#指数分布
sp.stats.norm.cdf()#累积分布函数
sp.stats.norm.sf()#残存函数
sp.stats.norm.isf()#逆残存函数
sp.stats.t()#t分布
sp.stats.beta()#beta分布
sp.stats.gamma()#gamma分布
sp.stats.hypergeom()#超几何分布
sp.stats.lognorm()#对数正态分布
sp.stats.uniform()#均匀分布
sp.stats.chi2()#卡方分布
sp.stats.cauchy()#柯西分布
sp.stats.laplace()#拉普拉斯分布
sp.stats.rayleigh()#瑞利分布
sp.stats.randint()#离散均匀分布
sp.stats.f()#f分布
sp.stats.binom()#二项分布
sp.stats.poisson()#泊松分布
sp.stats.rv_continuous()#自定义连续分布
sp.stats.rv_discrete()#自定义离散分布
sp.stats.mode()#计算数据的众数
sp.stats.skew()#计算数据的偏度
sp.stats.kurtosis()#计算数据的峰度
sp.stats.ttest_rel()#配对样本t检验
sp.stats.ttest_ind()#独立样本t检验
sp.stats.ttest_1samp()#单样本t检验

```

```python
# 导入包
from scipy import stats
import matplotlib.pyplot as plt

# 1. 使用 scipy.stats 按照正态分布生成随机数
generated = stats.norm.rvs(size=900)

# 2. 用正态分布去拟合生成的数据，得到其均值和标准差
print("Mean", "Std", stats.norm.fit(generated))
# Mean Std (0.027757190138192445, 0.9967555892878)

# 3. 偏度（skewness）描述的是概率分布的偏斜（非对称）程度。我们来做一个偏度检验。
# 该检验有两个返回值，其中第二个返回值为p-value，即观察到的数据服从正态分布的概率，取值范围为0~1。
print("Skewest", "pvalue", stats.skewtest(generated))
# Skewtest Result(skewstatistic=-0.221280533102111, pvalue=0.827306519288437)
# 因此，该数据有很大的概率服从正态分布

# 4. 峰度（kurtosis）描述的是概率分布曲线的陡峭程度。我们来做一个峰度检验。
# 偏度检验类似，当然这里是针对峰度。
print("Kurtosistest", "pvalue", stats.kurtosistest(generated))
# Kurtosistest Result(kurtstatistic=0.2421563044089, pvalue=0.520733173445296)

# 5. 正态性检验（normality test）可以检查数据服从正态分布的程度。我们采取一个正态性检验。
# 该检验同样有两个返回值，其中第二个返回值为p-value。
print("Normaltest", "pvalue", stats.normaltest(generated))
# Normaltest pvalue NormaltestResult(statistic=0.59112406146223, pvalue=0.741129374456706)

# 6. 使用scipy我们可以很方便地得到数据所在的区段中某一百分比处的数值
print("95 percentile", stats.scoreatpercentile(generated, 95)) # 95 percentile 1.641949784025992

# 7. 将前一步反过来，我们也可以从数值1出发找到对应的百分比
print("Percentile at 1", stats.percentileofscore(generated, 1)) # Percentile at 1 84.44444444444444

# 8. 使用matplotlib绘制生成数据的分布直方图
plt.hist(generated)
plt.show()
```

```python
from scipy import stats
import numpy as np

np.random.seed(12345678)
rv1 = stats.norm.rvs(loc=-5, scale=10, size=500)
rv2 = stats.norm.rvs(loc=0, scale=10, size=500)
print(stats.ttest_ind(rv1, rv2))

# 两独立样本t检验
# 用于比较两组数据是否来自同一正态分布的总体。
# 注意：如果要比较的两组数据不满足方差齐性，需要在ttest_ind()函数中添加参数equal_var = False
# Ttest_indResult(statistic=-1.302244006355476, pvalue=0.1931343989106407)
```





```python
from scipy import stats

np.random.seed(12345678)
rv1 = stats.norm.rvs(loc=-5, scale=10, size=500)
rv2 = stats.norm.rvs(loc=0, scale=10, size=500)
print(stats.ttest_rel(rv1, rv2))
# 配对样本t检验可视为单样本t检验的扩展，检验的对象由一群来自正态分布独立样本更改二配对样本观测值之差。
# 它常用于比较同一受试对象处理前后，或者按照某一条件进行两两配对分别给与不同处理的受试对象之间是否存在差异。
# 运行结果：Ttest_relResult(statistic=-0.2410176496530079, pvalue=0.8096434435811551)
```



```python
from scipy import stats
import numpy as np
import pandas as pd

np.random.seed(12)
rvs = stats.norm.rvs(loc=5, scale=2, size=100)
# rvs_list = list(rvs)
# data = pd.DataFrame(rvs_list,columns=['数字'])
# data.to_excel('data.xlsx')
# scipy.stats.ttest_lsamp(a,popmean,axis = 0,nan_policy='propagate',alternative = 'two-sided')
# 单样本t检验，用于检验数据是否来自一致均值的总体，t检验主要是以均值为核心的检验
# 这是对独立观测样本t的期望(均值)等于给定总体均值 popmean 的零假设的检验。
print(stats.ttest_1samp(rvs, popmean=5))

```



```python
from scipy import stats
import numpy as np

np.random.seed(12345678)
rv1 = stats.norm.rvs(loc=-5, scale=10, size=500)
rv2 = stats.norm.rvs(loc=25, scale=9, size=500)
# rv3 = stats.norm.rvs(loc=10,scale=1,size=10) 生成10个服从正态分布的随机数
# 其主要用法为参数loc默认为0, scale默认为1, 不用管它们，下从stats.norm.pdf等
# stats.norm.pdf(x)，计算概率密度函数
# stats.norm.cdf(x)，计算累计密度函数
# stats.norm.ppf(q)，计算z值
# stats.describe(rv1).stddev()/stats.norm.mean()/stats.norm.median()，计算分布的方差/标准差/均值/中位数
# 方差齐性检验用以检验两组或多组数据离散程度是否存在差异，也是很多检验算法的先决条件
print(stats.levene(rv1, rv2))
```



```python
from scipy import stats
import numpy as np

np.random.seed(12345678)
x = stats.norm.rvs(loc=0, scale=1, size=300)
#科尔莫戈罗夫检验(Kolmogorov-Smirnov test)，检验样本数据是否服从某一分布，仅适用于连续分布的检验
print(stats.kstest(x, 'norm'))
```



```python
#### 代码示例：导入包
import numpy as np
from scipy import stats

#### 代码示例：定义随机分布
#Sample Size
N = 10
#Gaussian distributed data with mean = 2 and var = 1
a = np.random.randn(N) + 2
#Gaussian distributed data with with mean = 0 and var = 1
b = np.random.randn(N)

#### 代码示例：计算标准差
#Calculate the variance to get the standard deviation

#For unbiased max likelihood estimate we have to divide the var by N-1, and therefore the parameter ddof = 1
var_a = a.var(ddof=1)
var_b = b.var(ddof=1)

#std deviation
s = np.sqrt((var_a + var_b)/2)

#### 代码示例：计算t统计量
t = (a.mean() - b.mean())/(s*np.sqrt(2/N))

#### 代码示例：比较临界t值
#Degrees of freedom
df = 2*N - 2

#p-value after comparison with the t
p = 1 - stats.t.cdf(t,df=df)

print("t = " + str(t))
print("p = " + str(2*p))
#Note that we multiply the p value by 2 because its a two tail t-test
### 结果分析

#### 代码示例：用SciPy交叉验证
t2, p2 = stats.ttest_ind(a,b)
print("t = " + str(t2))
print("p = " + str(2*p2))

# t分数是两个组之间的差值与组内差的比值。
# t分数越大，组间的差异越大。
# t分数越小，组间的相似度就越大。t分数为3代表这些组是彼此之间的三倍。
# t检验,比较两个平均值（均值），然后告诉你它们彼此是否有差异。并且，t检验还会告诉你这个差异有没有意义，换句话说，它让你知道这些差异是否可能是偶然发生的。

```



```python
import pandas as pd
from sklearn.datasets import load_iris
import numpy as np
from scipy import stats

iris = load_iris()
# print(iris)
data = pd.DataFrame(iris.data, columns=['sepal_length','sepal_width','petal_length','petal_width'])
# data.to_csv('iris.csv')

#计算样本均值、标准差
mean = data['petal_length'].mean()
std = data['petal_length'].std()
print('样本均值：',mean)
print('样本标准差：',std)

#计算t统计量
t = (mean - 3.5)/(std/np.sqrt(len(data['petal_length'])))
print('t统计量：', t)

#计算p值
#df：自由度，即变量可以自由取值的个数
p = 2*stats.t.sf(abs(t), df=len(data['petal_length'])-1)
print('P-Value值：', p)
```



```python
import numpy as np
from scipy.stats import f

# 大数定律：随着样本量的增大，样本均值逐渐趋于总体均值。
# 中心极限定理：样本可能不是正态分布，但是样本均值都是呈正态分布。
# F分布
# 分析两个正态分布方差比值的分布情况，F分布是由两个独立的卡方分布的比值所得到的分布
# 检验两个样本方差是否相等。常被用于统计分析、财务分析、市场分析、医学研究等领域
# 案例
# 假设有两个班级，每个班级各有30名学生，两个班级的数学成绩分别为：
# 班级1：{70, 71, 68, 75, 72, 69, 74, 73, 71, 70, 75, 72, 69, 68, 72, 70, 71, 73, 72, 74, 73, 72, 70, 69, 71, 73, 75, 74, 73, 72, 75}
# 班级2：{71, 72, 69, 73, 74, 70, 68, 72, 75, 76, 74, 73, 72, 75, 73, 70, 69, 71, 70, 72, 73, 74, 73, 75, 72, 71, 70, 74, 73, 75, 73}
# 现在需要检验这两个班级的数学成绩方差是否相等。

class1 = [70, 71, 68, 75, 72, 69, 74, 73, 71, 70, 75, 72, 69, 68, 72, 70, 71, 73, 72, 74, 73, 72, 70, 69, 71, 73, 75, 74, 73, 72, 75]
class2 = [71, 72, 69, 73, 74, 70, 68, 72, 75, 76, 74, 73, 72, 75, 73, 70, 69, 71, 70, 72, 73, 74, 73, 75, 72, 71, 70, 74, 73, 75, 73]

s1 = np.var(class1, ddof=1)
s2 = np.var(class2, ddof=1)

print("s1: ", s1)
print("s2: ", s2)

F = s1 / s2
print("F值: ", F)

df1 = len(class1) - 1
df2 = len(class2) - 1
p_value = f.sf(F, df1, df2)

print("P值: ", p_value)
```



```python
from numpy import array, sqrt
from scipy.stats import t

a = array([506, 508, 499, 503, 504, 510, 497, 512, 514, 505, 493, 496, 506, 502, 509, 496])

# numpy.std() 求标准差的时候默认是除以 n 的，即是有偏的，np.std无偏样本标准差方式为 ddof = 1；
# pandas.std() 默认是除以n-1 的，即是无偏的，如果想和numpy.std() 一样有偏，需要加上参数ddof=0 ，即pandas.std(ddof=0)
alpha = 0.05
n = len(a)
mu = a.mean()
s = a.std(ddof=1)  #计算均值和标准差
print(mu, s)

val = (mu - s/sqrt(n)*t.ppf(1-alpha/2, n-1), mu + s/sqrt(n)*t.ppf(1-alpha/2, n-1))
print("置信区间为：", val)
```



```python
import pandas as pd
import numpy as np
from scipy import stats

def confidence_interval_u(data, sigma=-1, alpha=0.05, side_both=True):
    xb = np.mean(data)
    s = np.std(data, ddof=1)
    if sigma > 0:
        # sigma已知，Z分布（正态分布）
        Z = stats.norm(loc=0, scale=1.)
        if side_both:
            # 双侧置信区间
            tmp = sigma / np.sqrt(len(data)) * Z.ppf(1 - alpha / 2)
            return (xb - tmp, xb + tmp)
        else:
            # 单侧
            tmp = sigma / np.sqrt(len(data)) * Z.ppf(1 - alpha)
            return {'bottom_limit': xb - tmp, 'top_limit': xb + tmp}
    else:
        # sigma未知，t分布
        T = stats.t(df=len(data)-1)
        if side_both:
            tmp = s / np.sqrt(len(data)) * T.ppf(1 - alpha / 2)
            return (xb - tmp, xb + tmp)
        else:
            tmp = s / np.sqrt(len(data)) * T.ppf(1 - alpha)
            return {'bottom_limit': xb - tmp, 'top_limit': xb + tmp}

data = np.array([101.3, 96.6, 100.4, 98.8, 94.6, 103.1, 102.3, 97.5, 105.4, 100.2])
res = confidence_interval_u(data, 3)
print(res)
```



```python\
import numpy as np
import scipy.stats as st

# data of goals scored by 20 footballers in a calendar year
fb_data = [10, 11, 10, 14, 16, 24, 10, 6, 8, 10,
           11, 27, 28, 21, 13, 10, 6, 7, 8, 10]

# 置信水平 create 90% confidence interval
confidence_level = 0.90

# t.interval() 计算置信区间 （n<30，总体方差未知，使用t分布）
print("置信区间为：", st.t.interval(confidence_level, df=len(fb_data)-1,
                                   loc=np.mean(fb_data),
                                   scale=st.sem(fb_data)))

# 标准误用来衡量抽样误差，是统计推断可靠性的指标。
# scipy.stats.sem(a, axis=0, ddof=1, nan_policy='propagate')
# axis:默认为 0。如果没有，则计算整个数组a
# ddof:Delta 自由度。
```



```python
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# norm.rvs：生成正态分布随机样本
# loc=0 均值，scale=1标准差，size=400生成400个样本，标准正态分布N(0,1)
data = norm.rvs(loc=0, scale=1, size=400)
mu1 = np.mean(data)       # 直接计算样本均值
std1 = np.std(data)       # 直接计算样本标准差（ddof=0，除以n，有偏）
print("样本均值mu1：", mu1)
print("样本标准差std1：", std1)

# norm.fit()：极大似然估计，根据样本拟合正态分布，返回估计均值、标准差
mu, std = norm.fit(data)

# 绘制样本直方图
# bins=25 分25个区间；density=True 转为概率密度；alpha透明度；color绿色
plt.hist(data, bins=25, density=True, alpha=0.6, color='g')

# 获取直方图当前x轴左右边界
xmin, xmax = plt.xlim()
# 在x区间生成100个均匀坐标点，用于绘制连续概率密度曲线
x = np.linspace(xmin, xmax, 100)

# norm.pdf：正态分布概率密度函数
# p：使用fit拟合得到的均值、标准差绘制PDF曲线
p = norm.pdf(x, mu, std)
# q：使用直接np.mean、np.std计算出的均值标准差绘制PDF曲线
q = norm.pdf(x, mu1, std1)

# 黑色粗线：拟合参数的正态密度曲线
plt.plot(x, p, 'k', linewidth=2)
# 红色细线：直接统计计算参数的正态密度曲线
plt.plot(x, q, 'r', linewidth=1)

# 设置图表标题，打印拟合出来的均值与标准差，保留5位小数
title = "Fit results: mu = %.5f,  std = %.5f" % (mu, std)
plt.title(title)

# 展示图像
plt.show()

# plt.plot(x,y,format_string,**kwargs)
# x：横坐标数据
# y：纵坐标数据
# format_string：控制线条样式、颜色
# **kwargs：额外绘图参数（线宽、透明度等）
```



```python
from scipy import stats
import pandas as pd
# scipy包是一个高级的科学计算库，它和Numpy联系很密切，Scipy一般都是操控Numpy数组来进行科学计算

# 样本数据，35位健康男性在未进食之前的血糖浓度
data = [87,77,92,68,80,78,84,77,81,80,80,77,92,86,
        76,80,81,75,77,72,81,72,84,86,80,68,77,87,
        76,77,78,92,75,80,78]

df = pd.DataFrame(data, columns=['value'])
u = df['value'].mean()  # 计算均值
std = df['value'].std()  # 计算标准差
stats.kstest(df['value'], 'norm', (u, std))
# .kstest方法：KS检验，参数分别是：待检验的数据，检验方法（这里设置成norm正态分布），均值与标准差
# 结果返回两个值：statistic → D值，pvalue → P值
# p值大于0.05，为正态分布
```



```python
import numpy as np
from scipy import stats

#函数返回间隔类均匀分布的数值序列
b = np.linspace(0, 10, 100)

# Shapiro-Wilk test
#scipy.stats.shapiro适用于小样本数据，只能检查正态分布。
s,p = stats.shapiro(b)
print(s,p)

# kstest
#statistic,p=scipy.stats.kstest(rvs, cdf, args=(), N=20, alternative='two-sided', mode='approx')
# scipy.stats.kstest是一种K-S检验。它不局限于正态检验，可以检验多种分布。
s,p = stats.kstest(b, 'norm')
print(s,p)

# normaltest
# statistic,p=scipy.stats.normaltest(a, axis=0, nan_policy='propagate')
# 输出结果中第一个为统计量，第二个为P值（注：p值大于显著性水平0.05，认为样本数据符合正态分布）
s,p = stats.normaltest(b)
print(s,p)

# Anderson-Darling test
# 不局限于正态检验，可以检验多种分布（正态分布、指数分布、logistic 或者 Gumbel等分布）。
# statistic,critical_values,significance_level=scipy.stats.anderson(x, dist='norm')
# 返回值中，第一个表示统计值，第二个表示评价值，第三个是显著性水平；评价值和显著性水平对应，不同分布显著性水平不同。
s,c,p = stats.anderson(b,dist='norm')
print(s,c,p)

# 0.9547253251075745 0.0017218869179487228
# 0.7870260992168561 1.9947630078489827e-66
# 33.62988681076687 4.9815313241000405e-08
# 1.0837094127385427 [0.555 0.632 0.759 0.885 1.053] [15. 10.  5.  2.5  1. ]
```





