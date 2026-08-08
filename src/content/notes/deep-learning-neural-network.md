---
title: '深度学习：神经网络'
description: '矩阵计算、激活函数、softmax多分类与反向传播'
pubDate: 2026-08-07
category: '深度学习'
tags: ['神经网络','反向传播','激活函数','Softmax','多分类']
---

## 神经网络

> 在深度学习的神经网络里，一般按层进行划分。整体上神经网络会被分为输入层，隐藏层，和输出层。输入层就是原始特征，输出层就神经网络最终的输出。除了输入层和输出层，剩下的层都是隐藏层。

<img src="E:\selfdocument\web\public\images/deep-learning/0806.png" alt="img" style="zoom: 67%;" />

> 神经网络中的一层我们叫做线性层（Linear Layer），因为其中每个神经元都是一个线性回归加上激活函数。或者叫做全连接层、稠密层（Dense Layer），因为每一层的每个神经元都和前一层的每个神经元进行连接，所以叫做全连接、稠密连接。

### 神经网络与矩阵计算

<img src="E:\selfdocument\web\public\images/deep-learning/0807.png" alt="img" style="zoom:67%;" />

解释:

1. $x_1,x_2,x_3,x_4$是一个样本的4个特征值，也就是输入层的输入。
2. $w_{1,1}^1$​表示一个权重值
3. 上标1，表示这是第一层的参数。
4. 下标$(1,1)$​，
   * 第一个1表示这是针对第一个输入的权重。
   * 第二个1表示这是第一层的第1个神经元。

$w_{1,1}^1$表示第一层的第一个神经元对第一个输入的权重参数。

**第一层第一个神经元线性计算**

$$ z_1^1 = \left[ x_1,\ x_2,\ x_3,\ x_4 \right] \begin{bmatrix} w_{1,1}^1 \\ w_{2,1}^1 \\ w_{3,1}^1 \\ w_{4,1}^1 \end{bmatrix} $$​

**第一层第二个神经元线性计算**

$$ z_2^1 = \left[ x_1,\ x_2,\ x_3,\ x_4 \right] \begin{bmatrix} w_{1,2}^1 \\ w_{2,2}^1 \\ w_{3,2}^1 \\ w_{4,2}^1 \end{bmatrix} $$

**第一层第三个神经元线性计算**

$$ z_3^1 = \left[ x_1,\ x_2,\ x_3,\ x_4 \right] \begin{bmatrix} w_{1,3}^1 \\ w_{2,3}^1 \\ w_{3,3}^1 \\ w_{4,3}^1 \end{bmatrix} $$

**一层全部神经元合并矩阵形式**

$$ z^1 = \left[ z_1^1,\ z_2^1,\ z_3^1 \right] = \left[ x_1,\ x_2,\ x_3,\ x_4 \right] \begin{bmatrix} w_{1,1}^1 & w_{1,2}^1 & w_{1,3}^1 \\ w_{2,1}^1 & w_{2,2}^1 & w_{2,3}^1 \\ w_{3,1}^1 & w_{3,2}^1 & w_{3,3}^1 \\ w_{4,1}^1 & w_{4,2}^1 & w_{4,3}^1 \end{bmatrix} $$

**批量样本矩阵运算**

$$ \begin{bmatrix} x_{11} & x_{12} & x_{13} & x_{14} \\ x_{21} & x_{22} & x_{23} & x_{24} \end{bmatrix} \begin{bmatrix} w_{1,1}^1 & w_{1,2}^1 & w_{1,3}^1 \\ w_{2,1}^1 & w_{2,2}^1 & w_{2,3}^1 \\ w_{3,1}^1 & w_{3,2}^1 & w_{3,3}^1 \\ w_{4,1}^1 & w_{4,2}^1 & w_{4,3}^1 \end{bmatrix} $$ 

  $z^1 = \left[ z_1^1,\ z_2^1,\ z_3^1 \right]$，可以给线性回归结果加上偏置值 $\left[ b_1^1,\ b_2^1,\ b_3^1 \right]$。

$b_2^1$ 表示第一层第二个神经元的偏置值。

 $$ z^1 = \left[ z_1^1,\ z_2^1,\ z_3^1 \right] + \left[ b_1^1,\ b_2^1,\ b_3^1 \right] $$ 

对每个元素应用sigmoid函数就得到了第1层的激活值：

 $$ a^1 = sigmoid\left(\left[ z_1^1,\ z_2^1,\ z_3^1 \right]\right) $$ 

 $a^1$ 成为神经网络第二层的输入，第二层的计算可以用矩阵运算完成。

 $$ z^2 = \left[ z_1^2,\ z_2^2,\ z_3^2 \right] = \left[ a_1^1,\ a_2^1,\ a_3^1 \right] \begin{bmatrix} w_{1,1}^2 & w_{1,2}^2 & w_{1,3}^2 \\ w_{2,1}^2 & w_{2,2}^2 & w_{2,3}^2 \\ w_{3,1}^2 & w_{3,2}^2 & w_{3,3}^2 \end{bmatrix} $$ 

### 激活函数

> 激活函数的作用是引入非线性。

> 如果没有激活函数，不论有几层线性回归，最终都等价于一层的线性回归。
>
> 引入激活函数，模拟大脑神经元里的抑制和激活。神经网络才可以拟合任意函数。

1. $$ sigmoid(x) = \frac{1}{1 + e^{-x}} $$

<img src="E:\selfdocument\web\public\images/deep-learning/0703.png" alt="img" style="zoom: 50%;" />

> 可以将x映射到0到1之间。
>
> 好处:
>
> ​	0-1自然映射到概率值范围。
>
> 它非常适合作为二分类问题的神经网络的最后一层唯一神经元的激活函数。

2. $$ tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} $$

<img src="E:\selfdocument\web\public\images/deep-learning/0815.png" alt="img" style="zoom: 80%;" />

3. $ReLU(x)=max(x,0)$

<img src="E:\selfdocument\web\public\images/deep-learning/0816.png" alt="img" style="zoom:67%;" />

> 当输入当x>0 时，输出为 x；当输入 x≤0 时，输出为 0。

4. Leaky ReLU： 
   * 当$x > 0$时： $$ LeakyReLU(x) = x $$ 
   * 当$x \le 0$时： $$ LeakyReLU(x) = \alpha x $$

> 其中α*α*一般取小于1的数，比如0.1。这样当x取负值是也会有一个微小的梯度，可以更新参数。

<img src="E:\selfdocument\web\public\images/deep-learning/0820.png" alt="img" style="zoom:67%;" />

### 神经网络的多分类

**名词解释:**

对于神经网络，***输出层经过线性回归，还没有经过激活函数的值***，***叫做logits***。***logits经过激活函数后就是最终的输出值***

**激活函数的选择**

> 希望神经网络最后一层的logtis经过激活函数后可以代表样本属于每个类别的概率值。既然是概率值，则要求它们的和为1。

> 错误想法:

> 用每个神经元logits的值，除以所有神经元logits值的和作为最终的概率
>
> 1. logits的值有正有负，这样得到每个神经元的激活值，也就是概率值可能为负。
> 2. 这样做对不同类别的差异是线性的，我们希望能放大不同类别的差异，加速训练过程。

1. ***$softmax$​函数***

   我们假设输出层3个神经元的logits值为$z_1,z_2,z_3$，经过$$softmax$$输出的概率值为$O_1,O_2,O_3$，

   $$softmax$$函数的计算公式如下：

    $$ O_1 = \frac{e^{z_1}}{e^{z_1}+e^{z_2}+e^{z_3}} $$ 

   $$ O_2 = \frac{e^{z_2}}{e^{z_1}+e^{z_2}+e^{z_3}} $$ 

   $$ O_3 = \frac{e^{z_3}}{e^{z_1}+e^{z_2}+e^{z_3}} $$ 

   然后我们给出通用的$$softmax$$的公式：假设这个分类问题有$n$个类别，也就是输出层有$n$个神经元，输出的logits值有$n$个。

   对于第$i$个$$logits$$值输出的概率值，通过$$softmax$$计算公式如下：

    $$ O_i = \frac{e^{z_i}}{e^{z_1}+e^{z_2}+\dots+e^{z_n}} $$

2. **交叉熵损失函数**

   假设如下：

   1. 类别个数为$C$。 
   2. 真实标签用one-hot向量表示：$y = (y_1,y_2,\dots,y_C)$，其中$y$是一个样本的标签值，它的值是一个向量，有$C$个元素，里边只有一个元素$y_i = 1$，代表这个样本的真实类别，其余值都为0。 
   3. 预测的概率分布是：$p = (p_1,p_2,\dots,p_C)$，其中$p$是这个样本经过神经网络的softmax激活输出，它的值是一个向量，有$C$个元素，这$C$个元素因为是softmax的输出，代表概率，满足$\sum_{i=1}^C p_i = 1$。 

   交叉熵损失定义为： $$ loss = -\sum_{i=1}^C y_i \log(p_i) $$ 

   对于一个具体的样本，由于$y_i$只有一个为1，其余都为0，可以简化为：

    $$ loss = -\log(p_y) $$ 

   其中$p_y$是模型对真实类别$y$的预测概率。如果样本的真实类别为2，则这个样本的损失值就为$loss = -\log(p_2)$，当$p_2$等于1时（预测类别和真实类别一致），$loss$为0。当$p_2$越接近0，$loss$越接近无穷大。

    如果是对于批量样本计算loss，batch size 是$N$，则总的交叉熵损失就为：

    $$ loss = -\frac{1}{N}\sum_{n=1}^N \sum_{i=1}^C y_i^n \log(p_i^n) $$​



### 多分类神经网络的反向传播

#### **网络结构**

<img src="E:\selfdocument\web\public\images/deep-learning/0821.png" alt="神经网络图" style="zoom:67%;" />

- **输入**：2 个特征 x₁, x₂
- **隐藏层 1**：2 个神经元，权重矩阵 W¹ (2×2)，偏置 b¹ (1×2)
- **隐藏层 2**：2 个神经元，权重矩阵 W² (2×2)，偏置 b² (1×2)
- **输出层**：3 个神经元（对应 3 个类别），权重矩阵 W³ (2×3)，偏置 b³ (1×3)
- **激活函数**：act() 可选 Sigmoid、ReLU、Tanh 等
- **最终输出**：经过 Softmax 得到概率分布

**符号约定**：

- zᵏᵢ：第 k 层第 i 个神经元的 logits（激活前的值）
- aᵏᵢ：第 k 层第 i 个神经元的激活值
- wᵏᵢⱼ：第 k 层从第 j 个输入到第 i 个输出的权重

#### **前向传播过程**

输入： $$ \left[\begin{array}{ll} x_1 & x_2 \end{array}\right] $$ 

第一个隐藏层的logits： $$ \left[\begin{array}{ll} z_1^1 & z_2^1 \end{array}\right] = \left[\begin{array}{ll} x_1 & x_2 \end{array}\right] \begin{bmatrix} w_{1,1}^1 & w_{1,2}^1 \\ w_{2,1}^1 & w_{2,2}^1 \end{bmatrix} + \left[\begin{array}{ll} b_1^1 & b_2^1 \end{array}\right] $$ 

第一个隐藏层的输出如下。其中$act()$是激活函数，对logits的值逐个应用激活函数：

 $$ \left[\begin{array}{ll} a_1^1 & a_2^1 \end{array}\right] = \left[\begin{array}{ll} act(z_1^1) & act(z_2^1) \end{array}\right] $$ 

第一个隐藏层的输出作为第二个隐藏层的输入，则第二个隐藏层的logits为：

 $$ \left[\begin{array}{ll} z_1^2 & z_2^2 \end{array}\right] = \left[\begin{array}{ll} a_1^1 & a_2^1 \end{array}\right] \begin{bmatrix} w_{1,1}^2 & w_{1,2}^2 \\ w_{2,1}^2 & w_{2,2}^2 \end{bmatrix} + \left[\begin{array}{ll} b_1^2 & b_2^2 \end{array}\right] $$ 

第二个隐藏层的输出：

 $$ \left[\begin{array}{ll} a_1^2 & a_2^2 \end{array}\right] = \left[\begin{array}{ll} act(z_1^2) & act(z_2^2) \end{array}\right] $$ 

输出层的logits：

 $$ \left[\begin{array}{lll} z_1^3 & z_2^3 & z_3^3 \end{array}\right] = \left[\begin{array}{ll} a_1^2 & a_2^2 \end{array}\right] \begin{bmatrix} w_{1,1}^3 & w_{1,2}^3 & w_{1,3}^3 \\ w_{2,1}^3 & w_{2,2}^3 & w_{2,3}^3 \end{bmatrix} + \left[\begin{array}{lll} b_1^3 & b_2^3 & b_3^3 \end{array}\right] $$ 

输出层经过$softmax$​得到神经网络的输出：

> **Softmax — 把 logits 变成概率分布**
>
> $$Softmax(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}$$
>
> ```python
> # 例如：
> z = [2.0, 1.0, 0.1]
> e^z = [7.389, 2.718, 1.105]
> sum = 11.212
> softmax(z) = [0.659, 0.242, 0.099]   # 三个值加起来 = 1
> ```
>
> $$[\hat{y}_1 \quad \hat{y}_2 \quad \hat{y}_3] = [Softmax(z_1^3) \quad Softmax(z_2^3) \quad Softmax(z_3^3)]$$

 $$ a^3 = \left[\begin{array}{lll} a_1^3 & a_2^3 & a_3^3 \end{array}\right],\quad a_i^3 = \frac{e^{z_i^3}}{\sum_{j=1}^3 e^{z_j^3}} $$ 

网络输出结果和标签值利用交叉熵损失函数来计算loss：真实标签用一维的one-hot向量表示：

 $$ y = \left[\begin{array}{lll} y_1 & y_2 & y_3 \end{array}\right] $$ 

其中向量$y$​的元素中只有一个元素为1，其余元素为0。则交叉熵loss公式为：

> 对于多分类问题，损失函数是**交叉熵损失**：
>
> $$L = -\sum_{c=1}^{C} y_c \cdot \ln(\hat{y}_c)$$
>
> 其中：
>
> - C = 类别数（本例中 C=3）
> - y_c = 真实标签的 one-hot 编码（正确类别为 1，其余为 0）
> - ŷ_c = 模型预测的第 c 类的 Softmax 概率
>
> **例子**：真实标签是类别 0，模型预测的概率为 [0.7, 0.2, 0.1]
>
> ```python
> y_true = [1, 0, 0]         # one-hot 编码
> y_pred = [0.7, 0.2, 0.1]   # softmax 输出
> 
> Loss = -(1·ln(0.7) + 0·ln(0.2) + 0·ln(0.1))
>      = -ln(0.7)
>      = 0.357               # 预测越准，Loss 越小
> ```
>
> ---

 $$ loss = -\big(y_1 \ln a_1^3 + y_2 \ln a_2^3 + y_3 \ln a_3^3\big) $$​

#### **反向传播过程— 输出层梯度推导**

> 神经网络里每层的权重和偏置都可以看成是一个由多个参数构成的矩阵。反向传播时需要计算每个权重和偏置的梯度，实际上就是用最终的loss值对每一个参数求导，这些对单个参数的求导计算可以通过矩阵运算进行加速。

反向传播的核心思想：**链式法则从输出层向前逐层计算梯度**。

**loss对logits求导**

$$ \frac{\partial loss}{\partial z_i^3} = a_i^3 - y_i \quad (i = 1,2,3) $$

$$ \delta^3 = \left[\begin{array}{ccc} a_1^3 - y_1 & a_2^3 - y_2 & a_3^3 - y_3 \end{array}\right] $$

Softmax + 交叉熵组合的梯度特别简单：**预测概率 - 真实标签**。

> 具体看[推导过程](https://www.rethink.fun/chapter8/%E5%A4%9A%E5%88%86%E7%B1%BB%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%9A%84%E5%8F%8D%E5%90%91%E4%BC%A0%E6%92%AD.html)

输出层的logits计算公式如下：

 $$ \left[\begin{array}{ccc} z_1^3 & z_2^3 & z_3^3 \end{array}\right] = \left[\begin{array}{cc} a_1^2 & a_2^2 \end{array}\right] \left[\begin{array}{ccc} w_{11}^3 & w_{12}^3 & w_{13}^3 \\ w_{21}^3 & w_{22}^3 & w_{23}^3 \end{array}\right] + \left[\begin{array}{ccc} b_1^3 & b_2^3 & b_3^3 \end{array}\right] $$

下边分别**求loss对每一个 $w_{ij}^3$ 的梯度。 **

我们以loss对 $w_{11}^3$ 的偏导数为例： $$ \frac{\partial loss}{\partial w_{11}^3} = \frac{\partial loss}{\partial z_1^3} \cdot \frac{\partial z_1^3}{\partial w_{11}^3} + \frac{\partial loss}{\partial z_2^3} \cdot \frac{\partial z_2^3}{\partial w_{11}^3} + \frac{\partial loss}{\partial z_3^3} \cdot \frac{\partial z_3^3}{\partial w_{11}^3} $$ 

因为其中只有 $z_1^3$ 和 $w_{11}^3$ 有关，上边连加表达式的后两项 $z_2^3,z_3^3$ 对 $w_{11}^3$ 求导都为0，

所以有： $$ \frac{\partial loss}{\partial w_{11}^3} = \frac{\partial loss}{\partial z_1^3} \cdot \frac{\partial z_1^3}{\partial w_{11}^3} $$ 

其中 $\displaystyle \frac{\partial loss}{\partial z_1^3}$ 等于 $\delta_1^3$，而 $\displaystyle z_1^3 = a_1^2 \times w_{11}^3 + a_2^2 \times w_{21}^3 + b_1^3$，所以 $\displaystyle \frac{\partial z_1^3}{\partial w_{11}^3}$ 的结果为 $a_1^2$。

最终结果为： $$ \frac{\partial loss}{\partial w_{11}^3} = \delta_1^3 a_1^2 $$ 

依次类推，我们可以求出每个 $w_{ij}^3$ 的梯度

 $$ \left[\begin{array}{ccc} \delta_1^3 a_1^2 & \delta_2^3 a_1^2 & \delta_3^3 a_1^2 \\ \delta_1^3 a_2^2 & \delta_2^3 a_2^2 & \delta_3^3 a_2^2 \end{array}\right] $$ 

可以用矩阵运算表示如下：

 $$ \frac{\partial loss}{\partial w^3} = (a^2)^T \delta^3 = \left[\begin{array}{c} a_1^2 \\ a_2^2 \end{array}\right] \left[\begin{array}{ccc} \delta_1^3 & \delta_2^3 & \delta_3^3 \end{array}\right] $$

---

**偏置的梯度值**

以loss对$b_1^3$的偏导为例：

 loss对$b_1^3$求偏导，因为只有$b_1^3$只和$z_1^3$有关，所以：

 $$ \frac{\partial loss}{\partial b_1^3} = \frac{\partial loss}{\partial z_1^3} \cdot \frac{\partial z_1^3}{\partial b_1^3} $$ 

其中$\displaystyle \frac{\partial loss}{\partial z_1^3}$为$\delta_1^3$，又因为$\displaystyle z_1^3 = a_1^2 \times w_{11}^3 + a_2^2 \times w_{21}^3 + b_1^3$，所以$\displaystyle \frac{\partial z_1^3}{\partial b_1^3}$就等于1。

最终结果为：

 $$ \frac{\partial loss}{\partial b_1^3} = \frac{\partial loss}{\partial z_1^3} \cdot \frac{\partial z_1^3}{\partial b_1^3} = \delta_1^3 $$ 

同理，loss对$b_2^3,b_3^3$的偏导为：$\delta_2^3,\delta_3^3$，所以loss对于第三层偏置的偏导就等于$\delta^3$。

**计算loss对于第二层参数的偏导数**

利用链式法则，通过$a^2$进行传递，所以下边先计算loss对于$a^2$的偏导数。 

我们以loss对$a_1^2$为例：

 $$ \begin{aligned} \frac{\partial loss}{\partial a_1^2} &= \frac{\partial loss}{\partial z_1^3} \cdot \frac{\partial z_1^3}{\partial a_1^2} + \frac{\partial loss}{\partial z_2^3} \cdot \frac{\partial z_2^3}{\partial a_1^2} + \frac{\partial loss}{\partial z_3^3} \cdot \frac{\partial z_3^3}{\partial a_1^2} \\ &= \delta_1^3 w_{11}^3 + \delta_2^3 w_{12}^3 + \delta_3^3 w_{13}^3 \end{aligned} $$ 

同理，可以得到：

 $$ \frac{\partial loss}{\partial a_2^2} = \delta_1^3 w_{21}^3 + \delta_2^3 w_{22}^3 + \delta_3^3 w_{23}^3 $$ 

改为矩阵表示loss对第二层激活值的偏导为：

 $$ \frac{\partial loss}{\partial a^2} = \delta^3 (w^3)^T = \left[\begin{array}{ccc} \delta_1^3 & \delta_2^3 & \delta_3^3 \end{array}\right] \left[\begin{array}{cc} w_{11}^3 & w_{21}^3 \\ w_{12}^3 & w_{22}^3 \\ w_{13}^3 & w_{23}^3 \end{array}\right] $$​

接**求loss对第二层logits值的偏导**：

 $$ \delta^2 = \frac{\partial loss}{\partial z^2} = \frac{\partial loss}{\partial a^2} \cdot \frac{\partial a^2}{\partial z^2} = \delta^3 (w^3)^T \odot act'(z^2) $$ 

其中$\odot$是矩阵对应元素相乘（哈达玛积），因为激活函数是对每个$z$值单独应用的，所以这里求导也是逐个元素应用的。

第二层的梯度 

与上边对输出层的权重和偏置的求导方法一样，我们可以得到：

 $$ \frac{\partial loss}{\partial w^2} = (a^1)^T \delta^2 $$ 

loss对于第二层偏置的偏导就等于$\delta^2$。 

loss对于第一层logits值的偏导为：

 $$ \delta^1 = \delta^2 (w^2)^T \odot act'(z^1) $$ 

 第一层的梯度

 loss对于第一层权重的偏导为：

 $$ \frac{\partial loss}{\partial w^1} = x^T \delta^1 $$ 

loss对于第一层偏置的偏导就等于$\delta^1$

假设这个神经网络一共有$n$层，第$n$层是输出层。

$x$是输入向量，$y$是one-hot的label向量。

则：

 $$ \delta^n = a^n - y $$ 

对于第$i$层而言：

 $$ \delta^i = \delta^{i+1} (w^{i+1})^T \odot act'(z^i) $$

 $$ \frac{\partial loss}{\partial w^i} = (a^{i-1})^T \delta^i $$

 $$ \frac{\partial loss}{\partial b^i} = \delta^i $$

 第一层的输入是$x$：

 $$ a^0 = x $$

<img src="E:\selfdocument\web\public\images/deep-learning/backprop-1.png" alt="image-20260702104414688" style="zoom:67%;" />

