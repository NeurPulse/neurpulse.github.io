---
title: '深度学习:RNN 循环神经网络'
description: '理解序列问题、序列到序列任务形式、RNN 的隐状态记忆机制、循环层与普通层的区别，以及 RNN 为什么能够处理变长序列。'
pubDate: 2026-08-26
order: 47
category: '深度学习'
tags: ['RNN', '循环神经网络', '序列建模', 'NLP', '隐状态']
---

## 学习目标

学完这一节后，需要能回答三个问题：

1. 什么是序列问题，为什么它不能像普通表格数据一样随便打乱？
2. RNN 是如何用“隐状态”保存历史信息的？
3. RNN 中真正发生循环的是哪一层，普通输出层又负责什么？

一句话概括：**RNN 的核心是把上一个时刻的隐状态 $h_{t-1}$ 和当前输入 $x_t$ 一起用于计算当前隐状态 $h_t$，从而让模型在处理当前 token 时带着历史记忆。**

---

## 先修知识

学习 RNN 前，最好先理解下面几部分：

- [NLP 的发展：从规则到大语言模型](/notes/deep-learning-nlp-development/)：知道为什么文本、语音这类任务天然是序列任务。
- [NLP 常见任务：从文本分类到语言模型](/notes/deep-learning-nlp-common-tasks/)：理解文本分类、NER、翻译、摘要、语言模型这些任务的输入输出形式。
- [文本的计算机表示：从 Token 到 Embedding](/notes/deep-learning-token-encoding/)：理解 token、词向量、embedding 为什么可以作为神经网络输入。
- [深度学习:神经网络](/notes/deep-learning-neural-network/)：理解隐藏层、输出层、激活函数这些基础结构。
- [深度学习:反向传播](/notes/deep-learning-backpropagation/)：理解 RNN 训练时为什么要沿时间展开后反向传播。
- [深度学习:梯度消失与梯度爆炸](/notes/deep-learning-vanishing-exploding-gradients/)：理解普通 RNN 处理长序列时的主要困难。

---

## 1. 为什么需要 RNN

在普通神经网络里，输入通常被看成一组固定长度的特征，例如：

```text
[房屋面积, 房间数量, 楼层, 城市编码] -> 房价
```

这类数据的特征位置虽然有意义，但样本之间一般不强调“时间顺序”。

可是自然语言、语音、时间序列和视频不一样。它们的元素顺序本身就是信息的一部分。

例如：

```text
我喜欢你
你喜欢我
```

这两个句子包含相同的字，但顺序一变，语义关系就变了。

所以，**序列问题**指的是：输入、输出或两者都是有序数据，模型在处理某个位置的数据时，需要考虑它前面或后面的上下文信息。

典型序列任务包括：

- 自然语言处理：文本分类、NER、翻译、摘要、语言模型。
- 时间序列预测：股价、天气、流量、传感器数据。
- 语音识别：声音帧序列转文字。
- 视频分析：连续帧中识别动作或事件。
- 机器人控制：根据历史状态决定下一步动作。

---

## 2. 序列任务的输入输出形式

从输入和输出的角度看，序列任务常见有几种形式。

### 2.1 多对一

输入是一个序列，输出是一个结果。

例如情感分类：

```text
输入：这家店味道很好，下次还来
输出：正面评价
```

模型读完整句话后，给出一个分类结果。

这种任务可以表示为：

```text
(x_1, x_2, ..., x_T) -> y
```

### 2.2 等长多对多

输入是一个序列，输出也是一个序列，并且长度相同。

典型例子是命名实体识别 NER：

```text
输入：我 叫 李 宁
输出：O  O  B-N I-N
```

每个 token 都要对应一个标签，所以输入长度和输出长度一致。

### 2.3 不等长多对多

输入和输出都是序列，但长度不一定相同。

例如机器翻译：

```text
输入：I love deep learning.
输出：我喜欢深度学习。
```

再比如文本摘要：

```text
输入：一篇长文章
输出：几句话摘要
```

这类任务后来通常使用 Encoder-Decoder、Seq2Seq、Attention 和 Transformer 来处理。

---

## 3. RNN 的核心直觉：带着记忆逐步读序列

RNN 的设计思路很像人读句子。

我们不是一眼把所有词同时塞进脑子里，而是从左到右读：

```text
我 -> 叫 -> 李 -> 宁
```

读到“李”时，我们已经知道前面出现了“我叫”；读到“宁”时，我们又知道前面有“我叫李”。这些历史信息会影响我们判断“李宁”到底是人名、品牌名，还是别的实体。

RNN 用一个向量来保存这种历史信息，这个向量叫：

```text
隐状态 hidden state
```

记作：

$$
h_t
$$

其中 $t$ 表示当前时间步，也可以理解为当前 token 的位置。

---

## 4. RNN 的计算流程

设：

- $x_t$：第 $t$ 个时间步的输入，例如第 $t$ 个 token 的 embedding。
- $h_t$：第 $t$ 个时间步更新后的隐状态。
- $y_t$：第 $t$ 个时间步的输出。
- $h_{t-1}$：上一个时间步留下来的隐状态，也就是“历史记忆”。

RNN 在每一步做两件事：

1. 用当前输入 $x_t$ 和过去记忆 $h_{t-1}$ 计算新的记忆 $h_t$。
2. 用新的记忆 $h_t$ 计算当前输出 $y_t$。

流程可以写成：

```text
h_{t-1} + x_t -> RNN 循环层 -> h_t
h_t           -> 输出层       -> y_t
```

对于第一个时间步，因为还没有历史记忆，通常令：

$$
h_0 = 0
$$

也就是用一个全零向量表示“还没有读到任何内容”。

![RNN 从折叠结构到时间展开结构](/images/deep-learning/rnn/rnn-fold-unfold.svg)

---

## 5. RNN 的数学公式

RNN 的隐状态更新可以写成：

$$
h_t = \tanh([h_{t-1} \mid x_t]W_h + b_h)
$$

其中：

- $[h_{t-1} \mid x_t]$ 表示把上一时刻隐状态和当前输入拼接起来。
- $W_h$ 是循环层权重。
- $b_h$ 是循环层偏置。
- $\tanh$ 是激活函数，用来引入非线性，并把隐状态数值压到 $(-1, 1)$ 范围内。

输出层通常写成：

$$
y_t = \operatorname{softmax}(h_t W_y + b_y)
$$

如果当前任务是分类，例如 NER 中的 BIO 标签分类，$y_t$ 就是当前 token 属于各个类别的概率分布。

![一个 RNN 时间步的内部计算](/images/deep-learning/rnn/rnn-cell-state.svg)

---

## 6. 拼接写法与分开写法是等价的

有些教材会把公式写成：

$$
h_t = \tanh(x_t W_{xh} + h_{t-1}W_{hh} + b_h)
$$

这和上面的拼接写法本质等价。

可以这样理解：

```text
拼接写法：
[h_{t-1} | x_t] 乘以一个大矩阵 W_h

分开写法：
h_{t-1} 乘以 W_hh
x_t     乘以 W_xh
然后相加
```

两种写法只是表达方式不同，核心都一样：**当前隐状态同时依赖当前输入和上一时刻隐状态。**

---

## 7. Shape 检查

假设：

- 输入 embedding 维度为 $d_x$。
- 隐状态维度为 $d_h$。
- 输出类别数为 $c$。

那么：

| 符号 | 形状 | 含义 |
|---|---:|---|
| $x_t$ | $1 \times d_x$ | 当前 token 的 embedding |
| $h_{t-1}$ | $1 \times d_h$ | 上一时刻隐状态 |
| $[h_{t-1} \mid x_t]$ | $1 \times (d_h + d_x)$ | 拼接后的输入 |
| $W_h$ | $(d_h + d_x) \times d_h$ | 循环层权重 |
| $b_h$ | $1 \times d_h$ | 循环层偏置 |
| $h_t$ | $1 \times d_h$ | 当前隐状态 |
| $W_y$ | $d_h \times c$ | 输出层权重 |
| $b_y$ | $1 \times c$ | 输出层偏置 |
| $y_t$ | $1 \times c$ | 当前输出概率 |

如果是一批样本同时计算，批量大小为 $B$，则通常变成：

```text
x_t: B × d_x
h_t: B × d_h
y_t: B × c
```

---

## 8. 参数共享：RNN 能处理变长序列的关键

RNN 不是为每个位置单独准备一套参数，而是在所有时间步共享同一套参数：

```text
第 1 个 token 用 W_h, W_y
第 2 个 token 仍然用 W_h, W_y
第 3 个 token 仍然用 W_h, W_y
...
```

这带来两个好处：

1. **能处理任意长度的序列**：不需要提前固定输入有多少个 token。
2. **参数量不会随序列长度增加**：句子变长，计算步数增加，但参数数量不变。

这也是 RNN 和普通全连接网络的重要区别。

普通全连接网络通常要求输入维度固定；RNN 则通过“循环调用同一个模块”来处理变长输入。

---

## 9. 循环层与普通层

RNN 中真正“循环”的是循环层，也就是负责计算 $h_t$ 的那一层：

$$
h_t = \tanh([h_{t-1} \mid x_t]W_h + b_h)
$$

它每个时间步都会被调用，并且当前步依赖上一时间步的结果。

而输出层：

$$
y_t = \operatorname{softmax}(h_t W_y + b_y)
$$

本身不参与循环。它只是拿当前隐状态 $h_t$ 做分类或回归。

可以把两者理解成：

```text
循环层：负责读序列、更新记忆。
普通层：负责根据当前记忆完成任务。
```

所以 RNN 的本质不是“有很多层”，而是：**同一个循环层在时间维度上被重复调用。**

## 10. 以 NER 为例理解 RNN

假设任务是识别人名：

```text
句子 1：我的鞋是李宁的。
句子 2：我叫李宁。
```

同样是“李宁”，在第一句里更像品牌，在第二句里更像人名。

如果模型只看当前 token，很难判断“李宁”的语义。但 RNN 会在处理每个 token 时带上前面的隐状态：

```text
我 -> h_1
叫 -> h_2
李 -> h_3
宁 -> h_4
```

当模型处理“李”和“宁”时，隐状态里已经包含了“我叫”这个上下文，因此更容易判断它是人名。

对于 NER 这类等长多对多任务，每个时间步都可以输出一个标签：

```text
输入：我 叫 李 宁
输出：O  O  B-N I-N
```

![RNN 在 NER 任务中的上下文记忆](/images/deep-learning/rnn/rnn-ner-example.svg)

---

## 11. RNN 如何训练

RNN 的训练通常可以理解为：

1. 按时间步从前往后计算所有隐状态和输出。
2. 根据任务计算损失。
3. 沿时间展开后的计算图反向传播梯度。

这个过程叫：

```text
Backpropagation Through Time，BPTT
```

中文常译为“通过时间的反向传播”。

普通神经网络的反向传播是在层与层之间传递梯度；RNN 的反向传播还要沿着时间步传递梯度。

也就是说，第 $t$ 步的损失，不仅会更新当前步的计算，还会影响前面很多时间步共享的循环层参数。

---

## 12. RNN 的优势

RNN 的优势主要来自三个机制。

### 12.1 隐状态机制

隐状态让模型具有“记忆”能力，可以利用历史信息做当前判断。

例如在文本中，一个词的含义往往依赖它前面的词。

### 12.2 参数共享机制

所有时间步共享同一套参数，因此模型不会因为序列变长而引入更多参数。

这提升了效率，也让模型能在不同长度的序列上泛化。

### 12.3 支持变长输入

RNN 每次只处理一个序列元素，所以理论上可以处理任意长度的输入。

相比固定输入维度的全连接网络，RNN 更适合处理文本、语音、时间序列等任务。

---

## 13. RNN 的局限

虽然 RNN 很重要，但普通 RNN 也有明显问题。

### 13.1 长距离依赖困难

如果当前判断依赖很久以前的信息，普通 RNN 往往很难保留这么长的记忆。

例如：

```text
我昨天在图书馆借了一本关于深度学习的书，今天终于把它读完了。
```

“它”指的是“书”，但两者之间隔了很多词。普通 RNN 的隐状态在逐步更新时可能会冲淡早期信息。

### 13.2 梯度消失与梯度爆炸

RNN 训练时需要沿时间反向传播。序列越长，梯度经过的链条越长，越容易出现：

- 梯度消失：早期时间步几乎学不到东西。
- 梯度爆炸：参数更新过大，训练不稳定。

这也是后来 LSTM、GRU 和 Transformer 出现的重要原因。

---

## 14. 和后续模型的关系

RNN 是理解序列建模的一块地基。

后续模型可以看作是在解决它的不同问题：

| 模型 | 主要改进 |
|---|---|
| LSTM | 用门控机制缓解长距离依赖和梯度消失 |
| GRU | 用更简洁的门控结构改进 RNN |
| Seq2Seq + Attention | 让模型在生成输出时可以重新关注输入序列的不同位置 |
| Transformer | 用自注意力并行建模序列关系，减少 RNN 的串行计算瓶颈 |

因此，虽然现代 NLP 主要使用 Transformer，但理解 RNN 仍然很有价值，因为它清楚揭示了序列建模中的两个核心问题：

```text
如何保存历史信息？
如何让当前位置利用上下文？
```

---

## 15. 常见误区

### 误区一：RNN 每个时间步都有不同参数

不是。RNN 的关键正是参数共享。

每个时间步使用同一套 $W_h, b_h, W_y, b_y$。

### 误区二：RNN 的输出层也是循环结构

不是。循环主要发生在隐状态更新层。

输出层只是把每个时间步的 $h_t$ 转换成任务需要的结果。

### 误区三：RNN 只能处理 NLP

不是。RNN 适合所有有顺序依赖的数据，包括时间序列、语音、视频和控制任务。

### 误区四：RNN 理论上能处理任意长度，所以实践中也擅长超长序列

不完全对。RNN 确实可以接收变长序列，但普通 RNN 在长序列上容易出现记忆衰减和梯度问题。

---

## 16. 一句话总结

RNN 通过在时间步之间传递隐状态 $h_t$，让模型在处理当前输入 $x_t$ 时能够利用之前的上下文信息；它的核心是循环层的递归调用和参数共享，而不是输出层本身。

---

## 17. 复习卡片

```text
RNN 要解决什么问题？
  处理具有顺序依赖的序列数据。

RNN 的记忆是什么？
  隐状态 h_t。

h_t 怎么来？
  由当前输入 x_t 和上一时刻隐状态 h_{t-1} 共同计算。

第一个时间步没有历史怎么办？
  使用全零向量 h_0。

RNN 为什么能处理变长序列？
  每个时间步循环调用同一个模块，参数共享，不要求固定序列长度。

RNN 中哪一层是真正的循环层？
  计算 h_t 的隐状态更新层。

输出层做什么？
  根据 h_t 完成分类或回归。

普通 RNN 的主要问题？
  长距离依赖困难，容易出现梯度消失或梯度爆炸。
```

---

## 18. 自测问题

1. 为什么“我叫李宁”和“我的鞋是李宁的”需要上下文才能判断“李宁”的类别？
2. RNN 的 $h_t$ 和普通神经网络隐藏层输出有什么相同点和不同点？
3. 为什么 RNN 的参数共享机制可以支持变长序列？
4. 拼接公式 $[h_{t-1} \mid x_t]W_h$ 和分开公式 $x_tW_{xh} + h_{t-1}W_{hh}$ 为什么等价？
5. RNN 的循环层和输出层分别承担什么任务？
6. 为什么普通 RNN 在长序列上容易遇到梯度消失问题？

---

## 参考

- [RNN · GitBook](https://www.rethink.fun/chapter13/RNN.html)
- [What is a Recurrent Neural Network (RNN)? | IBM](https://www.ibm.com/think/topics/recurrent-neural-networks)
- [Recurrent Neural Network | NVIDIA Developer](https://developer.nvidia.com/discover/recurrent-neural-network)
- [Recurrent Neural Networks: From Basic to Advanced | Medium](https://medium.com/@sachinsoni600517/recurrent-neural-networks-rnn-from-basic-to-advanced-1da22aafa009)
- [循环神经网络 | 动手学深度学习](https://zh.d2l.ai/chapter_recurrent-neural-networks/index.html)
- [循环神经网络的从零开始实现 | 动手学深度学习](https://zh.d2l.ai/chapter_recurrent-neural-networks/rnn-scratch.html)
- [torch.nn.RNN | PyTorch Documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html)
- [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [Understanding LSTM Networks | colah's blog](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

## 推荐阅读路线

如果是为了学习掌握，而不是单纯查资料，建议按下面顺序读：

1. [RNN · GitBook](https://www.rethink.fun/chapter13/RNN.html)：先建立中文直觉，重点看“隐状态”和“循环层”。
2. [What is a Recurrent Neural Network (RNN)? | IBM](https://www.ibm.com/think/topics/recurrent-neural-networks)：补充 RNN 的整体应用、类型和局限。
3. [Recurrent Neural Network | NVIDIA Developer](https://developer.nvidia.com/discover/recurrent-neural-network)：从工程视角理解 RNN 为什么适合序列数据，以及为什么普通全连接网络不方便处理变长输入。
4. [循环神经网络 | 动手学深度学习](https://zh.d2l.ai/chapter_recurrent-neural-networks/index.html)：系统学习序列模型、语言模型、BPTT 和 RNN 的实现。
5. [循环神经网络的从零开始实现 | 动手学深度学习](https://zh.d2l.ai/chapter_recurrent-neural-networks/rnn-scratch.html)：想真正掌握代码时读，重点看隐状态初始化、时间步循环和梯度裁剪。
6. [torch.nn.RNN | PyTorch Documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html)：准备写代码时看，重点关注 `input_size`、`hidden_size`、`num_layers`、`batch_first` 和输出 shape。
7. [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)：看 RNN 在字符级语言模型上的直观威力，适合作为兴趣扩展。
8. [门控循环单元（GRU） | 动手学深度学习](https://zh.d2l.ai/chapter_recurrent-modern/gru.html)：学完普通 RNN 后继续看，理解 GRU 如何用门控机制缓解长距离依赖问题。
9. [Understanding LSTM Networks | colah's blog](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)：继续学习 LSTM 时读，图解非常经典，适合理解“门控”和“细胞状态”。

其中 Medium 那篇可以作为补充浏览，但优先级不如 D2L、PyTorch 官方文档、Karpathy 和 Colah。学习时建议先把一条主线读通，再用 Medium 这类文章补不同作者的表达方式。
