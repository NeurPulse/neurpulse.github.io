---
title: '数学建模：神经网络'
description: '多层感知机（MLP）与反向传播'
pubDate: 2026-08-07
category: '数学建模'
tags: ['神经网络','MLP','深度学习']
---

### 神经网络

> 神经网络是一种具有非常强大能力的模型，不仅能够做常规的分类、回归等任务，还可以处理非结构化数据。但神经网络需要在大量数据下使用才有其意义，需要注意使用条件。

多层感知机

> 多层感知机（MLP，Multilayer Perceptron）也叫人工神经网络（ANN，Artificial Neural Network），除了输入输出层，它中间可以有多个隐层。
>
> 多层感知机（MLP）是一种神经网络，它由三种类型的层组成：输入层、一个或多个隐藏层（中间层），以及输出层。在这些层之间，数据会经过线性映射和激活函数处理，然后传递到下一层。

步骤：

1. 输入层接收输入数据，并将其传递到第一个隐藏层。

2. 在每一层，数据会经过线性映射（即，每个神经元的输入是上一层输出的加权和），然后通过激活函数进行非线性转换。

3. 经过所有隐藏层的处理后，数据最终到达输出层，得到预测结果。

在建立了基本的神经网络模型之后，接下来的任务是训练模型，即学习模型的参数。由于神经网络包含多层结构，计算梯度变得非常复杂。为了解决这个问题，我们采用误差反向传播算法（Backpropagation）来更新网络参数。

误差反向传播算法步骤：

1. 从输入层开始，数据逐层传递到输出层，计算预测结果与实际数据之间的误差。

2. 对于每一层，我们计算误差与上一层输出的关系。这需要对每一层的激活函数求导，以找到误差对权重的影响。

3. 通过这种链式求导过程，我们可以计算出每一层权重对损失函数的梯度，从而更新权重以最小化误


我们通过矩阵$\boldsymbol{X} \in \mathbb{R}^{n\times d}$来表示$n$个样本的小批量，其中每个样本具有$d$个输入特征。对于具有$h$个隐藏单元的单隐藏层多层感知机，用$\boldsymbol{H} \in \mathbb{R}^{n\times h}$表示隐藏层的输出，称为**隐藏表示（hidden representations）**。在数学或代码中，$\boldsymbol{H}$也被称为隐藏层变量（hidden-layer variable）或隐藏变量（hidden variable）。因为隐藏层和输出层都是全连接的，所以我们有隐藏层权重$\boldsymbol{W}^{(1)} \in \mathbb{R}^{d\times h}$和隐藏层偏置$\boldsymbol{b}^{(1)} \in \mathbb{R}^{1\times h}$以及输出层权重$\boldsymbol{W}^{(2)} \in \mathbb{R}^{h\times q}$和输出层偏置$\boldsymbol{b}^{(2)} \in \mathbb{R}^{1\times q}$。形式上，我们按如下方式计算单隐藏层多层感知机的输出$\boldsymbol{O} \in \mathbb{R}^{n\times q}$：
$$
\begin{align} \boldsymbol{H} &= \boldsymbol{X}\boldsymbol{W}^{(1)} + \boldsymbol{b}^{(1)}, \\ \boldsymbol{O} &= \boldsymbol{H}\boldsymbol{W}^{(2)} + \boldsymbol{b}^{(2)}. \end{align} \tag{4.1.1} 
$$
注意在添加隐藏层之后，模型现在需要跟踪和更新额外的参数。可我们能从中得到什么好处呢？在上面定义的模型里，我们没有好处！原因很简单：上面的隐藏单元由输入的仿射函数给出，而输出（softmax操作前）只是隐藏单元的仿射函数。仿射函数的仿射函数本身就是仿射函数，但是我们之前的线性模型已经能够表示任何仿射函数。

 我们可以证明这一等价性，即对于任意权重值，我们只需合并隐藏层，便可产生具有参数$\boldsymbol{W} = \boldsymbol{W}^{(1)}\boldsymbol{W}^{(2)}$和$\boldsymbol{b} = \boldsymbol{b}^{(1)}\boldsymbol{W}^{(2)} + \boldsymbol{b}^{(2)}$的等价单层模型：
$$
\boldsymbol{O} = (\boldsymbol{X}\boldsymbol{W}^{(1)} + \boldsymbol{b}^{(1)})\boldsymbol{W}^{(2)} + \boldsymbol{b}^{(2)} = \boldsymbol{X}\boldsymbol{W}^{(1)}\boldsymbol{W}^{(2)} + \boldsymbol{b}^{(1)}\boldsymbol{W}^{(2)} + \boldsymbol{b}^{(2)} = \boldsymbol{X}\boldsymbol{W} + \boldsymbol{b}. \tag{4.1.2}
$$
为了发挥多层架构的潜力，我们还需要一个额外的关键要素：在仿射变换之后对每个隐藏单元应用非线性的**激活函数（activation function）**$\sigma$。激活函数的输出（例如，$\sigma(\cdot)$）被称为**活性值（activations）**。一般来说，有了激活函数，就不可能再将我们的多层感知机退化成线性模型：

 $$ \begin{align} \boldsymbol{H} &= \sigma(\boldsymbol{X}\boldsymbol{W}^{(1)} + \boldsymbol{b}^{(1)}), \\ \boldsymbol{O} &= \boldsymbol{H}\boldsymbol{W}^{(2)} + \boldsymbol{b}^{(2)}. \end{align} \tag{4.1.3} $$ 

由于$\boldsymbol{X}$中的每一行对应于小批量中的一个样本，出于记号习惯的考量，我们定义非线性函数$\sigma$也以按行的方式作用于其输入，即一次计算一个样本。应用于隐藏层的激活函数通常不仅按行操作，也按元素操作。这意味着在计算每一层的线性部分之后，我们可以计算每个活性值，而不需要查看其他隐藏单元所取的值。对于大多数激活函数都是这样。 

为了构建更通用的多层感知机，我们可以继续堆叠这样的隐藏层，例如$\boldsymbol{H}^{(1)} = \sigma_1(\boldsymbol{X}\boldsymbol{W}^{(1)} + \boldsymbol{b}^{(1)})$和$\boldsymbol{H}^{(2)} = \sigma_2(\boldsymbol{H}^{(1)}\boldsymbol{W}^{(2)} + \boldsymbol{b}^{(2)})$​​，一层叠一层，从而产生更有表达能力的模型。

**激活函数**：

1. *修正线性单元*（Rectified linear unit，*ReLU*）：

    给定元素，ReLU函数被定义为该元素与的最大值：
   $$
   RELU(x) = max(x,0)
   $$

   > ReLU函数通过将相应的活性值设为0，仅保留正元素并丢弃所有负元素。
   >
   >
   > 输入为负时，ReLU函数的导数为0，而当输入为正时，ReLU函数的导数为1。 注意，当输入值精确等于0时，ReLU函数不可导。 在此时，我们默认使用左侧的导数，即当输入为0时导数为0。 我们可以忽略这种情况，因为输入可能永远都不会是0。 这里引用一句古老的谚语，“如果微妙的边界条件很重要，我们很可能是在研究数学而非工程”， 这个观点正好适用于这里。 下面我们绘制ReLU函数的导数。
   >
   >
   > 使用ReLU的原因是，它求导表现得特别好：要么让参数消失，要么让参数通过。 这使得优化表现得更好，并且ReLU减轻了困扰以往神经网络的梯度消失问题

2. sigmoid函数:

   对于一个定义域在中的输入， *sigmoid函数*将输入变换为区间(0, 1)上的输出。 因此，sigmoid通常称为*挤压函数*（squashing function）： 它将范围（-inf, inf）中的任意输入压缩到区间（0, 1）中的某个值：
   $$
   sigmoid(x) = \frac{1}{1 + exp(-x)}
   $$

   > 对“激发”或“不激发”的生物神经元进行建模.sigmoid函数是一个自然的选择，因为它是一个平滑的、可微的阈值单元近似。 当我们想要将输出视作二元分类问题的概率时， sigmoid仍然被广泛用作输出单元上的激活函数 （sigmoid可以视为softmax的特例）。
   >

3.  tanh(双曲正切)函数:

   与sigmoid函数类似， tanh(双曲正切)函数也能将其输入压缩转换到区间(-1, 1)上。
   $$
   tanh(x) = \frac{1 - exp(-2x)}{1 + exp(-2x)}
   $$
   

   全连接神经网络搭建：

   ```python
   import numpy as np
   import torch
   from collections import Counter
   from sklearn import datasets
   import torch.nn.functional as Fun
   
   # 1. 数据准备
   dataset = datasets.load_iris()
   dataut=dataset['data']
   priciple=dataset['target']
   input=torch.FloatTensor(dataset['data'])
   label=torch.LongTensor(dataset['target'])
   
   # 2. 定义BP神经网络
   class Net(torch.nn.Module):
       def __init__(self, n_feature, n_hidden, n_output):
           super(Net, self).__init__()
           self.hidden = torch.nn.Linear(n_feature, n_hidden)   # 定义隐藏层网络
           self.out = torch.nn.Linear(n_hidden, n_output)   # 定义输出层网络
   
       def forward(self, x):
           x = Fun.relu(self.hidden(x))      # 隐藏层的激活函数,采用relu,也可以采用sigmod,tanh
           x = self.out(x)                   # 输出层不用激活函数
           return x
   
   # 3. 定义优化器损失函数
   net = Net(n_feature=4, n_hidden=20, n_output=3)    #n_feature:输入的特征维度,n_hiddenb:神经元个数,n_output:输出的类别个数
   optimizer = torch.optim.SGD(net.parameters(), lr=0.02) # 优化器选用随机梯度下降方式
   loss_func = torch.nn.CrossEntropyLoss() # 对于多分类一般采用的交叉熵损失函数,
   
   # 4. 训练数据
   for t in range(500):
       out = net(input)                 # 输入input,输出out
       loss = loss_func(out, label)     # 输出与label对比
       optimizer.zero_grad()   # 梯度清零
       loss.backward()         # 前馈操作
       optimizer.step()        # 使用梯度优化器
   
   # 5. 得出结果
   out = net(input) #out是一个计算矩阵，可以用Fun.softmax(out)转化为概率矩阵
   prediction = torch.max(out, 1)[1] # 返回index  0返回原值
   pred_y = prediction.data.numpy()
   target_y = label.data.numpy()
   
   # 6.衡量准确率
   accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
   print("莺尾花预测准确率",accuracy)
   
   ```

   ```python
   
   import numpy as np
   import pandas as pd
   import torch
   from torch.utils.data import Dataset, DataLoader
   
   # 准备数据集
   class TitanicDataset(Dataset):
       def __init__(self, filepath):
           xy = pd.read_csv(filepath)
           # xy.shape() 可以得到xy的行列数
           self.len = xy.shape[0]
           # 选取相关的数据特征
           feature = ["Pclass", "Sex", "SibSp", "Parch", "Fare"]
           # np.array()将数据转换成矩阵，方便进行接下来的计算
           self.x_data = torch.from_numpy(np.array(pd.get_dummies(xy[feature])))
           self.y_data = torch.from_numpy(np.array(xy["Survived"]))
   
       # getitem函数，可以使用索引拿到数据
       def __getitem__(self, index):
           return self.x_data[index], self.y_data[index]
   
       # 返回数据的条数/长度
       def __len__(self):
           return self.len
   
   # 实例化自定义类，并传入数据地址
   dataset = TitanicDataset('train.csv')
   # num_workers是否要进行多线程服务，num_worker=2 就是2个进程并行运行
   # 采用Mini-Batch的训练方法
   train_loader = DataLoader(dataset = dataset,batch_size = 1,shuffle = True,num_workers = 0)
   ```

   ```python
   # 定义模型
   class Model(torch.nn.Module):
       def __init__(self):
           super(Model, self).__init__()
           # 要先对选择的特征进行独热表示计算出维度，而后再选择神经网络开始的维度
           self.linear1 = torch.nn.Linear(6, 3)
           self.linear2 = torch.nn.Linear(3, 1)
   
           self.sigmoid = torch.nn.Sigmoid()
   
       # 前馈
       def forward(self, x):
           x = self.sigmoid(self.linear1(x))
           x = self.sigmoid(self.linear2(x))
           return x
   
       # 测试函数
       def test(self, x):
           with torch.no_grad():
               x = self.sigmoid(self.linear1(x))
               x = self.sigmoid(self.linear2(x))
               y = []
               # 根据二分法原理，划分y的值
               for i in x:
                   if i > 0.5:
                       y.append(1)
                   else:
                       y.append(0)
               return y
   
   # 实例化模型
   model = Model()
   ```

   ```python
   # 定义损失函数
   criterion = torch.nn.BCELoss(reduction='mean')
   # 定义优化器
   optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
   
   
   # 防止windows系统报错
   if __name__ == '__main__':
       # 采用Mini-Batch的方法训练要采用多层嵌套循环
       # 所有数据都跑100遍
       for epoch in range(100):
           # data从train_loader中取出数据（取出的是一个元组数据）：(x, y)
           # enumerate可以获得当前是第几次迭代，内部迭代每一次跑一个Mini-Batch
           for i, data in enumerate(train_loader, 0):
               # inputs获取到data中的x的值，labels获取到data中的y值
               x, y = data
               x = x.float()
               y = y.float()
               y_pred = model(x)
               y_pred = y_pred.squeeze(-1)
               loss = criterion(y_pred, y)
               print(epoch, i, loss.item())
               optimizer.zero_grad()
               loss.backward()
               optimizer.step()
   ```

   ```python
   # 测试
   test_url = "https://raw.githubusercontent.com/minsuk-heo/kaggle-titanic/master/input/test.csv"
   test_data=pd.read_csv(test_url)
   feature = ["Pclass", "Sex", "SibSp", "Parch", "Fare"]
   df_feature['Fare'] = df_feature['Fare'].fillna(df_feature['Fare'].mean())
   test=torch.from_numpy(np.array(pd.get_dummies(test_data[feature])))
   y=model.test(test.float())
   
   # 输出预测结果
   output=pd.DataFrame({'PassengerId':test_data.PassengerId,'Survived':y})
   output.to_csv('my_predict.csv',index=False)
   
   
   
   ```

   

```python
data=pd.read_csv("Bitcoin.csv")
all_data=data['Bitcoin'].to_numpy()

all_data

test_data_size = 200

train_data = all_data[:-test_data_size]
test_data = all_data[-test_data_size:]
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1))
train_data_normalized = scaler.fit_transform(train_data.reshape(-1, 1))

print(train_data_normalized[:5])

train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)
```



```python
train_window = 12
def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq

train_inout_seq = create_inout_sequences(train_data_normalized, train_window)
train_inout_seq[:5]
```



```python
fut_pred = 12
test_inputs = train_data_normalized[-train_window:].tolist()
print(test_inputs)
```



```python
model.eval()

for i in range(fut_pred):
    seq = torch.FloatTensor(test_inputs[-train_window:])
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
        test_inputs.append(model(seq).item())
```



```python
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.utils import np_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import RMSprop

# 数据导入
(x_train,y_train),(x_test,y_test) = mnist.load_data()
print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)

# 数据预处理
x_train = x_train.reshape(x_train.shape[0],-1) / 255.0
x_test = x_test.reshape(x_test.shape[0],-1) / 255.0
y_train = np_utils.to_categorical(y_train,num_classes=10)
y_test = np_utils.to_categorical(y_test,num_classes=10)

# 直接使用keras.Sequential()搭建全连接网络模型
model = Sequential()
model.add(Dense(128, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))

#lr为学习率，epsilon防止出现0，rho/decay分别对应公式中的beta_1和beta_2
rmsprop = RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-08)
model.compile(optimizer=rmsprop,loss='categorical_crossentropy',metrics=['accuracy'])
print("---------------training---------------")
model.fit(x_train,y_train,epochs=10,batch_size=32)
print('\n')
print("---------------testing---------------")
loss,accuracy = model.evaluate(x_test,y_test)
print('loss:',loss)
print('accuracy:',accuracy)
```



```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 模拟数据
X = all_data[:-1]  # 输入序列
y = all_data[1:]   # 目标序列
# 重塑数据以适应LSTM
X = X.reshape((X.shape[0], 1, 1))
y = y.reshape((y.shape[0], 1))

model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X.shape[1], X.shape[2])))  # 50个LSTM单元
model.add(Dense(1))  # 输出层，因为我们的任务是回归，所以这里是1个神经元
model.compile(optimizer='adam', loss='mse', metrics=['mse'])
history = model.fit(X, y, epochs=100, batch_size=32, validation_split=0.2)
test_loss = model.evaluate(X, y)  # 假设X_test和y_test是你的测试数据和标签
print('Test Mean Absolute Error:', test_loss)
```

