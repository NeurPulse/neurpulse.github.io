---
title: '深度学习:手写数字识别 — 从零手动实现 vs PyTorch 框架实现'
description: '基于 MNIST 数据集，先手动实现一个完整的多分类神经网络（前向/反向/参数更新），再用 PyTorch 的 nn.Module/CrossEntropyLoss/SGD 重写，对比两者理解框架帮我们做了什么'
pubDate: 2026-08-12
category: '深度学习'
tags: ['MNIST', '手写数字识别', '反向传播', 'PyTorch', 'nn.Module', '从零实现']
---

## Part A：手动实现

### 1. MNIST 数据集

- 70,000 张手写数字图片（训练 60,000 + 测试 10,000）
- 每张图片：28×28 像素灰度图，像素值 0~255
- 标签：0~9 的数字
- CSV 格式：785 列（第 1 列 = label，后 784 列 = 28×28 像素按行展开）

### 2. 关键实现细节

#### 2.1 数据预处理

```python
image = image / 255.0                     # 归一化到 [0, 1]
image = (image - 0.1307) / 0.3081         # 标准化（0.1307 和 0.3081 是全数据集统计值）
```

> 这里是对**整张图片所有像素整体**做标准化，而不是按像素位置分别做。因为图像的信息来自像素间的明暗对比，必须作为整体处理。

#### 2.2 参数初始化

```python
layer_sizes = [28*28, 128, 128, 128, 64, 10]  # 输入784 → 隐藏层 → 输出10
W = torch.randn(in_size, out_size) * torch.sqrt(torch.tensor(2 / in_size))  # Kaiming 初始化
b = torch.zeros(out_size)                                                      # 偏置初始化为 0
```

#### 2.3 激活函数实现

```python
def relu(x):
    return torch.clamp(x, min=0)         # max(0, x)

def relu_grad(x):
    return (x > 0).float()               # x>0 → 1.0, x≤0 → 0.0
```

#### 2.4 Softmax（含数值稳定技巧）

```python
def softmax(x):
    x_exp = torch.exp(x - x.max(dim=1, keepdim=True).values)  # 减去最大值防溢出
    return x_exp / x_exp.sum(dim=1, keepdim=True)
```

为什么要减 `x.max()`？ $e^{1000}$ 会超出 float 范围。分子分母同除以 $e^{x_{\max}}$ 等价于指数减最大值，数值稳定。

#### 2.5 交叉熵

```python
def cross_entropy(pred, labels):
    one_hot = torch.zeros_like(pred)
    one_hot[torch.arange(N), labels] = 1
    loss = -(one_hot * torch.log(pred + 1e-8)).sum() / N  # +1e-8 防止 log(0)
    return loss, one_hot
```

### 3. 训练循环 — 对照公式

| 步骤 | 代码 | 对应的公式 |
|------|------|-----------|
| 前向 | `z = a_prev @ W + b` → `a = relu(z)` | $z^l = a^{l-1}W^l + b^l$, $a^l = \text{ReLU}(z^l)$ |
| 输出层 | `y_pred = softmax(z_out)` | $a^n = \text{Softmax}(z^n)$ |
| $\delta^n$ | `dL_dz = (y_pred - one_hot) / N` | $\delta^n = \frac{1}{N}(a^n - y)$ |
| 权重梯度 | `grads_W[-1] = a_last.T @ dL_dz` | $\partial\text{loss}/\partial W^n = (a^{n-1})^T \cdot \delta^n$ |
| 偏置梯度 | `grads_b[-1] = dL_dz.sum(dim=0)` | $\partial\text{loss}/\partial b_j = \sum_{k=1}^N \delta_{kj}$ |
| $\delta$ 传播 | `dL_dz = dL_dz @ W_next.T * relu_grad(z)` | $\delta^i = \delta^{i+1}(W^{i+1})^T \odot \text{ReLU}'(z^i)$ |
| 参数更新 | `W -= lr * grad` | $W := W - lr \cdot \partial\text{loss}/\partial W$ |

**这就是之前推导的所有公式的代码版本。** 一行一行对照，反向传播不再神秘。

### 4. 完整手动训练代码骨架

```python
for epoch in range(num_epochs):
    for images, labels in train_loader:
        # -- 前向 --
        activations = [x]
        pre_acts = []
        for W, b in zip(weights[:-1], biases[:-1]):
            z = activations[-1] @ W + b
            pre_acts.append(z)
            activations.append(relu(z))
        z_out = activations[-1] @ weights[-1] + biases[-1]
        y_pred = softmax(z_out)

        loss, one_hot = cross_entropy(y_pred, y)

        # -- 反向 --
        dL_dz = (y_pred - one_hot) / N
        for i in reversed(range(len(weights))):
            grads_W[i] = activations[i].T @ dL_dz
            grads_b[i] = dL_dz.sum(dim=0)
            if i > 0:
                dL_dz = dL_dz @ weights[i].T * relu_grad(pre_acts[i-1])

        # -- 更新 --
        with torch.no_grad():
            for i in range(len(weights)):
                weights[i] -= lr * grads_W[i]
                biases[i]  -= lr * grads_b[i]
```

运行结果：测试准确率 **97%+**。

```python
import torch
from torch.utils.data import DataLoader, Dataset


class MNISTDataset(Dataset):
    def __init__(self, file_path):
        self.images, self.labels = self._read_file(file_path)

    def _read_file(self, file_path):
        images = []
        labels = []
        with open(file_path, 'r') as f:
            next(f)  # 跳过标题行
            for line in f:
                line = line.rstrip("\n")
                items = line.split(",")
                images.append([float(x) for x in items[1:]])
                labels.append(int(items[0]))
        return images, labels

    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = torch.tensor(image)
        image = image / 255.0  # 归一化
        image = (image - 0.1307) / 0.3081  # 标准化
        label = torch.tensor(label)
        return image, label

    def __len__(self):
        return len(self.images)


batch_size = 64
train_dataset = MNISTDataset(r'E:\电子书\RethinkFun深度学习\data\mnist\mnist_train.csv\mnist_train.csv')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = MNISTDataset(r"E:\电子书\RethinkFun深度学习\data\mnist\mnist_test.csv\mnist_test.csv")
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


learning_rate = 0.1
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 配置网络结构，包含输入层、隐藏层、输出层大小
layer_sizes = [28*28, 128, 128, 128, 64, 10]  # 可根据需要修改，例如 [输入, 隐层1, 隐层2, ..., 输出]
# 手动初始化参数
weights = []
biases = []
for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
    W = torch.randn(in_size, out_size, device=device) * torch.sqrt(torch.tensor(2 / in_size))
    b = torch.zeros(out_size, device=device)
    weights.append(W)
    biases.append(b)

# 激活函数及其导数
def relu(x):
    return torch.clamp(x, min=0)

def relu_grad(x):
    return (x > 0).float()

# Softmax + 交叉熵损失 (手动实现)
def softmax(x):
    x_exp = torch.exp(x - x.max(dim=1, keepdim=True).values)
    return x_exp / x_exp.sum(dim=1, keepdim=True)

def cross_entropy(pred, labels):
    N = pred.shape[0]
    one_hot = torch.zeros_like(pred)
    one_hot[torch.arange(N), labels] = 1  # 生成one-hot编码
    loss = - (one_hot * torch.log(pred + 1e-8)).sum() / N  # 计算平均loss，这里加上一个很小的数1e-8，是为了防止出现log(0)时出现负无穷大的情况。
    return loss, one_hot

# 训练循环
for epoch in range(num_epochs):
    total_loss = 0
    for images, labels in train_loader:
        x = images.to(device)
        y = labels.to(device)
        N = x.shape[0]

        # 前向传播
        activations = [x]
        pre_acts = []
        for W, b in zip(weights[:-1], biases[:-1]):
            z = activations[-1] @ W + b
            pre_acts.append(z)
            a = relu(z)
            activations.append(a)
        # 输出层
        z_out = activations[-1] @ weights[-1] + biases[-1]
        pre_acts.append(z_out)
        y_pred = softmax(z_out)

        # 损失
        loss, one_hot = cross_entropy(y_pred, y)
        total_loss += loss.item()

        # 反向传播
        grads_W = [None] * len(weights)
        grads_b = [None] * len(biases)
        # 输出层梯度
        dL_dz = (y_pred - one_hot) / N  # [N, output]
        grads_W[-1] = activations[-1].t() @ dL_dz
        grads_b[-1] = dL_dz.sum(dim=0)
        # 隐层梯度
        for i in range(len(weights)-2, -1, -1):
            dL_dz = dL_dz @ weights[i+1].t() * relu_grad(pre_acts[i])
            grads_W[i] = activations[i].t() @ dL_dz
            grads_b[i] = dL_dz.sum(dim=0)

        # 更新参数
        with torch.no_grad():
            for i in range(len(weights)):
                weights[i] -= learning_rate * grads_W[i]
                biases[i]  -= learning_rate * grads_b[i]

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")


# 测试
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        x = images.view(-1, layer_sizes[0]).to(device)
        y = labels.to(device)
        a = x
        for W, b in zip(weights[:-1], biases[:-1]):
            a = relu(a @ W + b)
        logits = a @ weights[-1] + biases[-1]
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    print(f"Test Accuracy: {correct/total*100:.2f}%")
```



---

## Part B：PyTorch 实现

同样的模型，PyTorch 让代码量大幅减少。

### 1. 模型定义

```python
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)
```

`nn.Sequential` 按顺序串联各个模块，forward 只需调用它。

### 2. 损失函数和优化器

```python
criterion = nn.CrossEntropyLoss()                   # 内置 Softmax + 交叉熵
optimizer = optim.SGD(model.parameters(), lr=0.1)   # 随机梯度下降
```

>`nn.CrossEntropyLoss` 内部已经包含了 Softmax，所以模型输出的 logits 直接传给 loss 函数即可，**不需要**在模型 forward 里加 Softmax。

### 3. 训练循环

```python
for epoch in range(num_epochs):
    for images, labels in train_loader:
        outputs = model(images)          # 前向
        loss = criterion(outputs, labels) # 损失

        optimizer.zero_grad()            # 清零
        loss.backward()                  # 反向传播（自动求所有梯度）
        optimizer.step()                 # 更新参数
```

### 4. 测试

```python
model.eval()                             # 切换到评估模式
with torch.no_grad():                    # 测试不需要梯度
    outputs = model(images)
    preds = torch.argmax(outputs, dim=1) # logits 最大位置 = 预测类别
```

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# 自定义数据集
class MNISTDataset(Dataset):
    def __init__(self, file_path):
        self.images, self.labels = self._read_file(file_path)

    def _read_file(self, file_path):
        images = []
        labels = []
        with open(file_path, 'r') as f:
            next(f)  # 跳过标题行
            for line in f:
                items = line.strip().split(",")
                images.append([float(x) for x in items[1:]])
                labels.append(int(items[0]))
        return images, labels

    def __getitem__(self, index):
        image = torch.tensor(self.images[index], dtype=torch.float32).view(-1)
        image = image / 255.0  # 归一化
        image = (image - 0.1307) / 0.3081  # 标准化
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return image, label

    def __len__(self):
        return len(self.images)


# 模型定义
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)


# 参数设置
batch_size = 64
learning_rate = 0.1
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据加载
train_dataset = MNISTDataset(r'E:\电子书\RethinkFun深度学习\data\mnist\mnist_train.csv\mnist_train.csv')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = MNISTDataset(r"E:\电子书\RethinkFun深度学习\data\mnist\mnist_test.csv\mnist_test.csv")
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 模型、损失函数、优化器
model = NeuralNetwork().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 训练过程
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# 测试过程
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {100 * correct / total:.2f}%")
```
---

## Part C：手动 vs PyTorch — 一一对应

| 手动实现 | PyTorch 替代 | 省掉了什么 |
|----------|-------------|-----------|
| 手写 `W, b` 的 Kaiming 初始化 | `nn.Linear` 自动完成 | 手算 $\sqrt{2/n}$ |
| 手写 `relu(x)`, `relu_grad(x)` | `nn.ReLU()` 自动完成 | 手写激活函数和导数 |
| 手写 `softmax(x)` | `nn.CrossEntropyLoss` 内部完成 | 手写 Softmax + 数值稳定 |
| 手写 `cross_entropy` | `nn.CrossEntropyLoss` | 手写 one-hot + log |
| 手写整个反向传播循环 | `loss.backward()` | **全部梯度计算** |
| 手动管理 6 个 W, b 的更新 | `optimizer.step()` | 手动逐参数更新 |

---

## 总结

```
手动实现的价值：彻底理解前向传播、δ 的递归传播、梯度计算
PyTorch 的价值：写一行业务逻辑，框架自动完成所有求导和优化

学习路径：
  第一步 → 手动实现（建立认知，消除神秘感）
  第二步 → PyTorch 重写（感受框架的便利，同时知道底层在做什么）
```

强烈建议自己动手敲一遍手动版本——当你看到自己的代码能把 MNIST 识别到 97%+ 时，神经网络对你就不再是黑盒了。
