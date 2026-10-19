---
title: 'PyTorch:数据读入'
description: '组织 Dataset 与 DataLoader，正确处理图片布局、CSV 标签、训练验证变换和多进程读入。'
pubDate: 2026-08-08
updatedDate: 2026-10-19
order: 4
category: 'PyTorch'
tags: ['Dataset', 'DataLoader']
---

## 数据读入

对 map-style 数据集，`Dataset` 定义“一个索引对应什么样本”，`DataLoader` 再负责采样、组批与可选的并行读入。自定义 Dataset 通常实现 `__init__`、`__getitem__` 和 `__len__`；流式 `IterableDataset` 使用另一套迭代接口。

下面先用内存中的小数据集验证批次形状，不依赖外部图片或下载：

```python
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader

train_data = TensorDataset(torch.rand(8, 3, 32, 32), torch.arange(8) % 2)
val_data = TensorDataset(torch.rand(5, 3, 32, 32), torch.arange(5) % 2)
```

## ImageFolder

真实图片可按类别分目录存放。训练集和验证集应使用一致的类别名称与索引映射：

```text
data/
  train/
    cat/001.jpg
    dog/002.jpg
  val/
    cat/003.jpg
    dog/004.jpg
```

下面定义读入函数，需要已安装与 torch 匹配的 torchvision，调用时传入实际存在的目录。

```python
from torchvision import datasets, transforms


def load_image_folders(train_path, val_path):
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    training = datasets.ImageFolder(train_path, transform=train_transform)
    validation = datasets.ImageFolder(val_path, transform=val_transform)
    if training.class_to_idx != validation.class_to_idx:
        raise ValueError("训练集和验证集的类别映射必须一致")
    return training, validation
```

训练集可以使用随机增强，验证集通常使用确定性的变换。`ToTensor()` 对常见的 uint8 图片将布局转为 `[C,H,W]`、数值缩放到 `[0,1]`；这不是对任意 dtype 输入都自动执行的归一化规则。若验证集缺少某类，应显式保持统一标签映射，不能直接信任两个目录各自生成的编号。

## 定制 Dataset

若图片路径和标签存储在 CSV 中，需要先约定文件格式。下面的 CSV **包含表头**，与 `pd.read_csv` 的默认行为对应：

```csv
filename,label
image1.jpg,0
image2.jpg,1
image3.jpg,0
```

如果文件没有表头，应显式传入 `header=None, names=["filename", "label"]`，否则第一条样本会被当作列名而丢失。

```python
from pathlib import Path
import pandas as pd
from PIL import Image


class MyDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        if not {"filename", "label"}.issubset(self.img_labels.columns):
            raise ValueError("CSV 需要 filename 和 label 两列")
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        row = self.img_labels.iloc[idx]
        with Image.open(self.img_dir / str(row["filename"])) as image_file:
            image = image_file.convert("RGB")
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label
```

本例约定标签是整数类别索引，数据文件应满足该约定。RGB 转换保证通道数一致。用于默认 DataLoader 组批时，还应传入包含 `Resize` 和 `ToTensor` 的变换，使每个样本都返回相同尺寸的 Tensor；未经转换的 PIL 图片不能直接按这里的默认方式组批。

## DataLoader 组批

```python
batch_size = 4
train_loader = DataLoader(
    train_data, batch_size=batch_size, shuffle=True,
    num_workers=0, drop_last=False,
)
val_loader = DataLoader(
    val_data, batch_size=batch_size, shuffle=False,
    num_workers=0, drop_last=False,
)
images, labels = next(iter(val_loader))
assert images.shape == (4, 3, 32, 32)
assert labels.shape == (4,) and labels.dtype == torch.long
assert len(list(val_loader)) == 2  # 最后一个 batch 只有 1 个样本
```

- `shuffle=True`：每次迭代打乱样本顺序，通常用于训练集。
- `num_workers=0`：在主进程读入，适合先排查问题。Windows 也支持大于 0，但多进程启动通常需要 `if __name__ == "__main__":` 保护，且 Dataset 等定义应能被子进程导入；不是“Windows 必须为 0”。
- `drop_last=True`：丢弃不足一整批的尾部样本，不是丢弃一个完整 epoch。验证通常保留全部样本；训练是否丢尾批取决于模型和训练要求。

## 查看一批图片

下面沿用最前面的未标准化内存样本。Tensor 从 CHW 转为 HWC，应使用 `permute(1,2,0)`，不能向 Tensor 的 `transpose` 传入三个轴。

```python
import matplotlib.pyplot as plt

images, labels = next(iter(val_loader))
plt.imshow(images[0].permute(1, 2, 0).cpu().numpy())
plt.title(f"label={labels[0].item()}")
plt.axis("off")
plt.show()
```

如果读入流程中使用了 `Normalize(mean, std)`，显示前还需要按通道执行 `image * std + mean` 还原；不能把标准化后的数值直接当成 `[0,1]` RGB 图片。

## 参考

- [PyTorch：Dataset 与 DataLoader](https://docs.pytorch.org/docs/stable/data.html)
- [torchvision：ImageFolder](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html)
- [torchvision：ToTensor](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.ToTensor.html)
