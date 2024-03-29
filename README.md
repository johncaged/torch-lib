# torch_lib

![License](https://img.shields.io/github/license/johncaged/torch-lib)
![PyPI](https://img.shields.io/pypi/v/torch_lib?color=blue)
![Stars](https://img.shields.io/github/stars/johncaged/torch-lib?color=ff69b4)

> **WARNING: this repository is deprecated, see https://github.com/johncaged/TorchSlime for latest updates.**

## 介绍

torch_lib 是一个基于深度学习框架 PyTorch 的开源训练库，对训练 pipeline 提供了一系列标准化的流程和预定义的接口，用于简化训练代码的编写，提高开发效率。

## 特性

### 快速构建

根据实际需要实现接口、配置参数，快速构建训练pipeline。

### 完全可定制化

torch_lib 使用组合模式构建一套标准的训练流程（具体详见xx部分）。除此之外，您还可以对这个流程进行自定义修改，灵活度相较于市面已有框架大幅度提高。

### 清晰可视化

torch_lib 支持清晰的控制台可视化功能，可以实现训练流程监控、模型结构预览等。

### 渐进式

torch_lib 与普通 PyTorch 代码完全兼容，您可以自由地使用 torch_lib 的部分实用工具或整个框架。

## 快速上手

此部分默认您已经熟悉 PyTorch 的基本训练流程。

### 模型与数据集

对于一个完整的 PyTorch 训练流程，模型和数据集的处理是必不可少的。在 torch_lib 中，此部分与 PyTorch 原生代码没有区别。


```python
from torch.nn import Module
from torch.utils.data import DataLoader

model: Module = Model()  # 标准PyTorch模型
dataset: DataLoader = DataLoader()  # 标准PyTorch数据集
```

### 开始训练

调用训练的过程分为三步：创建代理（Proxy）类、build 构建、train（eval、predict）。**此示例适用单输入单输出任务，即数据集的格式为（输入数据，标签），如果想要进行更复杂的任务构建，请阅读完此章节后继续阅读xx章节。**

#### 创建代理（Proxy）类

根据尽量使用关联而不是继承的原则，代理类只是对模型的一些行为进行调用，最大程度解耦合。

```python
from torch_lib import Proxy

# 此部分将 PyTorch 模型包裹起来，device 参数用于指定训练设备，当然也可以后续再进行设置。
proxy = Proxy(model, device='cpu')
```

#### build构建

使用 build 是为了方便进行一些通用不变的配置，比如评价指标（在训练、验证和测试的过程中往往使用相同的评价指标，因此只需要配置一次）。

```python
# 具体参数使用详见 API 文档。
proxy.build(
    loss=None,  # PyTorch 损失函数
    metrics=None,  # torch_lib 评价指标
    optimizer=None,  # PyTorch 优化器
    lr=None,  # 学习率
    lr_decay=None,  # 学习率衰减
    optimizer_options=None,
    lr_decay_options=None,
    data_parser=None  # 用于多输入多输出的数据转换器
)
```

训练流程必要的参数有损失函数和优化器，因此只需配置这两项即可开始训练。

```python
from torch.nn import CrossEntropyLoss
from torch.optim.adam import Adam

# 仅为示例，需要根据实际情况选择合适的损失函数和优化器。
proxy.build(
    loss=CrossEntropyLoss(),
    optimizer=Adam(model.parameters(), lr=1e-4)
)
```

#### 开始训练 / 评估 / 预测（train / eval / predict）

使用 build 配置好必要参数后，接下来就可以调用训练流程。此处仅以 train 为例。

```python
proxy.train(
    train_dataset=dataset,
    total_epochs=10
)
```

至此，torch_lib 配置和调用训练的基本流程就完成了。想要实现一些定制化流程和高级功能，可以继续阅读此文档的后续章节。

## 高级使用

想要熟练地使用 torch_lib 的高级功能，需要先理解 torch_lib 的一些核心概念。

### 核心概念


