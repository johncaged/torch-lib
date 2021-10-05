# torch-lib

#### 介绍
torch-lib是一个快速搭建pytorch模型训练过程的库，并且可以进行控制台训练过程可视化（与keras的风格相似）解放双手，值得拥有

#### 快速上手
此部分默认您已经熟悉pytorch的模型搭建过程和数据处理过程，若仍不熟悉请阅读pytorch官方文档进行学习

1.模型训练

```python
from torch_lib.core import fit
from torch.nn import Module
from torch.utils.data import DataLoader

model = Module() # 标准的pytorch建模过程，详见pytorch官方文档，此行代码仅表示model的类型
dataset = DataLoader() # 标准的数据集格式，详见pytorch官方文档，此行代码仅表示dataset的类型
loss_options = { 'option1': 'value' }
fit(model=model, train_dataset=dataset, epochs=100, loss_func='ce', optimizer='adam', metrics=None, learning_rate=1e-4, lr_decay='step', loss_options=loss_options, optimizer_options=None, lr_decay_options=None, device='cpu')

```
