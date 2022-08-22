# torch_lib

![License](https://img.shields.io/github/license/johncaged/torch-lib)
![PyPI](https://img.shields.io/pypi/v/torch_lib?color=blue)
![Stars](https://img.shields.io/github/stars/johncaged/torch-lib?color=ff69b4)

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

### Core

#### /handler.py

##### 1. from abc import abstractmethod:
从abc模块导入抽象方法注释，可以用于对一个方法进行标注，从而不用过多关心方法本身的内容，只需要知道有这个方法即可，不需要实现。而在子类继承的时候实现它即可，用子类实例化即可调用子类的具体方法来执行相应操作。

##### 2. from typing import Dict, Sequence, Union: 

###### (1) Dict：
dict的泛型(generic)版本，用于注解(annotate)返回类型。注解参数时，最好使用抽象集合类型(abstract collection type)，例如Mapping。Dict与dict之间没有真正的区别，但是Dict是泛型类型，它允许你指定key和value的类型，使其更加灵活。例如：
``` python
    def count_words(text: str)->Dict[str, int]: ...
    x: Dict[str, int] = {"Changsha", 1}
    print(x)
```

###### (2) Sequence: 
collections.abc.Sequence的泛型版本

###### (3) Union: 
联合类型，Union[X, Y]等价于X|Y，意味着X或Y。使用形如Union[int, str]的形式来定义一个联合体：参数必须是某种类型，且至少有一个；联合类型的联合类型会被展开(flattened)；仅有一个参数的联合类型就是该参数自身；冗余的参数会被跳过(skipped)；在比较联合类型的时候，参数顺序会被忽略(ignored)；不能继承或者实例化一个联合类型；不支持Union[X][Y]这种写法。例如：
``` python
    # 联合类型的联合类型会被展开（flattened)
    Union[Union[int, str], float] == Union[int, str, float]
    # 仅有一个参数的联合类型就是该参数本身
    Union[int] = int
    # 冗余的参数会被跳过（skipped）
    Union[int, str, int] = Union[int, str]
    # 在比较联合类型的时候，参数顺序会被忽略（ignored）
    Union[int, str] == Union[str, int]
```

##### 3. from functools import wraps: 
由于函数本身也是一个对象，而且也可以赋值给别人，可以被调用。为了增强函数本身的功能，比如在函数调用前后做一些事情且不改变原有函数的定义，就需要用到装饰器。用装饰器去处理函数后，原函数的__name__等属性也会被装饰器的替代，所以需要用functools.wraps去处理，将原函数的属性再次替换回装饰器中。例如：
``` python
    import functools
    
    def log(text):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kw):
                print('%s %s():' % (text, func.__name__))
                return func(*args, **kw)
            return wrapper
        return decorator
```
比如写了一个函数：
``` python
>>> def loveyou()
...     print('2020-9-14')
```
则通过添加@log(‘execute’)的注释，可以使得输出变为下图，即在原本的打印前输出调用执行了的函数名：
``` python
execute loveyou()
2020-9-14
```
##### 4. def TorchGrad(func)：
是一个装饰器，用于在具体函数执行前，设置梯度计算模式的开闭。set_grad_enabled(mode)中的mode参数是一个布尔值，通过T或者F来控制开闭。通过获取上下文中TRAIN表示的对象中的状态，来设置是否需要梯度计算模式。所谓的梯度计算就是对向量的计算过程中产生的梯度进行保留，用于反向传播。
``` python
    def TorchGrad(func):
        """
        Set grad enabled or not according to the context mode.
        """
        @wraps(func)
        def grad_switch(self, ctx: Context):
            # only when context status is in ['TRAIN'] is the grad enabled
            with set_grad_enabled(str(ctx.status) in ['TRAIN']):
                func(self, ctx)
        return grad_switch
```
##### 5. 对于矩阵求导，现在的模型是这样的：
![pic1](https://thumbnail0.baidupcs.com/thumbnail/9520b0c1epec22c814d7dcd42b12f10e?fid=2939904872-250528-629394732297922&time=1661173200&rt=sh&sign=FDTAER-DCb740ccc5511e5e8fedcff06b081203-PeRpTj0GZHc4%2Bo%2FRaRdbg9JLyjo%3D&expires=8h&chkv=0&chkbd=0&chkpc=&dp-logid=8883216842698396187&dp-callid=0&file_type=0&size=c710_u400&quality=100&vuk=-&ft=video)

<p>则计算结果如下：</p>

![pic2](https://thumbnail0.baidupcs.com/thumbnail/d243098d9r012b0717a6878594107fe2?fid=2939904872-250528-985143410624318&time=1661173200&rt=sh&sign=FDTAER-DCb740ccc5511e5e8fedcff06b081203-atLlKZmc6hy5seOxQyCgzsen72A%3D&expires=8h&chkv=0&chkbd=0&chkpc=&dp-logid=8883234007716238596&dp-callid=0&file_type=0&size=c710_u400&quality=100&vuk=-&ft=video)

##### 6. Handler类：
是所有处理器的基类，定义了一个__init__函数，一个抽象处理函数handle，以及一个__call__函数。其中__call__函数的作用是使类实例对象可以像调用普通函数那样，以“对象名()”的形式使用。例如：
``` python
    class Test:
        # 定义__call__方法
        def __call__(self, name, add):
            print("调用__call__方法", name, add)
    test = Test()
    test("Johncage", "Shark")
```

##### 7. EmptyHandler类：
是一个什么处理都不去做的处理器，继承自Handler，使用了InvocationDebug注释唤醒了Debug信息的输出。

##### 8. C_SEQ: 
是一个类型声明，可能是一个处理器，或者是一个由一系列处理器组成的序列。

##### 9. HandlerContainer类: 
处理器容器，采用了组合模式的设计理念，继承自Handler和BaseList，初始化时传入的handlers是C_SEQ类型的数据，通过BaseList的初始化确保生成一个列表。然后重载的handle函数将用自己内部包含的每一个处理器去对上下文进行处理。

##### 10. EpochIterationHandler类：
轮迭代处理器，继承自HandlerContainer，进行epoch.total次的操作，取每一轮，调用父类的handle方法，即把Container里每个处理器按顺序拿出来执行

##### 11. IterationHandler类：
步迭代处理器，继承自HandlerContainer，激活梯度计算模式，如果上下文中存在数据内容，就将向上下文的step中更新这一次的batch、progress等参数。

##### 12. ForwardHandler类：
前馈处理器，继承自HandlerContainer，将上下文中的数据解释后分别取出赋给x, y_true和extra，其中y_true作为标签进行类型转换，而x作为输入则进入模型的预测，并输出结果为y_pred，上下文中更新x，y_true等内容，方便后续计算Loss以及相关指标。

##### 13. LossHandler类：
损失处理器，继承自HandlerContainer，通过上下文中获取到的损失函数，将每个step的y_pred于y_true作为参数输入，得到损失结算结果，然后更新该步的损失值。注意run.loss是一个计算函数，而step.loss是一个存储的值。

##### 14. BackwardHandler类：
反向传播处理器，继承自HandlerContainer，总步数模梯度积累用于判断是否除以完整的grad_acc，剩下的步数是不是小于它，确保其记录的是这个loss对应的步数。然后loss/grad_acc， 再进行反向传播。

##### 15. OptimizerHandler类：
优化器处理器，继承自HandlerContainer，如果有优化器且处在每个梯度积累的最终步，则进行优化操作。

##### 16. MetricsHandler类：
指标处理器，继承自Handler，计算上下文中的指标们。

##### 17. AverageHandler类：
平均处理器，继承自Handler，用于处理各种计算结果的平均。

###### (1) average函数：
用来求损失和指标的平均值并保存回上下文中。

###### (2) clear函数：
用来重置之前求出的各种平均值。

###### (3) _compute_avg_loss函数：
计算summary信息中的loss结果与count结果的相除结果。

###### (4) _compute_avg_metrics函数：
计算summary信息中的各种指标的计算结果与count结果的相除结果。

##### 18. DisplayHandler类：
显示处理器，继承自Handler，用于设定终端内的显示属性，控制光标等操作。

##### 19. DatasetHandler类：
数据集处理器，继承自Handler，通过调用上下文状态的get_dataset函数，为对应的状态取出对应的数据集。

##### 20. StatusHandler类：
状态处理器，继承自Handler，设定状态并对上下文状态进行修改

##### 21. LRDecayHandler类：
学习率衰减处理器，继承自Handler，如果设置了衰减，则将调用run.lr_decay.step进行衰减操作

##### 22. BeginHandler等六个相关类：
开始、结束、步开始、步结束、轮开始、轮结束处理器，继承自Handler，是回调适配器，标志着回调的开始结束。

#### /context.py

##### 1. Context类：
上下文类，继承自Base类，贯穿于整个生命周期。有run, epoch, step, handler, custom, inner这六个MultiConst实例对象。其中当一个上下文被实例化，就会对这六个实例对象重新赋值为对应的上下文类对象。除此之外的device，model，status，dataset属性暂时为NOTHING，但是都各自具有类型。

###### （1）ctx_check函数：
内部有一个_check函数，调用Base类里的check函数来检测传入的项是否存在，silent参数表示的是输出的信息是debug（默认不输出）还是警告。然后在_check外，进行判断，如果传入的items是由list和tuple组成的元组实例化的对象，则遍历该对象的每一个项，然后进行判断，否则，即传入的items只是个单独的值，就只需要对它本身进行检测即可。

##### 2. TempContext类：
临时上下文类，或者叫做空上下文类，继承自Base类，用于定义一个初始化的方法来迅速重置上下文。Initialize方法是一个抽象方法。

##### 3. StepContext类：
步阶上下文，继承自临时上下文类，重写initialize方法，包含x, y_pred等多个与模型训练一个step相关的属性，并初始化为NOTHING。

##### 4. EpochContext类：
轮上下文，继承自临时上下文类，重写initialize方法，包含total, current等多个与模型训练一轮相关的属性，并初始化为NOTHING。

##### 5. RunContext类：
运行上下文，继承自临时上下文类，重写initialize方法，包含train, eval, predict等多个与实际运行torch-lib的训练、评估、预测有关的属性，并初始化为对应需要的值。

##### 6. HandlerContext类：
处理器上下文，继承自临时上下文类，重写initialize方法，包含Container，EpochIteration等各式各样的处理器。

##### 7. CustomContext类：
惯常上下文，继承自临时上下文类，重写initialize方法，将类内存储的属性字典清空，并输出信息“惯常上下文已被初始化“

##### 8. InnerContext类：
内部上下文，继承自临时上下文类，重写initialize方法，将类内存储的属性字典清空，并输出信息“内部上下文已被初始化“

#### /status.py

##### 1. proxy_status：
注册一个叫做proxy_status的内容。

##### 2. Status类：
状态类，声明了一系列方法用于设置模型的模式、获取数据集、获取损失与指标等操作，在_get_avg_inner_init_item函数中，返回一个初始化的字典内容，包括了计数、损失和指标字典。

##### 3. TrainStatus和EvalStatus类：
训练状态和评估状态，都继承自Status类，分别完善了基类的空方法，对实施计算展示的get系列函数进行了定义。

##### 4. ValStatus和PredictStatus类：
验证状态和预测状态，都继承自EvalStatus类，验证状态其实就是每一轮结束后的评估状态，所以原理就如评估状态。预测状态与评估状态是一致的。

#### /__init__.py

##### 1. T=TypeVar(‘T’, bound=’Proxy’)：
用泛型，T可以被允许的类型是Proxy类以及其子类。

##### 2. DATASET：
一个类型声明，可以是DataLoader类或者DataProvider类

##### 3. Proxy类：
代理类，继承自Context类。

###### (1) __init__函数：
先用Context的初始化函数，除此之外还会设置一下使用设备、模型。最后用链式调用训练、预测和评估的过程建立函数。

###### (2) train函数：
激活Debug信息的输出。需要参数如下：
``` python
    train_dataset: DATASET, # 训练所需的数据集
    total_epochs: int = 1,	# 训练总的轮数，默认为1
    eval_dataset: DATASET = NOTHING,	# 评估所需的数据集，默认没有
    callbacks: C_SEQ = NOTHING, # 回调序列，默认没有
    grad_acc: int = 1, # 梯度累积，默认为1
```
在设置完一系列参数并打印日志后，调用run的train（即一系列训练相关的Handler操作）进行训练过程。

###### (3) predict函数：
激活Debug信息输出，原理类似于train

###### (4) eval函数：
激活Debug信息输出，原理类似于train

###### (5) count_params函数：
激活Debug信息输出，调用util的count_params对模型参数进行统计打印

###### (6) build函数：
激活Debug信息输出，采用方法链。参数如下：
``` python
    loss = None,  # 损失函数，默认为None
    metrics: M_SEQ = None,  # 指标函数的队列，默认为None
    optimizer: Union[str, Optimizer] = None,  # 优化器，可以自己定义优化器，也可以用torch自带的，通过字符串获取，默认为None
    lr: NUMBER = None,   # 学习率，默认为None
    lr_decay: Any = None,   # 学习率衰减，默认为None
    optimizer_options: Optional[Dict] = None,  # 优化器选项， 默认为None
    lr_decay_options: Optional[Dict] = None,   # 学习率衰减选项， 默认为None
    data_parser: Optional[DataParser] = None   # 数据解释器，默认为None
```

###### (7) build_train函数：
激活Debug信息输出，采用方法链。该方法将对run.train进行赋值，利用一个Container来实例化，结构如下：
<p>▻开始</p>
<details>
<summary>▻轮迭代</summary>
   <p>▻▻轮开始</p>
   <p>▻▻轮状态</p>
   <p>▻▻轮数据集获取</p>
   <p>▻▻轮清除上一轮指标结果</p>
   <details><summary>▻▻步迭代</summary>
      <p>▻▻▻步开始</p>
      <p>▻▻▻步前向传播</p>
      <p>▻▻▻步计算损失</p>
      <p>▻▻▻步优化（包括反向传播）</p>
      <p>▻▻▻步计算指标</p>
      <p>▻▻▻步计算平均结果</p>
      <p>▻▻▻步展示结果或保存文件</p>
      <p>▻▻▻步结束</p>
    </details>
   <p>▻▻轮学习率衰减</p>
   <p>▻▻轮开始验证集处理</p>
   <p>▻▻轮获取数据集</p>
   <p>▻▻轮清除上一轮验证指标</p>
   <details><summary>▻▻轮验证迭代</summary>
      <p>▻▻▻轮验证前向传播</p>
      <p>▻▻▻轮验证计算损失</p>
      <p>▻▻▻轮验证计算指标</p>
      <p>▻▻▻轮验证计算平均结果</p>
      <p>▻▻▻轮验证展示结果或保存文件</p>
    </details>
   <p>▻▻轮结束</p>
</details>
<p>▻结束</p>

###### (8) build_eval函数：
激活Debug信息输出，采用方法链。该方法将对run.eval进行赋值，利用一个Container来实例化，结构原理同上。

###### (9) build_loss等一系列其他的build函数：
激活Debug信息输出，如果损失函数不为空则赋值损失函数给run下对应的内容中。

##### 4. 关于梯度积累：
机器不够、内存不够用。在深度学习训练的时候，数据的batch size大小受到GPU内存限制，batch size大小会影响模型最终的准确性和训练过程的性能。在GPU内存不变的情况下，模型越来越大，那么这就意味着数据的batch size智能缩小，这个时候，梯度累积（Gradient Accumulation）可以作为一种简单的解决方案来解决这个问题。大的Batch size可能陷入局部最小值，小的Batch size可能导致算法学习收敛速度慢。梯度累积就是先按顺序执行Mini-Batch，同时对梯度进行累积，累积的结果在最后一个Mini-Batch计算后求平均更新模型变量。

### Util

#### /__init__.py

##### 1. import threading以及from time import time：
考虑到python解释器通过全局解释器锁GIL来保证任意时刻只允许单个线程执行，而在机器学习上我们更需要并行计算来充分利用硬件环境，所以我们通过这两个模块来实现多线程。

##### 2. def Singleton(cls): 
单例装饰器，用于将类变成单例模式，每次在整个程序中只能实例化一个对象或者实例化的对象是同一个，保证了线程安全。用threading.Lock()函数获取一个线程锁对象，再建立一个_instance的空字典用于存放类以及对应的对象，如果传入的类不在这个字典中，表示还没有这么一个对象出现，就需要把传入的参数用于实例化，并把实例化对象和这个类成对放入字典中。无论类是否在字典中出现，都需要在最后返回这个对象。

##### 3.def InvocationDebug(module_name)：
debug信息装饰器，用于激活某个方法是否需要激活debug的信息输出，具体的实现是用debug输出debug部分的模块名以及begin提示，然后保留函数本身的内容执行，再在最后输出模块名以及end提示。

##### 4. Nothing类：
是一个单例类，生成的对象与python自带的None不同，常常由从某个目标对象获取属性或项的时候，该对象不含有或者仅仅是代表默认值。Nothing允许任何属性获取或者方法调用的操作不报错，使得程序更加稳定，相对的，会在控制台打印出对应的告警信息。该类内部的所有方法都进行相关的定义。在程序开始运行之时就会直接生成一个Nothing对象。

##### 5. def is_nothing(obj): 
用于判断一个对象是否是Nothing，返回类型为布尔值，即如果判断的对象是Nothing的实例，则返回True。

##### 6. def check_nothing(obj, x, y=NOTHING)：
如果传入的obj不是Nothing，则返回x，否则返回y。

##### 7. def dict_merge(dict1: Dict, dict2: Dict)：
用**结构两个字典并重新拼合在一起。

##### 8. def safe_divide(dividend, divisor): 
返回一个不会报错的除法，即对除数进行判断，除数为零时默认返回的结果为0，否则就正常除法运算。

##### 9. Base类：
是一个用于使子类可以使用”[ ]”运算符的基类，该操作类似于python自带的字典操作。且Base类中对于获取属性等发现目标不存在时不报错，而是返回Nothing。除此之外，该类允许子类用字典来指派属性。

###### (1) from_dict函数：
通过dict_merge函数将传入的dict参数中的键值对与原本类中存在的键值对合并。

###### (2) check函数：
判断某个对象是否具有某个确切的属性，一般在查询的时候用。支持用“.”操作来获取查看。传入的参数是字符串形式的描述。如果对象本身不存在这个属性，则返回False，或者报错的话，也返回False，否则返回True。

###### (3) process_exc函数：
通过日志器报错，并返回结果为Nothing的类内静态方法。

###### (4) 其他attr系列函数与item系列函数：
通过内置函数和特殊函数进行属性与项的增删改查。

##### 10. SingleConst类：
是一个用于定义不允许修改的常量的类，当将要被修改的时候，会显示告警信息，并且使改变不会生效。允许在初始化的时候设值为Nothing，表示将在后续对其进行重新赋值。一旦这个值不再是Nothing，那么就再也不可以被修改了。需要注意的是，一个类的属性不允许改变，表示的是对于同一个类，其每个实例对象都具有相同的属性，这种定义不适合在不同实例之间会发生变化的属性。内部方法比较简单，不多赘述。

##### 11. MultiConst类：
是表示不同实例之间的常量约束，与SingleConst类似。值得注意的是，MultiConst不是一种严格的模式，为了方便起见，它会在对象中使用前缀_来创建一个同名的私有属性，因此，这种约束无法阻止直接赋值给它所创建的私有属性。例如：
``` python
    class Example:
        attr = MultiConst()

    ex = Example()
    ex.attr = 'a' # 赋值成功
    ex.attr = 'b' # 赋值失败，无法对常量进行赋值.
    print(ex.attr) # 输出结果仍然是a.
    ex._attr = 'c' # **** 警告: 这种操作是允许的，因为MultiConst不阻止对该私有属性赋值 ****
    print(ex.attr) # 现在输出结果变成了c.
```
不要对一个已经用前缀'_'命名的属性使用MultiConst，因为MultiConst将创建一个以'__'为前缀的新属性，使得属性在类外无法通过'__foo'访问。

##### 12. Count类：
继承自SingleConst类，初始值为0。其中__get__方法将返回的是当前的值，然后再对内部值进行＋1操作。

##### 13. BaseList类：
继承自list类。初始化时，如果传入的参数为None或者Nothing，则返回一个空列表，否则，如果本身内容可迭代，就返回它自己，如果不可迭代，就在外加上一层中括号使其成为列表。

##### 14. def get_device(obj: T_M): 
获取模块或者张量的所使用的设备，如果传入的类型是模块，则首先查看模块有无参数，有的话获取模块的参数，否则获取None，再根据parameter的内容来决定是返回设备还是None。如果传入的类型是张量，则返回张量的设备。

##### 15. def get_dtype(obj: T_M): 
与上面的获取设备类似，来获取数据的类型。

##### 16. def type_cast(obj: T_M_SEQ, device=None, dtype=None)->Union[Tuple[Tensor, Module], Tensor, Module, None]: 
->后的内容表示返回的数据类型为模块或者张量或者由模块张量组成的元组或者None。该方法将T_M_SEQ强转为上述四种类型。如果传入的对象本身不是列表与元组组成的元组，即本身已经属于模块或张量了，就直接返回本身。否则进行如下操作，将obj内的所有元素都进行device与dtype的赋值，再将obj用tuple生成元组实例，如果元组长度大于1，则返回本身，否则返回元组的第一个元素。

##### 17. def list_take(list_like, index: Union[Sequence[int], int]): 
从一个列表形状的对象中，通过索引（或索引们）来获取内部项或者子列表。如果索引为None，则返回Nothing，否则根据索引（索引们）获取所需的内容。

##### 18. def MethodChaining(func)：
链式装饰器，用来循环调用自己。

##### 19. Iter类与IterTool类：
一对父子类，用来定义迭代器的工作。

##### 20. def count_params(model: Module, format: str=None, decimal: int=2): 
用来统计参数量，format参数用于设置显示的内容量级表示是原始、千、百万级。然后对模型的参数量进行计算统计并返回结果。

#### /type.py

##### 1. 文件说明：
存放了一些自定义的可以用一个类型来表示多个类型的类型，包括

###### (1) T_M：
Tensor或者Module

###### (2) T_M_SEQ：
Tensor或者Module或者是由T_M组成的序列

###### (3) NUMBER：
整型或者浮点型

###### (4) NUMBER_T：
由整型或浮点型组成的元组

#### /terminal.py

##### 1. 文件说明：
这个文件的作用是在控制台进行操控的，声明的包括ESC、CSI等这些变量都是为了处理光标、行、屏幕等的操作所需要的变量，该文件中的函数都是为了处理色彩、内容、行数、位置、可见性等所设立的。

#### /formatter.py

##### 1. ProgressStyle类：
这个类是用来设置命令行打印输出时候，进度条的样式的。声明类后在一个字典中实例化，目前设计的主要分为方块（|█ █ █   |），线（|━ ━ ━   |），箭头（|===> --|）三种进度条，使用者可以自己去添加或自定义。

##### 2. def progress_format(…): 
对进度条进行设置与定义，默认是方块条。包括对长度、样式、是否显示百分比、是否显示时间等进行设置。

##### 3. def period_time_format(_time: float)->：
第一个step的时候因为不清楚时间所以输出--，否则根据第一个step来计算每轮时间。

##### 4. def eta_format(…):
估算一轮所需要花费的时间。

#### /table.py [暂未更新]

### Module

#### /__init__.py

##### 1. 文件说明：
一个方便的模块注册工具，帮助你对网络结构进行参数化。

##### 2. RegistryMapper类：
登记映射表类，使用单例注解表示只允许有一张表。registries是一个SingleConst字典对象，表允许增删查三种操作，将登记名与登记信息绑定在一起。程序运行指出自动生成一个registry_mapper对象。

##### 3. Registry类：
登记类，每个登记对象初始化后会生成一个模块字典，对象本身会被放入登记映射表中。

###### (1) register函数：
内部包含一个装饰器。nonlocal只能在封装函数中使用，在外部函数先进行声明，在内部函数进行nonlocal声明，这样在内部函数中的变量与外部中的是同一个变量。这里的name就是传给register的name，如果name是空的，则获取给装饰器传入的类的名，然后向模块字典中加入名与类的对。

###### (2) build函数：
根据name获取模块，并向模块中传入参数，从而实例化该模块。

###### (3) build_single函数：
build一个模块。用两个assert断言，使在条件不满足程序运行的条件下直接返回错误，而不必等待程序运行后出现崩溃的情况。这两个断言分别对传入的item是否为字典以及’name’属性是否在item中进行判断，如果都满足要求，则对item的参数与字典参数解构，与item名字一起调用build函数，进行模块的实例化。

###### (4) build_sequential函数：
build一系列模块。对传入的可迭代对象进行遍历，对里面的每一个项进行build_single操作，并把实例化的结果用blocks这个列表包括起来。最后用torch,nn.Sequential函数将blocks的内容解构串联。

###### (5)get函数：
根据名字返回模块。

#### /config.py

##### 1. def load_json(path:str):
是一个读取json文件的函数，返回读取内容

### Metric

#### /__init__.py

##### 1.Metric类：
指标类，用于对模型评价指标进行定义及操作的类。初始化的时候会对名字赋初值。然后通过__call__内部方法，允许通过上下文来获取具体内容，然后返回一个字典。

##### 2.M_SEQ：
一个类型声明，表示是Metric或者是由一系列Metric组成的序列。

##### 3.MetricContainer：
继承自Metric和BaseList两个类，与HandlerContainer原理类似，不多赘述。

### Log

#### /__init__.py

##### 1.from ..util import Singleton：
从工具模块中导入自定义的Singleton装饰器，用于声明一些单例类。

##### 2. color_dict以及一系列xx_prefix：
这个类本身用于打印输出相关的日志，color_dict用于保存一些用得到的颜色参数，用字母来代替颜色，获取对应的整型数字。各种Prefix是用于输出时候的输出语句前用到的前缀，表示这条输出是某种类型的日志记录。

##### 3. def color_format(*args, color:str, sep:str=’ ’)：
‘\033[%dm’ % color_dict.get(color, 38)中前半部分是用Python控制输出颜色的语法，‘\033[显示方式;前景色;背景色m内容’，后半部分中从字典中通过输入的字母来获取，或者默认拿38。最终返回的结果同样用正则填充，用prefix来设置输出的色彩样式，sep的输出语句框架结合着args的具体输出参数作为prefix的m后面跟着的内容，最后再用后缀suffix恢复原样。

##### 4. Logger类：
###### (1) __init__函数：
存放日志告警等级对应的布尔值，目前的设置是debug信息不用输出，其他的需要。

###### (2) info、warn、error、debug函数：
根据告警信息严重性的不同设置不同的颜色与内容，通过调用output函数实现输出

###### (3) log函数：
直接输出所有传入进来的内容

###### (4) output函数：
默认显示字体颜色为白色，否则就根据传入的颜色参数去设置。同时根据传入的type来获取是否需要输出的布尔值，默认是不输出的。

###### (5) 运行效果：
当运行整个程序后，默认自动生成一个日志器。

#### /directory.py

##### 1. BASE_PATH等五个PATH变量：
用于存储torch-lib框架所需的一些必要路径，具体包含关系如下：
<details>
    <summary>▻BASE</summary>
    <details>
        <summary>▻▻NAMESPACE</summary>
        <p>▻▻▻runtime.log</p>
        <p>▻▻▻metrics.json</p>
        <p>▻▻▻xx.checkpoint</p>
    </details>
</details>

##### 2. def join_path(*args): 
根据输入的变量返回一个把这些变量连接成串路径的结果

##### 3. def set_base_path(path:str): 
若想在函数内部对函数外的变量进行操作，就需要在函数内部声明其为global，所以这里的BASE_PATH就是在最前面声明的BASE_PATH，将他赋值为输入的path，再检测是否存在，如果不存在则创立这个路径

##### 4. def safe_makedirs(path): 
判断路径是否存在，如果不存在则建立。

##### 5. def set_namespace(namespace: str): 
设置NAMESPACE的路径，拼接传入的namespace和BASE，然后判断形成的路径是否存在，不存在则新建，存在则报错提示可能会覆盖。该函数只对NAMESPACE进行修改，而不是把拼接结果赋值过去。

##### 6. def get_namespace_path(): 
获取拼接后的namespace路径。

##### 7. def get_log_path(): 
返回与LOG_PATH拼接的完整路径结果。

##### 8. def get_metric_path(): 
返回与METRIC_PATH拼接的完整路径结果。

##### 9. def get_checkpoint_path(): 
返回与CHECKPOINT_PATH拼接的完整路径结果。

### Data

#### /__init__.py

##### 1. DataProvider类: 
数据支持类，__call__函数内，根据上下文获取data_loader，如果其不是DataLoader的实例，则警报提示。最终返回这个数据加载对象。

##### 2. ConstantProvider类：
常量支持类，继承自DataProvider，重载get函数返回dataset属性。

##### 3. DataParser类：
数据解释器类，__call__函数内，根据上下文获取batch大小，如果batch不是元组或者长度不为3，则可能导致无法把数据完整打包，所以会警报。

##### 4. IndexParser类：
索引解释器类，继承自DataParser，包括三个参数，分别是x，y和extra，都是整型或者整型序列。get函数根据上下文获取batch大小，再调用util里的list_take函数将传入的数据按照batch大小截取返回。

### Callback

#### /__init__.py

##### 1. Callback类：
回调类，包括训练、验证和使用预测的时候，都可以添加回调操作的类。可以在上述操作、每一训练步骤、每一训练轮的开始结束添加回调。

##### 2. C_SEQ ：
一个类型声明，表示是回调或者由回调组成的序列

##### 3. CallbackContainer类：
回调容器类，继承自回调类和BaseList类，与HandlerContainer类似，不再赘述原理。对于容器内部的所有回调，执行容器的某个回调函数，就是单独把每个回调拿出来遍历执行。

#### /common.py

##### 1. EPOCH_SEQ：
类型声明，表示的是整型或者整型序列

##### 2. SaveCheckpoint类：
断点保存类，继承自Callback类，获取存放断点的路径，根据需要存储断点的频率要求，将模型的断点存放在路径下，这个操作将会由epoch_end来完成。而save_dict函数主要做的是根据传入的保存选项，来选择是对上下文的什么内容进行保存，比如是模型参数、优化器参数、还是轮数+1。

##### 3. SaveMetrics类：
指标保存类，继承自Callback类，获取存放指标的路径，根据需要存储指标的频率要求，将运行过程中的指标存放在路径下，这个操作将会由epoch_end来完成。而parse函数是根据传入的保存选项，来确定是更新训练集的还是验证集的loss与各种指标。Append_list函数则是对文件进行读写的函数。

#### /__init__.py

##### 1. 规定了版本号
