from torch.optim import Adam, SGD, Optimizer, Adagrad
from typing import Optional, Union
from torch_lib.utils import func_call


def get_optimizer(optimizer: Union[str, Optimizer, None], optimizer_options: Optional[dict] = None):
    """
    优化器的字符串-实例映射
    :param optimizer: 优化器的名字，一个字符串
    :param optimizer_options: 优化器的参数（用字典做配置）
    :return: 实例化的优化器
    """
    if optimizer is None:
        return None
    elif isinstance(optimizer, Optimizer):
        return optimizer

    optimizer_dict = {
        'adam': Adam,
        'sgd': SGD,
        'ada': Adagrad
    }
    optimizer = optimizer_dict.get(optimizer, None)
    assert optimizer is not None, 'optimizer not supported'
    return func_call(optimizer, arg_dict=optimizer_options)
