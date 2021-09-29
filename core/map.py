from torch.nn import MSELoss, BCELoss, KLDivLoss, CrossEntropyLoss
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, LambdaLR, MultiStepLR, ExponentialLR, CyclicLR

from typing import Optional

from common.util import func_call


def get_loss_func(loss: str, loss_options: Optional[dict] = None):
    """
    损失函数的字符串-实例映射
    :param loss: 损失函数名字，一个字符串
    :param loss_options: 损失函数的参数（用字典做配置）
    :return: 实例化的损失函数
    """
    loss_dict = {
        'mse': MSELoss,
        'bce': BCELoss,
        'kld': KLDivLoss,
        'ce': CrossEntropyLoss
    }
    loss_func = loss_dict.get(loss, None)
    assert loss_func is not None, 'loss function not supported'
    return func_call(loss_func, arg_dict=loss_options)


def get_optimizer(optimizer: str, optimizer_options: Optional[dict] = None):
    """
    优化器的字符串-实例映射
    :param optimizer: 优化器的名字，一个字符串
    :param optimizer_options: 优化器的参数（用字典做配置）
    :return: 实例化的优化器
    """
    optimizer_dict = {
        'adam': Adam,
        'sgd': SGD
    }
    optimizer = optimizer_dict.get(optimizer, None)
    assert optimizer is not None, 'optimizer not supported'
    return func_call(optimizer, arg_dict=optimizer_options)


def get_scheduler(scheduler: Optional[str], scheduler_options: Optional[dict] = None):
    """
    学习率调度的字符串-实例映射
    :param scheduler: 学习率衰减方法的名字，一个字符串
    :param scheduler_options: 该方法的自定义参数配置
    :return: 实例化之后的学习率衰减调度器
    """
    if scheduler is None:
        return None
    scheduler_dict = {
        'step': StepLR,
        'lambda': LambdaLR,
        'multi_step': MultiStepLR,
        'exponential': ExponentialLR,
        'cyclic': CyclicLR
    }
    scheduler = scheduler_dict.get(scheduler, None)
    assert scheduler is not None, 'scheduler not supported'
    return func_call(scheduler, arg_dict=scheduler_options)
