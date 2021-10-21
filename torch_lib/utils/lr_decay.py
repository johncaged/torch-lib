from typing import Optional, Union
from torch_lib.utils import func_call
from torch.optim.lr_scheduler import StepLR, LambdaLR, MultiStepLR, ExponentialLR, CyclicLR


def get_scheduler(scheduler: Union[str, None, object], scheduler_options: Optional[dict] = None):
    """
    学习率调度的字符串-实例映射
    :param scheduler: 学习率衰减方法的名字，一个字符串
    :param scheduler_options: 该方法的自定义参数配置
    :return: 实例化之后的学习率衰减调度器
    """
    if scheduler is None:
        return None
    # 非字符串类型则直接返回实例
    elif isinstance(scheduler, str) is not True:
        return scheduler

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
