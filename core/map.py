from torch.nn import MSELoss, BCELoss, KLDivLoss, CrossEntropyLoss
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, LambdaLR, MultiStepLR, ExponentialLR, CyclicLR

from typing import Optional

from common.util import func_call


def get_loss_func(loss: str, loss_options: Optional[dict] = None):
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
    optimizer_dict = {
        'adam': Adam,
        'sgd': SGD
    }
    optimizer = optimizer_dict.get(optimizer, None)
    assert optimizer is not None, 'optimizer not supported'
    return func_call(optimizer, arg_dict=optimizer_options)


def get_scheduler(scheduler: Optional[str], scheduler_options: Optional[dict] = None):
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
