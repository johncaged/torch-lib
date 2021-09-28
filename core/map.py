from torch.nn import MSELoss, BCELoss, KLDivLoss, CrossEntropyLoss
from torch.optim import Adam, SGD

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
