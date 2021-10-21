from typing import Optional, Union
from torch.nn import MSELoss, BCELoss, KLDivLoss, CrossEntropyLoss, Module
from torch_lib.utils import func_call


def get_loss_func(loss: Union[str, Module, None]) -> Optional[Module]:
    """
    损失函数的字符串-实例映射
    :param loss: 损失函数名字，一个字符串
    :return: 实例化的损失函数
    """
    if loss is None:
        return None
    # 自己实例化损失函数则返回它本身
    elif isinstance(loss, Module):
        return loss

    loss_dict = {
        'mse': MSELoss,
        'bce': BCELoss,
        'kld': KLDivLoss,
        'ce': CrossEntropyLoss
    }
    loss_func = loss_dict.get(loss, None)
    return func_call(loss_func) if loss_func is not None else None
