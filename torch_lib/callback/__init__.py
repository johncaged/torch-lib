from torch_lib.utils import func_call
from typing import Optional


def execute(callback_list: Optional[list], *args, **kwargs):
    """
    执行回调函数
    :param callback_list: 回调函数列表
    :param args: 参数
    :param kwargs: 参数
    :return: None
    """
    if callback_list is None:
        return

    for callback in callback_list:
        if callable(callback):
            func_call(callback, arg_list=args, arg_dict=kwargs)
