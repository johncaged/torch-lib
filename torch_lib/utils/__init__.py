from typing import Optional, Union, Tuple, List
from torch.nn import Module
from torch import Tensor
from time import time


class Nothing:
    """
    用于torch-lib中空类型的特殊标识，与python中的None区分，防止冲突
    """
    def __init__(self):
        pass


def is_nothing(obj):
    """
    判断是否是Nothing类型
    :param obj: 需判断的对象
    :return: bool
    """
    return isinstance(obj, Nothing)


def func_call(
        func,
        arg_list: Union[list, tuple, None] = None,
        arg_dict: Optional[dict] = None
):
    """
    函数的智能调用，可以传入列表和字典，该方法将列表和字典进行展开作为函数的参数
    :param func: 调用的主函数
    :param arg_list: 调用的参数列表
    :param arg_dict: 调用的参数字典
    :return: 函数func的执行结果
    """
    if arg_list is None:
        if arg_dict is None:
            return func()
        else:
            return func(**arg_dict)
    else:
        if arg_dict is None:
            return func(*arg_list)
        else:
            return func(*arg_list, **arg_dict)


def func_check(
        arg_list: Union[list, tuple, None] = None,
        arg_dict: Optional[dict] = None
):
    """
    函数的批量智能调用，用于一系列函数都需要用同一个参数进行调用的情况
    :param arg_list: 参数列表
    :param arg_dict: 参数字典
    :return: 用于调用函数的元函数
    """
    def call(func):
        return func_call(func, arg_list, arg_dict)
    return call


def execute_batch(callback_list: Optional[list], *args, **kwargs):
    """
    批量执行函数列表
    :param callback_list: 函数列表
    :param args: 参数
    :param kwargs: 参数
    :return: None
    """
    if callback_list is None:
        return

    call = func_check(arg_list=args, arg_dict=kwargs)
    for callback in callback_list:
        if callable(callback):
            call(callback)


def dict_merge(dict_1, dict_2):
    """
    字典的合并
    :param dict_1: 字典1
    :param dict_2: 字典2
    :return: 合并之后的字典
    """
    dict_1 = type_check(dict_1, dict, {})
    dict_2 = type_check(dict_2, dict, {})
    return dict(dict_1, **dict_2)


def type_check(value, type_, default):
    """
    类型检查，如果传入的参数不属于某个类型，则返回默认值
    :param value: 需要检查的变量
    :param type_: 变量需要属于的类型
    :param default: 如果变量不属于该类型，则传入的默认值
    :return: 属于type_类的值
    """
    return value if isinstance(value, type_) else default


def get_device(obj: Union[Tensor, Module]):
    """
    获取对象所在的设备，适用于model、tensor
    :param obj: tensor或model
    :return: obj所在的设备
    """
    if isinstance(obj, Module):
        parameter = next(obj.parameters(), None)
        return parameter.device if parameter is not None else None
    elif isinstance(obj, Tensor):
        return obj.device
    else:
        return None


def get_dtype(obj: Union[Tensor, Module]):
    """
    获取对象的数据类型，适用于model、tensor
    :param obj: tensor或model
    :return: obj所属的数据类型
    """
    if isinstance(obj, Module):
        parameter = next(obj.parameters(), None)
        return parameter.dtype if parameter is not None else None
    elif isinstance(obj, Tensor):
        return obj.dtype
    else:
        return None


def cast(obj: Union[Tensor, Module, tuple, list, None], device=None, dtype=None) -> Union[Tuple[Tensor, Module], Tensor, Module, None]:
    """
    数据类型和设备的转换，适用于model、tensor
    :param obj: tensor或model
    :param device: 需要转换的设备
    :param dtype: 需要转换的数据类型
    :return: 转换后的结果
    """
    obj = obj if isinstance(obj, (list, tuple)) else ((obj, ) if isinstance(obj, (Tensor, Module)) else obj)
    if isinstance(obj, (list, tuple)) is False:
        return obj
    if device is not None:
        obj = [item.to(device=device) for item in obj]
    if dtype is not None:
        obj = [item.to(dtype=dtype) for item in obj]
    obj = tuple(obj)
    return obj if len(obj) > 1 else obj[0]


def to_number(number_like):
    """
    将torch.Tensor类型转换为普通数字
    :param number_like:
    :return:
    """
    if isinstance(number_like, Tensor):
        return number_like.tolist()
    else:
        return number_like


def time_format(_time: float) -> str:
    """
    时间格式化
    :param _time: 待格式化时间
    :return: 格式化后的时间
    """
    if _time < 0:
        return '--'
    # 转换为秒
    _time = int(_time)

    m, s = divmod(_time, 60)
    h, m = divmod(m, 60)
    return '%d:%02d:%02d' % (h, m, s)


def list_to_str(arr: Union[list, tuple], sep=''):
    """
    列表转换成字符串
    :param arr: 列表
    :param sep: 分隔符
    :return: 字符串
    """
    return sep.join(str(i) for i in arr)


def unpack(data: Union[tuple, list, object], output_len: int):
    """
    元组智能拆包
    :param data: 待拆包元组
    :param output_len: 输出长度
    :return: 拆包后的元组（长度为output_len）
    """
    data = data if isinstance(data, (tuple, list)) else (data, )
    data_len = len(data)

    if data_len == output_len:
        return data if output_len > 1 else data[0]
    elif data_len > output_len:
        return data[0:output_len] if output_len > 1 else data[0]
    else:
        return tuple([data[i if i < data_len else data_len - 1] for i in range(output_len)])


def build_from_dict(self: object, kwargs: dict, required_params: Union[list, tuple, set, None] = None):
    """
    通过字典构建对象的参数
    :param self:
    :param kwargs:
    :param required_params:
    :return:
    """
    if required_params is None:
        required_params = set()
    assert set(kwargs.keys()) >= set(required_params), 'class init params missing'
    self.__dict__.update(kwargs)


def list_take(list_like, index: Union[List[int], Tuple[int], int]):
    """
    根据下标获取列表元素（或子列表）
    :param list_like: list、tuple或其他可取下标的类
    :param index: 元素索引
    :return: 一个列表元素或子列表
    """
    if index is None:
        return Nothing()
    elif isinstance(index, int):
        return list_like[index]
    elif isinstance(index, (list, tuple)):
        # TODO:list类型效率问题
        temp = []
        for i in index:
            temp.append(list_like[i])
        return tuple(temp)


class TimeRecord:
    """
    记录运行时间
    """

    def __init__(self):
        self.begin = 0
        self.end = 0

    def __enter__(self):
        self.begin = time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time()

    def time(self):
        return self.end - self.begin

    def __float__(self):
        return float(self.end - self.begin)
