from typing import Optional


def func_call(
        func,
        arg_list: Optional[list] = None,
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
        arg_list: Optional[list] = None,
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
    if value is not None and isinstance(value, type_):
        return value
    else:
        return default