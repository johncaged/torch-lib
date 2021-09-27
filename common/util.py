from typing import Optional


def func_call(
        func,
        arg_list: Optional[list] = None,
        arg_dict: Optional[dict] = None
):
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
    def call(func):
        return func_call(func, arg_list, arg_dict)
    return call
