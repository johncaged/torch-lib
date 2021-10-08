

_warning = True


def set_warning(use_warning: bool):
    global _warning
    _warning = use_warning


def _warning_output(*args):
    print('\033[31m', 'torch_lib warning:', *args, '\033[m')


class _CastWarning:
    """
    数据类型转换警告
    """

    def __init__(self):
        self._output = False

    def warn(self, dtype_1, dtype_2):
        if dtype_1 == dtype_2:
            return
        if self._output is False and _warning is True:
            self._output = True
            info = 'Your model and tensor are not of the same data type, and this may cause unnecessary data type casting. ' \
                   'If you do not want to see such warnings like this, use torch_lib.utils.warning.set_warning(False) to disable warning tips.'
            _warning_output(info)


cast_warning = _CastWarning()
