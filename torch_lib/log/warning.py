from torch_lib.log import _warning_output, _Warning

_warning = True


def set_warning(use_warning: bool):
    global _warning
    _warning = use_warning


class _CastWarning(_Warning):
    """
    数据类型转换警告
    """

    def __init__(self):
        super(_CastWarning, self).__init__()

    def warn(self, dtype_1, dtype_2):
        if dtype_1 == dtype_2:
            return
        if self._already is False and _warning:
            self._already = True
            warning = 'Your model and tensor are not of the same data type, and this may cause unnecessary data type casting. ' \
                      'If you do not want to see such warnings like this, use torch_lib.log.warning.set_warning(False) to disable warning tips.'
            _warning_output(warning)


cast_warning = _CastWarning()
