from torch_lib.log import _Info, _info_output, refresh_output


_info = True


def set_info(use_info: bool):
    global _info
    _info = use_info


class _DeviceInfo(_Info):
    """
    设备信息日志
    """

    def __init__(self):
        super(_DeviceInfo, self).__init__()

    def info(self, device):
        if self._already is False and _info:
            self._already = True
            _info_output('using device', device, 'to train and predict')


class PlainInfo(_Info):
    """
    普通输出控制流
    """

    def __init__(self, console_print: bool = True):
        super(PlainInfo, self).__init__()
        # 用于控制是否控制台打印信息
        self.console_print = console_print

    def info(self, *args, mode: str = 'p'):
        """
        输出信息
        :param args: 输出内容
        :param mode: 输出模式：'p'--普通打印， 'r'--控制台刷新
        :return: None
        """
        if self.console_print:
            if mode == 'p':
                print(*args)
            elif mode == 'r':
                refresh_output(*args)


device_info = _DeviceInfo()
