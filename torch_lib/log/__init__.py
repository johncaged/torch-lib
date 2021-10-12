from abc import abstractmethod


def _warning_output(*args):
    """
    警告输出样式
    :param args: 待输出的内容
    :return: None
    """
    print(*color_format('torch_lib warning:', *args, color='y'))


def _info_output(*args):
    """
    日志类型的输出样式
    :param args: 待输出的内容
    :return: None
    """
    print(*color_format('torch_lib info:', *args, color='b'))


def refresh_output(*args):
    """
    控制台刷新输出样式
    :param args: 待输出的内容
    :return: None
    """
    print('\r', *args, end='', flush=True)


def color_format(*args, color: str):
    color_dict = {
        'r': 31,
        'g': 32,
        'y': 33,
        'b': 34
    }
    color_index = color_dict.get(color, 38)
    return ['\033[%dm' % color_index, *args, '\033[0m']


class _Log:
    """
    所有控制台日志信息的基类
    """

    def __init__(self):
        # 用于一次性控制台输出，表示是否已经输出过了
        self._already = False


class _Warning(_Log):
    """
    控制台警告信息的基类
    """

    def __init__(self):
        super(_Warning, self).__init__()

    @abstractmethod
    def warn(self, *args, **kwargs):
        pass


class _Info(_Log):
    """
    控制台普通提示信息的基类
    """

    def __init__(self):
        super(_Info, self).__init__()

    @abstractmethod
    def info(self, *args, **kwargs):
        pass
