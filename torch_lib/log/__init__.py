from abc import abstractmethod


def _warning_output(*args):
    """
    警告输出样式
    :param args: 待输出的内容
    :return: None
    """
    print('\033[31m', 'torch_lib warning:', *args, '\033[m')


def _info_output(*args):
    """
    日志类型的输出样式
    :param args: 待输出的内容
    :return: None
    """
    print('torch_lib info:', *args)


def refresh_output(*args):
    """
    控制台刷新输出样式
    :param args: 待输出的内容
    :return: None
    """
    print('\r', *args, end='', flush=True)


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
