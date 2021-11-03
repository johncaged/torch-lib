from abc import abstractmethod
from torch_lib.utils import list_to_str, TimeRecord
from typing import Union
from torch_lib.utils import time_format


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


def refresh_output(*args, end=''):
    """
    控制台刷新输出样式
    :param args: 待输出的内容
    :param end: 打印结束符
    :return: None
    """
    print('\r', *args, end=end, flush=True)


def color_format(*args, color: str):
    color_dict = {
        'r': 31,
        'g': 32,
        'y': 33,
        'b': 34
    }
    color_index = color_dict.get(color, 38)
    return ['\033[%dm' % color_index, *args, '\033[0m']


def progress(current_step, total_steps, *args, step_time: Union[float, TimeRecord, None] = None, progress_len: int = 25, output: bool = True, newline: bool = False):
    """

    :param current_step:
    :param total_steps:
    :param args: 其余需要打印的参数
    :param step_time: 运行一步所需要的时间
    :param progress_len: 控制条的长度
    :param output: 直接打印结果还是返回字符串
    :param newline: 结束是否换行
    :return:
    """
    assert 0 <= current_step <= total_steps, 'current_step should be between 0 and total_steps'
    # 计算当前epoch的进度
    rate = int(current_step * progress_len / total_steps)
    info = '%d/%d [%s%s] ' % (current_step, total_steps, '=' * rate, '-' * (progress_len - rate))
    # 计算ETA
    if step_time is not None:
        step_time = float(step_time)
        info += list_to_str(color_format('ETA: %s ' % time_format(step_time * (total_steps - current_step)), color='b'))
    if output:
        end = '\n' if current_step == total_steps and newline is True else ''
        refresh_output(info, *args, end=end)
    else:
        return info + list_to_str(args)


class Progress:

    def __init__(self, obj):
        pass

    def __iter__(self):
        pass

    def __next__(self):
        pass


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
