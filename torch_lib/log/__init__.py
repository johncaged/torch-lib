from torch_lib.utils import Singleton


color_dict = {
    'r': 31,
    'g': 32,
    'y': 33,
    'b': 34,
    'w': 38
}

info_prefix = '[TORCH_LIB INFO]'
warn_prefix = '[TORCH_LIB WARN]'
error_prefix = '[TORCH_LIB ERROR]'


def color_format(*args, color: str, sep: str = ' '):
    color_prefix = '\033[%dm' % color_dict.get(color, 38)
    color_suffix = '\033[0m'
    return '%s%s%s' % (color_prefix, sep.join(args), color_suffix)


@Singleton
class Console:

    # TODO: 如果同时想要文件输出怎么设计
    def __init__(self):
        self._control = {
            'info': True,
            'warn': True,
            'error': True
        }

    def info(self, *args):
        self.output(info_prefix, *args, type='info', color='b')

    def warn(self, *args):
        self.output(warn_prefix, *args, type='warn', color='y')

    def error(self, *args):
        self.output(error_prefix, *args, type='error', color='r')

    def output(self, *args, type: str, color: str = 'w'):
        if self._control.get(type, False) is True:
            print(color_format(*args, color=color))


console = Console()
