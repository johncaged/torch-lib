from torch_lib.utils import Singleton


color_dict = {
    'r': 31,
    'g': 32,
    'y': 33,
    'b': 34
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

    def __init__(self):
        self._control = {
            'info': True,
            'warn': True,
            'error': True
        }

    def info(self, *args):
        self.output(color_format(info_prefix, *args, color='b'), type='info')

    def warn(self, *args):
        self.output(color_format(warn_prefix, *args, color='y'), type='warn')

    def error(self, *args):
        self.output(color_format(error_prefix, *args, color='r'), type='error')

    def output(self, *args, type: str):
        if self._control.get(type, False) is True:
            print(*args)


console = Console()
