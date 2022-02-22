from torch_lib.utils import Nothing, SingleConst


class Callback:
    """
    Base class for all the callbacks.
    """

    def __init__(self):
        self.instance_ctx = Nothing()

    def set_instance_ctx(self, instance_ctx):
        self.instance_ctx = instance_ctx
