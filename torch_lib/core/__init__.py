from torch_lib.utils import MultiConst, get_device, type_cast
from torch_lib.context.common import RunContext


class ModelProxy:

    config = MultiConst()
    def __init__(self, model, device=None):
        # set device
        self.device = device if device is not None else get_device(model)
        # set model and apply type cast
        self.model = type_cast(model, self.device)
        # set config
        self.config = RunContext()

    def fit(self):
        self.model.train()
        pass

    def evaluate(self):
        pass

