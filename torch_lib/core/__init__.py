from typing import TypeVar
from torch_lib.utils import MultiConst, get_device, type_cast, MethodChaining
from torch_lib.context import Context
from torch_lib.core.handler import BatchHandler


MP = TypeVar('MP', bound='ModelProxy')


class ModelProxy:

    ctx = MultiConst()
    def __init__(self, model, device=None):
        # set context
        self.ctx = Context()
        # set device
        self.ctx.device = device if device is not None else get_device(model)
        # set model and apply type cast
        self.ctx.model = type_cast(model, self.ctx.device)

    def train(self):
        self.ctx.build.train(self.ctx)

    def predict(self):
        self.ctx.build.predict(self.ctx)

    def eval(self):
        self.ctx.build.eval(self.ctx)

    @MethodChaining
    def build_train(self) -> MP:
        self.ctx.build.train = BatchHandler()

    @MethodChaining
    def build_predict(self) -> MP:
        self.ctx.build.predict = BatchHandler()

    @MethodChaining
    def build_eval(self) -> MP:
        self.ctx.build.eval = BatchHandler()

