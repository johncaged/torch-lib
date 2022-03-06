from typing import TypeVar
from torch_lib.utils import MultiConst, get_device, type_cast, MethodChaining
from torch_lib.context import Context
from torch_lib.core.handler import BatchHandler


MP = TypeVar('MP', bound='ModelProxy')


class ModelProxy:

    config = MultiConst()
    def __init__(self, model, device=None):
        # set config
        self.config = Context()
        # set device
        self.config.device = device if device is not None else get_device(model)
        # set model and apply type cast
        self.config.model = type_cast(model, self.config.device)

    def fit(self):
        self.config.fit_handlers(self.config)

    def predict(self):
        self.config.predict_handlers(self.config)

    def evaluate(self):
        self.config.evaluate_handlers(self.config)

    @MethodChaining
    def build_fit(self) -> MP:
        self.config.fit_handlers = BatchHandler()

    @MethodChaining
    def build_predict(self) -> MP:
        self.config.predict_handlers = BatchHandler()

    @MethodChaining
    def build_evaluate(self) -> MP:
        self.config.evaluate_handlers = BatchHandler()

