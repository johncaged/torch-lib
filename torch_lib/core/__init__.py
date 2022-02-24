from typing import TypeVar
from torch_lib.utils import MultiConst, get_device, type_cast, MethodChaining
from torch_lib.context import Context
from torch_lib.core.handler import BatchHandler


MP = TypeVar('MP', bound='ModelProxy')


class ModelProxy:

    config = MultiConst()
    def __init__(self, model, device=None):
        # set device
        self.device = device if device is not None else get_device(model)
        # set model and apply type cast
        self.model = type_cast(model, self.device)
        # set config
        self.config = Context()

    @MethodChaining
    def fit(self) -> MP:
        self.config.fit_handlers(self.config)

    @MethodChaining
    def predict(self) -> MP:
        self.config.predict_handlers(self.config)

    @MethodChaining
    def evaluate(self) -> MP:
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

