from torch_lib.callback.context import BeginContext, EndContext, StepBeginContext, StepEndContext, EpochBeginContext, EpochEndContext
from typing import List, Union


class Callback:
    """
    回调函数基类
    """

    def __init__(self):
        pass

    def begin(self, ctx: BeginContext):
        pass

    def end(self, ctx: EndContext):
        pass

    def step_begin(self, ctx: StepBeginContext):
        pass

    def step_end(self, ctx: StepEndContext):
        pass

    def epoch_begin(self, ctx: EpochBeginContext):
        pass

    def epoch_end(self, ctx: EpochEndContext):
        pass


class CallbackExecutor(Callback):
    """
    整合所有回调函数列表，进行顺序调用
    """
    def __init__(self, callbacks: Union[Callback, List[Callback]]):
        super(CallbackExecutor, self).__init__()
        assert callbacks is None or isinstance(callbacks, (Callback, list)), 'callbacks type check failed'
        if isinstance(callbacks, list):
            self.callbacks = callbacks
        elif isinstance(callbacks, Callback):
            self.callbacks = [callbacks]
        else:
            self.callbacks = []

    def begin(self, ctx: BeginContext):
        for callback in self.callbacks:
            callback.begin(ctx)

    def end(self, ctx: EndContext):
        for callback in self.callbacks:
            callback.end(ctx)

    def step_begin(self, ctx: StepBeginContext):
        for callback in self.callbacks:
            callback.step_begin(ctx)

    def step_end(self, ctx: StepEndContext):
        for callback in self.callbacks:
            callback.step_end(ctx)

    def epoch_begin(self, ctx: EpochBeginContext):
        for callback in self.callbacks:
            callback.epoch_begin(ctx)

    def epoch_end(self, ctx: EpochEndContext):
        for callback in self.callbacks:
            callback.epoch_end(ctx)
