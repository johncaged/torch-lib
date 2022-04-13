from torch_lib.core.context import Context
from torch_lib.util import AddAccessFilter, MultiConst, AccessFilter, ListAccessFilter
from typing import List, Union, Sequence


class Callback():
    """
    Callback for running operations(training, evaluation, prediction, etc.).
    """
    def __init__(self):
        super().__init__()

    def begin(self, ctx: Context):
        pass

    def end(self, ctx: Context):
        pass

    def step_begin(self, ctx: Context):
        pass
    
    def step_end(self, ctx: Context):
        pass

    def epoch_begin(self, ctx: Context):
        pass

    def epoch_end(self, ctx: Context):
        pass


# callback or sequence of callbacks
C_SEQ = Union[Callback, Sequence[Callback]]


@AddAccessFilter(ListAccessFilter('callbacks'))
@AccessFilter
class CallbackContainer(Callback):
    """
    Maintaining a list that contains callbacks, combination mode.
    """
    callbacks = MultiConst()

    def __init__(self, callbacks: C_SEQ = None):
        super().__init__()
        # Assign a const list to each run callback executor.
        self.callbacks: List[Callback] = []
        # Add callbacks
        if callbacks is not None:
            self.extend(callbacks)

    def begin(self, ctx: Context):
        for run_callback in self.callbacks:
            run_callback.begin(ctx)
    
    def end(self, ctx: Context):
        for run_callback in self.callbacks:
            run_callback.end(ctx)
    
    def step_begin(self, ctx: Context):
        for run_callback in self.callbacks:
            run_callback.step_begin(ctx)
    
    def step_end(self, ctx: Context):
        for run_callback in self.callbacks:
            run_callback.step_end(ctx)
    
    def epoch_begin(self, ctx: Context):
        for run_callback in self.callbacks:
            run_callback.epoch_begin(ctx)
    
    def epoch_end(self, ctx: Context):
        for run_callback in self.callbacks:
            run_callback.epoch_end(ctx)
