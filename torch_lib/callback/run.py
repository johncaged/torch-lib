from torch_lib.callback import Callback
from torch_lib.context import Context
from torch_lib.utils import AddAccessFilter, MultiConst, AccessFilter, ListAccessFilter
from typing import List, Union, Sequence


class RunCallback(Callback):
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


# run callback or sequence of run callbacks
R_SEQ = Union[RunCallback, Sequence[RunCallback]]


@AddAccessFilter(ListAccessFilter('run_callbacks'))
@AccessFilter
class RunCallbackExecutor(RunCallback):
    """
    Maintaining a list that contains run callbacks, combination mode.
    """
    run_callbacks = MultiConst()

    def __init__(self, run_callbacks: R_SEQ = None):
        super().__init__()
        # Assign a const list to each run callback executor.
        self.run_callbacks: List[RunCallback] = []
        # Add run_callbacks
        if run_callbacks is not None:
            self.extend(run_callbacks)

    def begin(self, ctx: Context):
        for run_callback in self.run_callbacks:
            run_callback.begin(ctx)
    
    def end(self, ctx: Context):
        for run_callback in self.run_callbacks:
            run_callback.end(ctx)
    
    def step_begin(self, ctx: Context):
        for run_callback in self.run_callbacks:
            run_callback.step_begin(ctx)
    
    def step_end(self, ctx: Context):
        for run_callback in self.run_callbacks:
            run_callback.step_end(ctx)
    
    def epoch_begin(self, ctx: Context):
        for run_callback in self.run_callbacks:
            run_callback.epoch_begin(ctx)
    
    def epoch_end(self, ctx: Context):
        for run_callback in self.run_callbacks:
            run_callback.epoch_end(ctx)
