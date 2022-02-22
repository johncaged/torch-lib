from torch_lib.callback import Callback
from torch_lib.context.common import RunContext
from torch_lib.utils import AddAccessFilter, MultiConst, AccessFilter, ListAccessFilter
from torch_lib.utils.type import ExtendedSequence
from typing import List


class RunCallback(Callback):
    """
    Callback for running operations(training, evaluation, prediction, etc.).
    """
    def __init__(self):
        super().__init__()

    def begin(self, ctx: RunContext):
        pass

    def end(self, ctx: RunContext):
        pass

    def step_begin(self, ctx: RunContext):
        pass
    
    def step_end(self, ctx: RunContext):
        pass

    def epoch_begin(self, ctx: RunContext):
        pass

    def epoch_end(self, ctx: RunContext):
        pass


@AddAccessFilter(ListAccessFilter('run_callbacks'))
@AccessFilter
class RunCallbackExecutor(RunCallback):
    """
    Maintaining a list that contains run callbacks, combination mode.
    """
    run_callbacks = MultiConst()

    def __init__(self, run_callbacks: ExtendedSequence(RunCallback)=None):
        super().__init__()
        # Assign a const list to each run callback executor.
        self.run_callbacks: List[RunCallback] = []
        # Add run_callbacks
        if run_callbacks is not None:
            self.extend(run_callbacks)

    def set_instance_ctx(self, instance_ctx):
        # Set instance context to itself.
        super().set_instance_ctx(instance_ctx)
        # Set instance context to all its sub run callbacks.
        for run_callback in self.run_callbacks:
            run_callback.set_instance_ctx(instance_ctx)
