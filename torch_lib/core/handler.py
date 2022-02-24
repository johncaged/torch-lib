from abc import abstractmethod
from typing import List
from torch_lib.utils import AddAccessFilter, AccessFilter, ListAccessFilter, MultiConst, IterTool, Nothing, type_cast
from torch_lib.utils.type import ExtendedSequence
import copy
from functools import wraps
from torch import set_grad_enabled


_mode_all = ['train', 'eval']
provider_dict = {
    'train': 'train_provider',
    'eval': 'val_provider'
}


def TorchGrad(func):
    """
    Set grad enabled or not according to the context mode.
    """
    @wraps(func)
    def grad_switch(self, ctx):
        # only when context mode is 'train' is the grad enabled
        set_grad_enabled(ctx.mode == 'train')(func)(self, ctx)
    return grad_switch


class CoreHandler:

    def __init__(self):
        super().__init__()

    @abstractmethod
    def handle(self, ctx):
        pass

    def __call__(self, ctx):
        self.handle(ctx)


@AddAccessFilter(ListAccessFilter('handlers'))
@AccessFilter
class BatchHandler(CoreHandler):

    handlers = MultiConst()
    def __init__(self, handlers: ExtendedSequence(CoreHandler) = None):
        super().__init__()
        self.handlers: List[CoreHandler] = []
        if handlers is not None:
            self.extend(handlers)
    
    def handle(self, ctx):
        for handler in self.handlers:
            handler(ctx)


class EpochIterationHandler(BatchHandler):

    def __init__(self, handlers: ExtendedSequence(CoreHandler) = None):
        super().__init__(handlers)

    def handle(self, ctx):
        # TODO
        # epoch loops
        for epoch in range(ctx.epochs):
            # set current epoch to the context
            ctx.epoch = epoch
            super()(ctx)


class IterationHandler(BatchHandler):

    def __init__(self, handlers: ExtendedSequence(CoreHandler) = None):
        super().__init__(handlers)

    @TorchGrad
    def handle(self, ctx):
        for item, progress, time in IterTool(ctx.epoch.dataset, True, True):
            # the context is copied to a new object because the original context object should not contain temporary data of a specific step.
            step_ctx = copy.copy(ctx)
            step_ctx.from_dict({
                'step': {
                    'item': item, # original batch data of the dataset
                    'progress': progress, # progress of iteration(includes step and total)
                    'time': time # time of the iter(current time)
                }
            })
            # carry out the subsequent actions
            super()(step_ctx)
            # release source
            del step_ctx


class ForwardHandler(CoreHandler):
    
    def __init__(self):
        super().__init__()

    def handle(self, ctx):
        # forward
        x, y_true, extra = ctx.data_parser.parse(ctx.item)
        y_pred = ctx.model(type_cast(x, ctx.device))
        y_true = type_cast(y_true, ctx.device)
        # clone and update context info
        ctx.from_dict({
            'step': { # the result of the forward progress
                'x': x,
                'y_true': y_true,
                'y_pred': y_pred,
                'extra': extra
            }
        })


class BackwardHandler(CoreHandler):

    def __init__(self):
        super().__init__()

    def handle(self, ctx):
        # backward
        loss = ctx.loss_func(ctx.step.y_pred, ctx.step.y_true)
        ctx.optimizer.zero_grad()
        loss.backward()
        ctx.optimizer.step()
        # add loss data to the context
        ctx.step.loss = loss


class MetricsHandler(CoreHandler):

    def __init__(self):
        super().__init__()
    
    def handle(self, ctx):
        # TODO
        # compute metrics
        return super().handle(ctx)


class DisplayHandler(CoreHandler):

    def __init__(self):
        super().__init__()
    
    def handle(self, ctx):
        # TODO
        # display results
        return super().handle(ctx)


class DatasetHandler(CoreHandler):

    def __init__(self):
        super().__init__()
    
    def handle(self, ctx):
        # get dataset through mode
        dataset = ctx[provider_dict.get(ctx.mode, Nothing())].get(ctx)
        ctx.from_dict({
            'epoch': {
                'dataset': dataset
            }
        })


class ModeHandler(CoreHandler):

    def __init__(self, mode: str = 'train'):
        super().__init__()
        assert mode in _mode_all
        self.mode = mode
    
    def handle(self, ctx):
        # set mode to the context
        ctx.mode = self.mode
        # change model mode to self.mode
        getattr(ctx.model, self.mode, Nothing())()


# run callback adapters
class BeginHandler(CoreHandler):
    
    def __init__(self):
        super().__init__()

    def handle(self, ctx):
        ctx.run_callback_exec.begin(ctx)


class EndHandler(CoreHandler):

    def __init__(self):
        super().__init__()
    
    def handle(self, ctx):
        ctx.run_callback_exec.end(ctx)


class StepBeginHandler(CoreHandler):

    def __init__(self):
        super().__init__()
    
    def handle(self, ctx):
        ctx.run_callback_exec.step_begin(ctx)


class StepEndHandler(CoreHandler):

    def __init__(self):
        super().__init__()
    
    def handle(self, ctx):
        ctx.run_callback_exec.step_end(ctx)


class EpochBeginHandler(CoreHandler):

    def __init__(self):
        super().__init__()
    
    def handle(self, ctx):
        ctx.run_callback_exec.epoch_begin(ctx)


class EpochEndHandler(CoreHandler):

    def __init__(self):
        super().__init__()
    
    def handle(self, ctx):
        ctx.run_callback_exec.epoch_end(ctx)
