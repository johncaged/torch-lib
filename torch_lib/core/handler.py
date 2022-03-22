from abc import abstractmethod
from typing import List, Sequence, Union
from torch_lib.util import AddAccessFilter, AccessFilter, ListAccessFilter, MultiConst, IterTool, NOTHING, type_cast, InvocationDebug
from torch_lib.context import Context
from functools import wraps
from torch import set_grad_enabled


_mode_all = ['train', 'eval']
provider_dict = {
    'train': 'train_provider',
    'eval': 'eval_provider'
}


def TorchGrad(func):
    """
    Set grad enabled or not according to the context mode.
    """
    @wraps(func)
    def grad_switch(self, ctx: Context):
        # only when context mode is 'train' is the grad enabled
        with set_grad_enabled(ctx.mode == 'train'):
            func(self, ctx)
    return grad_switch


class Handler:

    def __init__(self):
        super().__init__()

    @abstractmethod
    def handle(self, ctx: Context):
        pass

    def __call__(self, ctx: Context):
        return self.handle(ctx)


# core handler or sequence of core handlers
C_SEQ = Union[Handler, Sequence[Handler]]


@AddAccessFilter(ListAccessFilter('handlers'))
@AccessFilter
class HandlerContainer(Handler):

    handlers = MultiConst()
    def __init__(self, handlers: C_SEQ = None):
        super().__init__()
        self.handlers: List[Handler] = []
        if handlers is not None:
            self.extend(handlers)
    
    def handle(self, ctx: Context):
        for handler in self.handlers:
            handler(ctx)


class EpochIterationHandler(HandlerContainer):

    def __init__(self, handlers: C_SEQ = None):
        super().__init__(handlers)

    def handle(self, ctx: Context):
        # epoch loops
        for current in range(ctx.epoch.total):
            # set current epoch to the context
            ctx.epoch.current = current
            super().handle(ctx)

    @InvocationDebug('EpochIterationHandler')
    def __call__(self, ctx: Context):
        # context check
        ctx.check(['epoch.total'])
        return super().__call__(ctx)


class IterationHandler(HandlerContainer):

    def __init__(self, handlers: C_SEQ = None):
        super().__init__(handlers)

    @TorchGrad
    def handle(self, ctx: Context):
        for item, progress, time, current, total in IterTool(ctx.dataset, True, True, True, True):
            ctx.step.from_dict({
                'item': item, # original batch data of the dataset
                'progress': progress, # progress of iteration(includes current step and total steps)
                'time': time, # time of the iter(current time)
                'current': current, # the current step
                'total': total # total steps of iteration
            })
            # carry out the subsequent actions
            super().handle(ctx)

    @InvocationDebug('IterationHandler')
    def __call__(self, ctx: Context):
        # context check
        ctx.check(['dataset'])
        return super().__call__(ctx)


class ForwardHandler(Handler):
    
    def __init__(self):
        super().__init__()

    def handle(self, ctx: Context):
        # forward
        x, y_true, extra = ctx.build.data_parser(ctx)
        y_pred = ctx.model(type_cast(x, ctx.device))
        y_true = type_cast(y_true, ctx.device)
        # clone and update context info
        ctx.step.from_dict({
            # the result of the forward progress
            'x': x,
            'y_true': y_true,
            'y_pred': y_pred,
            'extra': extra
        })

    @InvocationDebug('ForwardHandler')
    def __call__(self, ctx: Context):
        # context check
        ctx.check([
            'model',
            'device',
            'build.data_parser',
            'step'
        ])
        return super().__call__(ctx)


class LossHandler(Handler):

    def __init__(self):
        super().__init__()
    
    def handle(self, ctx: Context):
        # compute loss
        loss = ctx.build.loss(ctx.step.y_pred, ctx.step.y_true)
        ctx.step.loss = loss

    @InvocationDebug('LossHandler')
    def __call__(self, ctx: Context):
        # context check
        ctx.check([
            'build.loss',
            'step.y_pred',
            'step.y_true'
        ])
        return super().__call__(ctx)


class BackwardHandler(Handler):

    def __init__(self):
        super().__init__()

    def handle(self, ctx: Context):
        # backward
        ctx.build.optimizer.zero_grad()
        ctx.step.loss.backward()
        ctx.build.optimizer.step()

    @InvocationDebug('BackwardHandler')
    def __call__(self, ctx: Context):
        # context check
        ctx.check([
            'step.loss',
            'build.optimizer'
        ])
        return super().__call__(ctx)


class MetricsHandler(Handler):

    def __init__(self):
        super().__init__()
    
    def handle(self, ctx: Context):
        ctx.step.metrics = ctx.build.metrics(ctx)

    @InvocationDebug('MetricsHandler')
    def __call__(self, ctx: Context):
        # context check
        ctx.check([
            'build.metrics',
            'step'
        ])
        return super().__call__(ctx)


class DisplayHandler(Handler):

    def __init__(self):
        super().__init__()
    
    @InvocationDebug('DisplayHandler')
    def handle(self, ctx: Context):
        # TODO
        # display results
        return super().handle(ctx)


class DatasetHandler(Handler):

    def __init__(self):
        super().__init__()
    
    def handle(self, ctx: Context):
        # get dataset through mode
        ctx.dataset = ctx.build[provider_dict.get(ctx.mode, NOTHING)](ctx)

    @InvocationDebug('DatasetHandler')
    def __call__(self, ctx: Context):
        # context check
        ctx.check([
            'build',
            'mode'
        ])
        return super().__call__(ctx)


class ModeHandler(Handler):

    def __init__(self, mode: str = 'train'):
        super().__init__()
        assert mode in _mode_all
        self.mode = mode
    
    def handle(self, ctx: Context):
        # set mode to the context
        ctx.mode = self.mode
        # change model mode to self.mode
        getattr(ctx.model, self.mode, NOTHING)()

    @InvocationDebug('ModeHandler')
    def __call__(self, ctx: Context):
        # context check
        ctx.check([
            'model'
        ])
        return super().__call__(ctx)


# run callback adapters
class BeginHandler(Handler):
    
    def __init__(self):
        super().__init__()

    def handle(self, ctx: Context):
        ctx.build.callbacks.begin(ctx)

    @InvocationDebug('BeginHandler')
    def __call__(self, ctx: Context):
        # context check
        ctx.check([
            'build.run_callback_exec'
        ])
        return super().__call__(ctx)


class EndHandler(Handler):

    def __init__(self):
        super().__init__()
    
    def handle(self, ctx: Context):
        ctx.build.callbacks.end(ctx)
    
    @InvocationDebug('EndHandler')
    def __call__(self, ctx: Context):
        # context check
        ctx.check([
            'build.run_callback_exec'
        ])
        return super().__call__(ctx)


class StepBeginHandler(Handler):

    def __init__(self):
        super().__init__()
    
    def handle(self, ctx: Context):
        ctx.build.callbacks.step_begin(ctx)
    
    @InvocationDebug('StepBeginHandler')
    def __call__(self, ctx: Context):
        # context check
        ctx.check([
            'build.run_callback_exec'
        ])
        return super().__call__(ctx)


class StepEndHandler(Handler):

    def __init__(self):
        super().__init__()
    
    def handle(self, ctx: Context):
        ctx.build.callbacks.step_end(ctx)
    
    @InvocationDebug('StepEndHandler')
    def __call__(self, ctx: Context):
        # context check
        ctx.check([
            'build.run_callback_exec'
        ])
        return super().__call__(ctx)


class EpochBeginHandler(Handler):

    def __init__(self):
        super().__init__()
    
    def handle(self, ctx: Context):
        ctx.build.callbacks.epoch_begin(ctx)
    
    @InvocationDebug('EpochBeginHandler')
    def __call__(self, ctx: Context):
        # context check
        ctx.check([
            'build.run_callback_exec'
        ])
        return super().__call__(ctx)


class EpochEndHandler(Handler):

    def __init__(self):
        super().__init__()
    
    def handle(self, ctx: Context):
        ctx.build.callbacks.epoch_end(ctx)
    
    @InvocationDebug('EpochEndHandler')
    def __call__(self, ctx: Context):
        # context check
        ctx.check([
            'build.run_callback_exec'
        ])
        return super().__call__(ctx)
