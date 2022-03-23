from abc import abstractmethod
from typing import Dict, List, Sequence, Union
from torch_lib.util import AddAccessFilter, AccessFilter, ListAccessFilter, MultiConst, IterTool, NOTHING, is_nothing, safe_divide, type_cast, InvocationDebug
from torch_lib.context import Context
from torch_lib.log import logger
from functools import wraps
from torch import set_grad_enabled


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


# handler or sequence of handlers
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
            handler.handle(ctx)


class EpochIterationHandler(HandlerContainer):

    def __init__(self, handlers: C_SEQ = None):
        super().__init__(handlers)

    @InvocationDebug('EpochIterationHandler')
    def handle(self, ctx: Context):
        # context check
        ctx.check('epoch.total', silent=False)
        # epoch loops
        for current in range(ctx.epoch.total):
            # set current epoch to the context
            ctx.epoch.current = current
            # output epoch info. TODO: change logger operation to a handler?
            logger.log('Epoch %d' % ctx.epoch.current)
            super().handle(ctx)


class IterationHandler(HandlerContainer):

    def __init__(self, handlers: C_SEQ = None):
        super().__init__(handlers)

    @InvocationDebug('IterationHandler')
    @TorchGrad
    def handle(self, ctx: Context):
        # context check
        ctx.check('dataset', silent=False)
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


class ForwardHandler(Handler):
    
    def __init__(self):
        super().__init__()

    @InvocationDebug('ForwardHandler')
    def handle(self, ctx: Context):
        # context check
        ctx.check([
            'model',
            'device',
            'build.data_parser',
            'step'
        ], silent=False)
        # forward
        x, y_true, extra = ctx.build.data_parser.obtain(ctx)
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


class LossHandler(Handler):

    def __init__(self):
        super().__init__()
    
    @InvocationDebug('LossHandler')
    def handle(self, ctx: Context):
        # context check
        if ctx.check('build.loss') is True:
            # compute loss
            loss = ctx.build.loss(ctx.step.y_pred, ctx.step.y_true)
            ctx.step.loss = loss


class BackwardHandler(Handler):

    def __init__(self):
        super().__init__()

    @InvocationDebug('BackwardHandler')
    def handle(self, ctx: Context):
        # context check
        ctx.check([
            'step.loss',
            'build.optimizer'
        ], silent=False)
        # backward
        ctx.build.optimizer.zero_grad()
        ctx.step.loss.backward()
        ctx.build.optimizer.step()


class MetricsHandler(Handler):

    def __init__(self):
        super().__init__()
    
    @InvocationDebug('MetricsHandler')
    def handle(self, ctx: Context):
        # context check
        ctx.check('step', silent=False)
        if ctx.check('build.metrics') is True:
            ctx.step.metrics = ctx.build.metrics.obtain(ctx)


# TODO: implementation to be optimized
class AverageHandler(Handler):

    # inner context key
    INNER_KEY = 'AVERAGE_INNER'

    def __init__(self, type: str = 'avg'):
        super().__init__()
        type_supported = ['avg', 'clear']
        if type not in type_supported:
            logger.warn('An unsupported average handler type is set.')
        self.type = type
    
    def handle(self, ctx: Context):
        if is_nothing(ctx.inner[self.INNER_KEY]):
            ctx.inner[self.INNER_KEY] = {
                'train': {
                    'count': 0,
                    'loss': 0,
                    'metrics': {}
                },
                'eval': {
                    'count': 0,
                    'loss': 0,
                    'metrics': {}
                }
            }

        if self.type == 'avg':
            self.average(ctx)
        elif self.type == 'clear':
            self.clear(ctx)

    def average(self, ctx: Context):
        # get inner context variables
        summary = ctx.inner[self.INNER_KEY].get(ctx.mode, NOTHING)
        summary['count'] += 1
        # get average loss and metrics
        avg_loss = self._compute_avg_loss(summary, ctx.step.loss)
        avg_metrics = self._compute_avg_metrics(summary, ctx.step.metrics)
        if ctx.mode == 'train':
            ctx.epoch.train_loss = avg_loss
            ctx.epoch.train_metrics = avg_metrics
        elif ctx.mode == 'eval':
            ctx.epoch.eval_loss = avg_loss
            ctx.epoch.eval_metrics = avg_metrics

    def clear(self, ctx: Context):
        # reset inner context variables
        ctx.inner[self.INNER_KEY][ctx.mode] = {
            'count': 0,
            'loss': 0,
            'metrics': {}
        }
        # reset epoch metrics and loss
        if ctx.mode == 'train':
            ctx.epoch.train_metrics = NOTHING
            ctx.epoch.train_loss = NOTHING
        elif ctx.mode == 'eval':
            ctx.epoch.eval_metrics = NOTHING
            ctx.epoch.eval_loss = NOTHING

    @staticmethod
    def _compute_avg_loss(summary, loss):
        if 'loss' in summary and 'count' in summary and is_nothing(loss) is False:
            summary['loss'] += float(loss)
            return safe_divide(summary['loss'], summary['count'])
        else:
            return NOTHING

    @staticmethod
    def _compute_avg_metrics(summary: Dict, metrics: Dict):
        if 'metrics' in summary and 'count' in summary:
            temp = {}
            _metrics = summary['metrics']
            for key, value in metrics.items():
                if key in _metrics:
                    _metrics[key] += value
                else:
                    _metrics[key] = value
                temp[key] = safe_divide(_metrics[key], summary['count'])
            return temp
        else:
            return NOTHING


class DisplayHandler(Handler):

    def __init__(self):
        super().__init__()
    
    @InvocationDebug('DisplayHandler')
    def handle(self, ctx: Context):
        logger.log('\r', '%d/%d' % (ctx.step.current + 1, ctx.step.total), 'loss: %f' % ctx.step.loss, 'metrics: %s' % str(ctx.step.metrics), end='', flush=True)
        if ctx.step.current + 1 == ctx.step.total:
            logger.log()


class DatasetHandler(Handler):

    def __init__(self):
        super().__init__()
    
    @InvocationDebug('DatasetHandler')
    def handle(self, ctx: Context):
        # context check
        ctx.check('mode', silent=False)
        # get dataset through mode
        if ctx.mode == 'train':
            ctx.check('build.train_provider', silent=False)
            ctx.dataset = ctx.build.train_provider.obtain(ctx)
        elif ctx.mode == 'eval':
            ctx.check('build.eval_provider', silent=False)
            ctx.dataset = ctx.build.eval_provider.obtain(ctx)


class ModeHandler(Handler):

    def __init__(self, mode: str = 'train'):
        super().__init__()
        # only two modes are supported
        mode_supported = ['train', 'eval']
        if mode not in mode_supported:
            logger.warn('An unsupported mode is set, this may cause some problems.')
        self.mode = mode
    
    @InvocationDebug('ModeHandler')
    def handle(self, ctx: Context):
        # context check
        ctx.check([
            'model'
        ], silent=False)
        # set mode to the context
        ctx.mode = self.mode
        # change pytorch model mode to self.mode
        getattr(ctx.model, self.mode, NOTHING)()


# callback adapters
class BeginHandler(Handler):
    
    def __init__(self):
        super().__init__()

    @InvocationDebug('BeginHandler')
    def handle(self, ctx: Context):
        # context check
        ctx.check([
            'build.callbacks'
        ])
        ctx.build.callbacks.begin(ctx)


class EndHandler(Handler):

    def __init__(self):
        super().__init__()
    
    @InvocationDebug('EndHandler')
    def handle(self, ctx: Context):
        # context check
        ctx.check([
            'build.callbacks'
        ])
        ctx.build.callbacks.end(ctx)


class StepBeginHandler(Handler):

    def __init__(self):
        super().__init__()
    
    @InvocationDebug('StepBeginHandler')
    def handle(self, ctx: Context):
        # context check
        ctx.check([
            'build.callbacks'
        ])
        ctx.build.callbacks.step_begin(ctx)


class StepEndHandler(Handler):

    def __init__(self):
        super().__init__()
    
    @InvocationDebug('StepEndHandler')
    def handle(self, ctx: Context):
        # context check
        ctx.check([
            'build.callbacks'
        ])
        ctx.build.callbacks.step_end(ctx)


class EpochBeginHandler(Handler):

    def __init__(self):
        super().__init__()
    
    @InvocationDebug('EpochBeginHandler')
    def handle(self, ctx: Context):
        # context check
        ctx.check([
            'build.callbacks'
        ])
        ctx.build.callbacks.epoch_begin(ctx)


class EpochEndHandler(Handler):

    def __init__(self):
        super().__init__()
    
    @InvocationDebug('EpochEndHandler')
    def handle(self, ctx: Context):
        # context check
        ctx.check([
            'build.callbacks'
        ])
        ctx.build.callbacks.epoch_end(ctx)
