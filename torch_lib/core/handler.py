from abc import abstractmethod
from typing import Dict, List, Sequence, Union
from torch_lib.util import AddAccessFilter, AccessFilter, ListAccessFilter, MultiConst, IterTool, NOTHING, is_nothing, safe_divide, type_cast, InvocationDebug
import torch_lib.util.terminal as Cursor
from torch_lib.util.formatter import progress_format, eta_format
from torch_lib.core.context import Context
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
    """Base class for all handlers.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def handle(self, ctx: Context):
        pass

    def __call__(self, ctx: Context):
        self.handle(ctx)


class EmptyHandler(Handler):
    """Empty handler that does nothing when called.

    Args:
        Handler (torch_lib.core.handler.Handler): _description_
    """

    def __init__(self):
        super().__init__()
    
    @InvocationDebug('EmptyHandler')
    def handle(self, _: Context):
        """do nothing"""
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
            handler(ctx)


class EpochIterationHandler(HandlerContainer):

    def __init__(self, handlers: C_SEQ = None):
        super().__init__(handlers)

    @InvocationDebug('EpochIterationHandler')
    def handle(self, ctx: Context):
        # context check
        ctx.ctx_check('epoch.total', silent=False)
        # epoch loops
        for current in range(ctx.epoch.total):
            # set current epoch to the context
            ctx.epoch.current = current
            # output epoch info. TODO: change logger operation to a handler?
            logger.log('Epoch %d' % (ctx.epoch.current + 1))
            super().handle(ctx)


class IterationHandler(HandlerContainer):

    def __init__(self, handlers: C_SEQ = None):
        super().__init__(handlers)

    @InvocationDebug('IterationHandler')
    @TorchGrad
    def handle(self, ctx: Context):
        # context check
        if ctx.ctx_check('dataset') is True:
            for batch, progress, time, current, total in IterTool(ctx.dataset, True, True, True, True):
                ctx.step.from_dict({
                    'batch': batch, # original batch data of the dataset
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
        ctx.ctx_check([
            'model',
            'device',
            'run.data_parser',
            'step'
        ], silent=False)
        # forward
        x, y_true, extra = ctx.run.data_parser(ctx)
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
        if ctx.ctx_check('run.loss') is True:
            # compute loss
            loss = ctx.run.loss(ctx.step.y_pred, ctx.step.y_true)
            ctx.step.loss = loss


class BackwardHandler(Handler):

    def __init__(self):
        super().__init__()

    @InvocationDebug('BackwardHandler')
    def handle(self, ctx: Context):
        # context check
        if ctx.ctx_check([
            'step.loss',
            'run.optimizer'
        ]) is True:
            # backward
            ctx.run.optimizer.zero_grad()
            ctx.step.loss.backward()
            ctx.run.optimizer.step()


class MetricsHandler(Handler):

    def __init__(self):
        super().__init__()
    
    @InvocationDebug('MetricsHandler')
    def handle(self, ctx: Context):
        # context check
        ctx.ctx_check('step', silent=False)
        if ctx.ctx_check('run.metrics') is True:
            ctx.step.metrics = ctx.run.metrics(ctx)


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
    
    @InvocationDebug('AverageHandler')
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
        current = ctx.step.current
        total = ctx.step.total

        data = []
        if ctx.mode == 'train':
            if is_nothing(ctx.epoch.train_loss) is False:
                data.append('loss: {0:.5f}'.format(ctx.epoch.train_loss))
            for key, value in ctx.epoch.train_metrics.items():
                data.append('{0}: {1:.5f}'.format(key, value))
        elif ctx.mode == 'eval':
            if is_nothing(ctx.epoch.eval_loss) is False:
                data.append('loss: {0:.5f}'.format(ctx.epoch.eval_loss))
            for key, value in ctx.epoch.eval_metrics.items():
                data.append('{0}: {1:.5f}'.format(key, value))
        data = ' '.join(data)

        with Cursor.cursor_invisible():
            Cursor.refresh_print(
                ctx.mode.upper(),
                # progress bar
                progress_format(ctx.step.progress, newline=False),
                # eta with color blue
                '{0}ETA: {1}{2}'.format(
                    Cursor.single_color('b'),
                    eta_format(ctx.step.time, total - current - 1),
                    Cursor.reset_style()
                ),
                # loss and metrics output
                data,
                # print new line if progress end
                end='\n' if current + 1 == total else ''
            )


class DatasetHandler(Handler):

    def __init__(self):
        super().__init__()
    
    @InvocationDebug('DatasetHandler')
    def handle(self, ctx: Context):
        # context check
        ctx.ctx_check('mode', silent=False)
        # get dataset through mode
        if ctx.mode == 'train':
            ctx.ctx_check('run.train_provider', silent=False)
            ctx.dataset = ctx.run.train_provider(ctx)
        elif ctx.mode == 'eval':
            ctx.ctx_check('run.eval_provider', silent=False)
            ctx.dataset = ctx.run.eval_provider(ctx)


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
        ctx.ctx_check([
            'model'
        ], silent=False)
        # set mode to the context
        ctx.mode = self.mode
        # change pytorch model mode to self.mode
        getattr(ctx.model, self.mode, NOTHING)()


class LRDecayHandler(Handler):

    def __init__(self):
        super().__init__()
    
    @InvocationDebug('LRDecayHandler')
    def handle(self, ctx: Context):
        if ctx.ctx_check(['run.lr_decay']) is True:
            ctx.run.lr_decay.step()


# callback adapters
class BeginHandler(Handler):
    
    def __init__(self):
        super().__init__()

    @InvocationDebug('BeginHandler')
    def handle(self, ctx: Context):
        # context check
        ctx.ctx_check([
            'run.callbacks'
        ])
        ctx.run.callbacks.begin(ctx)


class EndHandler(Handler):

    def __init__(self):
        super().__init__()
    
    @InvocationDebug('EndHandler')
    def handle(self, ctx: Context):
        # context check
        ctx.ctx_check([
            'run.callbacks'
        ])
        ctx.run.callbacks.end(ctx)


class StepBeginHandler(Handler):

    def __init__(self):
        super().__init__()
    
    @InvocationDebug('StepBeginHandler')
    def handle(self, ctx: Context):
        # context check
        ctx.ctx_check([
            'run.callbacks'
        ])
        ctx.run.callbacks.step_begin(ctx)


class StepEndHandler(Handler):

    def __init__(self):
        super().__init__()
    
    @InvocationDebug('StepEndHandler')
    def handle(self, ctx: Context):
        # context check
        ctx.ctx_check([
            'run.callbacks'
        ])
        ctx.run.callbacks.step_end(ctx)


class EpochBeginHandler(Handler):

    def __init__(self):
        super().__init__()
    
    @InvocationDebug('EpochBeginHandler')
    def handle(self, ctx: Context):
        # context check
        ctx.ctx_check([
            'run.callbacks'
        ])
        ctx.run.callbacks.epoch_begin(ctx)


class EpochEndHandler(Handler):

    def __init__(self):
        super().__init__()
    
    @InvocationDebug('EpochEndHandler')
    def handle(self, ctx: Context):
        # context check
        ctx.ctx_check([
            'run.callbacks'
        ])
        ctx.run.callbacks.epoch_end(ctx)
