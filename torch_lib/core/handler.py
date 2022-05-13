from abc import abstractmethod
from typing import Dict, Sequence, Union
from ..util import BaseList, IterTool, NOTHING, is_nothing, safe_divide, type_cast, InvocationDebug
import torch_lib.util.terminal as Cursor
from ..util.formatter import progress_format, eta_format
from .context import Context
from ..log import logger
from functools import wraps
from torch import set_grad_enabled


def TorchGrad(func):
    """
    Set grad enabled or not according to the context mode.
    """
    @wraps(func)
    def grad_switch(self, ctx: Context):
        # only when context status is in ['TRAIN'] is the grad enabled
        with set_grad_enabled(str(ctx.status) in ['TRAIN']):
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


class HandlerContainer(Handler, BaseList):

    def __init__(self, handlers: C_SEQ = None):
        super().__init__()
        BaseList.__init__(self, handlers)
    
    def handle(self, ctx: Context):
        for handler in self:
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
        if ctx.ctx_check(['step.loss']) is True:
            last = ctx.step.total % ctx.run.grad_acc
            grad_acc = ctx.run.grad_acc if (ctx.step.total - ctx.step.current - 1) >= last else last
            # backward
            (ctx.step.loss / grad_acc).backward()


class OptimizerHandler(HandlerContainer):

    def __init__(self, handlers: C_SEQ = None):
        super().__init__(handlers)
    
    @InvocationDebug('OptimizerHandler')
    def handle(self, ctx: Context):
        # backward handler
        super().handle(ctx)
        if ctx.ctx_check(['run.optimizer']) is True and \
            ((ctx.step.current + 1) % ctx.run.grad_acc == 0 or ctx.step.current + 1 == ctx.step.total):
            ctx.run.optimizer.step()
            ctx.run.optimizer.zero_grad()


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
        ctx.status.init_avg_inner_ctx(ctx, self.INNER_KEY)
        if self.type == 'avg':
            self.average(ctx)
        elif self.type == 'clear':
            self.clear(ctx)

    def average(self, ctx: Context):
        # get inner context variables
        summary = ctx.status.get_avg_inner_ctx(ctx, self.INNER_KEY)
        summary['count'] += 1
        # get average loss and metrics
        avg_loss = self._compute_avg_loss(summary, ctx.step.loss)
        avg_metrics = self._compute_avg_metrics(summary, ctx.step.metrics)
        ctx.status.set_avg_loss_and_metrics(ctx, avg_loss, avg_metrics)

    def clear(self, ctx: Context):
        # reset avg info
        ctx.status.clear_avg_info(ctx, self.INNER_KEY)

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

        data = ' '.join(ctx.status.get_avg_loss_and_metrics(ctx))

        with Cursor.cursor_invisible():
            Cursor.refresh_print(
                str(ctx.status),
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
        ctx.ctx_check('status', silent=False)
        # get dataset through status
        ctx.status.get_dataset(ctx)


class StatusHandler(Handler):

    def __init__(self, status: str = 'train'):
        super().__init__()
        # get status supported
        from .status import proxy_status
        mode_supported = list(proxy_status.modules.keys())
        if status not in mode_supported:
            logger.warn('An unsupported status is set, this may cause some problems.')
        self.status = status
    
    @InvocationDebug('ModeHandler')
    def handle(self, ctx: Context):
        # context check
        ctx.ctx_check([
            'model'
        ], silent=False)
        # set status to the context
        from .status import proxy_status
        ctx.status = proxy_status.build(self.status)
        # change pytorch model mode
        ctx.status.set_model_mode(ctx)


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
