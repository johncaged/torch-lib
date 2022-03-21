from typing import Any, Dict, Optional, Union, TypeVar
from torch_lib.callback.dataset import ConstantDataProvider, DataParser, DataProvider
from torch_lib.callback.metrics import M_SEQ, MetricCallbackExecutor
from torch_lib.callback.run import R_SEQ, RunCallbackExecutor
from torch_lib.utils import NOTHING, MultiConst, get_device, type_cast, MethodChaining, InvocationDebug, is_nothing, logger
from torch_lib.utils.type import NUMBER
from torch_lib.context import Context
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.optim import Optimizer


MP = TypeVar('MP', bound='ModelProxy')
DATASET = Union[DataLoader, DataProvider]


class ModelProxy:

    ctx = MultiConst()
    def __init__(self, model, device=None):
        # set context
        self.ctx = Context()
        # set device
        self.ctx.device = device if device is not None else get_device(model)
        # set model and apply type cast
        self.ctx.model = type_cast(model, self.ctx.device)
        # build train, predict and eval process
        self.build_train().build_predict().build_eval()

    @InvocationDebug('ModelProxy.Train')
    def train(
        self,
        train_dataset: DATASET,
        total_epochs: int = 1,
        eval_dataset: DATASET = NOTHING,
        run_callbacks: R_SEQ = NOTHING,
        log_option = None  # TODO: log system design
    ):
        self._build_total_epochs(total_epochs)
        self._build_run_callback_exec(run_callbacks)
        self._build_dataset(train_dataset, 'train')
        self._build_dataset(eval_dataset, 'eval')
        self.ctx.build.train(self.ctx)

    @InvocationDebug('ModelProxy.Predict')
    def predict(
        self,
        dataset: DATASET,
        run_callbacks: R_SEQ = NOTHING,
        log_option = None  # TODO: log system design
    ):
        self._build_run_callback_exec(run_callbacks)
        self._build_dataset(dataset, 'eval')
        self.ctx.build.predict(self.ctx)

    @InvocationDebug('ModelProxy.Eval')
    def eval(
        self,
        dataset: DATASET,
        run_callbacks: R_SEQ = NOTHING,
        log_option = None  # TODO: log system design
    ):
        self._build_run_callback_exec(run_callbacks)
        self._build_dataset(dataset, 'eval')
        self.ctx.build.eval(self.ctx)

    @InvocationDebug('ModelProxy.Summary')
    def summary(self):
        pass

    @InvocationDebug('ModelProxy.Build')
    @MethodChaining
    def build(
        self,
        loss_func = None,
        metric_callbacks: M_SEQ = None,
        optimizer: Union[str, Optimizer] = None,
        learning_rate: NUMBER = None,
        lr_decay: Any = None,
        optimizer_options: Optional[Dict] = None,
        lr_decay_options: Optional[Dict] = None,
        data_parser: Optional[DataParser] = None
    ) -> MP:
        self._build_loss(loss_func)
        self._build_metric_callback_exec(metric_callbacks)
        self._build_data_parser(data_parser)
        self._build_optimizer(optimizer, learning_rate, optimizer_options)
        self._build_lr_decay(lr_decay, lr_decay_options)

    @InvocationDebug('ModelProxy.TrainBuilder')
    @MethodChaining
    def build_train(self) -> MP:
        # get handler classes from context
        handler = self.ctx.handler
        # build training process using handlers
        self.ctx.build.train = handler.Batch([
            # begin callback
            handler.Begin(),
            # epoch iter
            handler.EpochIteration([
                # epoch begin callback
                handler.EpochBegin(),
                # set mode to 'train'
                handler.Mode('train'),
                # get dataset
                handler.Dataset(),
                # dataset iter
                handler.Iteration([
                    # step begin callback
                    handler.StepBegin(),
                    # forward
                    handler.Forward(),
                    # compute loss
                    handler.Loss(),
                    # backward
                    handler.Backward(),
                    # compute metrics
                    handler.Metrics(),
                    # display in console or in log files
                    handler.Display(),
                    # step end callback
                    handler.StepEnd()
                ]),
                # set mode to 'eval'
                handler.Mode('eval'),
                # get dataset
                handler.Dataset(),
                # dataset iter
                handler.Iteration([
                    # forward
                    handler.Forward(),
                    # compute loss
                    handler.Loss(),
                    # metrics
                    handler.Metrics()
                ]),
                # epoch end callback
                handler.EpochEnd()
            ]),
            # end callback
            handler.End()
        ])

    @InvocationDebug('ModelProxy.PredictBuilder')
    @MethodChaining
    def build_predict(self) -> MP:
        # get handler classes from context
        handler = self.ctx.handler
        # build predicting process using handlers
        self.ctx.build.predict = handler.Batch([
            # begin callback
            handler.Begin(),
            # set mode to 'eval'
            handler.Mode('eval'),
            # get dataset
            handler.Dataset(),
            # dataset iteration
            handler.Iteration([
                # step begin callback
                handler.StepBegin(),
                # forward
                handler.Forward(),
                # display
                handler.Display(),
                # step end callback
                handler.StepEnd()
            ]),
            # end callback
            handler.End()
        ])

    @InvocationDebug('ModelProxy.EvalBuilder')
    @MethodChaining
    def build_eval(self) -> MP:
        # get handler classes from context
        handler = self.ctx.handler
        # build evaluating process using handlers
        self.ctx.build.eval = handler.Batch([
            # begin callback
            handler.Begin(),
            # set mode to 'eval'
            handler.Mode('eval'),
            # get dataset
            handler.Dataset(),
            # dataset iteration
            handler.Iteration([
                # step begin callback
                handler.StepBegin(),
                # forward
                handler.Forward(),
                # compute loss
                handler.Loss(),
                # compute metrics
                handler.Metrics(),
                # display
                handler.Display(),
                # step end callback
                handler.StepEnd()
            ]),
            # end callback
            handler.End()
        ])

    @InvocationDebug('ModelProxy._build_loss')
    def _build_loss(self, loss_func):
        if loss_func is not None:
            self.ctx.build.loss_func = loss_func if is_nothing(loss_func) is False else NOTHING

    @InvocationDebug('ModelProxy._build_metric_callback_exec')
    def _build_metric_callback_exec(self, metric_callbacks):
        if metric_callbacks is not None:
            self.ctx.build.metric_callback_exec = MetricCallbackExecutor(metric_callbacks) if is_nothing(metric_callbacks) is False else NOTHING

    @InvocationDebug('ModelProxy._build_data_parser')
    def _build_data_parser(self, data_parser):
        if data_parser is not None:
            self.ctx.build.data_parser = data_parser if is_nothing(data_parser) is False else NOTHING

    @InvocationDebug('ModelProxy._build_run_callback_exec')
    def _build_run_callback_exec(self, run_callbacks):
        if run_callbacks is not None:
            self.ctx.build.run_callback_exec = RunCallbackExecutor(run_callbacks) if is_nothing(run_callbacks) is False else NOTHING

    @InvocationDebug('ModelProxy._build_optimizer')
    def _build_optimizer(self, optimizer, learning_rate, optimizer_options):
        if optimizer is not None:
            if isinstance(optimizer, Optimizer):
                self.ctx.build.optimizer = optimizer

    @InvocationDebug('ModelProxy._build_lr_decay')
    def _build_lr_decay(self, lr_decay, lr_decay_options):
        pass

    @InvocationDebug('ModelProxy._build_total_epochs')
    def _build_total_epochs(self, total_epochs):
        self.ctx.epoch.total = total_epochs if isinstance(total_epochs, int) else NOTHING

    @InvocationDebug('ModelProxy._build_dataset')
    def _build_dataset(self, dataset, mode: str):
        if dataset is not None:
            if is_nothing(dataset):
                dataset = NOTHING
            else:
                dataset = dataset if isinstance(dataset, DataProvider) else ConstantDataProvider(dataset)

            if mode == 'train':
                self.ctx.build.train_provider = dataset
            elif mode == 'eval':
                self.ctx.build.eval_provider = dataset
            else:
                logger.warn('_build_dataset mode not supported.')
