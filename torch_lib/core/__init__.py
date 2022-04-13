from typing import Any, Dict, Optional, Union, TypeVar
from torch_lib.data import ConstantProvider, DataParser, DataProvider, IndexParser
from torch_lib.metric import M_SEQ, MetricContainer
from torch_lib.callback import C_SEQ, CallbackContainer
from torch_lib.util import NOTHING, get_device, type_cast, MethodChaining, InvocationDebug, check_nothing, logger, is_nothing
from torch_lib.util.type import NUMBER
from torch_lib.core.context import Context
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.optim import Optimizer


T = TypeVar('T', bound='Proxy')
DATASET = Union[DataLoader, DataProvider]


class Proxy(Context):

    def __init__(self, model, device=None):
        # init context
        super().__init__()
        # set device
        self.device = device if device is not None else get_device(model)
        # set model and apply type cast
        self.model = type_cast(model, self.device)
        # build train, predict and eval process
        self.build_train().build_predict().build_eval()

    @InvocationDebug('Proxy.Train')
    def train(
        self,
        train_dataset: DATASET,
        total_epochs: int = 1,
        eval_dataset: DATASET = NOTHING,
        callbacks: C_SEQ = NOTHING,
        log_option = None  # TODO: log system design
    ):
        self.build_total_epochs(total_epochs)
        self.build_callbacks(callbacks)
        self.build_dataset(train_dataset, 'train')
        self.build_dataset(eval_dataset, 'eval')
        logger.info('Using device %s to train.' % str(self.device))
        self.run.train(self)

    @InvocationDebug('Proxy.Predict')
    def predict(
        self,
        dataset: DATASET,
        callbacks: C_SEQ = NOTHING,
        log_option = None  # TODO: log system design
    ):
        self.build_callbacks(callbacks)
        self.build_dataset(dataset, 'eval')
        logger.info('Using device %s to predict.' % str(self.device))
        self.run.predict(self)

    @InvocationDebug('Proxy.Eval')
    def eval(
        self,
        dataset: DATASET,
        callbacks: C_SEQ = NOTHING,
        log_option = None  # TODO: log system design
    ):
        self.build_callbacks(callbacks)
        self.build_dataset(dataset, 'eval')
        logger.info('Using device %s to eval.' % str(self.device))
        self.run.eval(self)

    @InvocationDebug('Proxy.Summary')
    def summary(self):
        pass

    @InvocationDebug('Proxy.Build')
    @MethodChaining
    def build(
        self,
        loss = None,
        metrics: M_SEQ = None,
        optimizer: Union[str, Optimizer] = None,
        lr: NUMBER = None,
        lr_decay: Any = None,
        optimizer_options: Optional[Dict] = None,
        lr_decay_options: Optional[Dict] = None,
        data_parser: Optional[DataParser] = None
    ) -> T:
        self.build_loss(loss)
        self.build_metrics(metrics)
        self.build_data_parser(data_parser)
        self.build_optimizer(optimizer, lr, optimizer_options)
        self.build_lr_decay(lr_decay, lr_decay_options)

    @InvocationDebug('Proxy.TrainBuilder')
    @MethodChaining
    def build_train(self) -> T:
        # get handler classes from context
        handler = self.handler
        # build training process using handlers
        self.run.train = handler.Container([
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
                # clear average metrics
                handler.Average('clear'),
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
                    # compute average metrics
                    handler.Average('avg'),
                    # display in console or in log files
                    handler.Display(),
                    # step end callback
                    handler.StepEnd()
                ]),
                # apply learning rate decay
                handler.LRDecay(),
                # set mode to 'eval'
                handler.Mode('eval'),
                # get dataset
                handler.Dataset(),
                # clear average metrics
                handler.Average('clear'),
                # dataset iter
                handler.Iteration([
                    # forward
                    handler.Forward(),
                    # compute loss
                    handler.Loss(),
                    # metrics
                    handler.Metrics(),
                    # compute average metrics
                    handler.Average('avg'),
                    # display in console or in log files; TODO: should eval output by step?
                    handler.Display()
                ]),
                # epoch end callback
                handler.EpochEnd()
            ]),
            # end callback
            handler.End()
        ])

    @InvocationDebug('Proxy.PredictBuilder')
    @MethodChaining
    def build_predict(self) -> T:
        # get handler classes from context
        handler = self.handler
        # build predicting process using handlers
        self.run.predict = handler.Container([
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

    @InvocationDebug('Proxy.EvalBuilder')
    @MethodChaining
    def build_eval(self) -> T:
        # get handler classes from context
        handler = self.handler
        # build evaluating process using handlers
        self.run.eval = handler.Container([
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

    @InvocationDebug('Proxy.build_loss')
    def build_loss(self, loss):
        if loss is not None:
            self.run.loss = check_nothing(loss, loss)

    @InvocationDebug('Proxy.build_metrics')
    def build_metrics(self, metrics):
        if metrics is not None:
            self.run.metrics = check_nothing(metrics, MetricContainer(metrics))

    @InvocationDebug('Proxy.build_data_parser')
    def build_data_parser(self, data_parser):
        if data_parser is not None:
            self.run.data_parser = check_nothing(data_parser, data_parser, IndexParser())

    @InvocationDebug('Proxy.build_callbacks')
    def build_callbacks(self, callbacks):
        if callbacks is not None:
            self.run.callbacks = check_nothing(callbacks, CallbackContainer(callbacks))

    @InvocationDebug('Proxy.build_optimizer')
    def build_optimizer(self, optimizer, lr, optimizer_options):
        if optimizer is not None:
            if isinstance(optimizer, Optimizer):
                self.run.optimizer = optimizer

    @InvocationDebug('Proxy.build_lr_decay')
    def build_lr_decay(self, lr_decay, lr_decay_options):
        if lr_decay is not None:
            if isinstance(lr_decay, str) is False:
                self.run.lr_decay = lr_decay

    @InvocationDebug('Proxy.build_total_epochs')
    def build_total_epochs(self, total_epochs):
        self.epoch.total = total_epochs if isinstance(total_epochs, int) else NOTHING

    @InvocationDebug('Proxy.build_dataset')
    def build_dataset(self, dataset, mode: str):
        if dataset is not None:
            if is_nothing(dataset):
                dataset = NOTHING
            else:
                dataset = dataset if isinstance(dataset, DataProvider) else ConstantProvider(dataset)

            if mode == 'train':
                self.run.train_provider = dataset
            elif mode == 'eval':
                self.run.eval_provider = dataset
            else:
                logger.warn('build_dataset mode not supported.')
