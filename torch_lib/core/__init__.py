from typing import Any, Dict, Optional, Union, TypeVar
from torch_lib.callback.dataset import DataParser, DataProvider
from torch_lib.callback.metrics import M_SEQ
from torch_lib.callback.run import R_SEQ
from torch_lib.utils import MultiConst, get_device, type_cast, MethodChaining, InvocationDebug
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
        eval_dataset: DATASET = None,
        run_callbacks: R_SEQ = None,
        log_option = None  # TODO: log system design
    ):
        self.ctx.build.train(self.ctx)

    @InvocationDebug('ModelProxy.Predict')
    def predict(
        self,
        dataset: DATASET,
        run_callbacks: R_SEQ = None,
        log_option = None  # TODO: log system design
    ):
        self.ctx.build.predict(self.ctx)

    @InvocationDebug('ModelProxy.Eval')
    def eval(
        self,
        dataset: DATASET,
        run_callbacks: R_SEQ = None,
        log_option = None  # TODO: log system design
    ):
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
        pass

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
