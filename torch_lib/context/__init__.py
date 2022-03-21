from torch_lib.utils import Base, NOTHING, MultiConst, is_nothing
from torch.nn import Module
from torch import device
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch_lib.utils.type import NUMBER
from typing import Any, Sequence, Union, Dict, Tuple, Type
from torch_lib.log import logger
from abc import abstractmethod


class TempContext(Base):
    """Temp context that defines a initialize method to quickly reset the context.

    Args:
        Base (_type_): _description_
    """
    def __init__(self):
        super().__init__()
        # initialize
        self.initialize()
    
    @abstractmethod
    def initialize(self):
        pass


class StepContext(TempContext):

    def __init__(self):
        super().__init__()
    
    def initialize(self):
        """
        step context attribute placeholders(for code hints)
        """
        # data input to the model
        self.x: Any = NOTHING
        # output of the model
        self.y_pred: Any = NOTHING
        # label of the data(or expected output in unsupervised learning)
        self.y_true: Any = NOTHING
        # metrics of the step
        self.metrics: Dict = NOTHING
        # loss of the step
        self.loss = NOTHING
        # extra data passed to the context
        self.extra: Any = NOTHING
        # current iteration step
        self.current: int = NOTHING
        # total steps of iteration
        self.total: int = NOTHING
        # timestamp at the beginning of the step
        self.time: Union[int, float] = NOTHING
        # tuple of current step and total steps, it's used for progress visualization in the console
        self.progress: Tuple[int, int] = NOTHING
        # original item of the iteration of dataloader
        self.item: Any = NOTHING


class EpochContext(TempContext):

    def __init__(self):
        super().__init__()

    def initialize(self):
        """
        epoch context attribute placeholders(for code hints)
        """
        # total epochs for training
        self.total: int = NOTHING
        # the current epoch
        self.current: int = NOTHING
        # average train metrics in one epoch
        self.train_metrics: Dict = NOTHING
        # average eval metrics in one epoch
        self.eval_metrics: Dict = NOTHING
        # average train loss in one epoch
        self.train_loss = NOTHING
        # average eval loss in one epoch
        self.eval_loss = NOTHING


class BuildContext(TempContext):

    def __init__(self):
        super().__init__()
    
    def initialize(self):
        # batch handlers that define the process of training, evaluating and predicting.
        from torch_lib.core.handler import BatchHandler
        self.train: BatchHandler = NOTHING
        self.eval: BatchHandler = NOTHING
        self.predict: BatchHandler = NOTHING
        
        # optimizer
        self.optimizer: Optimizer = NOTHING
        # loss_func
        self.loss_func: Module = NOTHING
        # learning rate
        self.learning_rate: NUMBER = NOTHING
        # learning rate decay
        self.lr_decay: Any = NOTHING
        # data provider
        from torch_lib.callback.dataset import DataProvider
        self.train_provider: DataProvider = NOTHING
        self.eval_provider: DataProvider = NOTHING
        # data parser
        from torch_lib.callback.dataset import DataParser
        self.data_parser: DataParser = NOTHING
        # run callback executor
        from torch_lib.callback.run import RunCallbackExecutor
        self.run_callback_exec: RunCallbackExecutor = NOTHING
        # metric callback executor
        from torch_lib.callback.metrics import MetricCallbackExecutor
        self.metric_callback_exec: MetricCallbackExecutor = NOTHING


class HandlerContext(TempContext):

    def __init__(self):
        super().__init__()
    
    def initialize(self):
        import torch_lib.core.handler as handler
        # handler class
        self.Batch = handler.BatchHandler
        self.EpochIteration = handler.EpochIterationHandler
        self.Iteration = handler.IterationHandler
        self.Core = handler.CoreHandler
        self.Forward = handler.ForwardHandler
        self.Loss = handler.LossHandler
        self.Backward = handler.BackwardHandler
        self.Metrics = handler.MetricsHandler
        self.Display = handler.DisplayHandler
        self.Dataset = handler.DatasetHandler
        self.Mode = handler.ModeHandler
        self.Begin = handler.BeginHandler
        self.End = handler.EndHandler
        self.StepBegin = handler.StepBeginHandler
        self.StepEnd = handler.StepEndHandler
        self.EpochBegin = handler.EpochBeginHandler
        self.EpochEnd = handler.EpochEndHandler


class Context(Base):
    """
    Context in the whole life time.
    """

    build = MultiConst()
    epoch = MultiConst()
    step = MultiConst()
    handler = MultiConst()

    def __init__(self):
        super().__init__()
        
        """
        context attribute placeholders(for code hints)
        """
        # device for pytorch
        self.device: Union[str, device] = NOTHING
        # model
        self.model: Module = NOTHING
        # running mode(train, eval, etc.)
        self.mode: str = NOTHING
        # the current dataset for running
        self.dataset: DataLoader = NOTHING
        # build context
        self.build: BuildContext = BuildContext()
        # information in one epoch
        self.epoch: EpochContext = EpochContext()
        # information in one step
        self.step: StepContext = StepContext()
        # handler context
        self.handler: HandlerContext = HandlerContext()

    def check(self, items: Union[str, Sequence[str]]):
        def _check(_item):
            _result = super(Context, self).check(_item)
            if is_nothing(_result):
                logger.warn(
                    'Context check failed: got NOTHING with key \'%s\'.' % _item
                )
            return _result

        result = NOTHING
        if isinstance(items, (list, tuple)):
            # sequence value
            result = tuple(_check(item) for item in items)
        else:
            # single value
            result = _check(str(items))
        return result
