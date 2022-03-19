from torch_lib.core.handler import BatchHandler
from torch_lib.utils import Base, NOTHING, MultiConst
from torch.nn import Module
from torch import device
from torch.utils.data import DataLoader
from typing import Any, Union, Dict, Tuple
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


class BuildContext(TempContext):

    def __init__(self):
        super().__init__()
    
    def initialize(self):
        # batch handlers that define the process of training, evaluating and predicting.
        self.train: BatchHandler = NOTHING
        self.eval: BatchHandler = NOTHING
        self.predict: BatchHandler = NOTHING
        
        # other build config used for running
        


class Context(Base):
    """
    Context in the whole life time.
    """

    build = MultiConst()
    epoch = MultiConst()
    step = MultiConst()

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
