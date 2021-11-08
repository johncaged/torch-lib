from torch_lib.utils import build_from_dict
from typing import Optional
from torch.nn import Module


class _Context:

    def __init__(self):
        pass


class BeginContext(_Context):

    def __init__(self, **kwargs):
        super(BeginContext, self).__init__()
        build_from_dict(self, kwargs, required_params=[])


class EndContext(_Context):

    def __init__(self, **kwargs):
        super(EndContext, self).__init__()
        build_from_dict(self, kwargs, required_params=[])


class StepBeginContext(_Context):

    def __init__(self, **kwargs):
        super(StepBeginContext, self).__init__()
        build_from_dict(self, kwargs, required_params=[])


class StepEndContext(_Context):

    def __init__(self, **kwargs):
        super(StepEndContext, self).__init__()
        self.step: Optional[int] = None
        self.total_steps: Optional[int] = None
        self.metrics: Optional[dict] = None
        self.model: Optional[Module] = None
        self.y_pred = None
        self.y_true = None
        build_from_dict(self, kwargs, required_params=[])


class EpochBeginContext(_Context):

    def __init__(self, **kwargs):
        super(EpochBeginContext, self).__init__()
        build_from_dict(self, kwargs, required_params=[])


class EpochEndContext(_Context):

    def __init__(self, **kwargs):
        super(EpochEndContext, self).__init__()
        self.metrics: Optional[dict] = None
        self.total_epochs: Optional[int] = None
        self.epoch: Optional[int] = None
        self.model: Optional[Module] = None
        build_from_dict(self, kwargs, required_params=[])
