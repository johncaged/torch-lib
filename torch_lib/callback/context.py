from torch_lib.utils import build_from_dict


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
        build_from_dict(self, kwargs, required_params=[])


class EpochBeginContext(_Context):

    def __init__(self, **kwargs):
        super(EpochBeginContext, self).__init__()
        build_from_dict(self, kwargs, required_params=[])


class EpochEndContext(_Context):

    def __init__(self, **kwargs):
        super(EpochEndContext, self).__init__()
        build_from_dict(self, kwargs, required_params=[])
