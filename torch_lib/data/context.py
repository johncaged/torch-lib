from torch_lib.utils import build_from_dict


class DataProviderContext:

    def __init__(self, **kwargs):
        build_from_dict(self, kwargs, required_params=['epoch'])
