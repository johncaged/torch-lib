from torch_lib.utils import build_from_dict
from typing import Optional


class DataProviderContext:

    def __init__(self, **kwargs):
        self.epoch: Optional[int] = None
        build_from_dict(self, kwargs, required_params=[])
