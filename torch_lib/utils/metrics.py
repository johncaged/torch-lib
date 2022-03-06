from abc import abstractmethod
from typing import Union, Dict
from torch_lib.utils.type import NUMBER, NUMBER_T
from torch_lib.utils import Count, Nothing


class Metrics:

    count = Count()
    def __init__(self, name: str = None):
        self.name = name

    @abstractmethod
    def get(self, y_pred, y_true) -> Union[Dict, NUMBER]:
        pass

    def __call__(self, y_pred, y_true) -> Union[Dict, Nothing]:
        result = self.get(y_pred, y_true)
        if isinstance(result, Dict):
            return result
        elif isinstance(result, NUMBER_T):
            if self.name is None:
                # use default name
                self.name = 'metrics_%d' % self.count
            return { self.name: result }
        return Nothing()
