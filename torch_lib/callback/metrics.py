from abc import abstractmethod
from typing import Union, Dict, List, Sequence
from torch_lib.utils.type import NUMBER, NUMBER_T
from torch_lib.utils import Count, Nothing, MultiConst, AccessFilter, AddAccessFilter, ListAccessFilter, is_nothing, dict_merge, NOTHING
from torch_lib.callback import Callback
from torch_lib.context import Context


class MetricsCallback(Callback):

    count = Count()
    def __init__(self, name: str = None):
        self.name = name

    @abstractmethod
    def get(self, ctx: Context) -> Union[Dict, NUMBER]:
        pass

    def __call__(self, ctx: Context) -> Union[Dict, Nothing]:
        result = self.get(ctx)
        if isinstance(result, Dict):
            return result
        elif isinstance(result, NUMBER_T):
            if self.name is None:
                # use default name
                self.name = 'metrics_%d' % self.count
            return { self.name: result }
        return NOTHING


# metrics callback or sequence of metrics callbacks
M_SEQ = Union[MetricsCallback, Sequence[MetricsCallback]]


@AddAccessFilter(ListAccessFilter('metrics_callbacks'))
@AccessFilter
class MetricsCallbackExecutor(MetricsCallback):

    metrics_callbacks = MultiConst()
    def __init__(self, metrics: M_SEQ = None):
        super().__init__()
        self.metrics_callbacks: List[MetricsCallback] = []
        # add metrics to the list
        if metrics is not None:
            self.extend(metrics)

    def __call__(self, ctx: Context) -> Union[Dict, Nothing]:
        result = {}
        for metrics in self.metrics_callbacks:
            _res = metrics(ctx)
            # is not Nothing
            if is_nothing(_res) is False:
                dict_merge(result, _res)
        return result
