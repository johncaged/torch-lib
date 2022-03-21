from abc import abstractmethod
from typing import Union, Dict, List, Sequence
from torch_lib.utils.type import NUMBER, NUMBER_T
from torch_lib.utils import Count, Nothing, MultiConst, AccessFilter, AddAccessFilter, ListAccessFilter, is_nothing, dict_merge, NOTHING
from torch_lib.callback import Callback
from torch_lib.context import Context


class MetricCallback(Callback):

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
                self.name = 'metric_%d' % self.count
            return { self.name: result }
        return NOTHING


# metric callback or sequence of metric callbacks
M_SEQ = Union[MetricCallback, Sequence[MetricCallback]]


@AddAccessFilter(ListAccessFilter('metric_callbacks'))
@AccessFilter
class MetricCallbackExecutor(MetricCallback):

    metric_callbacks = MultiConst()
    def __init__(self, metric_callbacks: M_SEQ = None):
        super().__init__()
        self.metric_callbacks: List[MetricCallback] = []
        # add metric callbacks to the list
        if metric_callbacks is not None:
            self.extend(metric_callbacks)

    def get(self, ctx: Context) -> Union[Dict, NUMBER]:
        result = {}
        for metric_callback in self.metric_callbacks:
            _res = metric_callback(ctx)
            # is not Nothing
            if is_nothing(_res) is False:
                dict_merge(result, _res)
        return result
