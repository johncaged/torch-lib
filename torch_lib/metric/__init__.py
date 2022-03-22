from abc import abstractmethod
from typing import Union, Dict, List, Sequence
from torch_lib.util.type import NUMBER, NUMBER_T
from torch_lib.util import Count, Nothing, MultiConst, AccessFilter, AddAccessFilter, ListAccessFilter, is_nothing, dict_merge, NOTHING
from torch_lib.context import Context


class Metric():

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
M_SEQ = Union[Metric, Sequence[Metric]]


@AddAccessFilter(ListAccessFilter('metrics'))
@AccessFilter
class MetricContainer(Metric):

    metrics = MultiConst()
    def __init__(self, metrics: M_SEQ = None):
        super().__init__()
        self.metrics: List[Metric] = []
        # add metric callbacks to the list
        if metrics is not None:
            self.extend(metrics)

    def get(self, ctx: Context) -> Union[Dict, NUMBER]:
        result = {}
        for metric in self.metrics:
            _res = metric(ctx)
            # is not Nothing
            if is_nothing(_res) is False:
                # TODO: change the dict merge operation
                result = dict_merge(result, _res)
        return result
