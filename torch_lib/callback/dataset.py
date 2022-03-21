from abc import abstractmethod
from torch.utils.data import DataLoader
from torch_lib.context import Context
from torch_lib.callback import Callback
from torch_lib.utils import list_take
from typing import Sequence, Tuple, Any, Union


class DataProvider(Callback):

    def __init__(self):
        pass

    @abstractmethod
    def get(self, ctx: Context) -> DataLoader:
        pass

    def __call__(self, ctx: Context) -> DataLoader:
        return self.get(ctx)


class ConstantDataProvider(DataProvider):

    def __init__(self, dataset: DataLoader):
        super().__init__()
        self.dataset = dataset

    def get(self, ctx: Context):
        return self.dataset


class DataParser(Callback):

    def __init__(self):
        pass

    @abstractmethod
    def parse(self, ctx: Context) -> Tuple[Any, Any, Any]:
        pass

    def __call__(self, ctx: Context) -> Tuple[Any, Any, Any]:
        return self.parse(ctx)


class IndexParser(DataParser):

    def __init__(
        self,
        x: Union[Sequence[int], int] = 0,
        y: Union[Sequence[int], int] = 1,
        extra: Union[Sequence[int], int] = None
    ):
        super(IndexParser, self).__init__()
        self.x = x
        self.y = y
        self.extra = extra

    def parse(self, ctx: Context) -> Tuple[Any, Any, Any]:
        item = ctx.step.item
        return list_take(item, self.x), list_take(item, self.y), list_take(item, self.extra)
