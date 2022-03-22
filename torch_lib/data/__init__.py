from abc import abstractmethod
from torch.utils.data import DataLoader
from torch_lib.context import Context
from torch_lib.util import list_take
from torch_lib.log import logger
from typing import Sequence, Tuple, Any, Union


class DataProvider:

    def __init__(self):
        pass

    @abstractmethod
    def get(self, ctx: Context) -> DataLoader:
        pass

    def obtain(self, ctx: Context) -> DataLoader:
        data_loader = self.get(ctx)
        if isinstance(data_loader, DataLoader) is False:
            logger.warn('DataProvider returns a non-DataLoader object, this may cause some problems.')
        return data_loader


class ConstantProvider(DataProvider):

    def __init__(self, dataset: DataLoader):
        super().__init__()
        self.dataset = dataset

    def get(self, _) -> DataLoader:
        return self.dataset


class DataParser:

    def __init__(self):
        pass

    @abstractmethod
    def get(self, ctx: Context) -> Tuple[Any, Any, Any]:
        pass

    def obtain(self, ctx: Context) -> Tuple[Any, Any, Any]:
        item = self.get(ctx)
        if isinstance(item, tuple) is False or len(item) != 3:
            logger.warn('DataParser returns a non-tuple object or the tuple length is not 3, this may cause value-unpack excpetions.')
        return item


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

    def get(self, ctx: Context) -> Tuple[Any, Any, Any]:
        item = ctx.step.item
        return list_take(item, self.x), list_take(item, self.y), list_take(item, self.extra)
