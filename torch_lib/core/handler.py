from abc import abstractmethod
from typing import List
from torch_lib.utils import AddAccessFilter, AccessFilter, ListAccessFilter, MultiConst, IterTool, type_cast
from torch_lib.utils.type import ExtendedSequence


class CoreHandler:

    def __init__(self):
        super().__init__()

    @abstractmethod
    def handle(self, ctx):
        pass

    def __call__(self, ctx):
        self.handle(ctx)


@AddAccessFilter(ListAccessFilter('handlers'))
@AccessFilter
class BatchHandler(CoreHandler):

    handlers = MultiConst()
    def __init__(self, handlers: ExtendedSequence(CoreHandler) = None):
        super().__init__()
        self.handlers: List[CoreHandler] = []
        if handlers is not None:
            self.extend(handlers)
    
    def handle(self, ctx):
        for handler in self.handlers:
            handler.handle(ctx)


class ForwardHandler(BatchHandler):
    
    def __init__(self, handlers: ExtendedSequence(CoreHandler) = None):
        super().__init__(handlers)

    def handle(self, ctx):
        for item, progress, time in IterTool(ctx.dataset, True, True):
            x, y_true, extra = ctx.data_parser.parse(item)
            y_pred = ctx.model(type_cast(x, ctx.device))
            
        return super().handle(ctx)
