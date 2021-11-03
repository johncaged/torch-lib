from typing import Sized, Optional, Union
from torch.utils.data import DataLoader, random_split, Dataset, Subset
from torch import Generator
from torch_lib.utils import func_call
from abc import abstractmethod
from torch_lib.data.context import DataProviderContext


def pack(dataset: Dataset, ratios: Optional[list] = None, random: bool = True, generator: Optional[Generator] = None, options: Union[dict, list, None] = None):
    """
    数据集分割以及打包成DataLoader
    :param dataset: 数据集
    :param ratios: 分割比例
    :param random: 随机分割还是顺序分割
    :param generator: 随机分割的种子
    :param options: DataLoader选项
    :return:
    """
    ratios = [1.0] if ratios is None else ratios
    assert sum(ratios) == 1.0, 'the sum of ratios must equals to one'
    assert min(ratios) >= 0, 'ratios must be no less than 0'
    assert hasattr(dataset, '__len__') or isinstance(dataset, Sized), 'dataset has no attr: __len__'

    # 判断dataloader_options是否是list
    list_options = isinstance(options, list)
    if list_options:
        assert len(options) == len(ratios), 'dataloader_options must either be a list and be the same size of ratios or be a dict'

    data_len = len(dataset)
    lengths = [int(round(ratio * data_len)) for ratio in ratios]
    lengths[-1] = data_len - sum(lengths[0:-1])

    if random is False:
        split_data = []
        indices = list(range(data_len))
        index = 0
        for length in lengths:
            split_data.append(Subset(dataset, indices[index:index + length]))
            index += length
    elif generator is None:
        split_data = random_split(dataset, lengths)
    else:
        split_data = random_split(dataset, lengths, generator)

    return tuple((func_call(DataLoader, [split_data[i]], options[i] if list_options else options) for i in range(len(ratios))))


class DataProvider:

    def __init__(self):
        pass

    @abstractmethod
    def get(self, ctx: DataProviderContext):
        pass


class ConstantDataProvider(DataProvider):

    def __init__(self, dataset: Optional[DataLoader]):
        super(ConstantDataProvider, self).__init__()
        self.dataset = dataset

    def get(self, ctx: DataProviderContext):
        return self.dataset
