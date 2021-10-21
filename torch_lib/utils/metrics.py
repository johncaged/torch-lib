from typing import Optional, Union, List, Callable
from torch_lib.utils import to_number, cast
from torch import Tensor
from torch.nn import Module
from torch_lib.utils.loss import get_loss_func


def precision(true_positive, false_positive):
    """
    计算查准率
    :param true_positive: 真阳性
    :param false_positive: 假阳性
    :return: 查准率计算结果
    """
    if true_positive + false_positive == 0:
        return 0.0
    else:
        return true_positive / (true_positive + false_positive)


def recall(true_positive, false_negative):
    """
    计算召回率
    :param true_positive: 真阳性
    :param false_negative: 假阴性
    :return: 召回率计算结果
    """
    if true_positive + false_negative == 0:
        return 0.0
    else:
        return true_positive / (true_positive + false_negative)


def get_metric(metric: Union[Callable, Module, str], default: str, device=None, dtype=None):
    """
    将metric转换为可调用的对象以及其名称
    :param metric:
    :param default:
    :param device:
    :param dtype:
    :return:
    """
    metric_dict = {

    }
    _metric = None
    _name = ''
    if isinstance(metric, (Callable, Module)):
        _name = getattr(metric, '__name__', default)
        _metric = metric
    elif isinstance(metric, str):
        # 损失函数映射优先
        _metric = get_loss_func(metric)
        _metric = metric_dict.get(metric, None) if _metric is None else _metric
        _name = metric
    elif isinstance(metric, tuple):
        # 具名metric
        _metric = get_metric(metric[0], default, device, dtype)[0]
        _name = metric[1]
    assert _metric is not None, 'can not find metric or loss function: %d' % str(metric)
    return cast(_metric, device, dtype), _name


def parse_metrics(metrics: List[Union[Callable, Module, str, tuple]], device=None, dtype=None, loss_first: bool = False):
    """
    将metrics列表转换为可调用的对象以及其名称
    :param metrics:
    :param device:
    :param dtype:
    :param loss_first: 列表中的第一个元素是否必须是损失函数（训练时必须）
    :return:
    """
    _metrics = []
    for i, metric in enumerate(metrics):
        default = 'metric_%d' % i
        if loss_first and i == 0:
            _metric = get_metric(metric, 'loss', device, dtype)
            assert isinstance(_metric[0], Module), 'the first metric should be a loss function to optimize in the fit method'
        else:
            _metric = get_metric(metric, default, device, dtype)
        _metrics.append(_metric)
    return _metrics


def compute_metrics(y_pred: Union[Tensor, tuple, list], y_true: Union[Tensor, tuple, list], metrics: Optional[list] = None, val=False):
    """
    计算评估指标
    :param y_pred: 模型预测结果
    :param y_true: 数据真实标签
    :param metrics: 模型评估指标（列表）
    :param val: 是否是验证集（只是在返回值的字段上有所不同，如： 训练集： { 'loss': 0.123456 }， 验证集: { 'val_loss': 0.123456 }）
    :return: 评估指标的字典（如： { 'loss': 0.123456, 'acc': 0.981234 }）
    """
    metric_dict = {}
    if metrics is None:
        return metric_dict
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.detach()
    if isinstance(y_pred, (tuple, list)):
        y_pred = tuple([(item.detach() if isinstance(item, Tensor) else item) for item in y_pred])

    for metric, name in metrics:
        if callable(metric) or isinstance(metric, Module):
            metric_dict[('val_' if val else '') + str(name)] = to_number(metric(y_pred, y_true))
    return metric_dict
