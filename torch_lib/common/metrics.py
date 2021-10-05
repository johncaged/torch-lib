from typing import Optional


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


def compute_metrics(y_pred, y_true, metrics: Optional[list] = None, val=False):
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

    for metric in metrics:
        if callable(metric):
            metric_dict[('val_' if val else '') + getattr(metric, '__name__')] = metric(y_pred, y_true)

        else:
            pass
    return metric_dict