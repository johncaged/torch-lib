from typing import Union, Optional

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from torch_train_tool.core.map import get_optimizer, get_loss_func, get_scheduler

from torch_train_tool.common.metrics import compute_metrics

from torch_train_tool.common.util import dict_merge


def fit(
        model: Module,
        train_dataset: DataLoader,
        epochs: int,
        loss_func: Union[str, Module],
        optimizer: Union[str, Optimizer] = 'adam',
        metrics: Optional[list] = None,
        learning_rate: float = 1e-4,
        lr_decay=None,
        val_dataset: DataLoader = None,
        loss_options: Optional[dict] = None,
        optimizer_options: Optional[dict] = None,
        lr_decay_options: Optional[dict] = None,
        device='cpu',
        epoch_callbacks: Optional[list] = None,
        step_callbacks: Optional[list] = None
):
    """
    最最最核心的训练函数
    :param model: 训练的模型
    :param train_dataset: 训练数据集
    :param epochs: 需要训练多少个epochs
    :param loss_func: 损失函数，可以自己实例化一个损失函数（Module），也可以传入损失函数的名字（str）
    :param optimizer: 优化器，可以自己实例化一个优化器（Optimizer），也可以传入优化器的名字（str）
    :param metrics: 评估指标，一个列表，可以用字符串表示评估指标的名字，也可以传入函数
    :param learning_rate: 学习率，默认1e-4
    :param lr_decay: 学习率衰减，同样的，可以传入字符串或者lr_scheduler的实例
    :param val_dataset: 验证集
    :param loss_options: 损失函数参数配置，与loss_func搭配使用，为None则使用pytorch的默认配置（仅当loss_func为字符串时生效）
    :param optimizer_options: 优化器配置，与optimizer搭配使用，为None则使用pytorch的默认配置（仅当optimizer为字符串时生效）
    :param lr_decay_options: 学习率衰减的配置，与损失函数配置和优化器配置同理
    :param device: 训练设备，默认cpu，传入cuda值即可使用gpu训练
    :param epoch_callbacks: 每一个epoch结束的回调函数（还没开发）
    :param step_callbacks: 每一个training step结束的回调函数（还没开发）
    :return: None
    """
    # 参数类型检查
    assert isinstance(loss_func, (str, Module)), 'loss function type check failed'
    assert isinstance(optimizer, (str, Optimizer)), 'optimizer type check failed'
    # 初始化损失函数
    loss_func = get_loss_func(loss_func, loss_options) if isinstance(loss_func, str) else loss_func
    # 初始化优化器
    optimizer_options = dict_merge(optimizer_options, {
        'lr': learning_rate,
        'params': model.parameters()
    })
    optimizer = get_optimizer(optimizer, optimizer_options) if isinstance(optimizer, str) else optimizer
    # 初始化学习率衰减调度器
    lr_decay_options = dict_merge(lr_decay_options, {
        'optimizer': optimizer
    })
    scheduler = get_scheduler(lr_decay, lr_decay_options) if isinstance(lr_decay, str) or lr_decay is None else lr_decay
    # 计算总training steps
    total_steps = len(train_dataset)
    # 返回一个计算平均metrics的函数（用于训练集的训练过程展示）
    avg_metrics, clear_metrics = average_metrics()

    # epoch循环
    for i in range(epochs):
        print('epoch %d' % (i + 1))
        # 切换到训练模式
        model.train()

        avg_train_metrics = {}
        # batch循环
        for step, (x, y_true) in enumerate(train_dataset):
            # 前向传播
            y_pred = model(x.to(device))
            # 计算损失
            loss = loss_func(y_pred, y_true.to(device))
            # 清除梯度
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 梯度更新
            optimizer.step()
            # 学习率衰减
            if scheduler is not None:
                scheduler.step()
            # 这个batch计算得到的metrics
            train_metrics = compute_metrics(y_pred, y_true, metrics)
            # 计算这个epoch上的平均metrics
            avg_train_metrics = avg_metrics(dict_merge({'loss': loss}, train_metrics), step + 1)
            # 控制台训练过程可视化
            visualize(step + 1, total_steps, avg_train_metrics)
            if step_callbacks is not None:
                for callback in step_callbacks:
                    pass

        # 验证集验证
        if val_dataset:
            val_y_pred, val_y_true, val_loss = calculate(model, val_dataset, loss_func, device)
            val_metrics = compute_metrics(val_y_pred, val_y_true, metrics, val=True)
            val_metrics = dict_merge({'val_loss': val_loss}, val_metrics)
            visualize(total_steps, total_steps, dict_merge(avg_train_metrics, val_metrics))
        if epoch_callbacks is not None:
            for callback in epoch_callbacks:
                pass
        # 清除这一epoch的平均metrics，用于计算下一个epoch的平均metrics（如果不清除的话会导致结果累加错误）
        clear_metrics()
        print()


def evaluate(
        model: Module,
        dataset: DataLoader,
        metrics: list
):
    """
    模型评估
    :param model: 模型
    :param dataset: 数据集
    :param metrics: 评估指标
    :return: 评估指标的字典（如： { 'loss': 0.123456, 'acc': 0.985612 }）
    """
    pass


def visualize(step: int, total_steps: int, metrics: Optional[dict] = None, progress_len: int = 25):
    """
    控制台可视化，像keras一样可视化训练过程，我最喜欢的部分，因为看起来很酷
    :param step: 当前训练step
    :param total_steps: 总training steps
    :param metrics: 评估指标，这里传入评估指标的字典，用于控制台展示
    :param progress_len: 进度条长度（值为25代表将训练过程分成25个小格展示进度，依此类推，基本可以不用动）
    :return: None
    """
    def format_metric(name: str, item: float):
        return '%s: %f  ' % (name, item)

    # 计算当前epoch的进度
    rate = int(step * progress_len / total_steps)
    info = '%d/%d [%s%s] ' % (step, total_steps, '=' * rate, '-' * (progress_len - rate))

    # 展示评估指标
    if metrics is not None:
        for key, value in metrics.items():
            info += format_metric(key, value)

    print('\r%s' % info, end='', flush=True)


def average_metrics():
    """

    :return:
    """
    metric_dict = {}

    def compute_avg(metrics: Optional[dict], step):
        temp = {}
        if metrics is None:
            return temp
        for key, value in metrics.items():
            if key in metric_dict.keys():
                metric_dict[key] += value
            else:
                metric_dict[key] = value
            temp[key] = metric_dict[key] / step
        return temp

    def clear_metrics():
        metric_dict.clear()

    return compute_avg, clear_metrics


def calculate(model: Module, dataset, loss_func, device='cpu'):
    """

    :param model:
    :param dataset:
    :param loss_func:
    :param device:
    :return:
    """
    y_true_total = []
    y_pred_total = []
    # 切换到预测模式
    model.eval()
    with torch.no_grad():
        for step, (x, y_true) in enumerate(dataset):
            y_true_total += y_true
            # 前向传播
            y_pred = model(x.to(device))
            # 计算损失
            loss = loss_func(y_pred, y_true.to(device))
            y_pred_total += y_pred
    return y_pred_total, y_true_total, loss
