from typing import Union, Optional, Callable, List

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from torch_lib.utils.optim import get_optimizer
from torch_lib.utils.lr_decay import get_scheduler
from torch_lib.utils.metrics import compute_metrics, parse_metrics
from torch_lib.utils import dict_merge, get_device, to_number, func_call, get_dtype, cast, type_check, time_format, list_to_str, unpack, TimeRecord
from torch_lib.log.warning import cast_warning
from torch_lib.log.info import device_info, PlainInfo
from torch_lib.log import color_format, progress
from torch_lib.callback import Callback, CallbackExecutor
from torch_lib.callback.context import BeginContext, EndContext, StepBeginContext, StepEndContext, EpochBeginContext, EpochEndContext
from torch_lib.data import DataProvider, ConstantDataProvider, DataParser, IndexParser
from torch_lib.data.context import DataProviderContext


def fit(
        model: Module,
        train_dataset: Union[DataLoader, DataProvider],
        metrics: List[Union[str, Module, Callable]],
        epochs: int = 1,
        optimizer: Union[str, Optimizer] = 'adam',
        learning_rate: float = 1e-4,
        lr_decay=None,
        val_dataset: Union[DataLoader, DataProvider, None] = None,
        optimizer_options: Optional[dict] = None,
        lr_decay_options: Optional[dict] = None,
        callbacks: Union[Callback, List[Callback]] = None,
        console_print: bool = True,
        data_parser: Optional[DataParser] = None
):
    """
    最最最核心的训练函数，训练使用的设备由模型所在设备决定:cpu/cuda
    :param model: 训练的模型
    :param train_dataset: 训练数据集
    :param metrics: 评估指标，一个列表，可以用字符串表示评估指标的名字，也可以传入函数
    :param epochs: 需要训练多少个epochs
    :param optimizer: 优化器，可以自己实例化一个优化器（Optimizer），也可以传入优化器的名字（str）
    :param learning_rate: 学习率，默认1e-4
    :param lr_decay: 学习率衰减，同样的，可以传入字符串或者lr_scheduler的实例
    :param val_dataset: 验证集
    :param optimizer_options: 优化器配置，与optimizer搭配使用，为None则使用pytorch的默认配置（仅当optimizer为字符串时生效）
    :param lr_decay_options: 学习率衰减的配置，与损失函数配置和优化器配置同理
    :param callbacks: 回调函数插件
    :param console_print: 是否控制台可视化
    :param data_parser: 数据转换类
    :return: None
    """
    # 检查模型所在设备
    device = get_device(model)
    dtype = get_dtype(model)
    # 输出设备日志
    device_info.info(device)
    # 控制台输出控制流
    console = PlainInfo(console_print)
    # 检查数据集是否是函数类型
    train_provider = type_check(train_dataset, DataProvider, ConstantDataProvider(train_dataset))
    val_provider = type_check(val_dataset, DataProvider, ConstantDataProvider(val_dataset))
    # 转化metrics，并且确定损失函数
    metrics = parse_metrics(metrics, device, dtype, loss_first=True)
    loss_func = metrics[0][0]
    # 初始化优化器
    optimizer_options = dict_merge({
        'lr': learning_rate,
        'params': model.parameters()
    }, optimizer_options)
    optimizer = get_optimizer(optimizer, optimizer_options)
    # 初始化学习率衰减调度器
    lr_decay_options = dict_merge({
        'optimizer': optimizer
    }, lr_decay_options)
    scheduler = get_scheduler(lr_decay, lr_decay_options)
    # 返回一个计算平均metrics的函数（用于训练集的训练过程展示）
    avg_metrics, clear_metrics = _average_metrics()
    # 将callbacks汇总
    callback_exec = CallbackExecutor(callbacks)
    # 数据转换类
    data_parser: DataParser = type_check(data_parser, DataParser, IndexParser())
    # 计时器
    time_record = TimeRecord()

    # epoch循环
    for i in range(epochs):
        console.info('epoch', i + 1)
        # 根据epoch动态获取数据集，适用于渐进式学习
        del train_dataset
        del val_dataset
        train_dataset = train_provider.get(DataProviderContext(
            epoch=i
        ))
        val_dataset = val_provider.get(DataProviderContext(
            epoch=i
        ))

        # 计算总training steps
        total_steps = len(train_dataset)
        # 切换到训练模式
        model.train()

        avg_train_metrics = {}
        # batch循环
        for step, temp in enumerate(train_dataset):
            # 记录运行时间
            with time_record:
                x, y_true, extra = data_parser.parse(temp)
                # 需要类型转换则警告
                cast_warning.warn(get_dtype(x), dtype)
                # 前向传播
                y_pred = model(cast(x, device, dtype))
                # 设备转换
                y_true = cast(y_true, device)
                # 计算损失
                loss = loss_func(y_pred, y_true)
                # 清除梯度
                optimizer.zero_grad()
                # 反向传播
                loss.backward()
                # 梯度更新
                optimizer.step()
                # 这个batch计算得到的metrics
                train_metrics = compute_metrics(y_pred, y_true, metrics)
                # 计算这个epoch上的平均metrics
                avg_train_metrics = avg_metrics(train_metrics, step + 1)
                # 执行step_end回调函数
                callback_exec.step_end(StepEndContext(
                    metrics=avg_train_metrics,
                    step=step,
                    total_steps=total_steps,
                    model=model
                ))
                # 释放资源
                del y_pred, y_true
            # 控制台训练过程可视化
            console.info(_visualize(step + 1, total_steps, avg_train_metrics, time_record), mode='r')

        # 学习率衰减
        if scheduler is not None:
            scheduler.step()
        epoch_metrics = avg_train_metrics
        # 验证集验证
        if val_dataset:
            val_metrics = evaluate(model, val_dataset, metrics, console_print=False, val=True)
            epoch_metrics = dict_merge(epoch_metrics, val_metrics)
            console.info(_visualize(total_steps, total_steps, epoch_metrics), mode='r')
        # 执行epoch_end回调函数
        callback_exec.epoch_end(EpochEndContext(
            metrics=epoch_metrics,
            total_epochs=epochs,
            epoch=i,
            model=model
        ))
        # 清除这一epoch的平均metrics，用于计算下一个epoch的平均metrics（如果不清除的话会导致结果累加错误）
        clear_metrics()
        console.info()


def evaluate(
        model: Module,
        dataset: DataLoader,
        metrics: list,
        console_print: bool = True,
        val: bool = False,
        data_parser: Optional[DataParser] = None
):
    """
    模型评估
    :param model: 模型
    :param dataset: 数据集
    :param metrics: 评估指标
    :param console_print: 是否将预测进度展示在控制台
    :param val: 是否是验证集
    :param data_parser: 数据转换类
    :return: 评估指标的字典（如： { 'loss': 0.123456, 'acc': 0.985612 }）
    """
    # 获取模型所在的设备及数据类型
    device = get_device(model)
    dtype = get_dtype(model)
    # 输出设备日志
    device_info.info(device)
    metrics = parse_metrics(metrics, device, dtype)
    return _forward(model, dataset, 'evaluate', console_print, metrics, val, data_parser=data_parser)


def predict(
        model: Module,
        dataset: DataLoader,
        console_print: bool = True,
        data_parser: Optional[DataParser] = None
):
    """

    :param model: 模型
    :param dataset: 数据集
    :param console_print: 是否将推断进度显示在控制台
    :param data_parser: 数据转换类
    :return: 返回结果的预测值和真实值
    """
    # 获取模型所在的设备及数据类型
    device = get_device(model)
    # 输出设备日志
    device_info.info(device)
    return _forward(model, dataset, 'predict', console_print, data_parser=data_parser)


def traverse(
        model: Module,
        dataset: DataLoader,
        callbacks: Union[Callback, List[Callback]],
        metrics: Optional[list] = None,
        console_print: bool = True,
        val: bool = False,
        data_parser: Optional[DataParser] = None
):
    """

    :param model: 模型
    :param dataset: 数据集
    :param callbacks: 批量预测过程中的回调函数
    :param metrics: 遍历时计算评估指标
    :param console_print: 是否将推断进度显示在控制台
    :param val: 是否是验证集
    :param data_parser: 数据转换类
    :return: None
    """
    # 获取模型所在的设备及数据类型
    device = get_device(model)
    dtype = get_dtype(model)
    metrics = parse_metrics(metrics, device, dtype)
    return _forward(model, dataset, 'traverse', console_print, metrics=metrics, callbacks=callbacks, val=val, data_parser=data_parser)


def _visualize(step: int, total_steps: int, metrics: Optional[dict] = None, step_time: Union[float, TimeRecord, None] = None, progress_len: int = 25):
    """
    控制台可视化，像keras一样可视化训练过程，我最喜欢的部分，因为看起来很酷
    :param step: 当前训练step
    :param total_steps: 总training steps
    :param metrics: 评估指标，这里传入评估指标的字典，用于控制台展示
    :param step_time: 进行一个step所消耗的时间，用于计算ETA
    :param progress_len: 进度条长度（值为25代表将训练过程分成25个小格展示进度，依此类推，基本可以不用动）
    :return: 用于可视化的格式化字符串
    """

    def format_metric(name: str, item: float):
        return '%s: %f  ' % (name, item)

    info = ''

    # 展示评估指标
    if metrics is not None:
        for key, value in metrics.items():
            info += format_metric(key, value)

    return progress(step, total_steps, info, step_time=step_time, progress_len=progress_len, output=False)


def _average_metrics():
    """

    :return:
    """
    metric_dict = {}

    def compute_avg(metrics: Optional[dict], step: int = 1):
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


@torch.no_grad()
def _forward(
        model: Module,
        dataset: DataLoader,
        mode: str,
        console_print: bool = True,
        metrics: list = None,
        val: bool = False,
        callbacks: Union[Callback, List[Callback], None] = None,
        data_parser: Optional[DataParser] = None
):
    # 切换到预测模式
    model.eval()
    # 获取模型所在的设备及数据类型
    device = get_device(model)
    dtype = get_dtype(model)
    # 输出设备日志
    device_info.info(device)
    # 控制台输出流
    console = PlainInfo(console_print)

    """
    evaluate_mode = False
    """
    # 用于拼接所有结果
    y_pred_total = []

    """
    evaluate_mode = True
    """
    # 获取模型的steps
    total_steps = len(dataset)
    # 评估指标平均函数
    compute_avg, _ = _average_metrics()
    # 评估指标
    metrics_result = {}
    # 回调执行器
    callback_exec = CallbackExecutor(callbacks)
    # 数据转换类
    data_parser: DataParser = type_check(data_parser, DataParser, IndexParser())
    # 记录运行时间
    time_record = TimeRecord()

    console.info('predicting...')
    for step, temp in enumerate(dataset):
        with time_record:
            x, y_true, extra = data_parser.parse(temp)
            # 需要类型转换则警告
            cast_warning.warn(get_dtype(x), dtype)
            # 前向传播
            y_pred = model(cast(x, device, dtype))
            # 设备转换
            y_true = cast(y_true, device)
            # 评估模式计算评估指标
            if mode == 'evaluate':
                metrics_result = compute_avg(compute_metrics(y_pred, y_true, metrics, val), step + 1)
            # 推断模式将结果拼接
            elif mode == 'predict':
                y_pred_total += y_pred
            elif mode == 'traverse':
                _metrics = compute_metrics(y_pred, y_true, metrics, val)
                callback_exec.step_end(StepEndContext(
                    step=step,
                    y_pred=y_pred,
                    y_true=y_true,
                    metrics=_metrics,
                    total_steps=total_steps
                ))
            del y_pred, y_true
        # 如果设置了控制台打印输出，则显示当前预测进度
        console.info(_visualize(step + 1, total_steps, step_time=time_record), mode='r')
    console.info()

    if mode == 'evaluate':
        return metrics_result
    elif mode == 'predict':
        return cast(torch.stack(y_pred_total), device, dtype)
