from typing import Union, Optional

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from core.map import get_optimizer, get_loss_func, get_scheduler

from common.metrics import compute_metrics

from common.util import dict_merge


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
    # type check
    assert isinstance(loss_func, (str, Module)), 'loss function type check failed'
    assert isinstance(optimizer, (str, Optimizer)), 'optimizer type check failed'
    # init loss function
    loss_func = get_loss_func(loss_func, loss_options) if isinstance(loss_func, str) else loss_func
    # init optimizer
    optimizer_options = dict_merge(optimizer_options, {
        'lr': learning_rate,
        'params': model.parameters()
    })
    optimizer = get_optimizer(optimizer, optimizer_options) if isinstance(optimizer, str) else optimizer
    # init learning rate decay
    lr_decay_options = dict_merge(lr_decay_options, {
        'optimizer': optimizer
    })
    scheduler = get_scheduler(lr_decay, lr_decay_options) if isinstance(lr_decay, str) or lr_decay is None else lr_decay
    # compute total steps
    total_steps = len(train_dataset)
    # compute avg metrics func
    avg_metrics, clear_metrics = average_metrics()

    # epoch loop
    for i in range(epochs):
        print('epoch %d' % (i + 1))
        model.train()

        avg_train_metrics = {}
        # batch loop
        for step, (x, y_true) in enumerate(train_dataset):
            # forward propagation
            y_pred = model(x.to(device))
            # compute loss
            loss = loss_func(y_pred, y_true.to(device))
            # clear grad
            optimizer.zero_grad()
            # backward propagation
            loss.backward()
            # update grad
            optimizer.step()
            # lr decay
            if scheduler is not None:
                scheduler.step()
            # train metrics
            train_metrics = compute_metrics(y_pred, y_true, metrics)
            # compute average metrics
            avg_train_metrics = avg_metrics(dict_merge({'loss': loss}, train_metrics), step + 1)
            # visualize step
            visualize(step + 1, total_steps, avg_train_metrics)

        if val_dataset:
            val_y_pred, val_y_true, val_loss = calculate(model, val_dataset, loss_func, device)
            val_metrics = compute_metrics(val_y_pred, val_y_true, metrics, val=True)
            val_metrics = dict_merge({'val_loss': val_loss}, val_metrics)
            visualize(total_steps, total_steps, dict_merge(avg_train_metrics, val_metrics))
        clear_metrics()
        print()


def evaluate(
        model: Module,
        dataset: DataLoader,
        metrics: list
):
    pass


def visualize(step: int, total_steps: int, metrics: Optional[dict] = None, progress_len: int = 25):
    def format_metric(name: str, item: float):
        return '%s: %f  ' % (name, item)

    # progress of one epoch
    rate = int(step * progress_len / total_steps)
    info = '%d/%d [%s%s] ' % (step, total_steps, '=' * rate, '-' * (progress_len - rate))

    # metrics
    if metrics is not None:
        for key, value in metrics.items():
            info += format_metric(key, value)

    print('\r%s' % info, end='', flush=True)


def average_metrics():
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
    y_true_total = []
    y_pred_total = []
    model.eval()
    with torch.no_grad():
        for step, (x, y_true) in enumerate(dataset):
            y_true_total += y_true
            # forward propagation
            y_pred = model(x.to(device))
            # compute loss
            loss = loss_func(y_pred, y_true.to(device))
            y_pred_total += y_pred
    return y_pred_total, y_true_total, loss
