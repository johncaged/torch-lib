import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from metrics import precision, recall


def fit(
        model: Module,
        train_dataset: DataLoader,
        epochs: int,
        loss_function: str | Module,
        optimizer: str,
        metrics: list,
        learning_rate: float = 1e-4,
        learning_rate_decay: float = None,
        val_dataset: DataLoader = None
):
    pass


def evaluate(
        model: Module,
        dataset: DataLoader,
        metrics: list
):
    pass


def __visualize():
    pass


def __compute():
    pass
