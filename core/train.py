from torch.nn import MSELoss, BCELoss, KLDivLoss, CrossEntropyLoss, Module
from torch.optim import Adam, SGD, Optimizer
from torch.utils.data import DataLoader
from typing import Union, Optional
from common.util import func_call


def fit(
        model: Module,
        train_dataset: DataLoader,
        epochs: int,
        loss_func: Union[str, Module],
        optimizer: Union[str, Optimizer],
        metrics: Optional[list] = None,
        learning_rate: float = 1e-4,
        learning_rate_decay: float = None,
        val_dataset: DataLoader = None,
        loss_options: Optional[dict] = None,
        optimizer_options: Optional[dict] = None,
        device='cpu'
):
    # type check
    assert isinstance(loss_func, (str, Module)), 'loss function type check failed'
    assert isinstance(optimizer, (str, Optimizer)), 'optimizer type check failed'
    # init loss function
    loss_func = get_loss_func(loss_func, loss_options) if isinstance(loss_func, str) else loss_func
    # init optimizer
    optimizer_options = optimizer_options if isinstance(optimizer_options, dict) else {}
    optimizer_options = dict(optimizer_options, **{
        'lr': learning_rate,
        'params': model.parameters()
    })
    optimizer = get_optimizer(optimizer, optimizer_options) if isinstance(optimizer, str) else optimizer
    # compute total steps
    total_steps = len(train_dataset)

    # epoch loop
    for i in range(epochs):
        print('epoch %d' % (i + 1))
        model.train()

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
            visualize()


def evaluate(
        model: Module,
        dataset: DataLoader,
        metrics: list
):
    pass


def visualize():
    print('ok')


def compute():
    pass


def get_loss_func(loss: str, loss_options: Optional[dict] = None):
    loss_dict = {
        'mse': MSELoss,
        'bce': BCELoss,
        'kld': KLDivLoss,
        'ce': CrossEntropyLoss
    }
    loss_func = loss_dict.get(loss, None)
    assert loss_func is not None, 'loss function not supported'
    return func_call(loss_func, arg_dict=loss_options)


def get_optimizer(optimizer: str, optimizer_options: Optional[dict] = None):
    optimizer_dict = {
        'adam': Adam,
        'sgd': SGD
    }
    optimizer = optimizer_dict.get(optimizer, None)
    assert optimizer is not None, 'optimizer not supported'
    return func_call(optimizer, arg_dict=optimizer_options)
