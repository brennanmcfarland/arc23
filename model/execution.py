from typing import Callable, Any, Tuple, Iterable, Optional, Union

import torch
from torch.optim.optimizer import Optimizer

from model.state import get_train_mode_tree, set_train_mode_tree
from __utils import try_reduce_list, run_callbacks, run_metrics
from __types import ModuleProperty, Method, Device, MetricFuncs, Tensor, Module, Loader


# description of how to train the model, ie what optimizer and loss function to use
class Trainer:

    def __init__(self, optimizer: Optimizer, loss):
        self.optimizer = optimizer
        self.loss = loss


# a mutable collection of all state pertaining to training (not the model itself)
class TrainState:

    def __init__(self, epoch: int = 0):
        # the current epoch we're on, may be nonzero if training is paused and resumed
        # (eg by saving and loading the train state)
        self.epoch = epoch


# runs the network once without modifying the loader's state as a test/for profiling
def dry_run(
        net: Module,
        loader: Loader,
        trainer: Union[Optional[Trainer], Callable[[Module], Optional[Trainer]]],
        train_step_func: Callable,
        device: Device = None
):
    tr = trainer

    def _apply():
        # the trainer will need to be reinitialized if the model has changed
        if callable(tr):
            trn = tr(net)
        else:
            trn = tr

        prev_mode = get_train_mode_tree(net)
        batch = next(iter(loader))
        result = train_step_func(net, trn, device=device)(batch['inputs'], batch['labels'])  # TODO: generalize?
        set_train_mode_tree(net, prev_mode)

        if trn is not None:
            trn.optimizer.zero_grad()
        return result
    return _apply


# TODO: fix squeeze_gtruth, shouldn't need
def train_step(net: Module, trainer: Trainer, device: Device = None, squeeze_gtruth: bool = False):
    def _apply(inputs: Tensor, gtruth: Tensor):
        inputs, gtruth = inputs.to(device, non_blocking=True), gtruth.to(device, non_blocking=True)
        trainer.optimizer.zero_grad()  # reset the gradients to zero

        # run the inputs through the network and compute loss relative to gtruth
        outputs = net(inputs)
        if squeeze_gtruth:
            gtruth = gtruth.squeeze(-1)
        loss = trainer.loss(outputs, gtruth)
        loss.backward()
        trainer.optimizer.step()
        return loss
    return _apply


def train(net: Module, loader: Loader, trainer: Trainer, callbacks=None, device=None, step_func=None, train_state: TrainState = None,
          epochs=1, squeeze_gtruth=False) -> None:
    if callbacks is None:
        callbacks = []
    if train_state is None:
        train_state = TrainState(0)

    steps_per_epoch = len(loader)
    callbacks = {hook: [callback(steps_per_epoch) for callback in callbacks] for hook, callbacks in callbacks.items()}
    take_step = step_func
    if step_func is None:
        take_step = train_step(net, trainer, device=device, squeeze_gtruth=squeeze_gtruth)
    initial_epoch = train_state.epoch

    run_callbacks("on_epoch", callbacks, 0)
    for epoch in range(initial_epoch, epochs):
        run_callbacks("on_epoch_start", callbacks, epoch)
        print('----BEGIN EPOCH ', epoch, '----')
        for step, datum in enumerate(loader):
            loss = take_step(datum['inputs'], datum['labels'])
            run_callbacks("on_step", callbacks, loss, step, epoch)
        run_callbacks("on_epoch_end", callbacks)
        run_callbacks("on_epoch", callbacks, epoch)
        train_state.epoch = epoch
    print('TRAINING COMPLETE!')


def test(
        net: Module,
        loader: Loader,
        metrics: Iterable[MetricFuncs] = None,
        device: Device = None,
        squeeze_gtruth: bool = False
):
    if metrics is None:
        metrics = []

    print('TESTING')
    prev_mode = get_train_mode_tree(net)
    net.eval()
    with torch.no_grad():
        for l in loader:
            (inputs, gtruth) = l['inputs'], l['labels']
            inputs, gtruth = inputs.to(device, non_blocking=True), gtruth.to(device, non_blocking=True)
            outputs = net(inputs)
            if squeeze_gtruth:
                gtruth = gtruth.squeeze(-1)
            run_metrics("on_item", metrics, inputs, outputs, gtruth)
        result = try_reduce_list(run_metrics("on_end", metrics))
    set_train_mode_tree(net, prev_mode)
    return result


# validation is just an alias for testing
validate = test


# run one pass of the network with a hook on each module
# NOTE: output_func cannot return anything or it will change the network's parameters
def run_with_hook(
        net: Module,
        run_func: Callable[[], Any],
        hook_func: Callable[[], Tuple[ModuleProperty, str]],
        output_func: Method[Any]
) -> Method:
    hooks = []

    def get_register_func(direction: str):
        if direction == "forward":
            return lambda l: l.register_forward_hook
        elif direction == "backward":
            return lambda l: l.register_backward_hook
        else:
            raise ValueError("Invalid pass direction for running with hooks")

    def _add_hook(layer: Module):
        func, direction = hook_func()
        hooks.append(get_register_func(direction)(layer)(lambda l, i, o: output_func(func(l, i, o))))

    def _run():
        net.apply(_add_hook)
        run_func()
        for h in hooks:
            h.remove()
    return _run
