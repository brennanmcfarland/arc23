import torch

from model.state import get_train_mode_tree, set_train_mode_tree
from utils import try_reduce_list, run_callbacks
import torchvision.transforms as transforms


class Trainer:

    def __init__(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss = loss


# runs the network once without modifying the loader's state as a test/for profiling
def dry_run(net, loader, trainer, train_step_func, device=None):
    def _apply():
        prev_mode = get_train_mode_tree(net)
        batch = iter(loader).next()
        result = train_step_func(net, trainer, device=device)(batch[0]['data'], batch[0]['label']) # TODO: batch size > 1
        set_train_mode_tree(net, prev_mode)
        return result
    return _apply


def train_step(net, trainer, device=None):
    def _apply(inputs, gtruth):
        inputs, gtruth = inputs.to(device, non_blocking=True), gtruth.to(device, non_blocking=True)
        trainer.optimizer.zero_grad()  # reset the gradients to zero

        # run the inputs through the network and compute loss relative to gtruth
        outputs = net(inputs)
        gtruth = gtruth.squeeze(-1) # TODO: remove? put in data loader? should be general
        loss = trainer.loss(outputs, gtruth)
        loss.backward()
        trainer.optimizer.step()
        return loss
    return _apply


def train(net, loader, trainer, callbacks=None, device=None, epochs=1):
    if callbacks is None:
        callbacks = []

    steps_per_epoch = len(loader) # TODO: make this part of wrapper also
    callbacks = [callback(steps_per_epoch) for callback in callbacks]
    take_step = train_step(net, trainer, device=device)

    for epoch in range(epochs):
        print('----BEGIN EPOCH ', epoch, '----')
        for step, (inputs, gtruth) in enumerate(loader):
            loss = take_step(inputs, gtruth)
            run_callbacks("on_step", callbacks, loss, step, epoch)
        run_callbacks("on_epoch_end", callbacks)
    print('TRAINING COMPLETE!')


def test(net, loader, metrics=None, device=None):
    if metrics is None:
        metrics = []

    print('TESTING')
    with torch.no_grad():
        for l in loader:
            (inputs, gtruth) = l[0]['data'], l[0]['label'] # TODO: fix, this will not work for batch size > 1, need to make a wrapper iterator around DALI iterator
            inputs, gtruth = inputs.to(device, non_blocking=True), gtruth.to(device, non_blocking=True)
            outputs = net(inputs)
            run_callbacks("on_item", metrics, inputs, outputs, gtruth)
    return try_reduce_list(run_callbacks("on_end", metrics))


# validation is just an alias for testing
validate = test