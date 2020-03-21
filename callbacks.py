from tabulate import tabulate
import collections

import torch
import torch.utils.tensorboard as tensorboard
import torch.nn as nn

# all callbacks should document their performance impact with respect to how often they are meant to be called
# TODO: move actual calculations like these into a "diagnostics" file, and make callbacks just have funcs that accept
# TODO: them and do things like print the output or log to tensorboard and/or a combination of those things, or at least
# TODO: start with a func that decides what to do with the output/computed result
# TODO: after making these changes, it may be a good time to merge and sync changes with the other repo (but don't cut a
# TODO: version until both are in a more stable state)


# high performance impact, should be run on a per-epoch basis
def validate(validation_func, net, loader, metrics=None, device=None):
    def _bind(steps_per_epoch):
        def _run():
            validation_func(net, loader, metrics, device)
        return _run
    return _bind


# low performance impact, may be evaluated each step
def tensorboard_record_loss():
    def _bind(steps_per_epoch):
        tensorboard_writer = tensorboard.SummaryWriter()

        def _run(loss, step, epoch):
            tensorboard_writer.add_scalar('loss', loss, epoch * steps_per_epoch + step)
        return _run
    return _bind


# low performance impact, may be evaluated each step
def calc_interval_avg_loss(print_interval):
    def _bind(steps_per_epoch):
        functor = IntervalAvgLoss(steps_per_epoch, print_interval)
        return functor._run
    return _bind


# TODO: cleanup and maybe merge w other func below
def _apply_to_tensors_in_iterable(tensors, func):
    if not isinstance(tensors, torch.Tensor):
        return [_apply_to_tensors_in_iterable(t, func) for t in tensors]
    else:
        return func(tensors)


# TODO: merge w that and stats on outputs?
# TODO: abstract out w profiling into general func?
def calc_layer_grad_stats(net, run_func, stat_funcs=(torch.var_mean,)):
    hooks = []
    results_table = []

    def hook_func(layer, grad_input, grad_output):
        if hasattr(layer, 'weight'):
            stats = [_apply_to_tensors_in_iterable(grad_output, func) for func in stat_funcs]
            stats = _tensor_items_in_iterable(stats)
            func_names = [func.__name__ for func in stat_funcs]
            labelled_stats = [type(layer).__name__] + list(zip(func_names, stats))
            results_table.append(labelled_stats)

    def _add_hook(net):
        hooks.append(
            net.register_backward_hook(hook_func))

    def _bind(steps_per_epoch):
        def _run():
            net.apply(_add_hook)
            run_func()
            print(tabulate(results_table, headers=['Layer', 'Stats']))
            for h in hooks:
                h.remove()
        return _run
    return _bind


# high performance impact, should be run on a per-epoch basis
def calc_layer_stats(net, stat_funcs=(torch.var_mean,)):
    def _bind(steps_per_epoch):
        def _run():
            layers = net.modules()
            stats = _calc_weighted_layer_stats(layers, stat_funcs, lambda l, layer, layers: layer.weight)
            return stats
        return _run
    return _bind


# high performance impact, should be run on a per-epoch basis
def calc_layer_output_stats(net, loader, stat_funcs=(torch.var_mean,)):
    def _bind(steps_per_epoch):
        def _run():
            layers = net.children()
            datum = next(iter(loader))
            stats = _calc_weighted_layer_stats(
                layers,
                stat_funcs,
                lambda l, layer, layers: nn.Sequential(layers[:l])(datum['inputs'])
            )
            # reset loader
            iter(loader)
            return stats
        return _run
    return _bind


# helper function to extract the values of scalar tensors from a possibly nested iterable
# may want to move somewhere more general if ever needed
def _tensor_items_in_iterable(tensors):
    if not isinstance(tensors, torch.Tensor):
        return [_tensor_items_in_iterable(t) for t in tensors]
    else:
        return tensors.item()


# helper func for calculating stats associated with weighted layers
def _calc_weighted_layer_stats(layers, stat_funcs, provide_tensor):
    stats_table = []
    for l, layer in enumerate(layers):
        if hasattr(layer, 'weight'):
            with torch.no_grad():
                stats = [func(provide_tensor(l, layer, layers)) for func in stat_funcs]
                stats = _tensor_items_in_iterable(stats)
                func_names = [func.__name__ for func in stat_funcs]
                labelled_stats = [type(layer).__name__] + list(zip(func_names, stats))
                stats_table.append(labelled_stats)
    print(tabulate(stats_table, headers=['Layer', 'Stats']))
    return stats_table


# helper functor for calc_interval_avg_loss, since it's stateful
class IntervalAvgLoss:

    def __init__(self, steps_per_epoch, print_interval):
        self.interval_avg_loss = 0.0
        self.steps_per_epoch = steps_per_epoch
        self.print_interval = print_interval

    def _run(self, loss, step, epoch):
        interval_avg_loss = self.interval_avg_loss + loss.item()
        if step % self.print_interval == 0:
            print(
                'EPOCH ', epoch,
                ' STEP ', step, '/', self.steps_per_epoch,
                interval_avg_loss / self.print_interval
            )
            interval_avg_loss = 0
        return interval_avg_loss
