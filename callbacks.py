from tabulate import tabulate

import torch.utils.tensorboard as tensorboard


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


# TODO: abstract out w profiling into general func?
# TODO: reorganize/combine wrapper funcs with hook funcs?
# TODO: may be able/need to delete after completing func in evaluation.py
def run_with_hooks(net, run_func, hook_funcs, wrapper_funcs):
    hooks = []

    def get_register_func(direction):
        if direction == "forward":
            return lambda l: l.register_forward_hook
        elif direction == "backward":
            return lambda l: l.register_backward_hook
        else:
            raise ValueError("Invalid pass direction for running with hooks")

    def _add_hook(layer):
        for hook_func, wrapper_func in zip(hook_funcs, wrapper_funcs):
            func, direction = hook_func()
            hooks.append(get_register_func(direction)(layer)(lambda l, i, o: wrapper_func(func(l, i, o))))

    def _run():
        net.apply(_add_hook)
        run_func()
        for h in hooks:
            h.remove()
    return _run


# high performance impact, should be run on a per-epoch basis
def calc_layer_stats(net, run_func, hook_funcs):

    results = [[]] * len(hook_funcs)

    wrappers = [lambda result: results[i].append(result) for i in range(len(hook_funcs))]

    runner = run_with_hooks(net, run_func, hook_funcs, wrappers)

    def _bind(steps_per_epoch):
        def _run():
            runner()
            for hook_results in results:
                print(tabulate(hook_results, headers=['Layer', 'Stats']))
        return _run
    return _bind


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
