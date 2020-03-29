from utils import on_interval
from model.execution import run_with_hook


# all callbacks should document their performance impact with respect to how often they are meant to be called


# high performance impact, should be run on a per-epoch basis
def validate(validation_func, net, loader, metrics=None, device=None):
    def _bind(steps_per_epoch):
        def _run():
            return validation_func(net, loader, metrics, device)
        return _run
    return _bind


# low performance impact, may be evaluated each step
def loss():
    def _bind(steps_per_epoch):
        def _run(loss, step, epoch):
            return {"label": "loss", "value": loss}
        return _run
    return _bind


# low performance impact, may be evaluated each step
def interval_avg_loss(interval):
    def _bind(steps_per_epoch):
        class IntervalAvgLoss:

            def __init__(self, steps_per_epoch, interval):
                self.interval = interval
                self.interval_func = on_interval(self.interval_callback, interval)
                self.interval_avg_loss = 0.0
                self.steps_per_epoch = steps_per_epoch

            def __call__(self, loss, step, epoch):
                self.interval_avg_loss = self.interval_avg_loss + loss.item()
                return self.interval_func(step, epoch)

            def interval_callback(self, step, epoch):
                interval_avg_loss = self.interval_avg_loss / self.interval
                self.interval_avg_loss = 0.0
                return {
                    "label": "interval_avg_loss",
                    "value": interval_avg_loss
                }
        return IntervalAvgLoss(steps_per_epoch, interval)
    return _bind


# high performance impact, should be run on a per-epoch basis
# NOTE: each hook requires the network to be rerun because otherwise the hook return values corrupt each other; this
# is believed to be due to a bug in pytorch running multiple hooks with nested modules
def layer_stats(net, run_func, hook_funcs):
    def _bind(steps_per_epoch):
        def _run():
            for hook_func in hook_funcs:
                results = []
                run_with_hook(
                    net,
                    run_func,
                    hook_func,
                    lambda result: results.append(result)
                )()
                yield {"label": hook_func.__name__, "value": results}
        return _run
    return _bind

