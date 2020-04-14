from typing import *
from __utils import on_interval
from model.execution import run_with_hook
from __types import Module, MetricFuncs, Device, Bindable, LabelledValue, FuncOfLoss, ModuleProperty, Loader, Tensor


# all callbacks should document their performance impact with respect to how often they are meant to be called


# high performance impact, should be run on a per-epoch basis
def validate(
        validation_func: Callable[[Any, Any, Iterable[Any], Any], Any],
        net: Module,
        loader: Loader,
        metrics: Iterable[MetricFuncs] = None,
        device: Device = None
) -> Bindable[int, Callable]:
    def _bind(steps_per_epoch: int):
        def _run():
            return validation_func(net, loader, metrics, device)
        return _run
    return _bind


# low performance impact, may be evaluated each step
def loss() -> Bindable[int, FuncOfLoss]:
    def _bind(steps_per_epoch: int):
        def _run(loss: float, step: int, epoch: int):
            return LabelledValue("loss", loss)
        return _run
    return _bind


# low performance impact, may be evaluated each step
def interval_avg_loss(interval) -> Bindable[int, FuncOfLoss]:
    def _bind(steps_per_epoch: int):
        class IntervalAvgLoss:

            def __init__(self, steps_per_epoch: int, interval: int):
                self.interval: int = interval
                self.interval_func = on_interval(self.interval_callback, interval)
                self.interval_avg_loss: float = 0.0
                self.steps_per_epoch: int = steps_per_epoch

            def __call__(self, loss: Tensor, step: int, epoch: int):
                self.interval_avg_loss = self.interval_avg_loss + loss.item()
                return self.interval_func(step, epoch)

            def interval_callback(self, step: int, epoch: int):
                interval_avg_loss = self.interval_avg_loss / self.interval
                self.interval_avg_loss = 0.0
                return LabelledValue("interval_avg_loss", interval_avg_loss)
        return IntervalAvgLoss(steps_per_epoch, interval)
    return _bind


# high performance impact, should be run on a per-epoch basis
# NOTE: each hook requires the network to be rerun because otherwise the hook return values corrupt each other; this
# is believed to be due to a bug in pytorch running multiple hooks with nested modules
def layer_stats(
        net: Module,
        run_func: Callable[[], Any],
        hook_funcs: Iterable[Callable[[], Tuple[ModuleProperty, str]]]
) -> Bindable[int, Callable[[], LabelledValue]]:
    def _bind(steps_per_epoch: int):
        def _run():
            for hook_func in hook_funcs:
                results = []
                run_with_hook(
                    net,
                    run_func,
                    hook_func,
                    lambda result: results.append(result)
                )()
                yield LabelledValue(hook_func.__name__, results)
        return _run
    return _bind

