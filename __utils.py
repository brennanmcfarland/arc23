from typing import *
from __types import MetricFuncs


# reduces a list to it's item if there's only one, else returns the list unchanged
def try_reduce_list(l):
    l = list(l)
    if len(l) == 1:
        return l[0]
    else:
        return l


# hook is the specific point at which the callbacks are being called, eg on_step
def run_callbacks(hook, callbacks, *args):
    results = []
    if hook in callbacks:
        for callback in callbacks[hook]:
            results.append(callback(*args))
    return results


# hook is the specific point at which the metrics are being called, eg on_item
def run_metrics(hook, metrics: Iterable[MetricFuncs], *args):
    results = []
    for metric in metrics:
        results.append(getattr(metric, hook)(*args))
    return results


# return the return value of func every nth time this is called, else return None
def on_interval(func, interval):
    class OnInterval:

        def __init__(self, func, interval):
            self.interval = interval
            self.call_counter = 0
            self.func = func

        def __call__(self, *args, **kwargs):
            self.call_counter = (self.call_counter + 1) % self.interval
            if self.call_counter == 0:
                return self.func(*args, **kwargs)
    return OnInterval(func, interval)
