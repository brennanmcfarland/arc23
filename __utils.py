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
# TODO: rename to return_on_interval? or does that not make as much semantic sense?
def on_interval(func, interval):
    def get_return_value(o, *args, **kwargs):
        if o.call_counter == 0:
            return func(*args, **kwargs)
    return _on_interval(get_return_value, interval)


# cache the result of the given function and only recalculate it every nth time it is called as well as initially
# the arguments to the function only matter every nth time it is called, otherwise the cached value is just returned
def cache_on_interval(func, interval):
    def get_return_value(o, *args, **kwargs):
        # 1 ensures it's calculated initially
        if o.call_counter == 1:
            o.return_value = func(*args, **kwargs)
        return o.return_value
    return _on_interval(get_return_value, interval)


# helper function to do/return based on func on an interval counter
def _on_interval(func, interval):
    class _OnInterval:

        def __init__(self, func, interval):
            self.interval = interval
            self.call_counter = 0
            self.func = func
            self.return_value = None

        def __call__(self, *args, **kwargs):
            self.call_counter = (self.call_counter + 1) % self.interval
            return func(self, *args, **kwargs)
    return _OnInterval(func, interval)
