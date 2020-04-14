import torch

from typing import Iterable, Callable, Any, Union, Optional

from __types import AnnotatedBindable, ModuleProperty, Tensor, Module


def weight_stats_hook(stat_funcs: Iterable[Callable]) -> AnnotatedBindable[None, ModuleProperty, str]:
    def _bind():
        def _run(layer, input, output):
            return _run_stats_hook(stat_funcs, layer, lambda: layer.weight)
        return _run, 'forward'
    return _bind


def output_stats_hook(stat_funcs: Iterable[Callable]) -> AnnotatedBindable[None, ModuleProperty, str]:
    def _bind():
        def _run(layer, input, output):
            return _run_stats_hook(stat_funcs, layer, lambda: output)
        return _run, 'forward'
    return _bind


def grad_stats_hook(stat_funcs: Iterable[Callable]) -> AnnotatedBindable[None, ModuleProperty, str]:
    def _bind():
        def _run(layer, grad_input, grad_output):
            return _run_stats_hook(stat_funcs, layer, lambda: grad_output)
        return _run, 'backward'
    return _bind


def _run_stats_hook(stat_funcs: Iterable[Callable], layer: Module, tensor_func: Callable[[], Tensor]) -> Iterable:
    stats: Union[Iterable[Optional[Tensor]]] = [None for _ in stat_funcs]
    if hasattr(layer, 'weight'):
        stats = [_apply_to_tensors_in_iterable(tensor_func(), func) for func in stat_funcs]
        stats = _tensor_items_in_iterable(stats)
    func_names = [func.__name__ for func in stat_funcs]
    labelled_stats = [type(layer).__name__] + list(map(str, zip(func_names, stats)))
    return labelled_stats


# helper function to extract the values of scalar tensors from a possibly nested iterable
# may want to move somewhere more general if ever needed
def _tensor_items_in_iterable(tensors: Iterable[Tensor]) -> Iterable[Tensor]:
    return _apply_to_tensors_in_iterable(tensors, lambda t: t.item())


# helper function to apply a function to each tensor in a possibly nested iterable
# may want to move somewhere more general if ever needed
def _apply_to_tensors_in_iterable(tensors: Iterable[Tensor], func: Callable[[Tensor], Any]):
    if not isinstance(tensors, Tensor):
        return [_apply_to_tensors_in_iterable(t, func) for t in tensors]
    else:
        return func(tensors)
