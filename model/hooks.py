import torch


def weight_stats_hook(stat_funcs):
    def _bind():
        def _run(layer, input, output):
            return _run_stats_hook(stat_funcs, layer, lambda: layer.weight)
        return _run, 'forward'
    return _bind


def output_stats_hook(stat_funcs):
    def _bind():
        def _run(layer, input, output):
            return _run_stats_hook(stat_funcs, layer, lambda: output)
        return _run, 'forward'
    return _bind


def grad_stats_hook(stat_funcs):
    def _bind():
        def _run(layer, grad_input, grad_output):
            return _run_stats_hook(stat_funcs, layer, lambda: grad_output)
        return _run, 'backward'
    return _bind


def _run_stats_hook(stat_funcs, layer, tensor_func):
    stats = [None for func in stat_funcs]
    if hasattr(layer, 'weight'):
        stats = [_apply_to_tensors_in_iterable(tensor_func(), func) for func in stat_funcs]
        stats = _tensor_items_in_iterable(stats)
    func_names = [func.__name__ for func in stat_funcs]
    labelled_stats = [type(layer).__name__] + list(zip(func_names, stats))
    return labelled_stats


# helper function to extract the values of scalar tensors from a possibly nested iterable
# may want to move somewhere more general if ever needed
def _tensor_items_in_iterable(tensors):
    return _apply_to_tensors_in_iterable(tensors, lambda t: t.item())


# helper function to apply a function to each tensor in a possibly nested iterable
# may want to move somewhere more general if ever needed
def _apply_to_tensors_in_iterable(tensors, func):
    if not isinstance(tensors, torch.Tensor):
        return [_apply_to_tensors_in_iterable(t, func) for t in tensors]
    else:
        return func(tensors)
