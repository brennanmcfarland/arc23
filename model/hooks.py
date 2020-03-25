import torch

# TODO: clean up this whole file
# TODO: the problem is hooks can't return except for modified output, which is screwing with the network data flow


def layer_stats_hook(stat_funcs):
    def _bind():
        def _run(layer, input, output):
            stats = [None for func in stat_funcs]
            if hasattr(layer, 'weight'):
                stats = [_apply_to_tensors_in_iterable(layer.weight, func) for func in stat_funcs]
                stats = _tensor_items_in_iterable(stats)
            func_names = [func.__name__ for func in stat_funcs]
            labelled_stats = [type(layer).__name__] + list(zip(func_names, stats))
            return labelled_stats
        return _run, 'forward'
    return _bind


def output_stats_hook(stat_funcs):
    def _bind():
        def _run(layer, input, output):
            stats = [None for func in stat_funcs]
            if hasattr(layer, 'weight'):
                stats = [_apply_to_tensors_in_iterable(output, func) for func in stat_funcs]
                stats = _tensor_items_in_iterable(stats)
            func_names = [func.__name__ for func in stat_funcs]
            labelled_stats = [type(layer).__name__] + list(zip(func_names, stats))
            return labelled_stats
        return _run, 'forward'
    return _bind


def grad_stats_hook(stat_funcs):
    def _bind():
        def _run(layer, grad_input, grad_output):
            stats = [None for func in stat_funcs]
            if hasattr(layer, 'weight'):
                stats = [_apply_to_tensors_in_iterable(grad_output, func) for func in stat_funcs]
                stats = _tensor_items_in_iterable(stats)
            func_names = [func.__name__ for func in stat_funcs]
            labelled_stats = [type(layer).__name__] + list(zip(func_names, stats))
            return labelled_stats
        return _run, 'backward'
    return _bind


# NOTE: only runs in the pass defined by pass_type, need to add two hooks to run it on both forward and backward passes
def cuda_profile_layer(pass_type, memory_allocated_func, memory_cached_func, device):
    def _bind():
        def _run(layer, *_):
            return [
                pass_type,
                type(layer).__name__,
                memory_allocated_func(device),
                memory_cached_func(device)
            ]
        return _run, pass_type
    return _bind


# TODO: cleanup and maybe merge w tensor_items_in_iterable?
def _apply_to_tensors_in_iterable(tensors, func):
    if not isinstance(tensors, torch.Tensor):
        return [_apply_to_tensors_in_iterable(t, func) for t in tensors]
    else:
        return func(tensors)


# helper function to extract the values of scalar tensors from a possibly nested iterable
# may want to move somewhere more general if ever needed
def _tensor_items_in_iterable(tensors):
    if not isinstance(tensors, torch.Tensor):
        return [_tensor_items_in_iterable(t) for t in tensors]
    else:
        return tensors.item()
