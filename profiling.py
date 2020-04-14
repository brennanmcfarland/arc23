import torch.cuda as cuda

from tabulate import tabulate
import functools  # TODO: may be a lot more here we can use, especially see single dispatch and cached property
import math
from typing import Callable, Any

from __types import Module, Device, Tensor


# NOTE: only works on CUDA
def profile_cuda_memory_by_layer(net: Module, run_func: Callable[[], Any], device: Device = None) -> None:
    assert cuda.is_available()

    profiler_hooks = []
    results_table = []

    def _profile_layer(net: Module, input: Tensor, output: Tensor, pass_type: str) -> None:
        results_table.append([
            pass_type,
            type(net).__name__,
            _format_memory_allocated(device),
            _format_memory_cached(device)
        ])

    def _add_profiler_hook(net: Module) -> None:
        profiler_hooks.append(net.register_forward_hook(functools.partial(_profile_layer, pass_type='FORWARD PASS')))
        profiler_hooks.append(net.register_backward_hook(functools.partial(_profile_layer, pass_type='BACKWARD PASS')))

    print("CUDA MEMORY PROFILE")

    cuda.empty_cache()
    cuda.reset_max_memory_allocated(device)
    cuda.reset_max_memory_cached(device)
    print("CUDA device initial allocated memory: ", _format_memory_allocated(device))
    print("CUDA device initial cached memory: ", _format_memory_cached(device))

    print('Name Allocated Cached')
    net.apply(_add_profiler_hook)

    # train step
    run_func()

    print(tabulate(results_table, headers=['Layer', 'Allocated', 'Cached']))

    print("CUDA device max allocated memory: ", _format_size(cuda.max_memory_allocated(device)))
    print("CUDA device max cached memory: ", _format_size(cuda.max_memory_cached(device)))

    for h in profiler_hooks:
        h.remove()

    cuda.empty_cache()
    cuda.reset_max_memory_allocated(device)
    cuda.reset_max_memory_cached(device)


def _format_memory_allocated(device: Device) -> str:
    return _format_size(cuda.memory_allocated(device))


def _format_memory_cached(device: Device) -> str:
    return _format_size(cuda.memory_cached(device))


def _format_size(bytes: int) -> str:
    if bytes == 0:
        return "0B"
    units = ('B', 'KB', 'MB', 'GB', 'TB')
    u = int(math.floor(math.log(bytes, 1024)))
    p = math.pow(1024, u)
    b = round(bytes / p, 2)
    return "%s %s" % (b, units[u])
