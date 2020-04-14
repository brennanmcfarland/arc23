from typing import Any, Callable

from functional.core import pipe as core_pipe


# maybe monad bind operation
def bind(f: Callable, x: Any):
    return None if x is None else f(x)


# maybe monad pipe operation
def pipe(*fs: Callable):
    return core_pipe([lambda x: bind(f, x) for f in fs])


# pipe until not none
def pipe_until(*fs: Callable):
    return core_pipe([lambda x: not bind(f, x) for f in fs])
