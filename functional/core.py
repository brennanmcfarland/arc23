from typing import Callable


def pipe(*fs: Callable) -> Callable:
    if len(fs) == 0:
        return lambda x: x

    def _apply(x):
        if len(fs) == 1:
            return fs[0](x)
        else:
            return pipe(*fs[1:])(fs[0](x))
    return _apply


def ffork(*fs: Callable) -> Callable:
    def _apply(*args, **kwargs):
        return (f(*args, **kwargs) for f in fs)
    return _apply
