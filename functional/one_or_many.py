from typing import Any, Callable, Iterable


# one or many monad bind operation
def bind(f: Callable, x: Any):
    if type(x) is Iterable:
        for xi in x:
            yield f(xi)
    else:
        return f(x)
