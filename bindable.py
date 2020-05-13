from __types import Bindable
from typing import Callable


# wrap the return value of a bindable in another function
def wrap_return(bindable: Bindable, func: Callable) -> Bindable:
    def _bind(*args, **kwargs):
        bound = bindable(*args, **kwargs)

        def _run(*args, **kwargs):
            return func(bound(*args, **kwargs))
        return _run
    return _bind
