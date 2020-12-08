from __types import Bindable, LabelledValue
from typing import Callable


# wrap the return value of a bindable in another function
def wrap_return(bindable: Bindable, func: Callable) -> Bindable:
    def _bind(*args, **kwargs):
        bound = bindable(*args, **kwargs)

        def _run(*args, **kwargs):
            return func(bound(*args, **kwargs))
        return _run
    return _bind


def item_from_tensor(bindable: Bindable) -> Bindable:
    return wrap_return(bindable, lambda x: LabelledValue(x.label, x.value.item()))
