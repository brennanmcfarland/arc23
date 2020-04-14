from typing import Iterable
from __types import Module


# get whether this module and each submodule recursively is in train or evaluation mode
def get_train_mode_tree(module: Module) -> Iterable[bool]:
    return [m.training for m in module.modules()]


# set the train or evaluation state of this module and each submodule recursively
def set_train_mode_tree(module: Module, mode: Iterable[bool]) -> None:
    for m, modl in zip(mode, module.modules()):
        modl.train(m)
