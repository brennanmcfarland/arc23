from typing import Sequence, Union

from __types import Module, Tensor


# NOTE: input and output dimensions must match, and only tensor(s) can be passed in input
# a residual block, in which the output of the submodule is added to the input
class Residual(Module):
    def __init__(self, submodule: Module):
        super(Residual, self).__init__()
        self.submodule = submodule

    def forward(self, input_tensors: Union[Tensor, Sequence[Tensor]]) -> Union[Tensor, Sequence[Tensor]]:
        return input_tensors + self.submodule(input_tensors)
