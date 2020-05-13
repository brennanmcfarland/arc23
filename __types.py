from typing import *
from enum import Enum
from collections import namedtuple

from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

# classes inherit from NamedTuple to enforce immutability
# TODO: "progress" type containing step + epoch?


# TYPE VARIABLES


C = TypeVar("C", bound=Callable)
T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


# DATA TYPES


Device = str


class LabelledValue(NamedTuple, Generic[T]):
    label: str
    value: T


# may refactor this if more than steps_per_epoch is needed in the future, that's why this is its own type
OutputBinding = int

ModuleType = type(Module)

Table = Sequence[Sequence[T]]

Loader = DataLoader

Shape = Tuple

OneOrMany = Union[T, Iterable[T]]


# shape inference implementation overrides for custom layers
CustomLayerTypes = Dict[str, Callable[[Module, Sequence[Module]], Shape]]

# shape inference implementation overrides for custom superclasses of layers
CustomLayerSuperclasses = Dict[Module, Callable[[Module, Sequence[Module]], Shape]]


# FUNCTIONS AND FUNCTION COLLECTIONS


class MetricFuncs(NamedTuple):
    on_item: Callable
    on_end: Callable


# a function that receives loss and progress and returns a labelled value from it
FuncOfLoss = Callable[[float, int, int], LabelledValue]


# a function of a module's weights, inputs, or gradients
ModuleProperty = Callable[[Module, Tensor, Tensor], Tensor]

# a function guaranteed to not return anything
Method = Callable[[Optional[T]], None]

# a function accepting arguments to bind to another function which it returns (typing system prevents enforcing the
# return type as a function though)
Bindable = Callable[[T], C]

# a bindable that also returns something else
AnnotatedBindable = Callable[[T], Tuple[C, U]]

# a function that receives and returns a tensor
TensorOp = Callable[[Tensor], Tensor]
