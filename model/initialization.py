import torch.nn as nn
from typing import Iterable
from __types import Module


# given an iterable of layers, create a network
def from_iterable(layers: Iterable[Module]) -> Module:
    return nn.Sequential(*layers)
