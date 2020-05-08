import torch.nn as nn

from typing import Sequence, MutableSequence

from __types import Module, Loader, CustomLayerTypes, CustomLayerSuperclasses, Tensor, ModuleType, Shape, Any

from model.execution import Trainer
from app import dry_run # TODO: fix


# assumes that any nested submodules have already had their shape inferred if necessary
# TODO: in the future may want a recursive function for that
def infer_shapes(layers: MutableSequence[Module], loader: Loader) -> Sequence[Module]:
    infer = ShapeInferer(loader)
    for l in range(len(layers[1:])):
        # TODO: fix, shouldn't have actual inference logic here (but we don't want inference object to be stateful, either) - maybe just make it recursive?
        if type(layers[l]).__name__ is 'Reshape':
            layers[l+1] = layers[l+1](infer(layers[:l]))
        else:
            layers[l+1] = layers[l+1](infer(layers[:l+1]))

    return layers[1:]


class Input(nn.Module):
    def __init__(self, input_size: Shape = None):
        super(Input, self).__init__()
        self.size = input_size

    def forward(self, input):
        return input


# can be passed a loader, in which case all supported shapes can utilize shape inference, or not, in which case
# only certain layers can utilize shape inference
# custom shape inference logic can be passed in via custom_layer_types and custom_layer_superclasses
# custom layers take precedence, then custom layer superclasses, then built-in implementation
# device isn't used since it's probably less efficient to move the incrementally created network each time, anyway
# if the loader is used, it's assumed every example will have the same number of channels/features
# TODO: abstract out base implementation into CustomLayerTypes and CustomLayerSuperclasses (and rename?)
class ShapeInferer:
    def __init__(
            self,
            loader: Loader = None,
            custom_layer_types: CustomLayerTypes = None,
            custom_layer_superclasses: CustomLayerSuperclasses = None
    ):
        self.shape: int
        self.loader = loader

        if custom_layer_types is None:
            self.custom_layer_types = {}
        else:
            self.custom_layer_types = custom_layer_types

        if custom_layer_superclasses is None:
            self.custom_layer_superclasses = {}
        else:
            self.custom_layer_superclasses = custom_layer_superclasses

    def __call__(self, prev_layers: Sequence[Module]) -> Shape:
        return self.infer(prev_layers)

    def infer(self, prev_layers: Sequence[Module]) -> Shape:
        self.shape = self._infer(prev_layers[-1], prev_layers)
        return self.shape

    # helper function
    # feed an example through the data loader for manually capturing the output shape, returns the output tensor
    def _run_prev_layers(self, prev_layers: Sequence[Module], layer_type: ModuleType) -> Tensor:
        model = nn.Sequential(*prev_layers)

        def _run(net, trainer, device):
            def _r(inputs, gtruth):
                return model(inputs.to('cpu'))  # all shape inference is done on cpu
            return _r

        if self.loader is not None:
            return dry_run(model, self.loader, None, _run)()
        else:
            raise TypeError("A data loader must be provided for shape inference with " + layer_type.__name__)

    def _infer(self, prev_layer: Module, prev_layers: Sequence[Module]) -> int:
        layer_type = type(prev_layer)
        if layer_type.__name__ in self.custom_layer_types:
            # the value returned from the custom layer inference
            return self.custom_layer_types[layer_type.__name__](prev_layer, prev_layers)
        elif any((issubclass(layer_type, c) for c in self.custom_layer_superclasses.keys())):
            # the value returned from the custom layer superclass inference
            return self.custom_layer_superclasses[layer_type.__name__](prev_layer, prev_layers)
        elif layer_type.__name__ is 'Input':
            # the Input's size if it has one, else get an example from the data loader
            if prev_layer.size is not None:
                return prev_layer.size
            else:
                if self.loader is None:
                    raise ValueError("must provide a loader to shape inference for layer type " + layer_type.__name__)
                inputs = next(iter(self.loader))['inputs']
                return inputs.size()[1]
        elif (issubclass(layer_type, nn.modules.conv._ConvNd)
              or layer_type.__name__ is 'Linear'
              ):
            # the out_features dim
            return prev_layer.weight.size()[0]
        elif layer_type.__name__ is 'Flatten':
            # feed an example through the data loader and manually capture the output shape
            return self._run_prev_layers(prev_layers, layer_type).size()[-1]
        elif (layer_type.__name__ is 'ReLU' or 'LeakyReLU' or 'Residual' or 'Upsample'
                or issubclass(layer_type, nn.modules.batchnorm._BatchNorm)
                or issubclass(layer_type, nn.modules.pooling._MaxPoolNd)
                or issubclass(layer_type, nn.modules.padding._ReflectionPadNd)
                or issubclass(layer_type, nn.modules.dropout._DropoutNd)):
            # reuse and pass along the previously inferred shape unchanged
            return self._infer(prev_layers[-2], prev_layers[:-1])
        elif layer_type.__name__ is 'Reshape':
            # TODO: cleanup/generalize, and this may not be correct (should be # filters/channels?)
            return prev_layer.weight.size[0]
        else:
            raise NotImplementedError("No shape inference implementation for layer of type " + layer_type.__name__)
