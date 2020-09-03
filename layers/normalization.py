import torch
import torch.nn.init
from functools import reduce
from operator import mul
from math import sqrt


class PixelNorm(torch.nn.Module):

    def __init__(self, eps=1e-8):
        super(PixelNorm, self).__init__()
        self.eps = eps

    def forward(self, x):
        return x * torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.eps)


class MinibatchStdDev(torch.nn.Module):

    def __init__(self):
        super(MinibatchStdDev, self).__init__()

    # TODO: get this to not error out, then it should be done
    def forward(self, x):
        # TODO: the problem is gradients are 0 because sometimes the fakes are all the same if untrained in the right way,
        # TODO: that will probably go away once noise is added to the generator
        # TODO: not sure if this should be with grad or not
        with torch.no_grad():
            stddev = torch.std(x, 0)
            stddev = torch.mean(stddev)
            repeat_dims = list(x.size())
            repeat_dims[1] = 1  # repeating over batch and location, but only want one extra feature map
            stddev = stddev.repeat(repeat_dims[0], 1, *repeat_dims[2:])
            return torch.cat([x, stddev], 1)


def EqualizedLR(sublayer):
    if hasattr(sublayer, 'weight_unmodded'):
        return _equalizeLR(sublayer, 'weight_unmodded')
    elif hasattr(sublayer, 'weight'):
        return _equalizeLR(sublayer)
    else:
        return sublayer


# TODO: cleanup, rename
class EQLR(torch.nn.Module):
    def __init__(self, sublayer):
        super(EQLR, self).__init__()
        self.scale_factor = 1
        weight_count = reduce(mul, sublayer.weight.size())
        if weight_count > 0:
            self.scale_factor = (2 / weight_count) ** .5
        self.sublayer = sublayer
        self.weight = self.sublayer.weight  # TODO: fix this, terribly hacky
        if hasattr(self.sublayer, 'bias'):
            self.bias = self.sublayer.bias
        self.weight_s = torch.nn.Parameter(self.weight * self.scale_factor)
        self.sublayer.register_forward_pre_hook(self.scaling_hook)
        self.submodule = torch.nn.ModuleList([sublayer])

    def forward(self, x):
        # setattr(self.sublayer, 'weight', getattr(self, 'weight_s'))
        return self.sublayer(x)

    def scaling_hook(self, module, input):
        weight = getattr(self.sublayer, 'weight')
        #setattr(self.sublayer, 'weight', self._apply_scaling(weight, y))
        setattr(self.sublayer, 'weight', torch.nn.Parameter(self._apply_scaling(weight)))
        return input

    def _apply_scaling(self, weight):
        return weight * self.scale_factor


# TODO: this is copy-pasted, adapt it
class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        if hasattr(module._parameters, name):
            del module._parameters[name]
        module.register_parameter(name + '_orig', torch.nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)
        # weight = fn.compute_weight(module)
        # setattr(module, fn.name, weight)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        # module.weight = module.weight.masked_fill(weight.new_ones(weight.size(), dtype=torch.bool).view(-1), weight.view(-1))
        # TODO: may or may not need nograd, not sure
        with torch.no_grad():
            setattr(module, self.name, torch.nn.Parameter(weight))


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module


# NOTE: will not work for layers without a weight attribute, since that's what's used to scale
def _equalizeLR(sublayer, param_name='weight'):
    EqualLR.apply(sublayer, param_name)
    # TODO: this is new, enforces initialization, may want to do that elsewhere though
    torch.nn.init.normal_(sublayer.weight_orig)
    return sublayer
    # return EQLR(sublayer)
    weight_count = reduce(mul, sublayer.weight.size())
    if weight_count == 0:
        scale_factor = 1.
    else:
        scale_factor = (2 / weight_count) ** .5

    def compute_equalized(module):
        return getattr(module, 'weight_s')

    def pre_hook(module, input):
        setattr(module, 'weight', compute_equalized(module))

    weight = getattr(sublayer, 'weight')
    del sublayer._parameters['weight']
    sublayer.register_parameter('weight_s', torch.nn.Parameter(weight * scale_factor))
    setattr(sublayer, 'weight', compute_equalized(sublayer))
    sublayer.register_forward_pre_hook(pre_hook)

    # TODO: this is new, enforces initialization, may want to do that elsewhere though
    torch.nn.init.normal_(weight)
    return sublayer


# TODO: need to abstract out changing weights with one or more intermediate variables so they don't just overwrite each other
# TODO: / aren't as order-dependent
class _ModDemod:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        w = getattr(module, self.name + '_unmodded')
        # use each scalar style from y to fill out a weight-sized feature map so they can be element-wise multiplied
        # the feature map (in channels) is dimension 1 of the weights
        projection_dims = [*w.size()[:2], *[1 for _ in w.size()[2:]]]
        # projection_dims[1] = y.size()[0]
        projection_size = [*projection_dims[:2], *w.size()[2:]]
        projected_y = torch.reshape(y, projection_dims).expand(projection_size)
        modded = w * projected_y
        eps_term = torch.ones(size=modded.size(), device='cuda:0') * self.eps
        normalizer = torch.norm((modded ** 2) + eps_term)
        demodded = modded / normalizer
        return demodded

    @staticmethod
    def apply(module, name):
        fn = _ModDemod(name)

        weight = getattr(module, name)
        if hasattr(module._parameters, name):
            del module._parameters[name]
        module.register_parameter(name + '_unmodded', torch.nn.Parameter(weight.data))
        module.register_parameter(name + '_modded', torch.nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)
        # weight = fn.compute_weight(module)
        # setattr(module, fn.name, weight)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        # module.weight = module.weight.masked_fill(weight.new_ones(weight.size(), dtype=torch.bool).view(-1), weight.view(-1))
        # TODO: may or may not need nograd, not sure
        with torch.no_grad():
            setattr(module, self.name, torch.nn.Parameter(weight))
            setattr(module, self.name + '_modded', torch.nn.Parameter(weight))


class ModDemod(torch.nn.Module):
    def __init__(self, sublayer, style_layer, eps=1e-8):
        super(ModDemod, self).__init__()
        self.sublayer = sublayer
        self.style_layer = style_layer
        self.eps = eps
        self.sublayer.register_forward_pre_hook(self.scaling_hook)

    def scaling_hook(self, module, input):
        (x, y) = input
        weight = getattr(self.sublayer, 'weight')
        #setattr(self.sublayer, 'weight', self._apply_scaling(weight, y))
        self.sublayer.weight = torch.nn.Parameter(self._apply_scaling(weight, y))
        return x  # the return value is set as the input to the layer


    # y is the style/scaling input, x is the nominal input
    # TODO: some of this can/should probably be moved to initialization for efficiency
    def forward(self, x, y):
        return self.sublayer(x, y)
        #weight = getattr(self.sublayer, 'weight')
        # TODO: fix this error
        #setattr(self.sublayer, 'weight', self._apply_scaling(weight, y))
        # with torch.no_grad():
        #     self.sublayer.weight.masked_fill_(torch.ones(size=(torch.numel(self.sublayer.weight),), dtype=torch.bool), self._apply_scaling(weight, y))

    # TODO: obviously clean up/make efficient
    def _apply_scaling(self, w, y):
        # use each scalar style from y to fill out a weight-sized feature map so they can be element-wise multiplied
        # the feature map (in channels) is dimension 1 of the weights
        projection_dims = [*w.size()[:2], *[1 for _ in w.size()[2:]]]
        #projection_dims[1] = y.size()[0]
        projection_size = [*projection_dims[:2], *w.size()[2:]]
        projected_y = torch.reshape(y, projection_dims).expand(projection_size)
        modded = w * projected_y
        eps_term = torch.ones(size=modded.size(), device='cuda:0') * self.eps
        normalizer = torch.norm((modded ** 2) + eps_term)
        demodded = modded / normalizer
        return demodded

# # TODO: generalized weight scaling func, for efficiency to combine w equalized LR?
# # NOTE: will not work for layers without a weight attribute, since that's what's used to scale
# def ModDemod(sublayer, style, eps=1e-8):
#
#     def pre_hook(module, input):
#         setattr(module, 'weight', compute_equalized(module))
#
#     weight = getattr(sublayer, 'weight')
#     del sublayer._parameters['weight']
#     sublayer.register_parameter('weight_s', torch.nn.Parameter(weight * scale_factor))
#     setattr(sublayer, 'weight', compute_equalized(sublayer))
#     sublayer.register_forward_pre_hook(pre_hook)
#     return sublayer



# # NOTE: will not work for layers without a weight attribute, since that's what's used to scale
# class _EqualizedLR(torch.nn.Module):
#
#     def __init__(self, sublayer):
#         super(_EqualizedLR, self).__init__()
#         self.sublayer = sublayer
#         with torch.no_grad():
#             weight_count = reduce(mul, sublayer.weight.size())
#             if weight_count == 0:
#                 self.scale_factor = 1.
#             else:
#                 self.scale_factor = (2 / weight_count)**.5
#
#     def forward(self, x):
#         with torch.no_grad():
#             self.sublayer.weight.mul_(self.scale_factor)
#         x = self.sublayer(x)
#         with torch.no_grad():
#             self.sublayer.weight.div_(self.scale_factor)
#         return x
