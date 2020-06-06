import torch
from functools import reduce
from operator import mul


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
        stddev = torch.std(x, 0)
        stddev = torch.mean(stddev)
        repeat_dims = list(x.size())
        repeat_dims[1] = 1  # repeating over batch and location, but only want one extra feature map
        stddev = stddev.repeat(repeat_dims[0], 1, *repeat_dims[2:])
        return torch.cat([x, stddev], 1)


def EqualizedLR(sublayer):
    if hasattr(sublayer, 'weight'):
        return _equalizeLR(sublayer)
    else:
        return sublayer


# NOTE: will not work for layers without a weight attribute, since that's what's used to scale
def _equalizeLR(sublayer):
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
    return sublayer


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
        # TODO: demod, ie the denominator part
        eps_term = torch.ones(size=modded.size(), device='cuda:0') * self.eps
        # TODO: this may not be the correct norm
        normalizer = torch.norm(modded + eps_term)
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
