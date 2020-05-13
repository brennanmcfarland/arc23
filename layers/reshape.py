from torch import nn


class Reshape(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Reshape, self).__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        # preserve batch size (the first size dimension)
        return x.view(x.size()[0], *self.args, **self.kwargs)
