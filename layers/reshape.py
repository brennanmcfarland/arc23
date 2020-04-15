from torch import nn


class Reshape(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Reshape, self).__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        return x.view(*self.args, **self.kwargs)
