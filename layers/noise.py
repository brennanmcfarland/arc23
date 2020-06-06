import torch


# TODO: generalize and clean up, rename based on generalized functionality?
# TODO: the noise itself should be generated in the training step, with this module just scaling it
class ScaledNoise(torch.nn.Module):
    def __init__(self, channels):
        super(ScaledNoise, self).__init__()
        self.channels = channels
        # TODO: better way to initialize?
        self.per_feature_scaling = torch.nn.Parameter(torch.normal(torch.zeros(channels), torch.ones(channels))) # amount to scale noise input to each feature map by

    # x is nominal input, y is noise input
    def forward(self, x, y):
        # TODO: not sure if this should be 1 because of batch dimension or not
        # broadcast y across all channels
        projection_size = list(y.size())
        projection_size[1] = self.channels
        y = y.expand(projection_size)
        per_feature_scaling = self.per_feature_scaling.expand(projection_size)
        #print(y.data.cpu().numpy())
        # for i in range(self.per_feature_scaling.size()[0]):
        #     y[:, i] = y[:, i] * self.per_feature_scaling[i]
        y = y * per_feature_scaling
        #print(y.data.cpu().numpy())
        return x + y  # TODO: make add it's own layer?
