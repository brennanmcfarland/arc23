from torch import nn


class GAN(nn.Module):
    def __init__(self, generator, discriminator):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator

    def forward(self, x):
        # preserve batch size (the first size dimension)
        return self.discriminator(self.generator(x))

