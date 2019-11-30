import numpy as np
import torch
from torch import nn
from modules.blocks_bunch import BlocksBunch
from path_generator import PathGenerator


def make_mnist_path_generator(latent_dim=100, branches_per_layer=8, img_size=28,
                              constant_noise=True, backprop_noise=False):
    def block_generator(in_feat, out_feat, normalize=True):
        def make_block():
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return torch.nn.Sequential(*layers)
        return make_block

    def last_layer_generator(out_dim):
        def make_last_layer():
            return nn.Linear(1024, out_dim)
        return make_last_layer

    img_shape = [1, img_size, img_size]

    if branches_per_layer is not tuple or branches_per_layer is not list:
        branches_per_layer = [branches_per_layer] * 5
    model = nn.Sequential(
        BlocksBunch(block_generator(latent_dim, 128, normalize=False), branches_per_layer[0]),
        BlocksBunch(block_generator(128, 256, normalize=False), branches_per_layer[1]),
        BlocksBunch(block_generator(256, 512, normalize=False), branches_per_layer[2]),
        BlocksBunch(block_generator(512, 1024, normalize=False), branches_per_layer[3]),
        BlocksBunch(last_layer_generator(int(np.prod(img_shape))), branches_per_layer[4]),
        nn.Tanh()
    )

    return PathGenerator(model, [latent_dim], img_shape,
                         constant_noise=constant_noise, backprop_noise=backprop_noise)


class MNISTDiscriminator(nn.Module):
    def __init__(self, img_size=28):
        super(MNISTDiscriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(img_size * img_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity
