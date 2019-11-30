import numpy as np
from torch import nn

from modules.utils import Identity
from modules.blocks_bunch import BlocksBunch
from path_generator import PathGenerator


WIDTH = 128


def make_sn_fc_path_generator(latent_dim=128, branches_per_layer=8, img_size=28, image_channels=1,
                              equal_split=False, leak=0.0, batch_norm='common',
                              **kwargs):
    def block_generator(in_feat, out_feat):
        def make_block():
            return nn.Sequential(
                nn.Linear(in_feat, out_feat),
                nn.LeakyReLU(leak)
            )
        return make_block

    def last_layer_generator(out_dim):
        def make_last_layer():
            return nn.Linear(8 * WIDTH, out_dim)
        return make_last_layer

    img_shape = [image_channels, img_size, img_size]

    if not (isinstance(branches_per_layer, tuple) or isinstance(branches_per_layer, list)):
        branches_per_layer = [branches_per_layer] * 5

    normalization_module = nn.BatchNorm1d if batch_norm == 'common' else Identity
    model = nn.Sequential(
        BlocksBunch(block_generator(latent_dim, WIDTH),
                    branches_per_layer[0], equal_split=equal_split),
        normalization_module(WIDTH),
        BlocksBunch(block_generator(WIDTH, 2 * WIDTH),
                    branches_per_layer[1], equal_split=equal_split),
        normalization_module(2 * WIDTH),
        BlocksBunch(block_generator(2 * WIDTH, 4 * WIDTH),
                    branches_per_layer[2], equal_split=equal_split),
        normalization_module(4 * WIDTH),
        BlocksBunch(block_generator(4 * WIDTH, 8 * WIDTH),
                    branches_per_layer[3], equal_split=equal_split),
        normalization_module(8 * WIDTH),
        BlocksBunch(last_layer_generator(int(np.prod(img_shape))),
                    branches_per_layer[4], equal_split=equal_split),
        nn.Tanh()
    )

    return PathGenerator(model, [latent_dim], img_shape, **kwargs)
