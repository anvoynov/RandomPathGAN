from collections import namedtuple
from torch import nn
import numpy as np

from path_generator import PathGenerator
from modules.blocks_bunch import BlocksBunch
from modules.utils import Reshape, Identity
from modules.lazy_batch_norm import LazyBatchNorm


GEN_SIZE=64


ResNetGenConfig = namedtuple('ResNetGenConfig', ['channels', 'seed_dim'])


SN_RES_GEN_CONFIGS = {
    'sn_resnet32': ResNetGenConfig([256, 256, 256, 256], 4),
    'sn_resnet48': ResNetGenConfig([512, 256, 128, 64], 6),
    'sn_resnet64': ResNetGenConfig([16 * 64, 8 * 64, 4 * 64, 2 * 64, 64], 4),
    'sn_resnet128': ResNetGenConfig([16 * 64, 16 * 64, 8 * 64, 4 * 64, 2 * 64, 64], 4),
    'sn_resnet256': ResNetGenConfig([16 * 64, 16 * 64, 8 * 64, 8 * 64, 4 * 64, 2 * 64, 64], 4)
}


class ResBlockGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=None, leak=0.0):
        super(ResBlockGenerator, self).__init__()

        self.bn1 = Identity() if batch_norm is None else batch_norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.bn2 = Identity() if batch_norm is None else batch_norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)

        nn.init.xavier_uniform_(self.conv1.weight.data, np.sqrt(2))
        nn.init.xavier_uniform_(self.conv2.weight.data, np.sqrt(2))

        self.model = nn.Sequential(
            self.bn1,
            nn.LeakyReLU(leak),
            nn.Upsample(scale_factor=2),
            self.conv1,
            self.bn2,
            nn.LeakyReLU(leak),
            self.conv2
            )

        if in_channels == out_channels:
            self.bypass = nn.Upsample(scale_factor=2)
        else:
            self.bypass = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
            )
            nn.init.xavier_uniform_(self.bypass[1].weight.data, 1.0)

    def forward(self, x):
        return self.model(x) + self.bypass(x)


def make_resnet_generator(resnet_gen_config, latent_dim=128, branches_per_layer=8,
                          img_size=128, batch_norm='none',
                          equal_split=False, leak=0.0,
                          image_channels=3, **kwargs):
    def make_dense():
        dense = nn.Linear(latent_dim, resnet_gen_config.seed_dim**2 * resnet_gen_config.channels[0])
        nn.init.xavier_uniform_(dense.weight.data, 1.)
        return dense

    def make_final():
        final = nn.Conv2d(resnet_gen_config.channels[-1], image_channels, 3, stride=1, padding=1)
        nn.init.xavier_uniform_(final.weight.data, 1.)
        return final

    def res_block_gen(in_channels, out_channels, batch_norm):
        if batch_norm == 'lazy':
            batch_norm = LazyBatchNorm
        elif batch_norm == 'per_block':
            batch_norm = nn.BatchNorm2d
        else:
            batch_norm = None

        def make_res_block():
            return ResBlockGenerator(in_channels, out_channels, batch_norm=batch_norm, leak=leak)
        return make_res_block

    channels = resnet_gen_config.channels
    if isinstance(branches_per_layer, int):
        branches_per_layer = [branches_per_layer] * (len(channels) + 1)

    input_layers = [
        BlocksBunch(make_dense, branches_per_layer[0], equal_split=equal_split),
        Reshape([-1, channels[0], 4, 4])
    ]
    res_blocks = [
        BlocksBunch(res_block_gen(channels[i], channels[i + 1], batch_norm),
                    branches_per_layer[i + 1], equal_split=equal_split)
        for i in range(len(channels) - 1)
    ]
    out_layers = [
        nn.BatchNorm2d(channels[-1]),
        nn.ReLU(),
        BlocksBunch(make_final, branches_per_layer[-1], equal_split=equal_split),
        nn.Tanh()
    ]

    model = nn.Sequential(*(input_layers + res_blocks + out_layers))

    return PathGenerator(model, [latent_dim], [image_channels, img_size, img_size], **kwargs)
