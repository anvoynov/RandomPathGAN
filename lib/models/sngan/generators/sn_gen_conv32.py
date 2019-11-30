# DCGAN-like generator and discriminator
from torch import nn


from modules.blocks_bunch import BlocksBunch
from modules.lazy_batch_norm import LazyBatchNorm
from modules.utils import Identity, Reshape
from path_generator import PathGenerator


def make_sn_conv32_path_generator(latent_dim=128, branches_per_layer=8, img_size=64,
                                  batch_norm='none', equal_split=False, leak=0.0, seed_dim=4,
                                  first_linear=True, **kwargs):
    def linear_layer_gen(in_dim, out_dim):
        def make_linear_layer():
            return nn.Linear(in_dim, out_dim)
        return make_linear_layer

    def upconv_layer_gen(in_channels, out_channels, kernel, stride, padding=(0, 0)):
        def make_upconv_layer():
            if batch_norm == 'per_block':
                bn = nn.BatchNorm2d(out_channels)
            elif batch_norm == 'lazy':
                bn = LazyBatchNorm(out_channels)
            else:
                bn = Identity()

            return nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding),
                bn,
            )
        return make_upconv_layer

    def make_blocks_bunch(in_channels, out_channels, blocks, kernel, stride, padding=(0, 0)):
        bn = nn.BatchNorm2d(out_channels) if batch_norm == 'common' else Identity()
        return nn.Sequential(
            BlocksBunch(upconv_layer_gen(in_channels, out_channels, kernel, stride, padding),
                        blocks, equal_split),
            bn,
            nn.LeakyReLU(leak),
        )


    if not (isinstance(branches_per_layer, list) or isinstance(branches_per_layer, tuple)):
        branches_per_layer = [branches_per_layer] * 5

    if first_linear:
        first_layer = [
            BlocksBunch(linear_layer_gen(
                latent_dim, seed_dim * seed_dim * 512), branches_per_layer[0], equal_split),
            Reshape([-1, 512, seed_dim, seed_dim])
        ]
    else:
        first_layer = [make_blocks_bunch(latent_dim, 512, branches_per_layer[0], 4, 1)]

    model = nn.Sequential(*(
        first_layer +
        [
            make_blocks_bunch(512, 256, branches_per_layer[1], 4, 2, (1, 1)),
            make_blocks_bunch(256, 128, branches_per_layer[2], 4, 2, (1, 1)),
            make_blocks_bunch(128, 64, branches_per_layer[3], 4, 2, (1, 1)),
            BlocksBunch(upconv_layer_gen(64, 1, 3, 1, (1, 1)), branches_per_layer[4], equal_split),
            nn.Tanh()
        ]))

    latent_dim = [latent_dim] if first_linear else [latent_dim, 1, 1]
    return PathGenerator(model, latent_dim, [1, img_size, img_size], **kwargs)
