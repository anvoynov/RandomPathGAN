# based on the original improved wgan architecture for upsamples = 3
import torch
from torch import nn
from path_generator import PathGenerator
from modules.blocks_bunch import BlocksBunch
from modules.utils import Crop, Reshape, Identity

DIM = 64


def make_conv_path_generator(
        latent_dim=128, branches_per_layer=8, img_size=64, upsamples=3, batch_norm=False,
        equal_split=False, leak=0.0, **kwargs):
    def linear_layer_generator(in_dim, out_dim):
        def make_layer():
            layer = nn.Linear(in_dim, out_dim)
            return layer
        return make_layer

    def upconv_layer_generator(input_channels, out_channels, kernel):
        def make_layer():
            layer = nn.ConvTranspose2d(input_channels, out_channels, kernel, stride=2, padding=1)
            torch.nn.init.kaiming_uniform_(layer.weight)
            return layer
        return make_layer

    if not (isinstance(branches_per_layer, list) or isinstance(branches_per_layer, tuple)):
        branches_per_layer = [branches_per_layer] * (1 + upsamples)

    relu = nn.LeakyReLU(leak, True)
    crop = Crop([1, 1])

    # I input linear
    linear_out_dim = 2**(upsamples - 1) * 4 * 4 * DIM

    linear = BlocksBunch(
        linear_layer_generator(latent_dim, linear_out_dim),
        branches_per_layer[0],
        equal_split=equal_split)

    reshape = Reshape([-1, 2**(upsamples - 1) * DIM, 4, 4])
    bn = nn.BatchNorm1d(linear_out_dim) if batch_norm else Identity()
    input = nn.Sequential(linear, bn, relu, reshape)

    # II upsampling
    upconvs = []
    for i in range(upsamples):
        in_channels = 2**(upsamples - 1 - i) * DIM
        out_channels = in_channels // 2 if i < upsamples - 1 else 3

        upconvs.append(BlocksBunch(
            upconv_layer_generator(in_channels, out_channels, kernel=5),
            branches_per_layer[i + 1],
            equal_split=equal_split))

        if i < upsamples - 1:
            if batch_norm:
                upconvs.append(nn.BatchNorm2d(out_channels))
            upconvs.append(relu)
        upconvs.append(crop)
    upconvs.append(nn.Tanh())

    model = nn.Sequential(*([input] + upconvs))
    return PathGenerator(model, [latent_dim], [3, img_size, img_size], **kwargs)


class ConvDiscriminator(nn.Module):
    def __init__(self, with_batch_norm=False, sigmoid=False, upsamples=3):
        super(ConvDiscriminator, self).__init__()
        def make_conv2d_layer(in_dim, out_dim, kernel, stride, bn):
            if bn:
                return nn.Sequential(
                    nn.Conv2d(in_dim, out_dim, kernel_size=kernel, stride=stride, padding=2),
                    nn.BatchNorm2d(out_dim),
                    nn.LeakyReLU(0.2, inplace=True))
            else:
                return nn.Sequential(
                    nn.Conv2d(in_dim, out_dim, kernel_size=kernel, stride=stride, padding=2),
                    nn.LeakyReLU(0.2, inplace=True))

        conv_modules = []
        for i in range(upsamples):
            in_channels = 3 if i == 0 else out_channels
            out_channels = DIM if i == 0 else 2 * in_channels
            conv_modules.append(
                make_conv2d_layer(in_channels, out_channels, 5, 2, with_batch_norm)
            )
        self.main = nn.Sequential(*conv_modules)

        self.features_count = 4 * 4 * out_channels
        self.scorer = nn.Linear(self.features_count, 1)
        self.sigmoid = nn.Sigmoid() if sigmoid else None

    def forward(self, input):
        output = self.main(input)
        output = output.view([-1, self.features_count])
        output = self.scorer(output)

        if self.sigmoid is not None:
            output = self.sigmoid(output)

        return output.view(-1, 1)
