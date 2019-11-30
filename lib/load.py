import os
import json
from numpy import log2
import torch

from data import make_mnist_dataloader, make_cifar10_dataloader, make_anime_faces_dataloader, \
                 make_lsun_bedroom_dataloader
from models.others.mnist_path_gan import MNISTDiscriminator, make_mnist_path_generator

from models.others.conv_path_gan import make_conv_path_generator, ConvDiscriminator

from models.sngan.generators.sn_gen_conv32 import make_sn_conv32_path_generator
from models.sngan.discriminators.sn_dis_conv32 import SNDiscriminator
from models.sngan.generators.sn_gen_fc import make_sn_fc_path_generator
from models.sngan.discriminators.sn_dis_fc import SNFCDiscriminator

from models.sngan.generators.sn_gen_resnet import SN_RES_GEN_CONFIGS, make_resnet_generator
from models.sngan.discriminators.sn_dis_resnet import SN_RES_DIS_CONFIGS, ResnetDiscriminator


MODELS = {
    'sn_conv32': 32,
    'sn_fc': 28,
    'sn_resnet32': 32,
    'sn_resnet48': 48,
    'sn_resnet64': 64,
    'sn_resnet128': 128,
    'sn_resnet256': 256,
    'fc': 28,
    'conv32': 32,
    'conv64': 64,
}


DATASETS = ['mnist', 'cifar10', 'lsun_bedroom', 'anime_faces']


class Args:
    def __init__(self, **kwargs):
        # old versions support
        self.nonfixed_noise = False
        self.noises_count = 1
        self.equal_split = False
        self.generator_batch_norm = False

        self.__dict__.update(kwargs)


def make_dataloader(args):
    if args.data == 'cifar10':
        return make_cifar10_dataloader(args.data_path, args.batch)
    elif args.data == 'mnist':
        return make_mnist_dataloader(args.data_path, args.batch, MODELS[args.model])
    elif args.data == 'lsun_bedroom':
        return make_lsun_bedroom_dataloader(args.data_path, args.batch, MODELS[args.model])
    elif args.data == 'anime_faces':
        return make_anime_faces_dataloader(args.data_path, args.batch)


def make_models(args):
    if isinstance(args.num_blocks, list) and len(args.num_blocks) == 1:
        args.num_blocks = args.num_blocks[0]

    gen_kwargs = {
        'backprop_noise': args.optimize_noise,
        'constant_noise': not args.nonfixed_noise,
        'latent_dim': args.latent_dim,
        'branches_per_layer': args.num_blocks,
        'img_size': MODELS[args.model],
        'noises_count': args.noises_count,
        'equal_split': args.equal_split,
        'batch_norm': args.generator_batch_norm,
        'leak': args.generator_leak}
    try:
        image_channels = args.image_channels
    except Exception:
        image_channels = 3

    if args.model == 'fc':
        generator = make_mnist_path_generator(**gen_kwargs)
        discriminator = MNISTDiscriminator()

    elif args.model.startswith('conv'):
        gen_kwargs['img_size'] = int(args.model[4:])
        upsamples = int(round(log2(gen_kwargs['img_size'] / 4)))

        generator = make_conv_path_generator(upsamples=upsamples, **gen_kwargs)
        discriminator = ConvDiscriminator(upsamples=upsamples)

    elif args.model == 'sn_conv32':
        generator = make_sn_conv32_path_generator(first_linear=args.first_linear, **gen_kwargs)
        discriminator = SNDiscriminator()

    elif args.model == 'sn_fc':
        generator = make_sn_fc_path_generator(
            image_channels=image_channels, **gen_kwargs)
        discriminator = SNFCDiscriminator(
            img_size=MODELS[args.model], image_channels=image_channels)

    elif args.model.startswith('sn_resnet'):
        gen_config = SN_RES_GEN_CONFIGS[args.model]
        dis_config = SN_RES_DIS_CONFIGS[args.model]
        generator = make_resnet_generator(gen_config, image_channels=image_channels, **gen_kwargs)
        discriminator = ResnetDiscriminator(dis_config, image_channels=image_channels)

    return generator, discriminator


def load_dataloader(root_dir):
    args = json.load(open(os.path.join(root_dir, 'args.json')))
    args = Args(**args)
    return make_dataloader(args)


def load_model(model_path, params_path, cuda=True):
    args = json.load(open(params_path))

    args = Args(**args)
    generator, _ = make_models(args)
    generator.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    if cuda:
        generator.cuda()
    generator.eval()

    return generator


def load_model_from_state_dict(root_dir, model_index=None, cuda=True):
    args_path = os.path.join(root_dir, 'args.json')

    models_dir = os.path.join(root_dir, 'models')
    if model_index is None:
        models = os.listdir(models_dir)
        model_index = max(
            [int(name.split('.')[0].split('_')[-1]) for name in models
             if name.startswith('generator')])

        print('using max index {}'.format(model_index))
    model_path = os.path.join(models_dir, 'generator_{}.pt'.format(model_index))

    return load_model(model_path, args_path, cuda)
