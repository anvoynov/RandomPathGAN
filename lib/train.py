import os
import sys
import argparse
import json
import random
import torch

from path_gan_train import train, TrainParams, LossType
from load import MODELS, DATASETS, make_models, make_dataloader


LOSS_TYPES_DICT = {
    'standard': LossType.standard,
    'wasserstein': LossType.wasserstein,
    'wasserstein_gp': LossType.wasserstein_gp,
    'hinge': LossType.hinge,
}
BATCH_NORM_TYPES = ['common', 'per_block', 'lazy', 'none']


def main():
    parser = argparse.ArgumentParser(description='Rangom Path GAN training')
    parser.add_argument('--args', type=str, default=None, help='json with all arguments')

    parser.add_argument('--out', type=str, help='out directory')
    parser.add_argument('--log', type=str, default=None, help='log directory')
    parser.add_argument('--checkpoint', type=str, default=None, help='path to checkpoint')
    parser.add_argument('--model', type=str, choices=list(MODELS.keys()), help='model')
    parser.add_argument('--data', type=str, choices=DATASETS, help='datasets')
    parser.add_argument('--image_channels', type=int, choices=[1, 3], default=3)
    parser.add_argument('--data_path', type=str, default=None, help='path to dataset')
    parser.add_argument('--device', type=int, default=0, help='CUDA device')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--num_blocks', type=int, default=[40], nargs='+',
                        help='number of independent blocks per bucket')
    parser.add_argument('--latent_dim', type=int, default=128, help='dimension of input noise')
    parser.add_argument('--nonfixed_noise', action='store_true', default=False, help='noise fixing')
    parser.add_argument('--noises_count', type=int, default=1, help='different noises count')
    parser.add_argument('--optimize_noise', action='store_true', default=False,
                        help='noise optimization')

    parser.add_argument('--batch', type=int, default=64, help='batch size')
    parser.add_argument('--equal_split', action='store_true', default=False,
                        help='split batch equaly between blocks')
    parser.add_argument('--generator_batch_norm', type=str, choices=BATCH_NORM_TYPES,
                        default='common', help='batch norm strategy for generator')
    parser.add_argument('--generator_leak', type=float, default=0.0, help='leak in generator lReLU')

    parser.add_argument('--discriminator_steps', type=int, default=5,
                        help='discriminator steps per generator step')
    parser.add_argument('--generator_steps', type=int, default=1,
                        help='generator steps per discriminator step')
    parser.add_argument('--loss', type=str, default='hinge', choices=LOSS_TYPES_DICT,
                        help='loss type')

    parser.add_argument('--diversity', type=float, default=1.0, help='blocks diversity')
    parser.add_argument('--diversity_margin', type=float, default=1.5,
                        help='max desired diversity')

    parser.add_argument('--steps', type=int, default=45e+4, help='learning steps')
    parser.add_argument('--rate', type=float, default=0.00025, help='learning rate')
    parser.add_argument('--betas', type=float, default=[0.5, 0.999], nargs=2, help='adam moments')

    parser.add_argument('--steps_per_save', type=int, default=5e+4, help='steps per model save')
    parser.add_argument('--steps_per_activations_log', type=int, default=None,
                        help='steps per activations histograms log (gpu memory wasteful)')

    args = parser.parse_args()

    if args.args is not None:
        with open(args.args) as args_json:
            args_dict = json.load(args_json)
            args.__dict__.update(**args_dict)
    if args.log is None:
        args.log = args.out
    if args.checkpoint is None:
        args.checkpoint = os.path.join(args.out, 'checkpoint.pt')

    if not os.path.isdir(args.out):
        os.makedirs(args.out)
    with open(os.path.join(args.out, 'args.json'), 'w') as args_file:
        json.dump(args.__dict__, args_file)
    with open(os.path.join(args.out, 'command.sh'), 'w') as command_file:
        command_file.write(' '.join(sys.argv))
        command_file.write('\n')

    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.set_device(args.device)

    dataloader = make_dataloader(args)
    generator, discriminator = make_models(args)

    # actual train
    train(generator, discriminator, dataloader,
          out_dir=args.out,
          log_dir=args.log,
          checkpoint=args.checkpoint,
          params=TrainParams(batch_size=args.batch,
                             critic_steps=args.discriminator_steps,
                             generator_steps=args.generator_steps,
                             diversity=args.diversity,
                             diversity_margin=args.diversity_margin,
                             rate=args.rate,
                             betas=args.betas,
                             steps=args.steps,
                             loss=LOSS_TYPES_DICT[args.loss],
                             steps_per_save=args.steps_per_save,
                             steps_per_activations_log=args.steps_per_activations_log))


if __name__ == '__main__':
    main()
