import os
import argparse
import random
import torch
import torchvision
from matplotlib import pyplot as plt

from load import load_model
from torch_tools.visualization import to_image


def inspect_path_generator_freeze(generator, samples_to_take=7, out_file=None):
    # inspect layers freeze
    buckets = list(generator.buckets())

    if generator.noise_is_discrete:
        noises_count = generator.const_noise.shape[0]
        noise_to_take = random.randint(0, noises_count - 1)
        generator.freeze_noise(noise_to_take)
    else:
        noises_count = 1
        generator.freeze_noise(True)

    blocks_to_take = \
        [random.randint(0, len(bucket.blocks) - 1) for bucket in buckets]

    def reset_model():
        for i, bucket in enumerate(generator.buckets()):
            bucket.freeze(blocks_to_take[i])

    def samples_to_grid(samples):
        return torchvision.utils.make_grid(torch.cat(samples), samples_to_take, pad_value=1)

    reset_model()
    original = generator(1).detach()

    grids_with_varying = []
    for i_layer, varying_bucket in enumerate(buckets):
        varying_images = []

        all_indices = list(range(len(varying_bucket.blocks)))
        if len(all_indices) > 1:
            all_indices.remove(blocks_to_take[i_layer])
        for i in all_indices[:samples_to_take]:
            varying_bucket.freeze(i)
            varying_images.append(generator(1).detach())

        grids_with_varying.append(samples_to_grid(varying_images))
        reset_model()

    # add noise variation images
    if noises_count > 1 or not generator.noise_is_discrete:
        varying_images = []

        if generator.noise_is_discrete:
            noise_indices = list(range(noises_count))
            noise_indices.remove(noise_to_take)
            for i in noise_indices[:samples_to_take]:
                generator.freeze_noise(i)
                varying_images.append(generator(1).detach())
        else:
            generator.freeze_noise(False)
            for i in range(samples_to_take):
                varying_images.append(generator(1).detach())

        grids_with_varying.insert(0, samples_to_grid(varying_images))
    generator.unfreeze_all()

    plt.subplot(len(grids_with_varying) + 1, 1, 1)
    plt.axis('off')
    plt.imshow(to_image(original, False))
    for i, grid_with_varying in enumerate(grids_with_varying):
        plt.subplot(len(grids_with_varying) + 1, 1, i + 2)
        plt.axis('off')
        plt.imshow(to_image(grid_with_varying, False))
    if out_file is not None:
        plt.savefig(out_file, dpi=200)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Random Path GAN visualization')
    parser.add_argument('--model', type=str, help='model weights')
    parser.add_argument('--model_params', type=str, help='model parameters json')
    parser.add_argument('--out_dir', type=str, help='out directory')
    parser.add_argument('--count', type=int, default=3, help='images to generate')
    parser.add_argument('--device', type=int, default=0, help='cuda device to use')

    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    generator = load_model(args.model, args.model_params)

    for i in range(args.count):
        inspect_path_generator_freeze(
            generator, out_file=os.path.join(args.out_dir, 'rpgan_inspection_{}.png'.format(i + 1)))
