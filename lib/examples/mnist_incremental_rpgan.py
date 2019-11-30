import os
import torch
import json
from torchvision import transforms, datasets
from torch_tools.data import FilteredDataset, LabeledDatasetImagesExtractor

from path_gan_train import train, TrainParams, LossType
from load import make_models


LABELS = list(range(0, 7, 1))
LABELS_EXTENDED = list(range(7, 10, 1))
BLOCKS = [20, 20, 20, 8, 1]
BLOCKS_EXTENDED = [25, 25, 20, 8, 1]
OUT_DIR = os.path.join(os.path.expanduser('~'), 'rpgan_inceremental_learning')  # place yours
MNIST_PATH = '/path/to/mnist'  # place path to torch mnist dataset


class _DataExtansionArgs:
    def __init__(self, **kwargs):
        params = {
            'model': 'sn_resnet32',
            'first_linear': True,
            'data': 'mnist',
            'data_path': MNIST_PATH,
            'num_blocks': BLOCKS,
            'num_blocks_extended': BLOCKS_EXTENDED,
            'latent_dim': 128,
            'nonfixed_noise': False,
            'noises_count': 1,
            'optimize_noise': True,
            'batch': 64,
            'equal_split': False,
            'generator_batch_norm': 'common',
            'generator_leak': 0.0,
            'discriminator_steps': 5,
            'generator_steps': 1,
            'loss': LossType.hinge,
            'diversity': 0,
            'diversity_margin': 0.0,
            'steps': 6e+4,
            'rate': 0.0002,
            'betas': [0.5, 0.999],
            'steps_per_save': 5000,
            'steps_per_img_save': 400,
            'steps_per_log': 20,
            'image_channels': 1
        }
        params.update(kwargs)
        for key, value in params.items():
            setattr(self, key, value)


def make_mnist_dataset(data_dir):
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])])
    return datasets.MNIST(data_dir, train=True, transform=transform)


def train_with_labels_filtration(generator, discriminator,
                                 dataset, batch_size, target_labels, train_params, out_dir):

    dataset = FilteredDataset(
        dataset, filterer=lambda i, s: s[1], target=target_labels)
    dataset = LabeledDatasetImagesExtractor(dataset)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    out_dir = os.path.join(out_dir, 'filtered_{}-{}'.format(target_labels[0], target_labels[-1]))
    train(generator, discriminator, dataloader,
          out_dir=out_dir,
          log_dir=out_dir,
          checkpoint=os.path.join(out_dir, 'checkpoint.pt'),
          params=train_params)


def train_two_stage(labels_1, labels_2, out_dir):
    args = _DataExtansionArgs()

    if not os.path.isdir(OUT_DIR):
        os.makedirs(OUT_DIR)
    with open(os.path.join(OUT_DIR, 'args.json'), 'w') as args_file:
        json.dump(
            {name: value for name, value in args.__dict__.items() if name !='loss'}, args_file)

    train_params = TrainParams()
    train_params.__dict__.update(args.__dict__)
    num_blocks, num_blocks_extended = args.num_blocks, args.num_blocks_extended

    args.num_blocks = num_blocks_extended
    generator, discriminator = make_models(args)
    # use only subset
    for bucket, num_blocks_to_use in zip(generator.buckets(), num_blocks):
        bucket.blocks_to_use = list(range(num_blocks_to_use))

    dataset = make_mnist_dataset(args.data_path)
    train_with_labels_filtration(
        generator, discriminator, dataset, args.batch, labels_1, train_params, out_dir)

    generator.unfreeze_all()
    generator.eval()

    def exclude_original_from_params(generator, num_blocks):
        named_params_to_leave = []
        for bucket, num_blocks_to_use in zip(generator.buckets(), num_blocks):
            for block in bucket.blocks[num_blocks_to_use:]:
                named_params_to_leave += list(block.named_parameters())

        def new_named_parameters(recurse):
            return named_params_to_leave
        generator.named_parameters = new_named_parameters

    exclude_original_from_params(generator, num_blocks)
    train_with_labels_filtration(
        generator, discriminator, dataset, args.batch, labels_1 + labels_2, train_params, out_dir)


if __name__ == '__main__':
    train_two_stage(LABELS, LABELS_EXTENDED, OUT_DIR)
