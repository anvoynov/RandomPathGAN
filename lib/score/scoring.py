import os
import argparse
import torch
import json
import numpy as np
from torch_tools.gan_sampling import GeneratorDataloader

from score.fid_score import calculate_fid_for_generators
from load import load_model_from_state_dict, load_dataloader


def inspect_directory(
        dir, target_model=None, samples_to_take=None,
        models_count=5, start_from=0,
        batch_size=64, inception_model=None, train_mode=False):
    if target_model is None:
        models_dir = os.path.join(dir, 'models')
        all_models_indices = \
            [int(name.split('.')[0].split('_')[-1]) for name in os.listdir(models_dir)
             if name.startswith('generator')]
        all_models_indices.sort()
        all_models_indices = np.array(all_models_indices)
        all_models_indices = all_models_indices[all_models_indices >= start_from]
        models_indices = [all_models_indices[i] for i in \
            range(len(all_models_indices) - 1, -1, -len(all_models_indices) // models_count)]
    else:
        models_indices = [target_model]

    original_dataloader = load_dataloader(dir)
    gen_dataloaders = []
    generators = []

    for model_index in models_indices:
        generators.append(load_model_from_state_dict(dir, model_index))
        if train_mode:
            generators[-1].train()
        else:
            generators[-1].eval()
        generators[-1].cpu()
        gen_dataloaders.append(
            GeneratorDataloader(generators[-1], batch_size,
                                length=len(original_dataloader), rand_sampler=lambda: batch_size))

    print('taking models: {}'.format(models_indices))

    out_dict = {'index': models_indices}
    with torch.no_grad():
        fids = calculate_fid_for_generators(original_dataloader, gen_dataloaders,
                                            cuda=True,
                                            verbose=True,
                                            model_path=inception_model,
                                            samples_to_take=samples_to_take)
    out_dict['fid'] = fids
    print('fid: {}'.format(fids))

    out_file = os.path.join(dir, 'score.json')
    if os.path.isfile(out_file):
        with open(out_file, 'r') as f:
            existed_dict = json.load(f)
            existed_dict.update(out_dict)
            out_dict = existed_dict
    with open(out_file, 'w') as f:
        json.dump(out_dict, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rangom Path GAN scoring')
    parser.add_argument('--dir', type=str, help='root directory for runs')
    parser.add_argument('--target_model', type=str, default=None, help='model to take')
    parser.add_argument('--recursive', action='store_true', default=False, help='dir recursion')
    parser.add_argument('--skip_existed', action='store_true', default=False,
                        help='skip runs with score.json presented')
    parser.add_argument('--start_from', type=int, default=0, help='epoch to start from')
    parser.add_argument('--count', type=int, default=3, help='epochs slices')
    parser.add_argument('--samples_to_take', type=int, default=None, help='evaluation subset')
    parser.add_argument('--device', type=int, default=0, help='cuda device to use')
    parser.add_argument('--model', type=str, default=None, help='path to InceptionV3 model')
    parser.add_argument('--train_mode', action='store_true', default=False,
                        help='generate images in train mode')

    args = parser.parse_args()
    torch.cuda.set_device(args.device)

    dirs = [os.path.join(args.dir, subrun) for subrun in os.listdir(args.dir)] if args.recursive \
        else [args.dir]
    for subrun in dirs:
        print('\n\n * * * \n\nProcessing {}'.format(os.path.basename(subrun)))
        try:
            if args.skip_existed and os.path.isfile(os.path.join(subrun, 'score.json')):
                print('    already processed, skipping')
                continue
            inspect_directory(subrun, args.target_model, args.samples_to_take,
                              args.count, args.start_from,
                              inception_model=args.model,
                              train_mode=args.train_mode)
        except Exception as e:
            print(e)
