import os
import itertools
import random
from enum import Enum
import torch
import torch.nn.functional as F
import torchvision
import numpy as np

from torch_tools.visualization import to_image
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception as e:
    print('Failed to load torch tensorboard; Trying tensorboardX')
    from tensorboardX import SummaryWriter

from modules.lazy_batch_norm import update_all_lazy_batch_norms


class LossType(Enum):
    standard = 0,
    wasserstein = 1,
    wasserstein_gp = 2,
    hinge = 3


class TrainParams:
    def __init__(self, **kwargs):
        params = {
            'steps': 45e+4,
            'batch_size': 64,
            'rate': 0.0002,
            'betas': (0.5, 0.999),
            'critic_steps': 5,
            'generator_steps': 1,
            'clip': 0.01,
            'diversity': 50,
            'diversity_margin': 0.003,
            'gradient_penalty': 10,
            'loss': LossType.hinge,
            'lazy_batch_norm_updates': 5,
            'steps_per_save': 5e+4,
            'steps_per_img_save': 400,
            'steps_per_log': 20,
            'steps_per_activations_log': None
        }
        params.update(kwargs)
        for key, value in params.items():
            setattr(self, key, value)


def diversity_loss(block_1, block_2):
    block_1_params = dict(block_1.named_parameters())
    block_2_params = dict(block_2.named_parameters())
    assert block_1_params.keys() == block_2_params.keys(), 'blocks should have the same parameters'

    def normalize(w1, w2):
        joint = torch.cat([w1, w2])
        mean, std = joint.mean().item(), joint.std().item()
        return (w1 - mean) / std, (w2 - mean) / std

    count = 0
    diff = 0.0
    for name in block_1_params.keys():
        if 'weight' in name and 'Batch' not in name:
            weight_1, weight_2 = normalize(block_1_params[name], block_2_params[name])
            diff += F.mse_loss(weight_1, weight_2)
            count += 1
    return diff / count


def generator_diversity_loss(generator, diversity_loss_scale=0.0, diversity_margin=np.inf):
    diversity = torch.zeros([], device='cuda')
    if diversity_loss_scale > 0.0:
        count = 0
        for bucket in generator.buckets():
            blocks_count = len(bucket.blocks)
            if blocks_count == 1:
                continue

            blocks_indices = list(range(blocks_count))
            random.shuffle(blocks_indices)
            b1, b2 = bucket.blocks[blocks_indices[0]], bucket.blocks[blocks_indices[1]]

            d = diversity_loss(b1, b2)
            if d < diversity_margin:
                diversity += d
                count += 1

        if count > 0:
            diversity = diversity / count

    return -diversity_loss_scale * diversity


def compute_gradient_penalty(discriminator, real_samples, fake_samples, device='cuda'):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand([real_samples.size(0), 1, 1, 1], device=device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1.0 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    fake = torch.ones([real_samples.shape[0], 1], requires_grad=False, device=device)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def generator_loss(discriminator, fake, loss_type):
    if loss_type == LossType.standard:
        return torch.mean(torch.log(1.0 - discriminator(fake)))
    elif loss_type in [LossType.wasserstein, LossType.wasserstein_gp, LossType.hinge]:
        return -torch.mean(discriminator(fake))


def discriminator_loss(discriminator, real, fake, params):
    if params.loss == LossType.standard:
        return torch.mean(torch.log(discriminator(real))) + \
            torch.mean(torch.log(1.0 - discriminator(fake)))
    elif params.loss in [LossType.wasserstein, LossType.wasserstein_gp]:
        real_validity = discriminator(real)
        fake_validity = discriminator(fake)
        w_loss = -torch.mean(real_validity) + torch.mean(fake_validity)
        if params.loss == LossType.wasserstein:
            return w_loss
        else:
            gradient_penalty = compute_gradient_penalty(discriminator, real.data, fake.data)
            return w_loss + params.gradient_penalty * gradient_penalty
    elif params.loss == LossType.hinge:
        return F.relu(1.0 - discriminator(real)).mean() + F.relu(1.0 + discriminator(fake)).mean()


def train(generator, discriminator, dataloader, out_dir, log_dir, checkpoint, params=TrainParams()):
    generator.train().cuda()
    discriminator.train().cuda()

    imgs_dir = os.path.join(log_dir, 'images')
    models_dir = os.path.join(out_dir, 'models')
    tboard_dir = os.path.join(log_dir, 'tensorboard')
    for dir in [out_dir, log_dir, imgs_dir, models_dir, tboard_dir]:
        if not os.path.isdir(dir):
            os.makedirs(dir)

    writer = SummaryWriter(tboard_dir)
    try:
        writer.add_graph(generator, 1)
        writer.add_graph(discriminator, generator(1))
    except Exception as e:
        print('failed to write graph: {}'.format(e))

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=params.rate, betas=params.betas)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=params.rate, betas=params.betas)

    step = 0
    start_epoch = 0
    if os.path.isfile(checkpoint):
        step, start_epoch = \
            load_from_checkpoint(checkpoint, generator, discriminator, optimizer_G, optimizer_D)
        print('start from checkpoint: step {} / epoch {}'.format(step, start_epoch))
    elif not os.path.isdir(os.path.dirname(checkpoint)):
        os.makedirs(os.path.dirname(checkpoint))

    for epoch in itertools.count(start_epoch):
        print('---- epoch {} ----'.format(epoch))
        for i, imgs in enumerate(dataloader):
            real_imgs = imgs.cuda()
            fake_imgs = generator(params.batch_size)

            optimizer_D.zero_grad()
            d_loss = discriminator_loss(discriminator, real_imgs, fake_imgs, params)
            d_loss.backward()
            optimizer_D.step()

            if i % params.critic_steps == 0:
                for i_generator in range(params.generator_steps):
                    optimizer_G.zero_grad()
                    fake_imgs = generator(params.batch_size)

                    g_loss = generator_loss(discriminator, fake_imgs, params.loss)
                    g_diversity = generator_diversity_loss(
                        generator, params.diversity, params.diversity_margin)

                    (g_loss + g_diversity).backward()
                    optimizer_G.step()

                if (step // params.critic_steps) % params.lazy_batch_norm_updates == 0:
                    update_all_lazy_batch_norms(generator)

            log_training(writer, imgs_dir, params, step, i, epoch,
                         d_loss, g_loss, g_diversity, generator, fake_imgs)
            if step % params.steps_per_save == 0 or step == params.steps:
                torch.save(generator.state_dict(),
                           os.path.join(models_dir, 'generator_{}.pt'.format(step)))
                torch.save(discriminator.state_dict(),
                           os.path.join(models_dir, 'discriminator_{}.pt'.format(step)))

            step += 1
            if step > params.steps:
                break

        save_checkpoint(checkpoint, generator, discriminator, optimizer_G, optimizer_D, step, epoch)
        if step > params.steps:
            break


def log_training(writer, imgs_dir, params,
                 iteration, batch_it, epoch,
                 d_loss, g_loss, g_diversity,
                 generator, fake_imgs):
    if batch_it % params.steps_per_log == 0:
        print('{}% | Step {} | Epoch {}: [D loss: {}] [G loss: {}] [Diversiy: {}]'.format(
            int(100.0 * iteration / params.steps), iteration, epoch,
            d_loss.item(), g_loss.item(), g_diversity.item()))

        writer.add_scalar('discriminator loss', d_loss.item(), iteration)
        writer.add_scalar('generator loss', g_loss.item(), iteration)
        writer.add_scalar('epoch', epoch, iteration)

    if params.steps_per_activations_log is not None and \
            batch_it % params.steps_per_activations_log == 0:
        log_buckets_activations(writer, generator, iteration)

    if batch_it % params.steps_per_img_save == 0:
        torchvision.utils.save_image(
            fake_imgs.data[:25],
            os.path.join(imgs_dir, 'e{}_{}.png'.format(epoch, iteration)),
            nrow=5, normalize=True)
        writer.add_image(
            'generated',
            torchvision.transforms.ToTensor()(
                to_image(torchvision.utils.make_grid(fake_imgs[:25], 5))),
            iteration)


def log_buckets_activations(writer, generator, it):
    def backup_hook(module, input, output):
        setattr(module, 'output', output)

    hooks = {}
    equal_split_backup = {}
    max_blocks_count = 0
    for bucket in generator.buckets():
        equal_split_backup[bucket] = bucket.equal_split
        bucket.equal_split = True
        max_blocks_count = max(max_blocks_count, len(bucket.blocks))
        for block in bucket.get_blocks_to_use():
            hooks[block] = block.register_forward_hook(backup_hook)

    batch_size = 8 * max_blocks_count
    is_train_backup = generator.training
    generator.eval()
    with torch.no_grad():
        generator(batch_size)
    generator.train(is_train_backup)

    for bucket in generator.buckets():
        bucket.equal_split = equal_split_backup[bucket]

    for i_bucket, bucket in enumerate(generator.buckets()):
        for i_block, block in enumerate(bucket.get_blocks_to_use()):
            writer.add_histogram(
                'gen_bucket_{}/{}'.format(i_bucket, i_block), block.output.cpu().numpy(), it)
            block.output = None
            hooks[block].remove()


def load_from_checkpoint(checkpoint, generator, discriminator, optimizer_G, optimizer_D):
    data = torch.load(checkpoint)
    generator.load_state_dict(data['generator'])
    discriminator.load_state_dict(data['discriminator'])
    optimizer_G.load_state_dict(data['optimizer_G'])
    optimizer_D.load_state_dict(data['optimizer_D'])

    return data['step'], data['epoch']


def save_checkpoint(checkpoint, generator, discriminator, optimizer_G, optimizer_D, step, epoch):
    torch.save({
        'step': step,
        'epoch': epoch,
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'optimizer_G': optimizer_G.state_dict(),
        'optimizer_D': optimizer_D.state_dict()
    }, checkpoint)
