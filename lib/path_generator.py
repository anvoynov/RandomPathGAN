import random
import torch
from torch import nn
import numpy as np
from modules.blocks_bunch import BlocksBunch

class PathGenerator(nn.Module):
    def __init__(self, model, latent_dim, out_img_shape,
                 noises_count=1, constant_noise=True, backprop_noise=False):
        super(PathGenerator, self).__init__()

        self.latent_dim = latent_dim

        self.noise_is_discrete = constant_noise
        self.const_noise = None
        if constant_noise:
            self.const_noise = nn.Parameter(torch.randn([noises_count] + latent_dim))
            self.const_noise.requires_grad = backprop_noise
        self.active_noise_indices = np.arange(noises_count)

        self.model = model
        self.out_img_shape = out_img_shape
        self.device = 'cpu'
        self.to(self.device)

    def buckets(self):
        for module in self.modules():
            if isinstance(module, BlocksBunch):
                yield module

    def freeze_noise(self, indices_to_take=False):
        if self.noise_is_discrete:
            if indices_to_take == False:
                self.active_noise_indices = np.arange(self.const_noise.shape[0])
            else:
                self.active_noise_indices = [indices_to_take] if isinstance(indices_to_take, int) \
                    else indices_to_take
        else:
            if indices_to_take is True:
                self.const_noise = self.make_noise(1).detach()
                self.active_noise_indices = [0]
            else:
                self.const_noise = None

    def unfreeze_all(self):
        for bucket in self.buckets():
            bucket.freeze(False)
        self.freeze_noise(False)

    def make_noise(self, batch_size):
        if self.const_noise is not None:
            noise = torch.empty([batch_size] + list(self.const_noise.shape[1:]))

            for i in range(batch_size):
                noise[i] = self.const_noise[random.choice(self.active_noise_indices)]
            return noise.to(self.device)
        else:
            return torch.randn([batch_size] + self.latent_dim).to(self.device)

    def cuda(self):
        super(PathGenerator, self).cuda()
        self.device = 'cuda'
        if self.const_noise is not None:
            self.const_noise = self.const_noise.cuda()

    def cpu(self):
        super(PathGenerator, self).cpu()
        self.device='cpu'
        if self.const_noise is not None:
            self.const_noise = self.const_noise.cpu()

    def to(self, device):
        super(PathGenerator, self).to(device)
        self.device = device
        if self.const_noise is not None:
            self.const_noise = self.const_noise.to(device)

    def forward(self, batch_size):
        img = self.model(self.make_noise(batch_size))
        img = img.view(img.shape[0], *self.out_img_shape)
        return img
