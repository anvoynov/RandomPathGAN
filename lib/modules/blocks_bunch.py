import torch
import random
import numpy as np


class BlocksBunch(torch.nn.Module):
    def __init__(self, block_generator, num_blocks, equal_split=False):
        super(BlocksBunch, self).__init__()
        self.blocks = torch.nn.ModuleList([
            block_generator() for _ in range(num_blocks)
        ])

        self.equal_split = equal_split
        self.blocks_to_use = range(len(self.blocks))

    def freeze(self, index=False):
        if index is False:
            self.blocks_to_use = range(len(self.blocks))
        else:
            self.blocks_to_use = [index]

    def get_blocks_to_use(self):
        return [self.blocks[i] for i in self.blocks_to_use]

    def forward(self, x):
        batch_size = x.shape[0]
        if self.equal_split:
            shuffled_blocks_to_use = list(self.blocks_to_use)
            random.shuffle(shuffled_blocks_to_use)

            blocks_indices = []
            splits = np.arange(0, batch_size, float(batch_size) / len(self.blocks_to_use))
            splits = list(splits.astype(int)) + [batch_size]

            for i in range(len(self.blocks_to_use)):
                blocks_indices = blocks_indices + \
                                 [shuffled_blocks_to_use[i]] * (splits[i + 1] - splits[i])

            random.shuffle(blocks_indices)
        else:
            blocks_indices = [random.choice(self.blocks_to_use) for _ in range(batch_size)]

        blocks_to_use = [self.blocks[i] for i in blocks_indices]

        if len(blocks_to_use) == 1:
            return blocks_to_use[0](x)
        else:
            return torch.cat([blocks_to_use[i](x[i].unsqueeze(0)) for i in range(batch_size)])
