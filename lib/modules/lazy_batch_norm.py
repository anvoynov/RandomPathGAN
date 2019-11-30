import torch
from torch.nn.modules.batchnorm import _BatchNorm


class LazyBatchNorm(_BatchNorm):
    def __init__(self, num_features, update_thr=4, **kwargs):
        super(LazyBatchNorm, self).__init__(num_features, **kwargs)
        self.accumulated_input = []
        self.update_thr = update_thr

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def update_params(self):
        if len(self.accumulated_input) < self.update_thr:
            return

        self.accumulated_input = torch.cat(self.accumulated_input)
        super(LazyBatchNorm, self).forward(self.accumulated_input)
        self.accumulated_input = []


    def forward(self, input):
        is_training = self.training
        if is_training:
            self.accumulated_input.append(input.detach())

        self.training = False
        out = super(LazyBatchNorm, self).forward(input)
        self.training = is_training

        return out


def update_all_lazy_batch_norms(module):
    for m in module.modules():
        if isinstance(m, LazyBatchNorm):
            m.update_params()
