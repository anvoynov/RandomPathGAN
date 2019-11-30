import torch
import torchvision


def to_image(tensor, adaptive=False):
    if len(tensor.shape) == 4:
        tensor = tensor[0]
    if adaptive:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
        return torchvision.transforms.ToPILImage()((255 * tensor.cpu().detach()).to(torch.uint8))
    else:
        tensor = (tensor + 1) / 2
        tensor.clamp(0, 1)
        return torchvision.transforms.ToPILImage()((255 * tensor.cpu().detach()).to(torch.uint8))
