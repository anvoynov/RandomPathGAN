from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch_tools.data import LabeledDatasetImagesExtractor, UnannotatedDataset


def make_cifar10_dataloader(data_dir, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])])

    ds = LabeledDatasetImagesExtractor(
        datasets.CIFAR10(root=data_dir, download=True, transform=transform))

    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)


def make_lsun_bedroom_dataloader(data_dir, batch_size, size=256):
    transform = transforms.Compose([
        transforms.Resize([size, size]),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])])
    ds = LabeledDatasetImagesExtractor(
        datasets.LSUN(data_dir, classes=['bedroom_train'], transform=transform))

    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)


def make_mnist_dataloader(data_dir, batch_size, size=28):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])])
    ds = LabeledDatasetImagesExtractor(datasets.MNIST(data_dir, train=True, transform=transform))

    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)


def make_anime_faces_dataloader(data_dir, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])])

    ds = UnannotatedDataset(data_dir, sorted=False, transform=transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1)
