import os
from torch.utils.data import Dataset
from torchvision import transforms
from  PIL import Image

from torch_tools.utils import numerical_order, wrap_with_tqdm, make_verbose


class UnannotatedDataset(Dataset):
    def __init__(self, root_dir, sorted=False,
                 transform=transforms.Compose(
                     [
                         transforms.ToTensor(),
                         transforms.Normalize([0.5], [0.5])
                     ])):
        self.img_files = []
        for root, _, files in os.walk(root_dir):
            for file in numerical_order(files) if sorted else files:
                if UnannotatedDataset.file_is_img(file):
                    self.img_files.append(os.path.join(root, file))
        self.transform = transform

    @staticmethod
    def file_is_img(name):
        return name.endswith('jpg') or name.endswith('png')

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, item):
        img = Image.open(self.img_files[item])
        if self.transform is not None:
            return self.transform(img)
        else:
            return img


class LabeledDatasetImagesExtractor(Dataset):
    def __init__(self, ds, img_field=0):
        self.source = ds
        self.img_field = img_field

    def __len__(self):
        return len(self.source)

    def __getitem__(self, item):
        return self.source[item][self.img_field]


class FilteredDataset(Dataset):
    def __init__(self, source, filterer, target, verbosity=make_verbose()):
        self.source = source
        if not isinstance(target, list):
            target = [target]
        self.indices = [i for i, s in wrap_with_tqdm(enumerate(source), verbosity)
                        if filterer(i, s) in target]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.source[self.indices[index]]
