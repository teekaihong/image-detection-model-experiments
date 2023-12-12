import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms
from torchvision.io import read_image


default_resizer = transforms.Compose([
    transforms.Resize((192,240), antialias=True)
])

class AEDataset(Dataset):
    def __init__(self, filepaths, resize = default_resizer, transform=None):
        self.filepaths = filepaths
        self.transform = transform
        self.resize = resize

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        image = read_image(self.filepaths[idx]) / 255.0
        image = self.resize(image)
        if self.transform:
            image = self.transform(image)
        return image
