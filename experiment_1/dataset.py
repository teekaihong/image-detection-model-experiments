import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image
import pandas as pd


default_resizer = transforms.Compose([
    transforms.Resize((192,240), antialias=True),
])

class AEDataset(Dataset):
    def __init__(self, filepaths, resize = default_resizer, transform=None):
        if isinstance(filepaths, pd.Series):
            filepaths = filepaths.tolist()
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
