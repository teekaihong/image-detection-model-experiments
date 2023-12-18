import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image
import pandas as pd
from pathlib import Path


default_resizer = transforms.Compose([
    transforms.Resize((192,240), antialias=True),
])

mask_blur = transforms.Compose([
    transforms.GaussianBlur(kernel_size=15, sigma=9.)
])

class AEDataset(Dataset):
    def __init__(self, image_df:pd.DataFrame, resize = default_resizer, transform=None, mask_transform=mask_blur):
        self.image_df = image_df
        self.labeled_df = image_df[image_df['implement_class'] != '<no_data>']
        self.transform = transform
        self.mask_transform = mask_transform
        self.resize = resize

    def __len__(self):
        return len(self.image_df)

    def __getitem__(self, idx):
        image_item:pd.Series = self.image_df.iloc[idx]
        image = self.read_image(image_item['image_path'])
        mask = self.read_image(image_item['mask_path'])
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        
        anchor_item:pd.Series = self.labeled_df.sample(n=1).iloc[0]
        positve_item:pd.Series = self.labeled_df[self.labeled_df['implement_class'] == anchor_item['implement_class']].sample(n=1).iloc[0]
        negative_item:pd.Series = self.labeled_df[self.labeled_df['implement_class'] != anchor_item['implement_class']].sample(n=1).iloc[0]

        anchor = self.read_image(anchor_item['image_path'])
        positive = self.read_image(positve_item['image_path'])
        negative = self.read_image(negative_item['image_path'])

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return {
            'image': image,
            'mask': mask,
            'anchor': anchor,
            'positive': positive,
            'negative': negative,
            'anchor_class': anchor_item['implement_class'],
        }

    
    def read_image(self, fp:str):
        return self.resize(read_image(fp) / 255.0)

        

