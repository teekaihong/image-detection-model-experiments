import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image
import pandas as pd
from pathlib import Path


default_resizer = transforms.Compose([
    transforms.Resize((192,240), antialias=True),
])

class AEDataset(Dataset):
    def __init__(self, labeled_df:pd.DataFrame, resize = default_resizer, transform=None):
        self.labeled_df = labeled_df
        self.transform = transform
        self.resize = resize
        self.implement_classes = labeled_df['implement_class'].unique().tolist()
        self.class_map = {
            k:v for v, k in enumerate(self.implement_classes)
        }
        


    def __len__(self):
        return len(self.labeled_df)

    def __getitem__(self, idx):
        image_item:pd.Series = self.labeled_df.iloc[idx]
        image, w, h = self.read_image(image_item['image_path'])
        implement_class = self.class_map[image_item['implement_class']]
        left, right, upper, bottom = image_item['left'] / w, image_item['right'] / w, image_item['upper'] / h, image_item['bottom'] / h
        if self.transform:
            image = self.transform(image)
        bbox_tensor = torch.tensor([left, right, upper, bottom])
        return {
            'image': image,
            'implement_class': implement_class,
            'bbox': bbox_tensor
        }

    
    def read_image(self, fp:str):
        img = read_image(fp)
        h, w = img.shape[1], img.shape[2]
        return self.resize(img / 255.0), w, h

        

