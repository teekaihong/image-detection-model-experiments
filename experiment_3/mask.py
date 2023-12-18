import os
from PIL import Image
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
import cv2
from scipy.interpolate import splprep, splev
import yaml
from easydict import EasyDict as ed
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv

##Imports required for Inspyrenet Inference
from masking.lib.InSPyReNet import InSPyReNet_SwinB
from masking.lib.InSPyReNet import InSPyReNet_Res2Net50
from masking.lib import *
from masking.utils.misc import *
from masking.data.dataloader import *
from masking.data.custom_transforms import *


load_dotenv()

##for getting the mask of all the frames that has been extracted from the videos
def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g',        action='store_true', default=True)
    return parser.parse_args()


##Transforming the data and loading the dataset
def get_transform(tfs):
    comp = []
    for key, value in zip(tfs.keys(), tfs.values()):
        if value is not None:
            tf = eval(key)(**value)
        else:
            tf = eval(key)()
        comp.append(tf)
    return transforms.Compose(comp)


def load_config(config_dir, easy=True):
    cfg = yaml.load(open(config_dir), yaml.FullLoader)
    if easy is True:
        cfg = ed(cfg)
    return cfg


####################################################################################################################

##dataloader for InspyRenet Inference
class ImageLoader:
    def __init__(self, root, tfs):
        if os.path.isdir(root):
            self.images = [os.path.join(root, f) for f in os.listdir(root) if f.lower().endswith(('.jpg', '.jpeg'))]
            
            self.images = sort(self.images)
        elif os.path.isfile(root):
            self.images = [root]
       
        self.size = len(self.images)
        self.transform = get_transform(tfs)

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index == self.size:
            raise StopIteration
        image = Image.open(self.images[self.index]).convert('RGB')
        shape = image.size[::-1]
        name = '/'.join(self.images[self.index].split(os.sep)[1:])
      
            
        sample = {'image': image, 'name': name, 'shape': shape, 'original': image}
        sample['image'] = sample['image'].resize([384, 384], Image.BILINEAR)
        sample['image_resized'] = sample['image'].resize([384, 384], Image.BILINEAR)

            
        for key in sample.keys():
            if key in ['image', 'image_resized', 'gt', 'gt_resized']:
                sample[key] = np.array(sample[key], dtype=np.float32)

            # normalize
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        div = 255

        sample['image'] /= div
        sample['image'] -= mean
        sample['image'] /= std
            
        sample['image_resized'] /= div
        sample['image_resized'] -= mean
        sample['image_resized'] /= std

            # to tensor
        if 'image' in sample.keys():
            sample['image'] = sample['image'].transpose((2, 0, 1))
            sample['image'] = torch.from_numpy(sample['image']).float()
            
        if 'image_resized' in sample.keys():
            sample['image_resized'] = sample['image_resized'].transpose((2, 0, 1))
            sample['image_resized'] = torch.from_numpy(sample['image_resized']).float()

        sample['image'] = sample['image'].unsqueeze(0)
        if 'image_resized' in sample.keys():
            sample['image_resized'] = sample['image_resized'].unsqueeze(0)
            
        
        self.index += 1
        return sample

    def __len__(self):
        return self.size


###Function for loading samples on cuda
def to_cuda(sample):
        for key in sample.keys():
            if type(sample[key]) == torch.Tensor:
                sample[key] = sample[key].cuda()
        return sample
  
###function for reshaping the pred and changing the 
def to_numpy(pred, shape):
        pred = F.interpolate(pred, shape, mode='bilinear', align_corners=True)
        pred = pred.data.cpu()
        pred = pred.numpy().squeeze()
        return pred


#################################################################################################################3
##Main code for inspyrenet inference

def get_model(gpu_flag):
    yaml_path="masking/data/swinB.yaml"
    opt=load_config(yaml_path)
    model=eval(opt.Model.name)(**opt.Model)
    model.load_state_dict(torch.load(os.path.join(
    opt.Test.Checkpoint.checkpoint_dir, 'latest.pth'), map_location=torch.device('cpu')), strict=True)
    if gpu_flag is True:
        model = model.cuda()
    model.eval()
    return model

@torch.no_grad()
def inspyrenet_mask(frame_path,Output_masks,gpu_flag,model,pbar=None):
    
    yaml_path="masking/data/swinB.yaml"
    opt=load_config(yaml_path)

    samples=ImageLoader(frame_path,opt.Test.Dataset.transforms)
    for sample in samples:
        original_img = np.array(sample['original'])
        
        h, w, _ = original_img.shape
        if h == 1080:
            ratio = 0.80
        else:
            ratio = 0.75
        
        if gpu_flag is True:
            sample=to_cuda(sample)
        out=model(sample)
        shape=(h//1,w//1)
        pred = to_numpy(out['pred'], shape)
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        img = (np.stack([pred] * 3, axis=-1) * 255).astype(np.uint8)
        img = img.astype(np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, final = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        #final[int(ratio*(h//4)):,:] = 0
    
        final_rgb = cv2.cvtColor(final, cv2.COLOR_GRAY2RGB)
        final_rgb = final_rgb.astype(np.uint8)   
        gray = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2GRAY)
        name=sample['name'].split('.')[0] 
        Image.fromarray(final_rgb).save(os.path.join(Output_masks,name+'.png'))
        if pbar is not None:
            pbar.update(1)