import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
import torch
import torchvision
from torchvision.transforms import ToTensor, Compose, ConvertImageDtype
import torch.nn.functional as F


class MaskRCNNDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform or Compose([ToTensor(), ConvertImageDtype(torch.float)])
        self.images = sorted([f for f in os.listdir(root_dir) if f.endswith('.tif')])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        base_name = os.path.splitext(img_name)[0]
        
        img_path = os.path.join(self.root_dir, img_name)
        subdir = os.path.join(self.root_dir, base_name)
        # print(img_path)
        image = Image.open(img_path).convert("RGB")
        
        labels_dir = os.path.join(subdir, 'labels')
        mask_files = sorted(os.listdir(labels_dir), key=lambda x: int(x.split('.')[0]))
        masks = []
        for mf in mask_files:
            mask_path = os.path.join(labels_dir, mf)
            mask = Image.open(mask_path).convert('L')
            masks.append(mask)
        
        primary_dir = os.path.join(subdir, 'primary')
        center_csv = os.path.join(primary_dir, base_name + '_Center.csv')
        size_roundness_csv = os.path.join(primary_dir, base_name + '_size_roundness.csv')
        
        info = pd.concat([pd.read_csv(center_csv, names=['x', 'y']), pd.read_csv(size_roundness_csv, names=['size', 'roundness'])], axis = 1)
        info['classes'] = info['size'].apply(lambda x: 1 if x <= 40 else 2)
        # info = info.to_numpy()

        mask_tensors = []
        boxes = []
        W, H = image.size[0], image.size[1]
        for mask in masks:
            mask_tensor = self.transform(mask)  # [1 x H x W] одна маска
            mask_tensor = (mask_tensor > 0).float() 
            
            if mask_tensor.sum() == 0: 
                continue
            y_indices, x_indices = torch.nonzero(mask_tensor[0] > 0, as_tuple=True)
            if len(x_indices) == 0 or len(y_indices) == 0:
                continue
            x_min = max(0, x_indices.min().item())
            x_max = min(W, x_indices.max().item() + 1)
            y_min = max(0, y_indices.min().item())
            y_max = min(H, y_indices.max().item() + 1)
            

            boxes.append([x_min, y_min, x_max, y_max])  
            
            mask_tensors.append(mask_tensor)

        if len(mask_tensors) == 0:
            target = {'boxes': torch.empty((0, 4), dtype=torch.float32),
                      'labels': torch.empty((0,), dtype=torch.int64),
                      'masks': torch.empty((0, H, W), dtype=torch.uint8)}
        else:
            target = {
                'boxes': torch.tensor(boxes, dtype=torch.float32),
                'labels': torch.tensor(info['classes'].values[:len(boxes)], dtype=torch.int64), 
                'masks': torch.stack(mask_tensors)  # [N x 1 x H x W]
            }
            target['masks'] = target['masks'].squeeze(1)  # [N x H x W]
        
        image = self.transform(image)
        

        return image, target

def mask_collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets

