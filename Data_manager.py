import os
import numpy as np
from torch.utils.data import Dataset
import torch
from PIL import Image
import tifffile
from cellpose_plus import transforms, dynamics
import cv2
import random


class NucleiDataset(Dataset):
    def __init__(self, root_dir, resize_to=None, augment=False):

        self.root_dir = root_dir
        self.resize_to = resize_to
        self.augment = augment

        self.images = sorted([f for f in os.listdir(root_dir) if f.endswith('.tif')])

    def __len__(self):
        return len(self.images)

    def load_image(self, path):
        img = tifffile.imread(path).astype(np.float32)
        if img.ndim == 3:
            if img.shape[-1] <= 4:
                img = np.mean(img, axis=-1)
            elif img.shape[0] <= 4:
                img = np.mean(img, axis=0)
            else:
                img = np.max(img, axis=0)

        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = np.power(img, 0.8).astype(np.float32)

        return np.stack([img, np.zeros_like(img)], axis=0)

    def load_masks(self, base_name, H, W):
        labels_dir = os.path.join(self.root_dir, base_name, "labels")
        mask_files = sorted(os.listdir(labels_dir), key=lambda x: int(x.split('.')[0]))
        combined = np.zeros((H, W), dtype=np.int32)
        for i, mf in enumerate(mask_files):
            mask = np.array(Image.open(os.path.join(labels_dir, mf)).convert("L"))
            if mask.shape != (H, W):
                mask = np.array(Image.fromarray(mask).resize((W, H), resample=Image.NEAREST))
            combined[mask > 0] = i + 1
        return combined
    
    def augment_pair(self, img, masks):
        # flip horizontal
        if random.random() < 0.5:
            img = img[:, :, ::-1].copy()
            masks = masks[:, ::-1].copy()

        # flip vertical
        if random.random() < 0.5:
            img = img[:, ::-1, :].copy()
            masks = masks[::-1, :].copy()

        # rotation 
        k = random.randint(0, 3)
        if k > 0:
            img = np.rot90(img, k, axes=(1, 2)).copy()
            masks = np.rot90(masks, k, axes=(0, 1)).copy()

        # gamma augmentation
        if random.random() < 0.4:
            gamma = random.uniform(0.7, 1.5)
            img = img ** gamma

        # Gaussian noise
        if random.random() < 0.3:
            noise = np.random.normal(0, 0.03, img.shape).astype(np.float32)
            img = np.clip(img + noise, 0, 1)

        return img, masks
       
    def __getitem__(self, idx):
        img_name = self.images[idx]
        base_name = os.path.splitext(img_name)[0]
        img_path = os.path.join(self.root_dir, img_name)

        image = self.load_image(img_path)
        H, W = image.shape[-2:]
        combined_masks = self.load_masks(base_name, H, W)

        # resize
        if self.resize_to:
            image = np.transpose(image, (1, 2, 0))

            image = transforms.resize_image(image, Ly=self.resize_to[0], Lx=self.resize_to[1], interpolation=1)

            image = np.transpose(image, (2, 0, 1))

            combined_masks = transforms.resize_image(combined_masks, Ly=self.resize_to[0], Lx=self.resize_to[1],
                                                    interpolation=0, no_channels=True)
            
        # augment
        if self.augment:
            image, combined_masks = self.augment_pair(image, combined_masks)

        def pad_32(arr, div=32):
            H, W = arr.shape[-2:]
            pad_h = (div - H % div) % div
            pad_w = (div - W % div) % div
            if pad_h > 0 or pad_w > 0:
                if arr.ndim == 3:
                    arr = np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
                elif arr.ndim == 2:
                    arr = np.pad(arr, ((0, pad_h), (0, pad_w)), mode='reflect')
            return arr    
        image = pad_32(image)
        combined_masks = pad_32(combined_masks)
        
        cellprob = (combined_masks > 0).astype(np.float32) 
        
        gy, gx = np.gradient(cellprob) # cellprob - бинарная маска, получаем градиент для вычисления вертикального потока и горизонтального, которые требуются в cellpose моделях
        dy, dx = -gy, -gx  
        lbl = np.stack([cellprob, dy, dx], axis=0).astype(np.float32)
        
        return torch.from_numpy(image).float(), torch.from_numpy(lbl).float()


class NucleiDataset2(Dataset):
    def __init__(self, root_dir, resize_to=None, augment=False,):
        self.root_dir = root_dir
        self.resize_to = resize_to
        self.augment = augment

        self.images = sorted([f for f in os.listdir(root_dir) if f.endswith('.tif')])

    def __len__(self):
        return len(self.images)
    
    def load_image(self, path):
        img = tifffile.imread(path).astype(np.float32)
        if img.ndim == 3:
            if img.shape[-1] <= 4:
                img = np.mean(img, axis=-1)
            elif img.shape[0] <= 4:
                img = np.mean(img, axis=0)
            else:
                img = np.max(img, axis=0)

        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = np.power(img, 0.8).astype(np.float32)

        return np.stack([img, np.zeros_like(img)], axis=0)
    
    def load_masks(self, base_name, H, W):
        labels_dir = os.path.join(self.root_dir, base_name, "labels")
        mask_files = sorted(os.listdir(labels_dir), key=lambda x: int(x.split('.')[0]))
        combined = np.zeros((H, W), dtype=np.int32)
        for i, mf in enumerate(mask_files):
            mask = np.array(Image.open(os.path.join(labels_dir, mf)).convert("L"))
            if mask.shape != (H, W):
                mask = np.array(Image.fromarray(mask).resize((W, H), resample=Image.NEAREST))
            combined[mask > 0] = i + 1

        return combined

    def augment_pair(self, img, masks):
        # flip horizontal
        if random.random() < 0.5:
            img = img[:, :, ::-1].copy()
            masks = masks[:, ::-1].copy()

        # flip vertical
        if random.random() < 0.5:
            img = img[:, ::-1, :].copy()
            masks = masks[::-1, :].copy()

        # rotation 
        k = random.randint(0, 3)
        if k > 0:
            img = np.rot90(img, k, axes=(1, 2)).copy()
            masks = np.rot90(masks, k, axes=(0, 1)).copy()

        # gamma augmentation
        if random.random() < 0.4:
            gamma = random.uniform(0.7, 1.5)
            img = img ** gamma

        # Gaussian noise
        if random.random() < 0.3:
            noise = np.random.normal(0, 0.03, img.shape).astype(np.float32)
            img = np.clip(img + noise, 0, 1)

        return img, masks
    


    def __getitem__(self, idx):
        img_name = self.images[idx]
        base_name = os.path.splitext(img_name)[0]
        img_path = os.path.join(self.root_dir, img_name)

        image = self.load_image(img_path)      
        H, W = image.shape[-2:]

        combined_masks = self.load_masks(base_name, H, W)

        # resize
        if self.resize_to:

            image = np.transpose(image, (1, 2, 0))

            image = transforms.resize_image(image, Ly=self.resize_to[0], Lx=self.resize_to[1], interpolation=1)

            image = np.transpose(image, (2, 0, 1))
            
            combined_masks = transforms.resize_image(combined_masks, Ly=self.resize_to[0], Lx=self.resize_to[1], interpolation=0, no_channels=True)

        # augment
        if self.augment:
            image, combined_masks = self.augment_pair(image, combined_masks)

        def pad_32(arr, div=32):
            H, W = arr.shape[-2:]
            pad_h = (div - H % div) % div
            pad_w = (div - W % div) % div
            if pad_h > 0 or pad_w > 0:
                if arr.ndim == 3:
                    arr = np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
                elif arr.ndim == 2:
                    arr = np.pad(arr, ((0, pad_h), (0, pad_w)), mode='reflect')
            return arr    
        image = pad_32(image)
        combined_masks = pad_32(combined_masks)

        device = torch.device("cuda")
        flows_list = dynamics.labels_to_flows([combined_masks], device=device)


        dy, dx, cellprob, _ = flows_list[0]

        lbl = np.stack([cellprob, dy, dx], axis=0).astype(np.float32)

        return torch.from_numpy(image).float(), torch.from_numpy(lbl).float()
