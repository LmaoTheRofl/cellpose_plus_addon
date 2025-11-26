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
        if random.random() < 0.5:
            img = np.rot90(img, 2, axes=(1, 2)).copy()
            masks = np.rot90(masks, 2, axes=(0, 1)).copy()

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
        mask_files = sorted(os.listdir(labels_dir), key=lambda x: int(os.path.splitext(x)[0]))
        combined = np.zeros((H, W), dtype=np.int32)
        instance_id = 1
        for mf in mask_files:
            mask = np.array(Image.open(os.path.join(labels_dir, mf)).convert("L"))
            mask = (mask > 0).astype(np.uint8)
            if mask.shape != (H, W): 
                mask = np.array(Image.fromarray(mask).resize((W, H), resample=Image.NEAREST))

            combined[(mask == 1) & (combined == 0)] = instance_id
            # combined[mask == 1] = instance_id
            instance_id += 1
        return combined

    def augment_pair(self, img, masks):
        flip_h = random.random() < 0.5
        if flip_h:
            img = img[:, :, ::-1].copy()
            masks = masks[:, ::-1].copy()

        flip_v = random.random() < 0.5
        if flip_v:
            img = img[:, ::-1, :].copy()
            masks = masks[::-1, :].copy()

        rot180 = random.random() < 0.5
        if rot180:
            img = np.rot90(img, 2, axes=(1,2)).copy()
            masks = np.rot90(masks, 2).copy()

        if random.random() < 0.4:
            gamma = random.uniform(0.7, 1.5)
            img = img ** gamma

        if random.random() < 0.3:
            noise = np.random.normal(0, 0.03, img.shape).astype(np.float32)
            img = np.clip(img + noise, 0, 1)

        return img, masks, flip_h, flip_v, rot180

    def augment_flows(self, dy, dx, cell_logit, flip_h, flip_v, rot180):

        if flip_h:
            dy = dy[:, ::-1]
            dx = dx[:, ::-1]
            dx = dx * (-1)
            cell_logit = cell_logit[:, ::-1]

        if flip_v:
            dy = dy[::-1, :]
            dx = dx[::-1, :]
            dy = dy * (-1)
            cell_logit = cell_logit[::-1, :]

        if rot180:
            dy = np.rot90(dy, 2)
            dx = np.rot90(dx, 2)
            dy = -dy
            dx = -dx
            cell_logit = np.rot90(cell_logit, 2)

        return dy, dx, cell_logit

    def pad_32(self, arr, mode="reflect", constant_values=0):
        H, W = arr.shape[-2:]
        pad_h = (32 - H % 32) % 32
        pad_w = (32 - W % 32) % 32

        if arr.ndim == 3:
            if mode == "constant":
                return np.pad(arr, ((0,0),(0,pad_h),(0,pad_w)), mode=mode, constant_values=constant_values)
            else:
                return np.pad(arr, ((0,0),(0,pad_h),(0,pad_w)), mode=mode)
        else:
            if mode == "constant":
                return np.pad(arr, ((0,pad_h),(0,pad_w)), mode=mode, constant_values=constant_values)
            else:
                return np.pad(arr, ((0,pad_h),(0,pad_w)), mode=mode)

    def __getitem__(self, idx):

        img_name = self.images[idx]
        base_name = os.path.splitext(img_name)[0]


        img_path = os.path.join(self.root_dir, img_name)
        image = self.load_image(img_path)
        H, W = image.shape[-2:]


        masks = self.load_masks(base_name, H, W)

        dy, dx, cellprob, _ = dynamics.labels_to_flows([masks], device=torch.device("cpu"))[0]

        dy = np.nan_to_num(dy).astype(np.float32)
        dx = np.nan_to_num(dx).astype(np.float32)
        cellprob = np.nan_to_num(cellprob).astype(np.float32)

        max_abs = max(
            np.max(np.abs(dy)),
            np.max(np.abs(dx)),
            1e-6
        )
        dy = dy / max_abs
        dx = dx / max_abs

        eps = 1e-6
        cellprob = np.clip(cellprob, eps, 1 - eps)
        cell_logit = np.log(cellprob / (1 - cellprob)).astype(np.float32)

        if self.augment:
            image, masks, flip_h, flip_v, rot180 = self.augment_pair(image, masks)
            dy, dx, cell_logit = self.augment_flows(dy, dx, cell_logit, flip_h, flip_v, rot180)

        if self.resize_to:
            Ly, Lx = self.resize_to
            image = transforms.resize_image(image.transpose(1,2,0), Ly=Ly, Lx=Lx).transpose(2,0,1)
            dy = transforms.resize_image(dy, Ly=Ly, Lx=Lx)
            dx = transforms.resize_image(dx, Ly=Ly, Lx=Lx)
            cell_logit = transforms.resize_image(cell_logit, Ly=Ly, Lx=Lx)

        image = self.pad_32(image, mode="reflect")
        masks = self.pad_32(masks, mode="constant", constant_values=0)
        dy = self.pad_32(dy, mode="constant", constant_values=0)
        dx = self.pad_32(dx, mode="constant", constant_values=0)

        tiny_logit = np.log(1e-6 / (1 - 1e-6))
        cell_logit = self.pad_32(cell_logit, mode="constant", constant_values=tiny_logit)

        lbl = np.stack([dy, dx, cell_logit], axis=0).astype(np.float32)

        return (
            torch.from_numpy(image).float(),
            torch.from_numpy(lbl).float(),
            torch.from_numpy(masks).long()
        )
    
    
class NucleiDataset3(Dataset):

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
        mask_files = sorted(os.listdir(labels_dir), key=lambda x: int(os.path.splitext(x)[0]))
        combined = np.zeros((H, W), dtype=np.int32)
        instance_id = 1
        for mf in mask_files:
            mask = np.array(Image.open(os.path.join(labels_dir, mf)).convert("L"))
            mask = (mask > 0).astype(np.uint8)
            if mask.shape != (H, W): 
                mask = np.array(Image.fromarray(mask).resize((W, H), resample=Image.NEAREST))

            combined[(mask == 1) & (combined == 0)] = instance_id
            # combined[mask == 1] = instance_id
            instance_id += 1
        return combined
    
    # def load_masks(self, base_name, H, W):
    #     labels_dir = os.path.join(self.root_dir, base_name, "labels")
    #     mask_files = sorted(os.listdir(labels_dir), key=lambda x: int(x.split('.')[0]))
    #     combined = np.zeros((H, W), dtype=np.int32)
    #     for i, mf in enumerate(mask_files):
    #         mask = np.array(Image.open(os.path.join(labels_dir, mf)).convert("L"))
    #         if mask.shape != (H, W):
    #             mask = np.array(Image.fromarray(mask).resize((W, H), resample=Image.NEAREST))
    #         # combined[(mask == 1) & (combined == 0)] = i + 1
    #         combined[mask > 0] = i + 1
    #     return combined

    def augment_pair(self, img, masks):
        flip_h = random.random() < 0.5
        if flip_h:
            img = img[:, :, ::-1].copy()
            masks = masks[:, ::-1].copy()

        flip_v = random.random() < 0.5
        if flip_v:
            img = img[:, ::-1, :].copy()
            masks = masks[::-1, :].copy()

        rot180 = random.random() < 0.5
        if rot180:
            img = np.rot90(img, 2, axes=(1,2)).copy()
            masks = np.rot90(masks, 2).copy()

        if random.random() < 0.4:
            gamma = random.uniform(0.7, 1.5)
            img = img ** gamma

        if random.random() < 0.3:
            noise = np.random.normal(0, 0.03, img.shape).astype(np.float32)
            img = np.clip(img + noise, 0, 1)

        return img, masks

    def pad_32(self, arr, mode="reflect", constant_values=0):
        H, W = arr.shape[-2:]
        pad_h = (32 - H % 32) % 32
        pad_w = (32 - W % 32) % 32

        if arr.ndim == 3:
            if mode == "constant":
                return np.pad(arr, ((0,0),(0,pad_h),(0,pad_w)), mode=mode, constant_values=constant_values)
            else:
                return np.pad(arr, ((0,0),(0,pad_h),(0,pad_w)), mode=mode)
        else:
            if mode == "constant":
                return np.pad(arr, ((0,pad_h),(0,pad_w)), mode=mode, constant_values=constant_values)
            else:
                return np.pad(arr, ((0,pad_h),(0,pad_w)), mode=mode)

    def __getitem__(self, idx):

        img_name = self.images[idx]
        base_name = os.path.splitext(img_name)[0]


        img_path = os.path.join(self.root_dir, img_name)
        image = self.load_image(img_path)
        H, W = image.shape[-2:]


        masks = self.load_masks(base_name, H, W)

        image = self.pad_32(image, mode="constant", constant_values=0)
        masks = self.pad_32(masks, mode="constant", constant_values=0)

        if self.augment:
            image, masks = self.augment_pair(image, masks)
        if self.resize_to:
            Ly, Lx = self.resize_to
            image = transforms.resize_image(image.transpose(1,2,0), Ly=Ly, Lx=Lx).transpose(2,0,1)

        dy, dx, cellprob2, _ = dynamics.labels_to_flows([masks], device=torch.device("cpu"))[0]

        dy = np.nan_to_num(dy).astype(np.float32)
        dx = np.nan_to_num(dx).astype(np.float32)
        # cellprob2 = np.nan_to_num(cellprob).astype(np.float32)

        cellprob = (masks > 0).astype(np.float32) 
        
        # gy, gx = np.gradient(cellprob) # cellprob - бинарная маска, получаем градиент для вычисления вертикального потока и горизонтального, которые требуются в cellpose моделях
        # dy, dx = -gy, -gx 
        
        # norm = np.sqrt(dy**2 + dx**2) + 1e-6
        # dy = dy / norm
        # dx = dx / norm
        max_abs = max(
            np.max(np.abs(dy)),
            np.max(np.abs(dx)),
            1e-6
        )
        dy = dy / max_abs
        dx = dx / max_abs

        # eps = 1e-6
        # cellprob = np.clip(cellprob, eps, 1 - eps)
        # cell_logit = np.log(cellprob / (1 - cellprob)).astype(np.float32)

        lbl = np.stack([dy, dx, cellprob], axis=0).astype(np.float32)
 
        return (
            torch.from_numpy(image).float(),
            torch.from_numpy(lbl).float(),
            torch.from_numpy(masks).long()
        )
    