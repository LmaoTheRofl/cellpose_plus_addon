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
    
    # def load_masks(self, base_name, H, W):

    #     labels_dir = os.path.join(self.root_dir, base_name, "labels")
    #     mask_files = sorted(os.listdir(labels_dir), key=lambda x: int(os.path.splitext(x)[0]))
    #     combined = np.zeros((H, W), dtype=np.uint8)
    #     for mf in mask_files:
    #         mask = np.array(Image.open(os.path.join(labels_dir, mf)).convert("L"))
    #         if mask.shape != (H, W): 
    #             mask = np.array(Image.fromarray(mask).resize((W, H), resample=Image.NEAREST))
    #         combined[mask > 0] = 1
    #     return combined

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

        return img, masks
    
    def compute_flows_from_mask(self, mask):

        H, W = mask.shape
        dy = np.zeros((H, W), np.float32)
        dx = np.zeros((H, W), np.float32)
        cellprob = (mask > 0).astype(np.float32)

        instance_ids = np.unique(mask)
        instance_ids = instance_ids[instance_ids != 0]

        for inst in instance_ids:
            ys, xs = np.where(mask == inst)
            if len(xs) == 0:
                continue

            cy = ys.mean()
            cx = xs.mean()

            vy = cy - ys
            vx = cx - xs

            norm = np.sqrt(vy**2 + vx**2) + 1e-6
            vy /= norm
            vx /= norm

            dy[ys, xs] = vy
            dx[ys, xs] = vx

        return dy, dx, cellprob
    
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

        # cellprob = masks.astype(np.float32)
        dy, dx, cellprob = self.compute_flows_from_mask(masks)

        lbl = np.stack([dy, dx, cellprob], axis=0).astype(np.float32)

        return (
            torch.from_numpy(image).float(),
            torch.from_numpy(lbl).float(),   # torch.from_numpy(cellprob).unsqueeze(0).float(),
            torch.from_numpy(masks).long()
        )
    

    
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

        flows = dynamics.labels_to_flows([masks], device=torch.device("cpu"))

        labels, cellprob, dy, dx = flows[0]

        dy = np.nan_to_num(dy).astype(np.float32)
        dx = np.nan_to_num(dx).astype(np.float32)
        cellprob = np.nan_to_num(cellprob).astype(np.float32)

        # cellprob = (masks > 0).astype(np.float32) 
        
        lbl = np.stack([dy, dx, cellprob], axis=0).astype(np.float32)
 
        return (
            torch.from_numpy(image).float(),
            torch.from_numpy(lbl).float(),
            torch.from_numpy(masks).long()
        )
    