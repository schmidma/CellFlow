from pathlib import Path

import albumentations
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
import tifffile
from torch.utils.data import Dataset


class CellDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.transform = transform

        root_dir = Path(root_dir)
        if split == "train":
            self.img_files = sorted(list(root_dir.glob(r"[ab]" + "/*.tif")))
            self.mask_files = sorted(list(root_dir.glob(r"[ab]" + "_GT/*.tif")))
        elif split == "val":
            self.img_files = sorted(list(root_dir.glob(r"[c]" + "/*.tif")))
            self.mask_files = sorted(list(root_dir.glob(r"[c]" + "_GT/*.tif")))
        elif split == "test":
            self.img_files = sorted(list(root_dir.glob(r"[de]" + "/*.tif")))
            self.mask_files = None

    def __getitem__(self, idx):
        img = tifffile.imread(self.img_files[idx])
        mask = (
            tifffile.imread(self.mask_files[idx]).astype(np.float32)
            if self.mask_files
            else None
        )

        if self.transform is not None:
            if self.mask_files:
                transformed = self.transform(image=img, mask=mask)
            else:
                transformed = self.transform(image=img)
            img = transformed["image"]
            if self.mask_files:
                mask = transformed["mask"].long()

        return img, mask if mask is not None else 1

    def __len__(self):
        return len(self.img_files)


def train_transform():
    transform = albumentations.Compose(
        [
            # albumentations.Resize(512, 512, interpolation=cv2.INTER_NEAREST),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            # albumentations.RandomBrightnessContrast(p=0.2),
            albumentations.Normalize(
                mean=33.53029578908284 / 255,
                std=23.36764441145509 / 255,
            ),
            ToTensorV2(),
        ]
    )

    return transform


def val_transform():
    transform = albumentations.Compose(
        [
            # albumentations.Resize(512, 512, interpolation=cv2.INTER_NEAREST),
            albumentations.Normalize(
                mean=33.53029578908284 / 255,
                std=23.36764441145509 / 255,
            ),
            ToTensorV2(),
        ]
    )

    return transform
