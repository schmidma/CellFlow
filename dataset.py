from pathlib import Path

import albumentations
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
import lightning
import tifffile
import torch
from torch.utils.data import Dataset, DataLoader


class CellDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.transform = transform

        root_dir = Path(root_dir)
        if split == "train":
            self.image_files = sorted(list(root_dir.glob(r"[ab]" + "/*.tif")))
            self.flow_gradient_files = sorted(
                list(root_dir.glob(r"[ab]" + "_flow/*.tif"))
            )
        elif split == "validation":
            self.image_files = sorted(list(root_dir.glob(r"[c]" + "/*.tif")))
            self.flow_gradient_files = sorted(
                list(root_dir.glob(r"[c]" + "_flow/*.tif"))
            )
        elif split == "test":
            self.image_files = sorted(list(root_dir.glob(r"[de]" + "/*.tif")))
            self.flow_gradient_files = None

    def __getitem__(self, idx):
        image = tifffile.imread(self.image_files[idx])
        if self.flow_gradient_files:
            flow_gradient = torch.tensor(
                tifffile.imread(self.flow_gradient_files[idx]).astype(np.float32)
            )
        else:
            flow_gradient = torch.zeros((2, image.shape[0], image.shape[1]))

        if self.transform is not None:
            image = self.transform(image=image)["image"]
        mask = torch.sum(flow_gradient**2, axis=0) != 0
        return image, mask, flow_gradient

    def __len__(self):
        return len(self.image_files)


def train_transform():
    transform = albumentations.Compose(
        [
            # albumentations.Resize(512, 512, interpolation=cv2.INTER_NEAREST),
            # albumentations.HorizontalFlip(p=0.5),
            # albumentations.VerticalFlip(p=0.5),
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


class CellDataModule(lightning.LightningDataModule):
    def __init__(self, batch_size=2, num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_data = CellDataset(
                root_dir="../train/",
                split="train",
                transform=train_transform(),
            )
            self.validation_data = CellDataset(
                root_dir="../train/",
                split="validation",
                transform=val_transform(),
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )


if __name__ == "__main__":
    cell_dataset = CellDataset(
        root_dir="../train/", split="train", transform=train_transform()
    )
    for image, mask, gradient in DataLoader(cell_dataset):
        print(f"image: {image.shape}, mask: {mask.shape}, gradient: {gradient.shape}")
