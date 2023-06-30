from pathlib import Path

import albumentations
from albumentations.pytorch.transforms import ToTensorV2
import lightning
import numpy as np
import tifffile
import torch
from torch.utils.data import DataLoader, Dataset

from flow import compute_flow


class CellDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        root_dir = Path(root_dir)
        if split == "train":
            runs = "0[0|1]"
        elif split == "validation":
            runs = "0[2]"
        elif split == "test":
            runs = "0[3|4]"
        else:
            raise ValueError(f"Invalid split: {split}")

        self.image_files = sorted(list(root_dir.glob(f"{runs}/*.tif")))
        self.segmentation_files = sorted(list(root_dir.glob(f"{runs}_GT/SEG/*.tif")))

        # TODO: move transform to optional parameter
        if split == "train":
            self.transform = albumentations.Compose(
                [
                    albumentations.Resize(480, 384),
                    albumentations.RandomBrightnessContrast(),
                    albumentations.GaussNoise(var_limit=(20, 70)),
                    albumentations.Normalize(
                        mean=33.53029578908284 / 255,
                        std=23.36764441145509 / 255,
                    ),
                ]
            )
        else:
            self.transform = albumentations.Compose(
                [
                    albumentations.Resize(480, 384),
                    albumentations.Normalize(
                        mean=33.53029578908284 / 255,
                        std=23.36764441145509 / 255,
                    ),
                ]
            )

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = tifffile.imread(image_path)

        segmentation_path = self.segmentation_files[idx]
        segmentation = tifffile.imread(segmentation_path).astype(np.int32)

        transformed = self.transform(image=image, mask=segmentation)
        image = transformed["image"]
        segmentation = transformed["mask"]

        flow = compute_flow(segmentation)

        # TODO: this is not very pretty
        transformed = ToTensorV2()(image=image, mask=segmentation)
        image = transformed["image"]
        segmentation = transformed["mask"]
        flow = ToTensorV2()(image=flow)["image"]

        is_foreground = segmentation != 0

        return image, is_foreground, flow

    def __len__(self):
        return len(self.image_files)


class CellDataModule(lightning.LightningDataModule):
    def __init__(self, root_dir, batch_size=16, num_workers=32):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_data = CellDataset(
                root_dir=self.root_dir,
                split="train",
            )
            self.validation_data = CellDataset(
                root_dir=self.root_dir,
                split="validation",
            )
        elif stage == "test" or stage == "predict" or stage is None:
            self.test_data = CellDataset(
                root_dir=self.root_dir,
                split="test",
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

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
