from pathlib import Path

import albumentations
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
import lightning
import tifffile
import torch
from torch.utils.data import Dataset, DataLoader


class CellDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        root_dir = Path(root_dir)
        if split == "train":
            self.image_files = sorted(list(root_dir.glob(r"[abc]" + "/*.tif")))
            self.flow_gradient_files = sorted(
                list(root_dir.glob(r"[abc]" + "_flow/*.tif"))
            )
        elif split == "validation":
            self.image_files = sorted(list(root_dir.glob(r"[c]" + "/*.tif")))
            self.flow_gradient_files = sorted(
                list(root_dir.glob(r"[c]" + "_flow/*.tif"))
            )
        elif split == "test":
            self.image_files = sorted(list(root_dir.glob(r"[de]" + "/*.tif")))
            self.flow_gradient_files = None

        if split == "train":
            self.image_transform = albumentations.Compose(
                [
                    albumentations.Resize(480, 384),
                    albumentations.RandomBrightnessContrast(),
                    albumentations.GaussNoise(var_limit=(20, 70)),
                    albumentations.Normalize(
                        mean=33.53029578908284 / 255,
                        std=23.36764441145509 / 255,
                    ),
                    ToTensorV2(),
                ]
            )
        else:
            self.image_transform = albumentations.Compose(
                [
                    albumentations.Resize(480, 384),
                    albumentations.Normalize(
                        mean=33.53029578908284 / 255,
                        std=23.36764441145509 / 255,
                    ),
                    ToTensorV2(),
                ]
            )
        self.flow_transform = albumentations.Compose(
            [
                albumentations.Resize(480, 384),
                ToTensorV2(),
            ]
        )

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = tifffile.imread(image_path)
        if self.flow_gradient_files:
            flow_gradient = tifffile.imread(self.flow_gradient_files[idx])
            flow_gradient = self.flow_transform(image=flow_gradient)["image"]
            flow_gradient = flow_gradient.type(torch.float32)
        else:
            flow_gradient = torch.zeros((2, image.shape[0], image.shape[1]))

        image = self.image_transform(image=image)["image"]
        mask = torch.sum(flow_gradient**2, axis=0) != 0
        file_name = "/".join(str(image_path).split("/")[-2:])
        return image, mask, flow_gradient, file_name

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
            print(
                f"DATASET_LENGTH: train: {len(self.train_data)}, validation: {len(self.validation_data)}"
            )
        elif stage == "test" or stage == "predict" or stage is None:
            self.test_data = CellDataset(
                root_dir=self.root_dir,
                split="test",
            )
            print(f"DATASET_LENGTH: test: {len(self.test_data)}")

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
            batch_size=16,
            num_workers=self.num_workers,
        )


if __name__ == "__main__":
    cell_dataset = CellDataset(root_dir="../train/", split="train")
    for image, mask, gradient in DataLoader(cell_dataset):
        print(f"image: {image.shape}, mask: {mask.shape}, gradient: {gradient.shape}")
