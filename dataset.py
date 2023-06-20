from pathlib import Path

import albumentations
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset


class CellDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.transform = transform

        root_dir = Path(root_dir)
        if split == "train":
            self.image_files = sorted(list(root_dir.glob(r"[ab]" + "/*.tif")))
            self.flow_gradient_files = sorted(
                list(root_dir.glob(r"[ab]" + "_flow/*.tif"))
            )
        elif split == "val":
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
            transformed_image = self.transform(image=image)["image"]
        mask = torch.sum(flow_gradient**2, axis=0) != 0
        return transformed_image, mask, flow_gradient

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

if __name__ == "__main__":
    cell_dataset = CellDataset()