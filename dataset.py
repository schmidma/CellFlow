import os
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union, List

import albumentations
from albumentations.pytorch.transforms import ToTensorV2
import lightning
import numpy as np
import tifffile
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.utils import download_and_extract_archive
from tqdm.contrib.concurrent import process_map

from flow import process_image


class ObiWanMicrobi(Dataset):
    mirror = "https://zenodo.org/record/7260137/files/ctc_format.zip"
    archive_md5 = "f29fddadcee5c18c86716d3958c5e0da"

    def __init__(
        self,
        root_dir: Union[str, os.PathLike],
        split: str = "train",
        transform: Optional[Callable] = None,
        download: bool = False,
        precompute_flow: bool = True,
        max_workers: int = os.cpu_count() or 1,
    ):
        self.root_dir = Path(root_dir)

        if download and not self._check_download_exists():
            download_and_extract_archive(
                self.mirror,
                str(self.root_dir),
                md5=self.archive_md5,
            )

        if split == "train":
            self.runs = ["00", "01"]
        elif split == "validation":
            self.runs = ["02"]
        elif split == "test":
            self.runs = ["03", "04"]
        else:
            raise ValueError(f"Invalid split: {split}")

        self.image_files = self._image_files(self.runs)
        self.segmentation_files = self._segmentation_files(self.runs)
        if len(self.image_files) != len(self.segmentation_files):
            raise ValueError(
                f"Number of images ({len(self.image_files)}) does not match number of segmentations ({len(self.segmentation_files)})"
            )

        if precompute_flow and not self._check_flow_exists(self.runs):
            print(f"Precomputing flow for run {', '.join(self.runs)} ...")
            process_map(
                process_image,
                self.segmentation_files,
                max_workers=max_workers,
                chunksize=1,
                total=len(self.segmentation_files),
            )

        self.flow_files = self._flow_files(self.runs)
        if len(self.image_files) != len(self.flow_files):
            raise ValueError(
                f"Number of images ({len(self.image_files)}) does not match number of flows ({len(self.flow_files)})"
            )

        self.transform = transform

    def _image_files(self, runs: List[str]) -> List[Path]:
        return sorted(
            [path for run in runs for path in self.root_dir.glob(f"{run}/*.tif")]
        )

    def _segmentation_files(self, runs: List[str]) -> List[Path]:
        return sorted(
            [path for run in runs for path in self.root_dir.glob(f"{run}_GT/SEG/*.tif")]
        )

    def _flow_files(self, runs: List[str]) -> List[Path]:
        return sorted(
            [
                path
                for run in runs
                for path in self.root_dir.glob(f"{run}_GT/FLOW/*.tif")
            ]
        )

    def _check_download_exists(self) -> bool:
        runs = ["00", "01", "02", "03", "04"]
        return (
            len(self._image_files(runs)) != 0
            and len(self._segmentation_files(runs)) != 0
        )

    def _check_flow_exists(self, runs: List[str]) -> bool:
        return len(self._flow_files(runs)) != 0

    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any]:
        image_path = self.image_files[idx]
        image = tifffile.imread(image_path)

        segmentation_path = self.segmentation_files[idx]
        segmentation = tifffile.imread(segmentation_path).astype(np.int32)

        flow_path = self.flow_files[idx]
        flow = tifffile.imread(flow_path)

        if self.transform is not None:
            image, segmentation, flow = self.transform(image, segmentation, flow)

        return image, segmentation, flow

    def __len__(self):
        return len(self.image_files)


class CellData(lightning.LightningDataModule):
    def __init__(self, root_dir, batch_size=8, num_workers=os.cpu_count() or 1):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _train_transform(self, image, segmentation, flow):
        resize = albumentations.Resize(480, 384)
        transform = albumentations.Compose(
            [
                resize,
                albumentations.RandomBrightnessContrast(),
                albumentations.GaussNoise(var_limit=(20, 70)),
                albumentations.Normalize(
                    mean=33.53029578908284 / 255,
                    std=23.36764441145509 / 255,
                ),
                ToTensorV2(),
            ]
        )
        transformed = transform(image=image, mask=segmentation)
        image = transformed["image"]
        segmentation = transformed["mask"]

        flow_transform = albumentations.Compose(
            [
                resize,
                ToTensorV2(),
            ]
        )
        flow = flow_transform(image=flow)["image"]

        return image, segmentation, flow

    def _test_transform(self, image, segmentation, flow):
        resize = albumentations.Resize(480, 384)
        transform = albumentations.Compose(
            [
                resize,
                albumentations.Normalize(
                    mean=33.53029578908284 / 255,
                    std=23.36764441145509 / 255,
                ),
                ToTensorV2(),
            ]
        )
        transformed = transform(image=image, mask=segmentation)
        image = transformed["image"]
        segmentation = transformed["mask"]

        flow_transform = albumentations.Compose(
            [
                resize,
                ToTensorV2(),
            ]
        )
        flow = flow_transform(image=flow)["image"]

        return image, segmentation, flow

    def prepare_data(self):
        ObiWanMicrobi(
            self.root_dir, split="train", download=True, max_workers=self.num_workers
        )
        ObiWanMicrobi(
            self.root_dir,
            split="validation",
            download=True,
            max_workers=self.num_workers,
        )
        ObiWanMicrobi(
            self.root_dir, split="test", download=True, max_workers=self.num_workers
        )

    def setup(self, stage=None):
        if stage == "fit" or stage == "validate" or stage is None:
            self.train_data = ObiWanMicrobi(
                root_dir=self.root_dir,
                split="train",
                download=False,
                precompute_flow=True,
                max_workers=self.num_workers,
                transform=self._train_transform,
            )
            self.validation_data = ObiWanMicrobi(
                root_dir=self.root_dir,
                split="validation",
                download=False,
                precompute_flow=True,
                max_workers=self.num_workers,
                transform=self._test_transform,
            )
        elif stage == "test" or stage == "predict" or stage is None:
            self.test_data = ObiWanMicrobi(
                root_dir=self.root_dir,
                split="test",
                download=False,
                precompute_flow=True,
                max_workers=self.num_workers,
                transform=self._test_transform,
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
