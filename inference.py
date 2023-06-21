import argparse
from pathlib import Path

import torch
import albumentations
import lightning
import os
import tifffile
import cv2

from dataset import CellDataModule
from unet import UNet


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=Path, required=True)
    parser.add_argument("--from_checkpoint", type=Path, required=True)
    parser.add_argument("--pred_dir", type=Path, required=True)

    arguments = parser.parse_args()
    model = UNet.load_from_checkpoint(arguments.from_checkpoint)
    data = CellDataModule(root_dir=arguments.root_dir)
    trainer = lightning.Trainer(gpus=4)

    with torch.no_grad():
        instances, filenames = trainer.predict(model, data)
        for instance, filename in zip(instances, filenames):
            resize = albumentations.Resize(256, 256, interpolation=cv2.INTER_NEAREST)
            instance = resize(image=instance.numpy())["image"]
            save_path = arguments.pred_dir / filename
            os.makedirs(save_path.parent, exist_ok=True)
            tifffile.imwrite(
                save_path,
                instance,
                compression="lzw"
            )
