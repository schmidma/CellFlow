import argparse
from pathlib import Path

import torch
import lightning

from dataset import CellData
from unet import UNet


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=Path, required=True)
    parser.add_argument("--from_checkpoint", type=Path, required=True)

    arguments = parser.parse_args()
    model = UNet.load_from_checkpoint(arguments.from_checkpoint)
    data = CellData(root_dir=arguments.root_dir)
    trainer = lightning.Trainer(accelerator="gpu", devices=2)

    with torch.no_grad():
        instances = trainer.predict(model, data)
