import argparse
from pathlib import Path

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import torch

from dataset import CellDataModule
from unet import UNet

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=Path, required=True)
    arguments = parser.parse_args()

    torch.set_float32_matmul_precision("medium")

    study_name = "unet"
    model = UNet(learning_rate=1e-2)
    data = CellDataModule(root_dir=arguments.root_dir / "train")
    objective_metric = "validation/iou"
    checkpoint_callback = ModelCheckpoint(
        arguments.root_dir / f"checkpoints/{study_name}/",
        monitor=objective_metric,
        mode="min",
        save_last=True,
    )
    tensorboard_logger = TensorBoardLogger(
        save_dir=arguments.root_dir / "logs",
        name=study_name,
        default_hp_metric=False,
    )

    trainer = Trainer(
        accelerator="gpu",
        devices=4,
        strategy="ddp",
        max_epochs=100,
        callbacks=[checkpoint_callback],
        logger=tensorboard_logger,
    )
    trainer.fit(model, data)
