from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from dataset import CellDataModule
from unet import UNet

if __name__ == "__main__":
    study_name = "unet"
    model = UNet()
    data = CellDataModule()
    objective_metric = "validation/iou"
    checkpoint_callback = ModelCheckpoint(
        f"checkpoints/{study_name}/",
        monitor=objective_metric,
        mode="min",
        save_last=True,
    )
    tensorboard_logger = TensorBoardLogger(
        save_dir="logs",
        name=study_name,
        default_hp_metric=False,
    )

    trainer = Trainer(
        max_epochs=10,
        callbacks=[checkpoint_callback],
        logger=tensorboard_logger,
    )
    trainer.fit(model, data)
