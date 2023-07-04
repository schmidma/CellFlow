from lightning.pytorch.cli import LightningCLI

from dataset import CellData
from unet import UNet


def cli_main():
    cli = LightningCLI(UNet, CellData)


if __name__ == "__main__":
    cli_main()
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--root_dir", type=Path, default=Path.cwd())
    # arguments = parser.parse_args()
    #
    # torch.set_float32_matmul_precision("medium")
    #
    # study_name = "unet"
    # model = UNet(learning_rate=1e-2)
    # data = CellData(root_dir=arguments.root_dir / "data", batch_size=8)
    #
    # objective_metric = "validation/iou"
    # checkpoint_callback = ModelCheckpoint(
    #     arguments.root_dir / f"checkpoints/{study_name}/",
    #     monitor=objective_metric,
    #     mode="min",
    #     save_last=True,
    # )
    # tensorboard_logger = TensorBoardLogger(
    #     save_dir=arguments.root_dir / "logs",
    #     name=study_name,
    #     default_hp_metric=False,
    # )
    #
    # trainer = Trainer(
    #     fast_dev_run=True,
    #     max_epochs=20,
    #     log_every_n_steps=25,
    #     callbacks=[checkpoint_callback],
    #     logger=tensorboard_logger,
    # )
    # trainer.fit(model, data)
