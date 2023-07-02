import argparse
from pathlib import Path

import torch
import albumentations
import lightning
import os
import tifffile
import cv2
from torch.nn.functional import sigmoid
from postprocessing import apply_flow, cluster

from dataset import CellDataModule
from unet import UNet


def gradients_to_instances(prediction, class_threshold=0.5):
    # batch, channels, height, width
    logits = prediction[:, 0, :, :]
    flow = prediction[:, [1, 2], :, :]
    positions = apply_flow(flow)
    probability = sigmoid(logits)
    mask = probability > class_threshold
    ids = cluster(positions, mask)
    return ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=Path, required=True)
    parser.add_argument("--from_checkpoint", type=Path, required=True)
    parser.add_argument("--pred_dir", type=Path, required=True)

    arguments = parser.parse_args()
    model = UNet.load_from_checkpoint(arguments.from_checkpoint, map_location="cpu")
    data = CellDataModule(root_dir=arguments.root_dir, batch_size=1, num_workers=0)
    trainer = lightning.Trainer(accelerator="cpu", devices=1)

    with torch.no_grad():
        predictions = trainer.predict(model, data)
        for prediction_batch in predictions:
            instances = gradients_to_instances(prediction_batch.detach())
