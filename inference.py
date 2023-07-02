import argparse
from pathlib import Path

import tqdm
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

    arguments = parser.parse_args()
    model = UNet.load_from_checkpoint(arguments.from_checkpoint)
    data = CellDataModule(root_dir=arguments.root_dir)
    trainer = lightning.Trainer(accelerator="gpu", devices=2)

    with torch.no_grad():
        predictions = trainer.predict(model, data)
        for prediction_batch in tqdm.tqdm(predictions):
            instances = gradients_to_instances(prediction_batch)
