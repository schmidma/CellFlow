import argparse
from pathlib import Path

import torch
import albumentations
import lightning
import os
from multiprocessing import Pool
import tifffile
import cv2
from torch.nn.functional import sigmoid
from torch.utils.data import DataLoader

from dataset import CellDataModule
from postprocessing import apply_flow, cluster
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

def infer_batch(batch):
    image, _, _, filenames = batch
    prediction = model(image)
    instances = gradients_to_instances(prediction)
    print(f"instances.shape: {instances.shape}")
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
