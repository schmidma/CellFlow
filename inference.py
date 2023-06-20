import argparse
from pathlib import Path

import torch
from torch.nn.functional import sigmoid
from torch.utils.data import DataLoader

from dataset import CellDataset
from unet import UNet


def gradients_to_instances(prediction, class_threshold=0.5):
    # batch, channels, height, width
    logits = prediction[:, 0, :, :]
    dx = prediction[:, 1, :, :]
    dy = prediction[:, 2, :, :]
    batch_size = logits.shape[0]
    image_size_x = logits.shape[1]
    image_size_y = logits.shape[2]

    probability = sigmoid(logits)
    foreground = probability > class_threshold
    positions = (
        torch.cartesian_prod(
            torch.arange(image_size_x),
            torch.arange(image_size_y),
        )
        .reshape(image_size_x, image_size_y, 2)
        .unsqueeze(0)
        .repeat(batch_size, 1, 1, 1)
    )
    # scatter_add
    positions.scatter_add(dim=1, index=dx, src=dx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=Path, required=True)
    parser.add_argument("--from_checkpoint", type=Path, required=True)
    parser.add_argument("--pred_dir", type=Path, required=True)
    parser.add_argument("--split", type=str, default="validation")

    arguments = parser.parse_args()
    model = UNet.load_from_checkpoint(arguments.from_checkpoint)
    data = CellDataset(root_dir=arguments.root_dir, split=arguments.split)

    for batch in DataLoader(data, batch_size=16):
        image, _, _ = batch
        prediction = model(image)
        instances = gradients_to_instances(prediction)
