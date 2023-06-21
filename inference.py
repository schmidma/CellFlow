import argparse
from pathlib import Path

import torch
from torch.nn.functional import sigmoid
from torch.utils.data import DataLoader

from dataset import CellDataset
from postprocessing import apply_flow
from unet import UNet


def gradients_to_instances(prediction, class_threshold=0.5):
    # batch, channels, height, width
    logits = prediction[:, 0, :, :]
    flow = prediction[:, [1, 2], :, :]
    positions = apply_flow(flow)
    probability = sigmoid(logits)
    mask = probability > class_threshold
    


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
