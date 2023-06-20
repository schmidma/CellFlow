from dataset import CellDataset
import numpy as np

ROOT_DIR = "/hkfs/work/workspace/scratch/hgf_pdv3669-health_train_data/train"

dataset = CellDataset(root_dir=ROOT_DIR, split="train")

images = []

for img, mask in dataset:
    images.append(img)

images = np.array(images)

mean = np.mean(images)
std = np.std(images)

print(f"mean: {mean}, std: {std}")
