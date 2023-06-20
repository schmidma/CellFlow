from pathlib import Path
from multiprocessing import Pool

import numpy as np
from scipy.signal import convolve2d
import tifffile
from tqdm import tqdm
import multiprocessing


ROOT_DIR = Path("/hkfs/work/workspace/scratch/hgf_pdv3669-H3/train/")


def simulate_heat_diffusion(indices, heat_center):
    index_range = np.max(indices, axis=0) - np.min(indices, axis=0) + 1
    if index_range[0] < 3 or index_range[1] < 3:
        return None
    heat_map = np.zeros(index_range)
    heat_source = heat_center - np.min(indices, axis=0)
    heat_indices = indices - np.min(indices, axis=0)

    N = 2 * np.sum(index_range)
    for _ in range(N):
        heat_map[heat_source[0], heat_source[1]] += 1.0
        next = convolve2d(heat_map, np.ones((3, 3)) / 9, mode="same")
        heat_map[heat_indices[:, 0], heat_indices[:, 1]] = next[
            heat_indices[:, 0], heat_indices[:, 1]
        ]

    gradients = np.gradient(heat_map)
    dx = gradients[0]
    dy = gradients[1]
    return np.stack([dx, dy], axis=0), heat_indices


def normalize_gradients(gradients):
    norms = np.linalg.norm(gradients, axis=0)
    gradients[:, norms != 0] /= norms[np.newaxis, norms != 0]


def compute_gradient_mask(mask_path):
    mask = tifffile.imread(mask_path)
    ids = np.unique(mask)
    mask_gradients = np.zeros((2, mask.shape[0], mask.shape[1]))

    progress = tqdm(
        ids[ids != 0],
        desc="Simulate per ID",
        leave=False,
        position=multiprocessing.current_process()._identity[0],
    )
    for id in progress:
        progress.set_postfix_str(f"ID: {id}")
        indices = np.argwhere(mask == id)
        if len(indices) < 9:
            progress.write(
                f"In {mask_path.name}: skipping ID {id} with {len(indices)} pixels"
            )
            continue
        median = np.median(indices, axis=0)
        distances = np.sum((indices - median) ** 2, axis=1)
        center = indices[np.argmin(distances)]

        result = simulate_heat_diffusion(indices, center)
        if result is None:
            progress.write(f"In {mask_path.name}: skipping ID {id} not large enough")
            continue
        gradients, heat_indices = result
        normalize_gradients(gradients)

        mask_gradients[:, indices[:, 0], indices[:, 1]] = gradients[
            :, heat_indices[:, 0], heat_indices[:, 1]
        ]
    return mask_gradients


def compute_gradient_masks(mask_path):
    gradients_file_name = mask_path.stem.split("_")[1] + "_gradients.tif"
    gradients_path = mask_path.parent / gradients_file_name
    if gradients_path.exists():
        tqdm.write(f"Skipping {mask_path.name}")
        return

    mask_gradients = compute_gradient_mask(mask_path)
    tifffile.imwrite(gradients_path, mask_gradients)


if __name__ == "__main__":
    mask_files = sorted(list(ROOT_DIR.glob(r"[abc]" + "_GT/man_*.tif")))
    # progress = tqdm(mask_files, desc="Mask Files")
    with Pool(15) as p:
        for _ in tqdm(
            p.imap_unordered(compute_gradient_masks, mask_files),
            total=len(mask_files),
        ):
            pass
