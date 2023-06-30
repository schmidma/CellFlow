import numpy as np
from scipy.signal import convolve2d


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
    dy = gradients[0]
    dx = gradients[1]
    return np.stack([dy, dx], axis=-1), heat_indices


def normalize_gradients(gradients):
    norms = np.linalg.norm(gradients, axis=-1)
    gradients[norms != 0] /= norms[norms != 0, np.newaxis]


def compute_flow(mask):
    ids = np.unique(mask)
    image_height, image_width = mask.shape
    flow = np.zeros((image_height, image_width, 2), dtype=np.float32)

    for id in ids[ids != 0]:
        indices = np.argwhere(mask == id)
        median = np.median(indices, axis=0)
        distances = np.sum((indices - median) ** 2, axis=1)
        center = indices[np.argmin(distances)]

        result = simulate_heat_diffusion(indices, center)
        if result is None:
            continue
        gradients, heat_indices = result
        normalize_gradients(gradients)

        flow[indices[:, 0], indices[:, 1]] = gradients[
            heat_indices[:, 0], heat_indices[:, 1]
        ]
    return flow
