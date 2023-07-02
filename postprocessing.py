import torch


def apply_flow(flow, n_steps=200):
    batch_size = flow.shape[0]
    image_height = flow.shape[2]
    image_width = flow.shape[3]
    device = flow.device
    # positions in (batch_size, image_height, image_width, 2)
    positions = (
        torch.cartesian_prod(
            torch.arange(image_height, device=device, dtype=torch.float),
            torch.arange(image_width, device=device, dtype=torch.float),
        )
        .reshape(image_height, image_width, 2)
        .unsqueeze(0)
        .repeat(batch_size, 1, 1, 1)
    )

    positions /= (
        torch.tensor([image_height, image_width], device=device, dtype=torch.float)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    positions = positions * 2 - 1

    flow *= 2.0 / torch.tensor([image_height, image_width], device=device).unsqueeze(
        -1
    ).unsqueeze(-1)

    positions = positions[:, :, :, [1, 0]]
    flow = flow[:, [1, 0], :, :]
    for _ in range(n_steps):
        gradients = torch.nn.functional.grid_sample(
            flow, positions, align_corners=False
        )
        for k in range(2):
            positions[:, :, :, k] = positions[:, :, :, k] + gradients[:, k, :, :]
    torch.clamp(positions, -1, 1)
    positions = positions[:, :, :, [1, 0]]
    flow = flow[:, [1, 0], :, :]

    positions = (positions + 1) / 2
    positions *= (
        torch.tensor([image_height, image_width], device=device)
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
    )

    return positions


def cluster(positions, is_foreground, max_threshold=10):
    batch_size, image_height, image_width, _ = positions.shape
    device = positions.device
    ids = torch.zeros(
        (batch_size, image_height, image_width), dtype=torch.long, device=device
    )
    for i in range(batch_size):
        masked_points = positions[i][is_foreground[i]]
        reshaped_positions = positions[i].reshape(-1, 2)
        bin_y = torch.arange(
            -0.5, image_height + 0.5, 1, dtype=torch.float, device=device
        )
        bin_x = torch.arange(
            -0.5, image_width + 0.5, 1, dtype=torch.float, device=device
        )
        hist, _ = torch.histogramdd(reshaped_positions, (bin_y, bin_x))
        max_hist = hist > max_threshold
        max_indices = max_hist.nonzero()
        distances = torch.sum(
            (masked_points.unsqueeze(1) - max_indices.unsqueeze(0)) ** 2, axis=2
        )
        best_cluster_index = torch.argmin(distances, axis=1)
        ids[i][is_foreground[i]] = best_cluster_index
    return ids
