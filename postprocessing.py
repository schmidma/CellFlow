
import torch

def apply_flow(flow, n_steps=200):
    batch_size = flow.shape[0]
    image_height = flow.shape[2]
    image_width = flow.shape[3]

    # positions in (batch_size, image_height, image_width, 2)
    positions = (
        torch.cartesian_prod(
            torch.arange(image_height),
            torch.arange(image_width),
        )
        .reshape(image_height, image_width, 2)
        .unsqueeze(0)
        .repeat(batch_size, 1, 1, 1)
    ).double()

    positions /= (
        torch.tensor([image_height, image_width]).double().unsqueeze(0).unsqueeze(0)
    )
    positions = positions * 2 - 1

    flow *= 2.0 / torch.tensor([image_height, image_width]).unsqueeze(-1).unsqueeze(-1)

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
        torch.tensor([image_height, image_width]).double().unsqueeze(0).unsqueeze(0)
    )

    return positions
