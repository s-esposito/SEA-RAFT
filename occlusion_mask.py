import numpy as np
import torch
import matplotlib.pyplot as plt


def get_pixels(height: int, width: int, device: str = "cpu") -> torch.Tensor:
    """returns all image pixels coords
    Args:
        height (int): frame height
        width (int): frame width
        device (str, optional): Defaults to "cpu".
    Returns:
        pixels (torch.Tensor): dtype int32, shape (W, H, 2), values in [0, W-1], [0, H-1]
    """

    pixels_x, pixels_y = torch.meshgrid(
        torch.arange(width, device=device),
        torch.arange(height, device=device),
        indexing="ij",
    )
    pixels = torch.stack([pixels_x, pixels_y], dim=-1).type(torch.int32)

    return pixels


def get_pixels_centers(pixels: torch.Tensor) -> torch.Tensor:
    """return the center of each pixel
    Args:
        pixels (torch.Tensor): (N, 2) list of pixels
    Returns:
        pixels_centers (torch.Tensor): (N, 2) list of pixels centers
    """

    points_2d_screen = pixels.float()  # cast to float32
    points_2d_screen = points_2d_screen + 0.5  # pixels centers

    return points_2d_screen


# load custom/bw_flow.npy
bw_flow = torch.tensor(np.load("custom/bw_flow.npy"))
# load  custom/fw_flow.npy
fw_flow = torch.tensor(np.load("custom/fw_flow.npy"))
#
print("bw_flow", bw_flow.shape, bw_flow.min(), bw_flow.max())  # (H, W, 2)
print("fw_flow", fw_flow.shape, fw_flow.min(), fw_flow.max())  # (H, W, 2)

# get all pixels in the image plane
width, height = bw_flow.shape[1], bw_flow.shape[0]
print("width", width, "height", height)

pixels = get_pixels(height, width)  # (W, H, 2)
pixels = pixels.permute(1, 0, 2)  # (H, W, 2)
points = get_pixels_centers(pixels)  # (H, W, 2)
print("points", points.shape)

# go forwards
points_fw = points + fw_flow  # (H, W, 2)
# go backwards
points_bw = points_fw + bw_flow  # (H, W, 2)

# calculate distance between points and points_fw
dist = torch.norm(points - points_bw, dim=-1)  # (H, W)

# # go backwards
# points_bw = points + bw_flow  # (H, W, 2)
# # go forwards
# points_fw = points_bw + fw_flow  # (H, W, 2)

# # calculate distance between points and points_fw
# dist = torch.norm(points - points_fw, dim=-1)  # (H, W)

tau = 10

mask = dist < tau

fig = plt.figure(figsize=(15, 5))
axs = fig.subplots(1, 2)
axs[0].imshow(mask.cpu().numpy(), cmap="gray")
axs[0].set_title(f"Occlusion Mask tau={tau}")
im = axs[1].imshow(dist.cpu().numpy(), cmap="hot")
axs[1].set_title("Distance")
fig.colorbar(im, ax=axs[1], orientation="vertical")
plt.show()