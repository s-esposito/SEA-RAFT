import torch


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
