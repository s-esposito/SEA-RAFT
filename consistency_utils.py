
import torch
import torch.nn.functional as F
from pixels_utils import get_pixels, get_pixels_centers




def consistency_mask(fw_flow, bw_flow, tau=1.0):
    
    # get all pixels in the image plane
    width, height = bw_flow.shape[1], bw_flow.shape[0]
    pixels = get_pixels(height, width)  # (W, H, 2)
    pixels = pixels.permute(1, 0, 2)  # (H, W, 2)
    points = get_pixels_centers(pixels)  # (H, W, 2)
    
    # forward points
    points_fw = points + fw_flow  # (H, W, 2)
    
    # Use grid_sample to fetch bw_flow at those coordinates, producing bw_flow_interpolated.
    #    grid_sample expects shape (N, C, H, W). We'll treat each flow component as a channel.
    bw_flow_4D = bw_flow.permute(2, 0, 1).unsqueeze(0)  # => (1, 2, H, W)
    
    # Build a sampling grid from points_fw, converting from [0..W-1, 0..H-1] to [-1..1].
    #    - x in [0..W-1] => x_norm in [-1..1]
    #    - y in [0..H-1] => y_norm in [-1..1]
    #  NOTE: grid_sample wants coords in (y_norm, x_norm) order if your channels=2, 
    #        but carefully check that you supply them correctly. 
    #        We'll do [batch, H, W, 2] => (y, x) in normalized coords.

    # split out x and y for clarity
    x_fw = points_fw[..., 0]  # (H, W)
    y_fw = points_fw[..., 1]  # (H, W)

    # convert to normalized coordinates
    x_norm = 2.0 * x_fw / (width - 1) - 1.0  # [-1..1]
    y_norm = 2.0 * y_fw / (height - 1) - 1.0  # [-1..1]

    # stack them in (H, W, 2) order
    grid = torch.stack([x_norm, y_norm], dim=-1)  # (H, W, 2)

    # finally reshape to (N=1, H, W, 2) so it can be passed to grid_sample
    grid_4D = grid.unsqueeze(0)  # => (1, H, W, 2)

    # Sample the backward flow at the forward positions
    #    This uses bilinear interpolation by default.
    bw_flow_interpolated_4D = F.grid_sample(
        bw_flow_4D,       # (1, 2, H, W)
        grid_4D,          # (1, H, W, 2)
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False
    )
    # shape => (1, 2, H, W)

    # Convert bw_flow_interpolated back to (H, W, 2)
    bw_flow_interpolated = bw_flow_interpolated_4D.squeeze(0).permute(1, 2, 0)  # => (H, W, 2)

    # Compute the backward-mapped points
    points_bw = points_fw + bw_flow_interpolated  # (H, W, 2)
        
    # dist = || points_bw - points ||
    error = torch.norm(points_bw - points, dim=-1)  # (H, W)
    
    mask_fw = error < tau  # (H, W) mask of regions occluded in frame 2
    # 0 occluded, 1 not occluded
    
    return mask_fw, error