import numpy as np
import torch
import glob
import cv2
import matplotlib.pyplot as plt
from consistency_utils import consistency_mask


if __name__ == "__main__":
    image1_path = "./custom/image1.*"
    image2_path = "./custom/image2.*"
    image1_path = glob.glob(image1_path)
    if len(image1_path) == 0:
        raise ValueError("No image1 found")
    image2_path = glob.glob(image2_path)
    if len(image2_path) == 0:
        raise ValueError("No image2 found")
    image1 = cv2.imread(image1_path[0])
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.imread(image2_path[0])
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    image1 = np.array(image1, dtype=np.float32) / 255.0
    image2 = np.array(image2, dtype=np.float32) / 255.0

    # load custom/bw_flow.npy
    bw_flow = torch.tensor(np.load("custom/bw_flow.npy"))
    # load  custom/fw_flow.npy
    fw_flow = torch.tensor(np.load("custom/fw_flow.npy"))
    #
    print("bw_flow", bw_flow.shape, bw_flow.min(), bw_flow.max())  # (H, W, 2)
    print("fw_flow", fw_flow.shape, fw_flow.min(), fw_flow.max())  # (H, W, 2)

    tau = 5.0

    mask_fw = consistency_mask(fw_flow, bw_flow, tau)

    mask_image1 = image1.copy()
    mask_image1[mask_fw] = 0

    plt.imshow(mask_image1)
    plt.show()

    mask_bw = consistency_mask(bw_flow, fw_flow, tau)

    mask_image2 = image2.copy()
    mask_image2[mask_bw] = 0

    plt.imshow(mask_image2)
    plt.show()
