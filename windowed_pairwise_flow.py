import sys
sys.path.append("core")
import argparse
import os
import cv2
import numpy as np
import glob
import torch
import torch.nn.functional as F
# import matplotlib.pyplot as plt
from config.parser import parse_args
# from consistency_utils import consistency_mask
from raft import RAFT
# from utils.flow_viz import flow_to_image
from utils.utils import load_ckpt
# from pathlib import Path


@torch.no_grad()
def forward_flow(args, model, image1, image2):
    output = model(image1, image2, iters=args.iters, test_mode=True)
    flow_final = output["flow"][-1]
    info_final = output["info"][-1]
    return flow_final, info_final


@torch.no_grad()
def calc_flow(args, model, image1, image2):
    img1 = F.interpolate(
        image1, scale_factor=2**args.scale, mode="bilinear", align_corners=False
    )
    img2 = F.interpolate(
        image2, scale_factor=2**args.scale, mode="bilinear", align_corners=False
    )
    flow, info = forward_flow(args, model, img1, img2)
    flow_down = F.interpolate(
        flow, scale_factor=0.5**args.scale, mode="bilinear", align_corners=False
    ) * (0.5**args.scale)
    info_down = F.interpolate(info, scale_factor=0.5**args.scale, mode="area")
    return flow_down, info_down


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene-path",
        help="path to the scene",
        type=str,
        default="/home/stefano/Codebase/DynSLAM/data/davis/car-turn",
    )
    parser.add_argument(
        "--cfg",
        help="experiment configure file name",
        type=str,
        default="config/eval/spring-L.json",
    )
    parser.add_argument('--path', help='checkpoint path', type=str, default="models/Tartan-C-T-TSKH-spring540x960-M.pth")
    parser.add_argument('--device', help='inference device', type=str, default='cuda')
    args = parse_args(parser)
    print(args)
    
    model = RAFT(args)
    load_ckpt(model, args.path)
    device = torch.device("cuda")
    model = model.to(device)
    model.eval()
    # # images_paths
    # images_paths = glob.glob(
    #     os.path.join(args.scene_path, "rgba/rgba_*.png")
    # )
    #
    # images_paths
    images_paths = glob.glob(
        os.path.join(args.scene_path, "rgb/*.jpg")
    )
    images_paths.sort()

    # seq lenght
    first_frame_idx = 0
    seq_len = -1

    #
    window_size = 10

    # select
    if seq_len != -1:
        images_paths = images_paths[:seq_len]
    else:
        # use all
        seq_len = len(images_paths)
    print("images_paths", len(images_paths))

    # results_path = f"./outputs/outs_{seq_len}_ws_{window_size}/"
    results_path = os.path.join(args.scene_path, "optical_flow")
    os.system(f"mkdir -p {results_path}")

    for window_start_idx in range(first_frame_idx, seq_len - window_size + 1, 1):

        print("window_start_idx", window_start_idx)

        for i_ in range(0, window_size - 1, 1):
            for j_ in range(i_, window_size, 1):
                # j indexes the neighbors

                if i_ == j_:
                    continue

                i = window_start_idx + i_
                j = window_start_idx + j_

                print("i", i, "j", j)

                i_str = format(i, "05d")
                j_str = format(j, "05d")

                image1 = cv2.imread(images_paths[i])
                image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
                # print("image1", image1.shape, image1.min(), image1.max())
                image1 = torch.tensor(image1, dtype=torch.float32).permute(2, 0, 1)
                H, W = image1.shape[1:]
                image1 = image1[None].to(device)
                # print("image1", image1.shape, image1.min(), image1.max())

                image2 = cv2.imread(images_paths[j])
                image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
                image2 = torch.tensor(image2, dtype=torch.float32).permute(2, 0, 1)
                image2 = image2[None].to(device)
                
                flow_i_to_j, info = calc_flow(args, model, image1, image2)
                flow_i_to_j = flow_i_to_j[0].permute(1, 2, 0)
                
                flow_j_to_i, info = calc_flow(args, model, image2, image1)
                flow_j_to_i = flow_j_to_i[0].permute(1, 2, 0)

                # save of as npy file
                optical_flow = np.zeros((2, H, W, 2), dtype=np.float32)
                optical_flow[0] = flow_i_to_j.cpu().numpy()
                optical_flow[1] = flow_j_to_i.cpu().numpy()
                np.save(os.path.join(results_path, f"flow_{i_str}_{j_str}.npy"), optical_flow)


if __name__ == "__main__":
    main()
