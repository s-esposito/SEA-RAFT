import sys
sys.path.append('core')
import argparse
import os
import cv2
import numpy as np
import glob
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from config.parser import parse_args
from consistency_utils import consistency_mask
from raft import RAFT
from utils.flow_viz import flow_to_image
from utils.utils import load_ckpt

def vis_heatmap(name, image, heatmap):
    # theta = 0.01
    # print(heatmap.max(), heatmap.min(), heatmap.mean())
    heatmap = heatmap[:, :, 0]
    # heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    
    fig = plt.figure(figsize=(15, 5))
    axs = fig.subplots(1, 2)
    cax = axs[0].matshow(heatmap, cmap='viridis', vmin=0)
    fig.colorbar(cax)
    
    # heatmap = heatmap > 0.01
    heatmap = (heatmap * 255).astype(np.uint8)
    colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_VIRIDIS)
    overlay = image * 0.3 + colored_heatmap * 0.7
    overlay = overlay.astype(np.uint8)
    
    axs[1].imshow(overlay)
    
    plt.savefig(name)
    plt.close()
    
    # Create a color bar
    # height, width = image.shape[:2]
    # color_bar = create_color_bar(25, width, cv2.COLORMAP_VIRIDIS)  # Adjust the height and colormap as needed
    # Add the color bar to the image
    # overlay = overlay.astype(np.uint8)
    # combined_image = add_color_bar_to_image(overlay, color_bar, 'vertical')
    # cv2.imwrite(name, cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))

def get_heatmap(info, args):
    raw_b = info[:, 2:]
    log_b = torch.zeros_like(raw_b)
    weight = info[:, :2].softmax(dim=1)              
    log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=args.var_max)
    log_b[:, 1] = torch.clamp(raw_b[:, 1], min=args.var_min, max=0)
    heatmap = (log_b * weight).sum(dim=1, keepdim=True)
    return heatmap

def forward_flow(args, model, image1, image2):
    output = model(image1, image2, iters=args.iters, test_mode=True)
    flow_final = output['flow'][-1]
    info_final = output['info'][-1]
    return flow_final, info_final

def calc_flow(args, model, image1, image2):
    img1 = F.interpolate(image1, scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
    img2 = F.interpolate(image2, scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
    flow, info = forward_flow(args, model, img1, img2)
    flow_down = F.interpolate(flow, scale_factor=0.5 ** args.scale, mode='bilinear', align_corners=False) * (0.5 ** args.scale)
    info_down = F.interpolate(info, scale_factor=0.5 ** args.scale, mode='area')
    return flow_down, info_down

@torch.no_grad()
def demo_data(path, mode, args, model, image1, image2):
    os.system(f"mkdir -p {path}")
    H, W = image1.shape[2:]
    
    flow, info = calc_flow(args, model, image1, image2)
    flow = flow[0].permute(1, 2, 0)
    
    # save as npy file
    np.save(f"{path}flow_{mode}.npy", flow.cpu().numpy())
    
    flow_vis = flow_to_image(flow.cpu().numpy(), convert_to_bgr=False)
    
    image1_np = image1[0].permute(1, 2, 0).cpu().numpy()
    image2_np = image2[0].permute(1, 2, 0).cpu().numpy()
    # rgb to bgr 
    image1_np = cv2.cvtColor(image1_np, cv2.COLOR_RGB2BGR)
    image2_np = cv2.cvtColor(image2_np, cv2.COLOR_RGB2BGR)
    # concatenate image1 to the left of flow_vis
    flow_vis = np.concatenate([image1_np, flow_vis], axis=1)
    # concatenate image2 to the right of flow_vis
    flow_vis = np.concatenate([flow_vis, image2_np], axis=1)
    
    cv2.imwrite(f"{path}flow_{mode}.jpg", flow_vis)
    
    heatmap = get_heatmap(info, args)
    heatmap = heatmap[0].permute(1, 2, 0)
    
    # save as npy file
    np.save(f"{path}heatmap_{mode}.npy", heatmap.cpu().numpy())
    vis_heatmap(f"{path}heatmap_{mode}.jpg", image1[0].permute(1, 2, 0).cpu().numpy(), heatmap.cpu().numpy())
    
    return flow, info, heatmap
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='experiment configure file name', type=str, default="config/eval/spring-L.json")
    parser.add_argument('--path', help='checkpoint path', type=str, default="models/Tartan-C-T-TSKH-spring540x960-M.pth")
    parser.add_argument('--url', help='checkpoint url', type=str, default=None)
    parser.add_argument('--device', help='inference device', type=str, default='cpu')
    args = parse_args(parser)
    
    if args.path is None and args.url is None:
        raise ValueError("Either --path or --url must be provided")
    if args.path is not None:
        model = RAFT(args)
        load_ckpt(model, args.path)
    else:
        model = RAFT.from_pretrained(args.url, args=args)
        
    if args.device == 'cuda':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = model.to(device)
    model.eval()
    
    # images_paths
    images_paths = glob.glob("/home/stefano/Codebase/DynSLAM/data/kubric/dynamic/rgba/rgba_*.png")
    images_paths.sort()
    
    # 
    tau = 10.0
    
    # seq lenght
    seq_len = 30
    
    #
    window_size = 30
    
    # select 
    images_paths = images_paths[:seq_len]
    print("images_paths", len(images_paths))
    
    results_path = f"./outputs/outs_{seq_len}_ws_{window_size}_{int(tau)}/"
    
    for window_start_idx in range(0, seq_len - window_size + 1, 1):
        
        print("window_start_idx", window_start_idx)
        
        for i_ in range(0, window_size - 1, 1):
            for j_ in range(i_, window_size, 1):
                # j indexes the neighbors
            
                if i_ == j_:
                    continue
                
                i = window_start_idx + i_
                j = window_start_idx + j_
        
                print("i", i, "j", j)
                
                i_str = format(i, '05d')
                j_str = format(j, '05d')
        
                image1 = cv2.imread(images_paths[i])
                image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
                print("image1", image1.shape, image1.min(), image1.max())
                image1 = torch.tensor(image1, dtype=torch.float32).permute(2, 0, 1)
                H, W = image1.shape[1:]
                image1 = image1[None].to(device)
                print("image1", image1.shape, image1.min(), image1.max())
            
                image2 = cv2.imread(images_paths[j])
                image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
                image2 = torch.tensor(image2, dtype=torch.float32).permute(2, 0, 1)
                image2 = image2[None].to(device)
        
                flow_i_to_j, info_fw, heatmap_fw = demo_data(results_path, f'fw_{i_str}_{j_str}', args, model, image1, image2)
                flow_j_to_i, info_bw, heatmap_bw = demo_data(results_path, f'bw_{i_str}_{j_str}', args, model, image2, image1)
                # flow_i_to_j (H, W, 2)
                # flow_j_to_i (H, W, 2)
                    
                mask_fw, error_fw = consistency_mask(flow_i_to_j, flow_j_to_i, tau)
                mask_bw, error_bw = consistency_mask(flow_j_to_i, flow_i_to_j, tau)
                
                # vis errors
                fig = plt.figure(figsize=(15, 5))
                axs = fig.subplots(1, 2)
                im = axs[0].matshow(error_fw, cmap='viridis')
                axs[0].set_title("error_fw")
                fig.colorbar(im, ax=axs[0])
                im = axs[1].matshow(error_bw, cmap='viridis')
                axs[1].set_title("error_bw")
                fig.colorbar(im, ax=axs[1])
                plt.savefig(f"{results_path}consistency_{i_str}_{j_str}.png")
                plt.close()

                # mask image2
                image2_masked = image2[0].permute(1, 2, 0).clone()
                # not mask
                image2_masked[~mask_bw] = image2_masked[~mask_bw] * 0.25
                image2_masked = image2_masked.cpu().numpy()
                image1_np = image1[0].permute(1, 2, 0).cpu().numpy()
                # concatenate image1 to the left of image2_masked
                image2_masked = np.concatenate([image1_np, image2_masked], axis=1)
                image2_masked = cv2.cvtColor(image2_masked, cv2.COLOR_RGB2BGR)
                # save as jpg
                cv2.imwrite(f"{results_path}rgb_masked_bw_{i_str}_{j_str}.jpg", image2_masked)
                
                # mask image1
                image1_masked = image1[0].permute(1, 2, 0).clone()
                # not mask
                image1_masked[~mask_fw] = image1_masked[~mask_fw] * 0.25
                image1_masked = image1_masked.cpu().numpy()
                image2_np = image2[0].permute(1, 2, 0).cpu().numpy()
                # concatenate image2 to the right of image1_masked
                image1_masked = np.concatenate([image1_masked, image2_np], axis=1)
                image1_masked = cv2.cvtColor(image1_masked, cv2.COLOR_RGB2BGR)
                # save as jpg
                cv2.imwrite(f"{results_path}rgb_masked_fw_{i_str}_{j_str}.jpg", image1_masked)
                
                # save as jpg
                mask_fw_vis = (mask_fw.cpu().numpy() * 255).astype(np.uint8)
                mask_bw_vis = (mask_bw.cpu().numpy() * 255).astype(np.uint8)
                cv2.imwrite(f"{results_path}mask_fw_{i_str}_{j_str}.jpg", mask_fw_vis)
                cv2.imwrite(f"{results_path}mask_bw_{i_str}_{j_str}.jpg", mask_bw_vis)
                
                # save of as npy file
                optical_flow = np.zeros((2, H, W, 2), dtype=np.float32)
                optical_flow[0] = flow_i_to_j.cpu().numpy()
                optical_flow[1] = flow_j_to_i.cpu().numpy()
                np.save(f"{results_path}flow_{i_str}_{j_str}.npy", optical_flow)
                
                # save masks as npy file
                masks = np.zeros((H, W, 2), dtype=np.bool)
                masks[:, :, 0] = mask_fw.cpu().numpy()
                masks[:, :, 1] = mask_bw.cpu().numpy()
                np.save(f"{results_path}masks_{i_str}_{j_str}.npy", masks)

if __name__ == '__main__':
    main()