def vis_heatmap(name, image, heatmap):
    # theta = 0.01
    # print(heatmap.max(), heatmap.min(), heatmap.mean())
    heatmap = heatmap[:, :, 0]
    # heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    fig = plt.figure(figsize=(15, 5))
    axs = fig.subplots(1, 2)
    cax = axs[0].matshow(heatmap, cmap="viridis", vmin=0)
    fig.colorbar(cax)

    # heatmap = heatmap > 0.01
    heatmap = (heatmap * 255).astype(np.uint8)
    colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_VIRIDIS)
    overlay = image * 0.3 + colored_heatmap * 0.7
    overlay = overlay.astype(np.uint8)

    axs[1].imshow(overlay)
    plt.savefig(name)
    plt.close("all")


def get_heatmap(info, args):
    raw_b = info[:, 2:]
    log_b = torch.zeros_like(raw_b)
    weight = info[:, :2].softmax(dim=1)
    log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=args.var_max)
    log_b[:, 1] = torch.clamp(raw_b[:, 1], min=args.var_min, max=0)
    heatmap = (log_b * weight).sum(dim=1, keepdim=True)
    return heatmap


@torch.no_grad()
def demo_data(path, mode, args, model, image1, image2, save_imgs=False):

    flow, info = calc_flow(args, model, image1, image2)
    flow = flow[0].permute(1, 2, 0)

    # save as npy file
    np.save(f"{path}flow_{mode}.npy", flow.cpu().numpy())

    if save_imgs:
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
        vis_heatmap(
            f"{path}heatmap_{mode}.jpg",
            image1[0].permute(1, 2, 0).cpu().numpy(),
            heatmap.cpu().numpy(),
        )

    return flow, info, heatmap