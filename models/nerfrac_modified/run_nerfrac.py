import os
import numpy as np
import imageio
import random
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter
from utils import *
import pandas as pd
from run_nerfrac_helpers import *

from load_llff import load_llff_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_consistency_rays(rays_o_full, rays_d_full, coords, H, W):
    """Build 9-neighbor ray batch for a set of pixel coordinates.

    :param rays_o_full: (H, W, 3) ray origins for full image
    :param rays_d_full: (H, W, 3) ray directions for full image
    :param coords: (N, 2) long tensor of [row, col] pixel coordinates
    :param H: image height
    :param W: image width
    :return: batch_rays (2, N*9, 3) tensor ready for render()
    """
    N = coords.shape[0]
    dev = rays_o_full.device
    neighbor_offsets = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]

    all_o = torch.zeros(N, 9, 3, device=dev)
    all_d = torch.zeros(N, 9, 3, device=dev)

    rows = coords[:, 0]  # (N,)
    cols = coords[:, 1]  # (N,)

    # Center ray (index 0)
    all_o[:, 0] = rays_o_full[rows, cols]
    all_d[:, 0] = rays_d_full[rows, cols]

    # 8 neighbors (indices 1-8)
    for idx, (dh, dw) in enumerate(neighbor_offsets):
        nh = torch.clamp(rows + dh, 0, H - 1)
        nw = torch.clamp(cols + dw, 0, W - 1)
        all_o[:, idx + 1] = rays_o_full[nh, nw]
        all_d[:, idx + 1] = rays_d_full[nh, nw]

    # Flatten to (N*9, 3) and stack as (2, N*9, 3)
    all_o = all_o.reshape(-1, 3)
    all_d = all_d.reshape(-1, 3)
    batch_rays = torch.stack([all_o, all_d], dim=0)  # (2, N*9, 3)

    return batch_rays


def compute_consistency_loss(
    poses, masks, H, W, K, i_train, render_kwargs, N_consist, chunk, device
):
    """Multi-view depth consistency loss.

    Picks two random training views, renders N_consist water pixels in each,
    projects 3D underwater points from view A into view B's image plane,
    finds nearest rendered pixel in B, and penalizes 3D position disagreement.

    :return: (loss_scalar, n_matched_pairs) tuple
    """
    torch.cuda.empty_cache()

    focal = float(K[0][0])

    # Pick 2 random training views
    pair = random.sample(list(i_train), 2)
    idx_a, idx_b = pair
    pose_a = poses[idx_a, :3, :4]  # (3, 4) c2w
    pose_b = poses[idx_b, :3, :4]

    mask_a_t = torch.tensor(masks[idx_a], dtype=torch.float32, device=device)
    mask_b_t = torch.tensor(masks[idx_b], dtype=torch.float32, device=device)
    water_coords_a = (mask_a_t > 0.5).nonzero(as_tuple=False)  # (N_wa, 2) [row, col]
    water_coords_b = (mask_b_t > 0.5).nonzero(as_tuple=False)
    del mask_a_t, mask_b_t  # free mask tensors

    if len(water_coords_a) < 10 or len(water_coords_b) < 10:
        return torch.tensor(0.0, device=device), 0

    # Subsample N_consist water pixels from each view
    n_a = min(N_consist, len(water_coords_a))
    n_b = min(N_consist, len(water_coords_b))
    perm_a = torch.randperm(len(water_coords_a), device=device)[:n_a]
    perm_b = torch.randperm(len(water_coords_b), device=device)[:n_b]
    coords_a = water_coords_a[perm_a]  # (n_a, 2) long
    coords_b = water_coords_b[perm_b]  # (n_b, 2) long
    del water_coords_a, water_coords_b

    # Build ray batches with 9-neighbor structure
    rays_o_a, rays_d_a = get_rays(H, W, K, pose_a)  # (H, W, 3) world-space
    batch_rays_a = build_consistency_rays(rays_o_a, rays_d_a, coords_a, H, W)
    del rays_o_a, rays_d_a  # free full-image ray maps

    rays_o_b, rays_d_b = get_rays(H, W, K, pose_b)
    batch_rays_b = build_consistency_rays(rays_o_b, rays_d_b, coords_b, H, W)
    del rays_o_b, rays_d_b

    wm_a = torch.ones(n_a, device=device)
    wm_b = torch.ones(n_b, device=device)

    # Render both batches through full pipeline
    consist_chunk = min(chunk, n_a * 9)
    rgb_a, _, _, Xs_a, depth_a, sray_a, extras_a = render(
        H,
        W,
        K,
        chunk=consist_chunk,
        rays=batch_rays_a,
        pixel_water_mask=wm_a,
        **render_kwargs,
    )
    del batch_rays_a, wm_a, rgb_a, extras_a  # free intermediates

    rgb_b, _, _, Xs_b, depth_b, sray_b, extras_b = render(
        H,
        W,
        K,
        chunk=consist_chunk,
        rays=batch_rays_b,
        pixel_water_mask=wm_b,
        **render_kwargs,
    )
    del batch_rays_b, wm_b, rgb_b, extras_b

    # Compute 3D underwater points in NDC: P_ndc = Xs + depth * sample_ray
    P_ndc_a = Xs_a + depth_a.unsqueeze(-1) * sray_a  # (n_a, 3)
    P_ndc_b = Xs_b + depth_b.unsqueeze(-1) * sray_b  # (n_b, 3)

    # Convert NDC to NeRF world coordinates
    model = render_kwargs["network_fn"]
    P_world_a = model.ndcp2camp(P_ndc_a, W=W, H=H, f=focal)  # (n_a, 3)
    P_world_b = model.ndcp2camp(P_ndc_b, W=W, H=H, f=focal)  # (n_b, 3)

    # Project A's 3D points into B's image plane
    R_b = pose_b[:3, :3]  # (3, 3)
    t_b = pose_b[:3, 3]  # (3,)
    # World-to-camera: P_cam = R^T @ (P - t) = (P - t) @ R
    P_cam_inB = (P_world_a - t_b.unsqueeze(0)) @ R_b  # (n_a, 3)
    # Pinhole projection (NeRF: camera looks in -Z)
    valid_front_ab = P_cam_inB[:, 2] < -1e-6
    u_b = focal * P_cam_inB[:, 0] / (-P_cam_inB[:, 2]) + W * 0.5
    # get_rays() uses Y_cam = -(j - cy)/fy, so inverse is j = -f*Y/(-Z) + H/2
    v_b = -focal * P_cam_inB[:, 1] / (-P_cam_inB[:, 2]) + H * 0.5
    valid_bounds_ab = (u_b >= 0) & (u_b < W) & (v_b >= 0) & (v_b < H)
    valid_ab = valid_front_ab & valid_bounds_ab

    # Project B's 3D points into A's image plane
    R_a = pose_a[:3, :3]
    t_a = pose_a[:3, 3]
    P_cam_inA = (P_world_b - t_a.unsqueeze(0)) @ R_a
    valid_front_ba = P_cam_inA[:, 2] < -1e-6
    u_a = focal * P_cam_inA[:, 0] / (-P_cam_inA[:, 2]) + W * 0.5
    # get_rays() uses Y_cam = -(j - cy)/fy, so inverse is j = -f*Y/(-Z) + H/2
    v_a = -focal * P_cam_inA[:, 1] / (-P_cam_inA[:, 2]) + H * 0.5
    valid_bounds_ba = (u_a >= 0) & (u_a < W) & (v_a >= 0) & (v_a < H)
    valid_ba = valid_front_ba & valid_bounds_ba

    # Debug logging
    n_front_ab = int(valid_front_ab.sum().item())
    n_bounds_ab = int(valid_bounds_ab.sum().item())
    n_valid_ab = int(valid_ab.sum().item())
    n_front_ba = int(valid_front_ba.sum().item())
    n_bounds_ba = int(valid_bounds_ba.sum().item())
    n_valid_ba = int(valid_ba.sum().item())
    tqdm.write(
        f"[CONSIST] views=({idx_a},{idx_b}) water_px=({n_a},{n_b}) "
        f"front_ab={n_front_ab} bounds_ab={n_bounds_ab} valid_ab={n_valid_ab} | "
        f"front_ba={n_front_ba} bounds_ba={n_bounds_ba} valid_ba={n_valid_ba}"
    )
    with torch.no_grad():
        tqdm.write(
            f"[CONSIST]   P_world_a: mean={P_world_a.mean(0).tolist()}, "
            f"std={P_world_a.std(0).tolist()}"
        )
        if n_valid_ab > 0:
            tqdm.write(
                f"[CONSIST]   proj A->B u: [{u_b[valid_front_ab].min():.1f}, {u_b[valid_front_ab].max():.1f}] "
                f"v: [{v_b[valid_front_ab].min():.1f}, {v_b[valid_front_ab].max():.1f}]"
            )

    # Match projected pixels to nearest rendered pixels, compute loss
    max_pixel_dist = 15.0
    loss = torch.tensor(0.0, device=device)
    n_pairs = 0

    # A -> B direction
    if valid_ab.sum() > 0:
        # Projected A pixels in B's image: (row, col)
        proj_ab = torch.stack([v_b[valid_ab], u_b[valid_ab]], dim=-1)  # (N_v, 2)
        coords_b_float = coords_b.float()  # (n_b, 2) [row, col]
        # Pairwise pixel distance
        dists_ab = torch.cdist(proj_ab, coords_b_float)  # (N_v, n_b)
        nn_dist_ab, nn_idx_ab = dists_ab.min(dim=1)  # (N_v,)
        close_ab = nn_dist_ab < max_pixel_dist
        tqdm.write(
            f"[CONSIST]   A->B nn_dist: min={nn_dist_ab.min():.1f} "
            f"median={nn_dist_ab.median():.1f} max={nn_dist_ab.max():.1f} "
            f"close={int(close_ab.sum().item())}"
        )

        if close_ab.sum() > 0:
            matched_P_b = P_world_b[nn_idx_ab[close_ab]]
            source_P_a = P_world_a[valid_ab][close_ab]
            loss = loss + torch.mean((source_P_a - matched_P_b) ** 2)
            n_pairs += int(close_ab.sum().item())

    # B -> A direction (symmetric)
    if valid_ba.sum() > 0:
        proj_ba = torch.stack([v_a[valid_ba], u_a[valid_ba]], dim=-1)
        coords_a_float = coords_a.float()
        dists_ba = torch.cdist(proj_ba, coords_a_float)
        nn_dist_ba, nn_idx_ba = dists_ba.min(dim=1)
        close_ba = nn_dist_ba < max_pixel_dist
        tqdm.write(
            f"[CONSIST]   B->A nn_dist: min={nn_dist_ba.min():.1f} "
            f"median={nn_dist_ba.median():.1f} max={nn_dist_ba.max():.1f} "
            f"close={int(close_ba.sum().item())}"
        )

        if close_ba.sum() > 0:
            matched_P_a = P_world_a[nn_idx_ba[close_ba]]
            source_P_b = P_world_b[valid_ba][close_ba]
            loss = loss + torch.mean((source_P_b - matched_P_a) ** 2)
            n_pairs += int(close_ba.sum().item())

    tqdm.write(f"[CONSIST]   => n_pairs={n_pairs}, loss={loss.item():.6f}")
    return loss, n_pairs


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches."""
    if chunk is None:
        return fn

    def ret(inputs, water_mask):
        outputs_flat = []
        Xs = []
        z_vals = []
        sample_ray = []
        dD_list = []

        for i in range(0, inputs.shape[0], chunk):
            wm_chunk = water_mask[i: i + chunk] if water_mask is not None else None
            outputs_flat_t, Xs_t, z_vals_t, sample_ray_t, dD_t = fn(
                inputs[i: i + chunk], wm_chunk
            )
            outputs_flat.append(outputs_flat_t)
            Xs.append(Xs_t)
            z_vals.append(z_vals_t)
            sample_ray.append(sample_ray_t)
            dD_list.append(dD_t)

        outputs_flat = torch.cat(outputs_flat, dim=0)
        Xs = torch.cat(Xs, dim=0)
        z_vals = torch.cat(z_vals, dim=0)
        sample_ray = torch.cat(sample_ray, dim=0)
        dD = torch.cat(dD_list, dim=0)

        return outputs_flat, Xs, z_vals, sample_ray, dD

    return ret


def run_network(rays_o, viewdirs, water_mask, fn, netchunk=1024, N_samples=None, N_importance=None):
    """Prepares inputs and applies network 'fn'."""
    embedded = torch.cat([viewdirs, rays_o], -1)

    embedded = torch.reshape(embedded, [-1, 9, 6])
    outputs_flat, Xs, z_vals, sample_ray, dD = batchify(fn, netchunk)(embedded, water_mask)
    outputs = torch.reshape(
        outputs_flat,
        [embedded.shape[0]] + [N_samples + N_importance] + [outputs_flat.shape[-1]],
    )

    return outputs, Xs, z_vals, sample_ray, dD


def batchify_rays(rays_flat, chunk=1024 * 32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM."""
    all_ret = {}

    # Extract water_mask from kwargs so we can slice it per-chunk
    water_mask = kwargs.pop("water_mask", None)

    for i in range(0, rays_flat.shape[0], chunk):
        wm_chunk = water_mask[i: i + chunk] if water_mask is not None else None
        ret = render_rays(rays_flat[i: i + chunk], water_mask=wm_chunk, **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(
    H,
    W,
    K,
    chunk=1024 * 32,
    rays=None,
    c2w=None,
    ndc=True,
    pixel_water_mask=None,
    **kwargs,
):
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)

        rays_od = torch.cat([rays_o.unsqueeze(0), rays_d.unsqueeze(0)], dim=0)
        rays_od = rays_od.permute([1, 2, 0, 3])
        rays_od = rays_od[:, :, None, :, :]

        # Vectorized neighbor gathering
        # Reminder: rays_od shape: [H, W, 1, 2, 3]
        H_img, W_img = rays_od.shape[0], rays_od.shape[1]

        # Squeeze to [H, W, 2, 3] and rearrange to [2, 3, H, W] for padding
        rays_squeezed = rays_od.squeeze(2)
        rays_permuted = rays_squeezed.permute(2, 3, 0, 1)

        # Pad with replicate mode
        rays_padded = F.pad(rays_permuted, (1, 1, 1, 1), mode="replicate")  # [2, 3, H+2, W+2]
        rays_padded = rays_padded.permute(2, 3, 0, 1)  # [H+2, W+2, 2, 3]

        # Gather center + 8 neighbors using vectorized slicing with original order
        neighbors = [
            rays_squeezed,  # Center
            rays_padded[0: H_img, 0: W_img, :, :],  # top-left
            rays_padded[0: H_img, 1: W_img + 1, :, :],  # top
            rays_padded[0: H_img, 2: W_img + 2, :, :],  # top-right
            rays_padded[1: H_img + 1, 0: W_img, :, :],  # left
            rays_padded[1: H_img + 1, 2: W_img + 2, :, :],  # right
            rays_padded[2: H_img + 2, 0: W_img, :, :],  # bottom-left
            rays_padded[2: H_img + 2, 1: W_img + 1, :, :],  # bottom
            rays_padded[2: H_img + 2, 2: W_img + 2, :, :],  # bottom-right
        ]

        rays_od = torch.stack(neighbors, dim=2) # [H, W, 9, 2, 3]

        rays_od = rays_od.view([-1, 2, 3])
        rays_o, rays_d = torch.split(rays_od, [1, 1], dim=-2)
        rays_o = rays_o.squeeze()
        rays_d = rays_d.squeeze()
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    rays_o = torch.reshape(rays_o, [-1, 9, 3])
    rays_d = torch.reshape(rays_d, [-1, 9, 3])

    # for forward facing scenes
    if ndc:
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1.0, rays_o, rays_d)

    rays = torch.cat([rays_o, rays_d], -1)

    # Build per-ray water_mask for batchify_rays
    if pixel_water_mask is not None:
        kwargs["water_mask"] = pixel_water_mask

    # Render
    all_ret = batchify_rays(rays, chunk, **kwargs)

    k_extract = ["rgb_map", "disp_map", "acc_map", "Xs", "depth_map", "sample_ray"]
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0, args=None, masks=None):

    H, W, focal = hwf

    if render_factor != 0:
        # Render downsampled for speed
        H = H // render_factor
        W = W // render_factor
        focal = focal / render_factor

    rgbs = []
    disps = []

    t = time.time()
    if gt_imgs is not None:
        lpips_model = util_of_lpips(net="vgg")

    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()

        # Build per-pixel water mask for this image
        pwm = None
        if masks is not None and i < len(masks):
            mask_i = masks[i]
            if mask_i.ndim == 3:
                mask_i = mask_i[:, :, 0]
            # Resize if render_factor changed H, W
            if mask_i.shape[0] != H or mask_i.shape[1] != W:
                from skimage.transform import resize

                mask_i = resize(mask_i, (H, W), order=0, preserve_range=True)
            pwm = torch.tensor(
                (mask_i.reshape(-1) > 0.5).astype(np.float32), device=c2w.device
            )

        rgb, disp, acc, Xs, depth_map, sample_ray, extras = render(
            H,
            W,
            K,
            chunk=chunk,
            c2w=c2w[:3, :4],
            pixel_water_mask=pwm,
            **render_kwargs,
        )
        depth_mode = extras["depth_mode"]
        if savedir is not None:
            np.save(
                os.path.join(savedir, f"{args.expname}{i:03d}_depth.npy"),
                depth_map.cpu().numpy().reshape(H, W).astype("float32"),
            )
            np.save(
                os.path.join(savedir, f"{args.expname}{i:03d}_acc.npy"),
                acc.cpu().numpy().reshape(H, W).astype("float32"),
            )
            np.save(
                os.path.join(savedir, f"{args.expname}{i:03d}_depth_mode.npy"),
                depth_mode.cpu().numpy().reshape(H, W).astype("float32"),
            )
        rgb = torch.reshape(rgb, [H, W, 3])
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i == 0:
            print(rgb.shape, disp.shape)

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, args.expname + "{:03d}.png".format(i))
            imageio.imwrite(filename, rgb8)

            # --- Save masked ---
            if masks is not None and i < len(masks):
                mask = masks[i]
                if mask.ndim == 3:
                    mask = mask[:, :, 0]  # Take first channel if multi-channel
                # Resize mask
                if mask.shape[0] != H or mask.shape[1] != W:
                    from skimage.transform import resize

                    mask = resize(mask, (H, W), order=0, preserve_range=True)

                # Create masked RGB
                mask_binary = (mask > 0.5).astype(np.float32)
                rgb_masked = rgb8.copy()
                rgb_masked[mask_binary < 0.5] = 0  # Black outside

                masked_filename = os.path.join(
                    savedir, args.expname + "{:03d}_masked.png".format(i)
                )
                imageio.imwrite(masked_filename, rgb_masked)

                # save with RGBA
                rgba = np.concatenate(
                    [rgb8, (mask_binary * 255).astype(np.uint8)[..., None]], axis=-1
                )
                rgba_filename = os.path.join(
                    savedir, args.expname + "{:03d}_rgba.png".format(i)
                )
                imageio.imwrite(rgba_filename, rgba)
            # --- END Save masked ---

            Xs_pts = Xs.cpu().numpy()
            ptsname = os.path.join(savedir, "Xs_pts{:03d}.npy".format(i))
            np.save(ptsname, Xs_pts)

            # --- Save camera pose ---
            posename = os.path.join(savedir, "pose_{:03d}.npy".format(i))
            np.save(posename, c2w[:3, :4].cpu().numpy())

            # --- Save rays ---
            ray_dir = sample_ray.cpu().numpy()
            np.save(os.path.join(savedir, "rays_{:03d}.npy".format(i)), ray_dir)

            if gt_imgs is not None:
                gt_img = to8b(gt_imgs[i].cpu().numpy())
                out_img = rgb8

                lpipsout = lpips_model.calc_lpips(gt_img, out_img)
                lpipsout = lpipsout[0][0][0][0].detach().cpu().numpy()
                
                psnrout = psnr(gt_img, out_img)

                # --- SSIM is broken --- //TODO
                try:
                    ssimout = ssim(gt_img, out_img)
                except Exception as e:
                    print(f"[warn] SSIM disabled this step: {e}")
                    ssimout = 0.0
                
                mse = 10 ** (-psnrout / 10)
                average = mse * np.sqrt(1 - ssimout) * lpipsout
                average = math.pow(average, 1.0 / 3)

                dic = {
                    "PSNR": psnrout,
                    "SSIM": ssimout,
                    "LPIPS": lpipsout,
                    "average": average,
                }
                df = pd.DataFrame([dic])
                df.to_csv(
                    os.path.join(savedir, "result.txt"),
                    sep="\t",
                    index=False,
                    float_format="%.3f",
                )

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def create_nerf(args):
    """Instantiate NeRF's MLP model."""
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    print("input_ch:", args.N_importance)
    model = NeRF.get_by_name(
        args.nerf_type,
        D=args.netdepth,
        W=args.netwidth,
        input_ch=input_ch,
        output_ch=output_ch,
        skips=skips,
        input_ch_views=input_ch_views,
        use_viewdirs=args.use_viewdirs,
        embed_fn=embed_fn,
        embeddirs_fn=embeddirs_fn,
        modelargs=args,
    ).to(device)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        # fine sampling is used in class NeRFrac forward and not here
        # never called, but its parameters are in every checkpoint's optimizer state
        model_fine = NeRF.get_by_name(
            args.nerf_type,
            D=args.netdepth_fine,
            W=args.netwidth_fine,
            input_ch=input_ch,
            output_ch=output_ch,
            skips=skips,
            input_ch_views=input_ch_views,
            use_viewdirs=args.use_viewdirs,
            embed_fn=embed_fn,
            embeddirs_fn=embeddirs_fn,
            modelargs=args,
        ).to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = (
        lambda rays_o,
        viewdirs,
        water_mask,
        network_fn,
        N_samples,
        N_importance: run_network(
            rays_o,
            viewdirs,
            water_mask,
            network_fn,
            netchunk=args.netchunk,
            N_samples=N_samples,
            N_importance=N_importance,
        )
    )

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path != "None":
        ckpts = [args.ft_path]
    else:
        ckpts = [
            os.path.join(basedir, expname, f)
            for f in sorted(os.listdir(os.path.join(basedir, expname)))
            if "tar" in f
        ]

    print("Found ckpts", ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print("Reloading from", ckpt_path)
        ckpt = torch.load(ckpt_path)
        start = ckpt["global_step"]
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        fn_state = {
            k.replace("_orig_mod.", ""): v
            for k, v in ckpt["network_fn_state_dict"].items()
        }
        model.load_state_dict(fn_state)
    ##########################

    render_kwargs_train = {
        "network_query_fn": network_query_fn,
        "water": args.water,
        "N_importance": args.N_importance,
        "N_samples": args.N_samples,
        "network_fn": model,
        "white_bkgd": args.white_bkgd,
        "raw_noise_std": args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.no_ndc:
        print("Not ndc!")
        render_kwargs_train["ndc"] = False

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test["raw_noise_std"] = 0.0

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False):
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.0 - torch.exp(-act_fn(raw) * dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    infinity = torch.tensor([1e10], device=dists.device).expand(dists[..., :1].shape)
    dists = torch.cat([dists, infinity], -1)

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
    rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
    noise = 0.0
    if raw_noise_std > 0.0:
        noise = torch.randn(raw[..., 3].shape, device=raw.device) * raw_noise_std

    alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    ones_col = torch.ones((alpha.shape[0], 1), device=alpha.device)
    weights = (
        alpha
        * torch.cumprod(torch.cat([ones_col, 1.0 - alpha + 1e-10], -1), -1)[:, :-1]
    )
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    depth_mode = z_vals[torch.arange(len(weights)), torch.argmax(weights, dim=-1)]
    disp_map = 1.0 / torch.max(
        1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1)
    )
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map, depth_mode

def distortion_loss(weights, z_vals):
    t_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
    delta = z_vals[..., 1:] - z_vals[..., :-1]
    w = weights[..., :-1]  # (N_rays, N_samples-1)
    w_cumsum = torch.cumsum(w, dim=-1)       # (N_rays, N_samples-1)
    wt_cumsum = torch.cumsum(w * t_mid, dim=-1)
    loss_uni = 2.0 * torch.sum(
        w * t_mid * w_cumsum - w * wt_cumsum, dim=-1
    )
    loss_bi = (1.0 / 3.0) * torch.sum(w ** 2 * delta, dim=-1)
    return torch.mean(loss_uni + loss_bi)

def render_rays(
    ray_batch,
    network_fn,
    network_query_fn,
    water,
    N_samples,
    N_importance,
    retraw=False,
    white_bkgd=False,
    raw_noise_std=0.0,
    water_mask=None,
):
    rays_o, rays_d = ray_batch[:, :, 0:3], ray_batch[:, :, 3:6]
    rays_o = torch.reshape(rays_o, [-1, 3])
    rays_d = torch.reshape(rays_d, [-1, 3])

    viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

    # Build per-ray water mask: use per-ray mask if provided, else fall back to scene-level flag
    if water_mask is not None:
        wm = water_mask
    elif water == 1:
        wm = torch.ones(ray_batch.shape[0], device=ray_batch.device)
    else:
        wm = torch.zeros(ray_batch.shape[0], device=ray_batch.device)

    raw, Xs, z_vals, sample_ray, dD = network_query_fn(
        rays_o, viewdirs, wm, network_fn, N_samples, N_importance
    )
    rgb_map, disp_map, acc_map, weights, depth_map, depth_mode = raw2outputs(
        raw, z_vals, sample_ray, raw_noise_std, white_bkgd
    )

    ret = {
        "rgb_map": rgb_map,
        "disp_map": disp_map,
        "acc_map": acc_map,
        "Xs": Xs,
        "sample_ray": sample_ray,
        "dD": dD,
    }
    ret["depth_map"] = depth_map
    ret["depth_mode"] = depth_mode  # NEW
    ret["water_mask"] = wm
    if retraw:
        ret["raw"] = raw
    ret["weights"] = weights
    ret["z_vals"] = z_vals

    return ret


def config_parser():
    import configargparse

    parser = configargparse.ArgumentParser()
    parser.add_argument("--config", is_config_file=True, help="config file path")
    parser.add_argument("--expname", type=str, help="experiment name")
    parser.add_argument(
        "--basedir", type=str, default="./logs/", help="where to store ckpts and logs"
    )
    parser.add_argument(
        "--datadir", type=str, default="./data/llff/fern", help="input data directory"
    )

    # training options
    parser.add_argument(
        "--nerf_type",
        type=str,
        default="original",
        help="nerf network type, original or nerof",
    )
    parser.add_argument(
        "--N_iter", type=int, default=200000, help="num training iterations"
    )
    parser.add_argument("--netdepth", type=int, default=8, help="layers in network")
    parser.add_argument("--netwidth", type=int, default=256, help="channels per layer")
    parser.add_argument(
        "--netdepth_fine", type=int, default=8, help="layers in fine network"
    )
    parser.add_argument(
        "--netwidth_fine",
        type=int,
        default=256,
        help="channels per layer in fine network",
    )
    parser.add_argument(
        "--N_rand",
        type=int,
        default=32 * 32 * 4,
        help="batch size (number of random rays per gradient step)",
    )
    parser.add_argument("--lrate", type=float, default=5e-4, help="learning rate")
    parser.add_argument(
        "--lrate_decay",
        type=int,
        default=250,
        help="exponential learning rate decay (in 1000 steps)",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=1024 * 32,
        help="number of rays processed in parallel, decrease if running out of memory",
    )
    parser.add_argument(
        "--netchunk",
        type=int,
        default=1024 * 2,
        help="number of pts sent through network in parallel, decrease if running out of memory",
    )
    parser.add_argument(
        "--no_batching",
        action="store_true",
        help="only take random rays from 1 image at a time",
    )
    parser.add_argument(
        "--no_reload", action="store_true", help="do not reload weights from saved ckpt"
    )
    parser.add_argument(
        "--ft_path",
        type=str,
        default=None,
        help="specific weights npy file to reload for coarse network",
    )

    # rendering options
    parser.add_argument(
        "--N_samples", type=int, default=64, help="number of coarse samples per ray"
    )
    parser.add_argument(
        "--N_importance",
        type=int,
        default=64,
        help="number of additional fine samples per ray",
    )
    parser.add_argument(
        "--use_viewdirs", action="store_true", help="use full 5D input instead of 3D"
    )
    parser.add_argument(
        "--i_embed",
        type=int,
        default=0,
        help="set 0 for default positional encoding, -1 for none",
    )
    parser.add_argument(
        "--multires",
        type=int,
        default=10,
        help="log2 of max freq for positional encoding (3D location)",
    )
    parser.add_argument(
        "--multires_views",
        type=int,
        default=4,
        help="log2 of max freq for positional encoding (2D direction)",
    )
    parser.add_argument(
        "--raw_noise_std",
        type=float,
        default=0.0,
        help="std dev of noise added to regularize sigma_a output, 1e0 recommended",
    )

    parser.add_argument(
        "--render_only",
        action="store_true",
        help="do not optimize, reload weights and render out render_poses path",
    )
    parser.add_argument(
        "--render_test",
        action="store_true",
        help="render the test set instead of render_poses path",
    )
    parser.add_argument(
        "--render_factor",
        type=int,
        default=0,
        help="downsampling factor to speed up rendering, set 4 or 8 for fast preview",
    )
    parser.add_argument(
        "--render_train",
        action="store_true",
        help="Render training views instead of test views",
    )
    parser.add_argument("--render_all", action="store_true", help="Render all views")

    # training options
    parser.add_argument(
        "--precrop_iters",
        type=int,
        default=0,
        help="number of steps to train on central crops",
    )
    parser.add_argument(
        "--precrop_frac",
        type=float,
        default=0.5,
        help="fraction of img taken for central crops",
    )

    # dataset options
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="llff",
        choices=["llff"],
        help="only LLFF-format scenes are supported",
    )
    parser.add_argument(
        "--use_mask_selection",
        action="store_true",
        help="Use water masks for selective ray sampling during training",
    )
    parser.add_argument(
        "--white_bkgd",
        action="store_true",
        help="composite rendered colors onto a white background",
    )

    # llff flags
    parser.add_argument(
        "--factor", type=int, default=8, help="downsample factor for LLFF images"
    )
    parser.add_argument(
        "--no_ndc",
        action="store_true",
        help="do not use normalized device coordinates (set for non-forward facing scenes)",
    )
    parser.add_argument(
        "--spherify", action="store_true", help="set for spherical 360 scenes"
    )
    parser.add_argument(
        "--llffhold",
        type=int,
        default=8,
        help="will take every 1/N images as LLFF test set, paper uses 8",
    )

    # logging/saving options
    parser.add_argument(
        "--i_print",
        type=int,
        default=1000,
        help="frequency of console printout and metric loggin",
    )
    parser.add_argument(
        "--i_weights", type=int, default=10000, help="frequency of weight ckpt saving"
    )
    parser.add_argument(
        "--i_testset", type=int, default=10000, help="frequency of testset saving"
    )
    parser.add_argument(
        "--i_video",
        type=int,
        default=400000,
        help="frequency of render_poses video saving",
    )

    parser.add_argument("--yita", type=float, default=1.0, help="refractive index")

    parser.add_argument("--water", type=int, default=1, help="with water surface")
    parser.add_argument("--initdepth", type=float, default=-0.6, help="initial depth")
    parser.add_argument(
        "--tframe", type=str, default=4, help="test frame, 4 is the center camera"
    )

    parser.add_argument(
        "--omultires",
        type=int,
        default=0,
        help="log2 of max freq for positional encoding (refractive field center)",
    )
    parser.add_argument(
        "--refracvmultires",
        type=int,
        default=0,
        help="log2 of max freq for positional encoding (refractive field dir)",
    )

    # --- Collapse prevention ---
    parser.add_argument(
        "--smooth_weight",
        type=float,
        default=0.01,
        help="weight for smoothness loss on refractive field (0 to disable)",
    )
    parser.add_argument(
        "--smooth_warmup",
        type=int,
        default=10000,
        help="number of iterations to linearly ramp smooth loss weight from 0",
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
        help="max gradient norm for clipping (0 to disable)",
    )

    # --- Multi-view depth consistency loss ---
    parser.add_argument(
        "--consistency_weight",
        type=float,
        default=0.0,
        help="weight for multi-view depth consistency loss (0 to disable)",
    )
    parser.add_argument(
        "--consistency_warmup",
        type=int,
        default=20000,
        help="iterations to linearly ramp consistency loss from 0",
    )
    parser.add_argument(
        "--consistency_every",
        type=int,
        default=50,
        help="compute consistency loss every N training iterations",
    )
    parser.add_argument(
        "--N_consist",
        type=int,
        default=256,
        help="number of water pixels to sample per view for consistency loss",
    )
    # --- Distortion loss (Mip-NeRF 360) ---
    parser.add_argument(
        "--distortion_weight",
        type=float,
        default=0.0,
        help="weight for distortion loss (0 to disable). Try 0.01-0.1",
    )
    parser.add_argument(
        "--distortion_warmup",
        type=int,
        default=10000,
        help="iterations to linearly ramp distortion loss from 0",
    )
    parser.add_argument(
        "--flatness_weight",
        type=float,
        default=0.0,
        help="weight for water surface flatness loss (0 to disable). Penalizes variance of Xs Z across water rays in each batch.",
    )
    parser.add_argument(
        "--flatness_warmup",
        type=int,
        default=10000,
        help="iterations to linearly ramp flatness loss from 0",
    )

    return parser


def train():
    set_seed(0)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")  # TF32 for Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    parser = config_parser()
    args = parser.parse_args()
    args.expname = (
        args.expname
        + "_"
        + args.nerf_type
        + "_d"
        + str(args.initdepth)
        + "_yita"
        + str(args.yita)
        + "_tframe"
        + str(args.tframe)
        + "_factor"
        + str(args.factor)
        + "multires"
        + str(args.omultires)
        + "_"
        + str(args.refracvmultires)
        + "_"
        + str(args.N_samples)
    )

    if not os.path.exists("./tensorboard/" + args.expname):
        os.makedirs("./tensorboard/" + args.expname)
    writer = SummaryWriter("./tensorboard/" + args.expname)

    # Load data
    images, masks, poses, _, render_poses, i_test = load_llff_data(
        args.datadir,
        args.factor,
        recenter=True,
        bd_factor=0.75,
        spherify=args.spherify,
        load_masks=True,
    )
    use_masks = masks is not None
    if use_masks:
        print(f"Using water masks for ray selection, shape: {masks.shape}")
        masks = np.transpose(masks, [2, 0, 1])  # [N, H, W]
    else:
        print("No masks found, training without mask-based ray selection")

    hwf = poses[0, :3, -1]
    poses = poses[:, :3, :4]
    print("Loaded llff", images.shape, render_poses.shape, hwf, args.datadir)
    if not isinstance(i_test, list):
        i_test = [i_test]

    if args.llffhold > 0:
        print("Auto LLFF holdout,", args.llffhold)
        i_test = np.arange(images.shape[0])[:: args.llffhold]
    i_test = [int(x) for x in args.tframe.split(",")]
    i_val = i_test
    i_train = np.array(
        [
            i
            for i in np.arange(int(images.shape[0]))
            if (i not in i_test and i not in i_val)
        ]
    )

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    print(H, W)
    hwf = [H, W, focal]

    # 定义内参矩阵
    K = np.array([[focal, 0, 0.5 * W], [0, focal, 0.5 * H], [0, 0, 1]])
    print("K:", K)
    args.H = H
    args.W = W
    args.f = focal
    if args.render_test:
        render_poses = np.array(poses[i_test])
    elif args.render_train:
        render_poses = np.array(poses[i_train])
        print(f"Rendering {len(i_train)} training views")
    elif args.render_all:
        i_comp = np.concatenate([i_train, i_test])
        i_comp.sort()
        render_poses = np.array(poses[i_comp])
        print(f"Rendering {len(i_comp)} training and test views")

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname

    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, "args.txt")

    with open(f, "w") as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write("{} = {}\n".format(arg, attr))
    if args.config is not None:
        dest_config = os.path.join(basedir, expname, "config.txt")
        src_config = os.path.abspath(args.config)
        if os.path.abspath(dest_config) != src_config:
            with open(dest_config, "w") as file:
                with open(args.config, "r") as config_file:
                    file.write(config_file.read())

    # Save coordinate system metadata for post-processing scripts
    # (c2w.npy was overwritten each run by recenter_poses)
    import shutil

    exp_dir = os.path.join(basedir, expname)
    c2w_src = os.path.join(os.getcwd(), "c2w.npy")
    c2w_dst = os.path.join(exp_dir, "c2w.npy")
    if os.path.exists(c2w_src):
        shutil.copy2(c2w_src, c2w_dst)
    pb = np.load(os.path.join(args.datadir, "poses_bounds.npy"))
    bds_raw = pb[:, -2:]
    sc = 1.0 / (bds_raw.min() * 0.75)
    np.save(os.path.join(exp_dir, "llff_scale.npy"), np.array([sc]))
    print(f"Saved c2w.npy and llff_scale.npy (sc={sc:.8f}) to {exp_dir}")

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(
        args
    )
    if hasattr(torch, "compile"):
        model = render_kwargs_train["network_fn"]
        if hasattr(model, "_occ"):
            model._occ = torch.compile(model._occ, mode="default")
            model._occfine = torch.compile(model._occfine, mode="default")
    global_step = start

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print("RENDER ONLY")
        torch.cuda.empty_cache()
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                render_idx = i_test
                images = images[i_test]
                render_masks = masks[i_test] if use_masks else None
            elif args.render_train:
                render_idx = i_train
                images = images[i_train]
                render_masks = masks[i_train] if use_masks else None
            elif args.render_all:
                render_idx = i_comp
                images = images[i_comp]
                render_masks = masks[i_comp] if use_masks else None

            else:
                # Default is smoother render_poses path
                render_idx = i_train
                images = images[i_train]
                render_masks = masks[i_train] if use_masks else None

            testsavedir = os.path.join(
                basedir,
                expname,
                "renderonly_{}_{:06d}".format(
                    "test" if args.render_test else "path", start
                ),
            )
            os.makedirs(testsavedir, exist_ok=True)
            print("test poses shape", render_poses.shape)
            rgbs, _ = render_path(
                render_poses,
                hwf,
                K,
                args.chunk,
                render_kwargs_test,
                savedir=testsavedir,
                render_factor=args.render_factor,
                args=args,
                masks=render_masks,
            )
            print("Done rendering", testsavedir)
            try:
                imageio.mimwrite(
                    os.path.join(testsavedir, "video.mp4"),
                    to8b(rgbs),
                    fps=30,
                    quality=8,
                )
            except Exception as e:
                print("[WARN] Could not write video with imageio/ffmpeg backend:", e)

            print(f"Generating error maps for {len(rgbs)} frames...")
            for k in range(len(rgbs)):
                pred = torch.tensor(rgbs[k])
                gt = torch.tensor(images[k])
                err_map = torch.sum((pred - gt) ** 2, dim=-1).cpu().numpy()
                err_map_colored = (colorize_np(err_map, range=(0.0, 1.0)) * 255).astype(
                    np.uint8
                )
                imageio.imwrite(
                    os.path.join(testsavedir, "{:03d}_err.png".format(k)),
                    err_map_colored,
                )
                if use_masks and args.use_mask_selection:
                    mask = masks[render_idx[k]]
                    masked_err = err_map * mask
                    err_map_masked_colored = (
                        colorize_np(masked_err, range=(0.0, 1.0)) * 255
                    ).astype(np.uint8)
                    imageio.imwrite(
                        os.path.join(testsavedir, "{:03d}_err_masked.png".format(k)),
                        err_map_masked_colored,
                    )

            return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        print("get rays")
        rays = np.stack(
            [get_rays_np(H, W, K, p) for p in poses[:, :3, :4]], 0
        )  # [N, ro+rd, H, W, 3]
        print("done, concats")
        rays_rgb = np.concatenate([rays, images[:, None]], 1)  # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])  # [N, H, W, ro+rd+rgb, 3]
        # Pre-compute neighbor arrangement for ALL training pixels
        print("Arranging neighbor rays for all training pixels...")
        filtered_rays_list = []
        water_flags_list = []
        for n in range(len(rays_rgb)):
            if n not in i_train:
                continue
            img_rays = rays_rgb[n]  # [H, W, 3, 3]
            mask_n = masks[n] if use_masks else np.ones((H, W))
            padded = np.pad(img_rays, ((1, 1), (1, 1), (0, 0), (0, 0)), mode='edge')
            # All 9 neighbors via slicing — shape (H, W, 9, 3, 3)
            neighbors = np.stack([
                padded[1:H + 1, 1:W + 1],  # 0: center
                padded[0:H, 0:W],  # 1: top-left
                padded[0:H, 1:W + 1],  # 2: top
                padded[0:H, 2:W + 2],  # 3: top-right
                padded[1:H + 1, 0:W],  # 4: left
                padded[1:H + 1, 2:W + 2],  # 5: right
                padded[2:H + 2, 0:W],  # 6: bottom-left
                padded[2:H + 2, 1:W + 1],  # 7: bottom
                padded[2:H + 2, 2:W + 2],  # 8: bottom-right
            ], axis=2)
            filtered_rays_list.append(neighbors.reshape(-1, 9, 3, 3))
            water_flags_list.append((mask_n.reshape(-1) > 0.5).astype(np.float32))
        rays_rgb = np.concatenate(filtered_rays_list, axis=0)  # [N_total, 9, 3, 3]
        rays_water_flags = np.concatenate(water_flags_list, axis=0)  # [N_total]
        n_water = int(np.sum(rays_water_flags > 0.5))
        n_nonwater = len(rays_water_flags) - n_water
        print(
            f"Total training rays: {len(rays_rgb)} (water: {n_water}, non-water: {n_nonwater})"
        )

        rays_rgb = rays_rgb.astype(np.float32)
        print("shuffle rays")
        # Shuffle both arrays in sync
        perm = np.random.permutation(len(rays_rgb))
        rays_rgb = rays_rgb[perm]
        rays_water_flags = rays_water_flags[perm]
        print("done")
        i_batch = 0
        
    # Move training data to GPU
    images = torch.Tensor(images).pin_memory().to(device, non_blocking=True)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb)
        rays_water_flags = (
            torch.Tensor(rays_water_flags)
        )
    poses = torch.Tensor(poses).pin_memory().to(device, non_blocking=True)

    N_iters = args.N_iter + 1
    print("Begin")
    print("TRAIN views are", i_train)
    print("TEST views are", i_test)
    print("VAL views are", i_val)

    start = start + 1
    for i in trange(start, N_iters):
        time0 = time.time()
        batch_wm = None  # per-ray water mask

        # Sample random ray batch
        if use_batching:
            # Random over all images
            batch = rays_rgb[i_batch : i_batch + N_rand].to(device)  # [B, 9, 2+1, 3*?]
            batch_wm = rays_water_flags[i_batch: i_batch + N_rand].to(device)  # [B]
            batch = torch.reshape(batch, [-1, 3, 3])
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                rays_water_flags = rays_water_flags[rand_idx]
                i_batch = 0

        else:
            # Random from one image
            img_i = np.random.choice(i_train)
            target = images[img_i]
            target = torch.Tensor(target).to(device)
            pose = poses[img_i, :3, :4]

            if N_rand is not None:
                rays_o, rays_d = get_rays(
                    H, W, K, torch.Tensor(pose)
                )  # (H, W, 3), (H, W, 3)

                if i < args.precrop_iters:
                    dH = int(H // 2 * args.precrop_frac)
                    dW = int(W // 2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H // 2 - dH, H // 2 + dH - 1, 2 * dH),
                            torch.linspace(W // 2 - dW, W // 2 + dW - 1, 2 * dW),
                        ),
                        -1,
                    )
                    if i == start:
                        print(
                            f"[Config] Center cropping of size {2 * dH} x {2 * dW} is enabled until iter {args.precrop_iters}"
                        )
                else:
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W)
                        ),
                        -1,
                    )  # (H, W, 2)

                coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)

                # --- Mask-based ray selection ---
                if use_masks and args.use_mask_selection:
                    current_mask = masks[img_i]  # (H, W)
                    mask_flat = current_mask.reshape(-1)  # (H * W,)
                    water_indices = np.where(mask_flat > 0.5)[0]
                    non_water_indices = np.where(mask_flat <= 0.5)[0]

                    # Sample from both water and non-water pixels
                    n_water_avail = len(water_indices)
                    n_nonwater_avail = len(non_water_indices)
                    # Use all water pixels up to N_rand, fill rest with non-water
                    n_water_sample = min(n_water_avail, N_rand)
                    n_nonwater_sample = N_rand - n_water_sample

                    w_inds = np.random.choice(
                        water_indices, size=n_water_sample, replace=False
                    )
                    if n_nonwater_sample > 0 and n_nonwater_avail > 0:
                        nw_inds = np.random.choice(
                            non_water_indices,
                            size=min(n_nonwater_sample, n_nonwater_avail),
                            replace=False,
                        )
                        select_inds = np.concatenate([w_inds, nw_inds])
                    else:
                        select_inds = w_inds

                    np.random.shuffle(select_inds)
                    # Build per-ray water mask for selected pixels
                    batch_wm = torch.tensor(
                        (mask_flat[select_inds] > 0.5).astype(np.float32),
                        device=device,
                    )
                else:
                    select_inds = np.random.choice(
                        coords.shape[0], size=[N_rand], replace=False
                    )
                    # No mask info: fall back to scene-level water flag (backup)
                    batch_wm = None
                # --- END Mask-based ray selection ---

                select_coords = coords[select_inds].long()  # (N_rand, 2)

                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[
                    select_coords[:, 0], select_coords[:, 1]
                ]  # (N_rand, 3)

        #####  Core optimization loop  #####
        # batch_wm is set by both use_batching and no_batching paths;
        # it may be None when no masks are available.
        rgb, disp, acc, Xs, _, _, extras = render(
            H,
            W,
            K,
            chunk=args.chunk,
            rays=batch_rays,
            retraw=True,
            pixel_water_mask=batch_wm,
            **render_kwargs_train,
        )

        optimizer.zero_grad(set_to_none=True)

        target_s = target_s.view([-1, 9, 3])
        target_s = target_s[:, 0, :].squeeze()

        img_loss = img2mse(rgb, target_s)

        trans = extras["raw"][..., -1]

        loss = img_loss
        model = render_kwargs_train["network_fn"]
        dD_out = extras["dD"]
        dD_reshaped = dD_out.view(-1, 9)
        smooth_loss = torch.tensor(0.0, device=device)
        if args.water == 1 and args.smooth_weight > 0:
            # Only compute smoothness loss on water rays
            wm_out = extras.get("water_mask", None)
            if wm_out is not None:
                water_ray_mask = wm_out > 0.5  # (N_rays,)
                if water_ray_mask.any():
                    dD_water = dD_reshaped[water_ray_mask]
                    smooth_loss = torch.mean(model.smooth(dD_water))
            else:
                # Fallback: all rays treated as water (backup)
                smooth_loss = torch.mean(model.smooth(dD_reshaped))

        effective_smooth_weight = args.smooth_weight
        if args.smooth_warmup > 0 and global_step < args.smooth_warmup:
            effective_smooth_weight = args.smooth_weight * (
                global_step / args.smooth_warmup
            )

        loss = img_loss + effective_smooth_weight * smooth_loss

        # Flatness loss
        flat_loss = torch.tensor(0.0, device=device)
        effective_flat_weight = 0.0
        if args.flatness_weight > 0 and Xs is not None:
            wm_out = extras.get("water_mask", None)
            if wm_out is not None:
                water_mask = wm_out > 0.5
                if water_mask.sum() > 1:
                    Xs_water_z = Xs[water_mask, 2]
                    flat_loss = torch.var(Xs_water_z)
            elif Xs.shape[0] > 1:
                flat_loss = torch.var(Xs[:, 2])
            if args.flatness_warmup > 0 and global_step < args.flatness_warmup:
                effective_flat_weight = args.flatness_weight * (global_step / args.flatness_warmup)
            else:
                effective_flat_weight = args.flatness_weight
            loss = loss + effective_flat_weight * flat_loss

        # Distortion loss (Mip-NeRF 360)
        dist_loss = torch.tensor(0.0, device=device)
        effective_dist_weight = 0.0
        if args.distortion_weight > 0:
            if global_step < args.distortion_warmup:
                effective_dist_weight = args.distortion_weight * (
                        global_step / args.distortion_warmup
                )
            else:
                effective_dist_weight = args.distortion_weight
            dist_loss = distortion_loss(extras["weights"], extras["z_vals"])
            loss = loss + effective_dist_weight * dist_loss

        psnr = mse2psnr(img_loss)

        # Coarse network loss
        if "rgb0" in extras:
            img_loss0 = img2mse(extras["rgb0"], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        # --- Multi-view depth consistency loss ---
        consistency_loss = torch.tensor(0.0, device=device)
        n_consist_pairs = 0
        effective_consist_weight = 0.0
        if (
            args.consistency_weight > 0
            and args.water == 1
            and use_masks
            and i % args.consistency_every == 0
            and i > 0
        ):
            if i < args.consistency_warmup:
                effective_consist_weight = args.consistency_weight * (
                    i / args.consistency_warmup
                )
            else:
                effective_consist_weight = args.consistency_weight

            try:
                consistency_loss, n_consist_pairs = compute_consistency_loss(
                    poses=poses,
                    masks=masks,
                    H=H,
                    W=W,
                    K=K,
                    i_train=i_train,
                    render_kwargs=render_kwargs_train,
                    N_consist=args.N_consist,
                    chunk=args.chunk,
                    device=device,
                )
                loss = loss + effective_consist_weight * consistency_loss
            except Exception as e:
                tqdm.write(f"[WARN] Consistency loss failed at iter {i}: {e}")

        # --- TensorBoard logging ---
        writer.add_scalar("loss_smooth", smooth_loss, global_step=i)
        writer.add_scalar("psnr", psnr, global_step=i, walltime=None)
        writer.add_scalar("loss", loss, global_step=i, walltime=None)
        writer.add_scalar(
            "smooth_weight_effective", effective_smooth_weight, global_step=i
        )
        if args.water == 1:
            with torch.no_grad():
                dD_abs = dD_out.abs()
                writer.add_scalar("dD/mean", dD_abs.mean().item(), global_step=i)
                writer.add_scalar("dD/max", dD_abs.max().item(), global_step=i)
                writer.add_scalar("dD/std", dD_out.std().item(), global_step=i)
        if args.consistency_weight > 0:
            writer.add_scalar(
                "loss/consistency", consistency_loss.item(), global_step=i
            )
            writer.add_scalar("consistency/n_pairs", n_consist_pairs, global_step=i)
            writer.add_scalar(
                "consistency/effective_weight",
                effective_consist_weight,
                global_step=i,
            )
        if args.distortion_weight > 0:
            writer.add_scalar("loss/distortion", dist_loss.item(), global_step=i)
            writer.add_scalar(
                "distortion/effective_weight",
                effective_dist_weight,
                global_step=i,
            )
        if args.flatness_weight > 0:
            writer.add_scalar("loss/flatness", flat_loss.item(), global_step=i)
            writer.add_scalar("flatness/effective_weight", effective_flat_weight, global_step=i)

        if torch.isnan(loss) or torch.isinf(loss):
            print(
                f"[WARN] Iter {i}: NaN/Inf loss detected ({loss.item():.4f}), skipping batch"
            )
            optimizer.zero_grad(set_to_none=True)
        else:
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(grad_vars, max_norm=args.grad_clip)
            optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        writer.add_scalar("lrate", new_lrate, global_step=i, walltime=None)
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lrate
        ################################

        dt = time.time() - time0

        # Rest is logging
        if i % args.i_weights == 0:
            path = os.path.join(basedir, expname, "{:06d}.tar".format(i))
            save_state = render_kwargs_train["network_fn"].state_dict()
            save_state = {k.replace("_orig_mod.", ""): v for k, v in save_state.items()}
            torch.save(
                {
                    "global_step": global_step,
                    "network_fn_state_dict": save_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                path,
            )
            print("Saved checkpoints at", path)

        if i % args.i_video == 0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_path(
                    render_poses, hwf, K, args.chunk, render_kwargs_test, args=args
                )
            print("Done, saving", rgbs.shape, disps.shape)
            moviebase = os.path.join(
                basedir, expname, "{}_spiral_{:06d}_".format(expname, i)
            )
            imageio.mimwrite(moviebase + "rgb.mp4", to8b(rgbs), fps=30, quality=8)

        if i % args.i_testset == 0 and i > 0:
            testsavedir = os.path.join(basedir, expname, "testset_{:06d}".format(i))
            os.makedirs(testsavedir, exist_ok=True)
            test_masks = masks[i_test] if use_masks else None
            print("test poses shape", poses[i_test].shape)
            with torch.no_grad():
                render_path(
                    torch.Tensor(poses[i_test]).to(device),
                    hwf,
                    K,
                    args.chunk,
                    render_kwargs_test,
                    gt_imgs=images[i_test],
                    savedir=testsavedir,
                    args=args,
                    masks=test_masks,
                )
            print("Saved test set")

        if i % args.i_print == 0:
            dD_info = ""
            if args.water == 1:
                dD_info = f" dD_mean: {dD_out.abs().mean().item():.6f} dD_max: {dD_out.abs().max().item():.4f}"
            smooth_val = (
                smooth_loss.item()
                if isinstance(smooth_loss, torch.Tensor)
                else smooth_loss
            )
            consist_info = ""
            if args.consistency_weight > 0:
                consist_val = (
                    consistency_loss.item()
                    if isinstance(consistency_loss, torch.Tensor)
                    else consistency_loss
                )
                consist_info = f" Consist: {consist_val:.6f} ({n_consist_pairs}p)"
            tqdm.write(
                f"[TRAIN] Iter: {i} Loss: {loss.item():.6f} PSNR: {psnr.item():.3f} "
                f"Smooth: {smooth_val:.6f}{dD_info}{consist_info}"
            )

        global_step += 1


if __name__ == "__main__":
    train()
