"""
Train NeRFrac on an LLFF scene with water masks, or render a trained model.

Run from models/nerfrac_modified/: basedir, datadir and tensorboard/ are relative to
the working directory.

Usage examples:
    # Train
    python run_nerfrac.py --config configs/pielach_oblique.txt

    # Render every view into logs/<experiment>/renderonly_all_<iteration>/
    python run_nerfrac.py --config configs/pielach_oblique.txt \\
        --render_only --render_all

    # Render the test views with a specific checkpoint
    python run_nerfrac.py --config configs/pielach_oblique.txt \\
        --render_only --render_test --ft_path logs/<experiment>/950000.tar
"""

import math
import os
import numpy as np
import imageio
import random
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter
from utils import *
import pandas as pd
from run_nerfrac_helpers import *

from load_llff import load_llff_data
from skimage.transform import resize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Order of the 3x3 neighborhood that every ray carries: center first, then row by row.
# NeRFrac.smooth() indexes into it (2 = top, 4 = left, 5 = right, 7 = bottom).
NEIGHBOR_OFFSETS = [
    (0, 0),
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
MAX_MATCH_DIST_PX = 15.0
# load_llff_data scales the scene by 1 / (smallest depth bound * BD_FACTOR), and
# llff_scale.npy stores that scale for the postprocessing
BD_FACTOR = 0.75


def gather_neighbor_rays(rays_o_full, rays_d_full, coords, H, W):
    rows, cols = coords[:, 0], coords[:, 1]
    neighbors_o, neighbors_d = [], []
    for dh, dw in NEIGHBOR_OFFSETS:
        r = torch.clamp(rows + dh, 0, H - 1)
        c = torch.clamp(cols + dw, 0, W - 1)
        neighbors_o.append(rays_o_full[r, c])
        neighbors_d.append(rays_d_full[r, c])
    rays_o = torch.stack(neighbors_o, dim=1).reshape(-1, 3)
    rays_d = torch.stack(neighbors_d, dim=1).reshape(-1, 3)
    return torch.stack([rays_o, rays_d], dim=0)


def render_water_points(pose, coords, H, W, K, render_kwargs, chunk):
    rays_o, rays_d = get_rays(H, W, K, pose)
    batch_rays = gather_neighbor_rays(rays_o, rays_d, coords, H, W)
    water_mask = torch.ones(len(coords), device=coords.device)
    _, _, _, Xs, depth, sample_ray, _ = render(
        H,
        W,
        K,
        chunk=chunk,
        rays=batch_rays,
        pixel_water_mask=water_mask,
        **render_kwargs,
    )
    P_ndc = Xs + depth.unsqueeze(-1) * sample_ray
    return render_kwargs["network_fn"].ndcp2camp(P_ndc, W=W, H=H, f=float(K[0][0]))


def project_to_view(P_world, pose, focal, H, W):
    P_cam = (P_world - pose[:3, 3]) @ pose[:3, :3]
    in_front = P_cam[:, 2] < -1e-6
    u = focal * P_cam[:, 0] / (-P_cam[:, 2]) + W * 0.5
    # get_rays() uses y = -(j - cy) / f, so the row is j = -f * y / (-z) + H / 2
    v = -focal * P_cam[:, 1] / (-P_cam[:, 2]) + H * 0.5
    in_view = in_front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return torch.stack([v, u], dim=-1), in_view


def match_points(proj_rc, in_view, P_src, coords_other, P_other):
    zero = torch.zeros((), device=P_src.device)
    if not in_view.any():
        return zero, 0
    nn_dist, nn_idx = torch.cdist(proj_rc[in_view], coords_other.float()).min(dim=1)
    close = nn_dist < MAX_MATCH_DIST_PX
    if not close.any():
        return zero, 0
    mse = torch.mean((P_src[in_view][close] - P_other[nn_idx[close]]) ** 2)
    return mse, int(close.sum().item())


def compute_consistency_loss(
    poses, masks, H, W, K, i_train, render_kwargs, N_consist, chunk, device
):
    """Multi-view depth consistency loss between two random training views.

    Renders up to N_consist water pixels in each view, projects the underwater points of
    each view into the other and penalizes the 3D distance to the nearest rendered pixel.
    """
    idx_a, idx_b = random.sample(list(i_train), 2)
    pose_a = poses[idx_a, :3, :4]
    pose_b = poses[idx_b, :3, :4]

    water_a = (
        torch.tensor(masks[idx_a], dtype=torch.float32, device=device) > 0.5
    ).nonzero()
    water_b = (
        torch.tensor(masks[idx_b], dtype=torch.float32, device=device) > 0.5
    ).nonzero()
    if len(water_a) < 10 or len(water_b) < 10:
        return torch.tensor(0.0, device=device), 0

    coords_a = water_a[
        torch.randperm(len(water_a), device=device)[: min(N_consist, len(water_a))]
    ]
    coords_b = water_b[
        torch.randperm(len(water_b), device=device)[: min(N_consist, len(water_b))]
    ]

    # the chunk size changes the random draws, so it stays as in the thesis runs
    chunk = min(chunk, 9 * len(coords_a))
    P_a = render_water_points(pose_a, coords_a, H, W, K, render_kwargs, chunk)
    P_b = render_water_points(pose_b, coords_b, H, W, K, render_kwargs, chunk)

    focal = float(K[0][0])
    proj_ab, in_view_ab = project_to_view(P_a, pose_b, focal, H, W)
    proj_ba, in_view_ba = project_to_view(P_b, pose_a, focal, H, W)
    loss_ab, n_ab = match_points(proj_ab, in_view_ab, P_a, coords_b, P_b)
    loss_ba, n_ba = match_points(proj_ba, in_view_ba, P_b, coords_a, P_a)
    return loss_ab + loss_ba, n_ab + n_ba


def ramp(weight, warmup, step):
    return weight * min(step / warmup, 1.0) if warmup > 0 else weight


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
            wm_chunk = water_mask[i : i + chunk] if water_mask is not None else None
            outputs_flat_t, Xs_t, z_vals_t, sample_ray_t, dD_t = fn(
                inputs[i : i + chunk], wm_chunk
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


def run_network(
    rays_o, viewdirs, water_mask, fn, netchunk=1024, N_samples=None, N_importance=None
):
    """Prepares inputs and applies network 'fn'."""
    embedded = torch.cat([viewdirs, rays_o], -1)

    embedded = torch.reshape(embedded, [-1, 9, 6])
    outputs_flat, Xs, z_vals, sample_ray, dD = batchify(fn, netchunk)(
        embedded, water_mask
    )
    outputs = torch.reshape(
        outputs_flat,
        [embedded.shape[0]] + [N_samples + N_importance] + [outputs_flat.shape[-1]],
    )

    return outputs, Xs, z_vals, sample_ray, dD


def batchify_rays(rays_flat, chunk=1024 * 32, water_mask=None, **kwargs):
    """Render rays in smaller minibatches to avoid OOM."""
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        wm_chunk = water_mask[i : i + chunk] if water_mask is not None else None
        ret = render_rays(rays_flat[i : i + chunk], water_mask=wm_chunk, **kwargs)
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
    """Render a full view from c2w, or a given ray batch, in groups of 9 rays.

    Every ray comes with its 3x3 neighbors (NEIGHBOR_OFFSETS), which NeRFrac uses for
    the surface normal and the smoothness loss.
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)

        rays_od = torch.stack([rays_o, rays_d], dim=2)  # [H, W, 2, 3]
        padded = F.pad(rays_od.permute(2, 3, 0, 1), (1, 1, 1, 1), mode="replicate")
        padded = padded.permute(2, 3, 0, 1)  # [H+2, W+2, 2, 3]
        rays_od = torch.stack(
            [
                padded[1 + dh : 1 + dh + H, 1 + dw : 1 + dw + W]
                for dh, dw in NEIGHBOR_OFFSETS
            ],
            dim=2,
        )  # [H, W, 9, 2, 3]
        rays_o, rays_d = rays_od[..., 0, :], rays_od[..., 1, :]
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    rays_o = torch.reshape(rays_o, [-1, 9, 3])
    rays_d = torch.reshape(rays_d, [-1, 9, 3])

    # for forward facing scenes
    if ndc:
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1.0, rays_o, rays_d)

    rays = torch.cat([rays_o, rays_d], -1)

    all_ret = batchify_rays(rays, chunk, water_mask=pixel_water_mask, **kwargs)

    k_extract = ["rgb_map", "disp_map", "acc_map", "Xs", "depth_map", "sample_ray"]
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(
    render_poses,
    hwf,
    K,
    chunk,
    render_kwargs,
    gt_imgs=None,
    savedir=None,
    render_factor=0,
    args=None,
    masks=None,
):
    """Render the poses and save the per-view outputs that the postprocessing reads.

    With savedir, view NNN gets <expname>NNN.png, <expname>NNN_depth.npy (expected depth),
    <expname>NNN_depth_mode.npy (depth of the sample with the highest weight),
    <expname>NNN_acc.npy, Xs_ptsNNN.npy (water surface points), rays_NNN.npy (refracted
    directions) and pose_NNN.npy, plus <expname>NNN_masked.png and <expname>NNN_rgba.png
    when masks are given. With gt_imgs, result.txt gets PSNR, SSIM and LPIPS per view.
    """
    H, W, focal = hwf

    if render_factor != 0:
        # Render downsampled for speed
        H = H // render_factor
        W = W // render_factor
        focal = focal / render_factor
        K = np.array([[focal, 0, 0.5 * W], [0, focal, 0.5 * H], [0, 0, 1]])

    rgbs = []
    disps = []

    metrics = []
    if gt_imgs is not None:
        lpips_model = util_of_lpips(net="vgg")

    for i, c2w in enumerate(tqdm(render_poses)):
        mask_i = None
        pwm = None
        if masks is not None:
            mask_i = masks[i]
            # resize if render_factor changed H, W
            if mask_i.shape != (H, W):
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

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, args.expname + "{:03d}.png".format(i))
            imageio.imwrite(filename, rgb8)

            if mask_i is not None:
                mask_binary = (mask_i > 0.5).astype(np.float32)
                rgb_masked = rgb8.copy()
                rgb_masked[mask_binary < 0.5] = 0  # Black outside

                masked_filename = os.path.join(
                    savedir, f"{args.expname}{i:03d}_masked.png"
                )
                imageio.imwrite(masked_filename, rgb_masked)

                rgba = np.concatenate(
                    [rgb8, (mask_binary * 255).astype(np.uint8)[..., None]], axis=-1
                )
                rgba_filename = os.path.join(savedir, f"{args.expname}{i:03d}_rgba.png")
                imageio.imwrite(rgba_filename, rgba)

            Xs_pts = Xs.cpu().numpy()
            ptsname = os.path.join(savedir, "Xs_pts{:03d}.npy".format(i))
            np.save(ptsname, Xs_pts)

            posename = os.path.join(savedir, f"pose_{i:03d}.npy")
            np.save(posename, c2w[:3, :4].cpu().numpy())

            ray_dir = sample_ray.cpu().numpy()
            np.save(os.path.join(savedir, f"rays_{i:03d}.npy"), ray_dir)

            if gt_imgs is not None:
                gt_img = to8b(gt_imgs[i].cpu().numpy())
                out_img = rgb8

                lpipsout = lpips_model.calc_lpips(gt_img, out_img)
                lpipsout = lpipsout[0][0][0][0].detach().cpu().numpy()

                psnrout = psnr(gt_img, out_img)

                ssimout = ssim(gt_img, out_img)

                mse = 10 ** (-psnrout / 10)
                average = mse * np.sqrt(1 - ssimout) * lpipsout
                average = math.pow(average, 1.0 / 3)

                metrics.append(
                    {
                        "PSNR": psnrout,
                        "SSIM": ssimout,
                        "LPIPS": lpipsout,
                        "average": average,
                    }
                )

    if metrics:
        pd.DataFrame(metrics).to_csv(
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
        # model_fine is never called, but its parameters are in the optimizer state of
        # every checkpoint, so it stays
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

    network_query_fn = lambda rays_o, viewdirs, water_mask, network_fn, N_samples, N_importance: run_network(
        rays_o,
        viewdirs,
        water_mask,
        network_fn,
        netchunk=args.netchunk,
        N_samples=N_samples,
        N_importance=N_importance,
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
            if f.endswith(".tar")
        ]

    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print("Reloading from", ckpt_path)
        ckpt = torch.load(ckpt_path)
        start = ckpt["global_step"]
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        # checkpoints saved from a torch.compile model carry an _orig_mod. prefix
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
    """Mip-NeRF 360 distortion loss (eq. 15) in prefix-sum form.

    The last, unbounded interval is left out.
    """
    t_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
    delta = z_vals[..., 1:] - z_vals[..., :-1]
    w = weights[..., :-1]  # (N_rays, N_samples-1)
    w_cumsum = torch.cumsum(w, dim=-1)  # (N_rays, N_samples-1)
    wt_cumsum = torch.cumsum(w * t_mid, dim=-1)
    loss_uni = 2.0 * torch.sum(w * t_mid * w_cumsum - w * wt_cumsum, dim=-1)
    loss_bi = (1.0 / 3.0) * torch.sum(w**2 * delta, dim=-1)
    return torch.mean(loss_uni + loss_bi)


def render_rays(
    ray_batch,
    network_fn,
    network_query_fn,
    water,
    N_samples,
    N_importance,
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

    return {
        "rgb_map": rgb_map,
        "disp_map": disp_map,
        "acc_map": acc_map,
        "Xs": Xs,
        "sample_ray": sample_ray,
        "dD": dD,
        "depth_map": depth_map,
        "depth_mode": depth_mode,
        "water_mask": wm,
        "weights": weights,
        "z_vals": z_vals,
    }


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
        default="nerfrac",
        choices=["nerfrac"],
        help="network type (only nerfrac is implemented)",
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
        "--no_reload", action="store_true", help="do not reload weights from saved ckpt"
    )
    parser.add_argument(
        "--ft_path",
        type=str,
        default=None,
        help=".tar checkpoint to load instead of the latest one in basedir/expname",
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
        help="render instead of training: the test views with --render_test, the training "
        "views with --render_train, all views with --render_all, otherwise the spiral path",
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
        help="render the training views",
    )
    parser.add_argument(
        "--render_all", action="store_true", help="render all views in dataset order"
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

    # logging/saving options
    parser.add_argument(
        "--i_print",
        type=int,
        default=1000,
        help="frequency of console printout and metric logging",
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

    parser.add_argument(
        "--yita",
        type=float,
        default=1.0,
        help="refractive index of the water (1.333 for Pielach, 1.33 in "
        "configs/real_*.txt)",
    )

    parser.add_argument(
        "--water",
        type=int,
        default=1,
        help="1 if the scene has a water surface: refracts rays when no masks are loaded "
        "and enables the smoothness and consistency losses",
    )
    parser.add_argument("--initdepth", type=float, default=-0.6, help="initial depth")
    parser.add_argument(
        "--tframe",
        type=str,
        default="4",
        help="comma-separated indices of the test views",
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

    # regularization
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

    # multi-view depth consistency
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

    # distortion and flatness
    parser.add_argument(
        "--distortion_weight",
        type=float,
        default=0.0,
        help="weight for distortion loss (0 to disable, the thesis models use 0.01)",
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
        help="weight for the flatness loss, the variance of the water surface height (0 to disable)",
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
        f"{args.expname}_{args.nerf_type}_d{args.initdepth}_yita{args.yita}_tframe{args.tframe}"
        f"_factor{args.factor}multires{args.omultires}_{args.refracvmultires}_{args.N_samples}"
    )

    if not os.path.exists("./tensorboard/" + args.expname):
        os.makedirs("./tensorboard/" + args.expname)
    writer = SummaryWriter("./tensorboard/" + args.expname)

    # Load data
    images, masks, poses, _, render_poses, _, c2w_recenter = load_llff_data(
        args.datadir,
        args.factor,
        recenter=True,
        bd_factor=BD_FACTOR,
        spherify=args.spherify,
        load_masks=True,
    )
    use_masks = masks is not None
    if use_masks:
        print(f"Using water masks for per-ray water/land routing, shape: {masks.shape}")
        masks = np.transpose(masks, [2, 0, 1])  # [N, H, W]
    else:
        surface = "water" if args.water == 1 else "land"
        print(
            f"No masks_{args.factor} folder in {args.datadir}: "
            f"every ray is treated as {surface} (--water {args.water})"
        )

    hwf = poses[0, :3, -1]
    poses = poses[:, :3, :4]
    print("Loaded llff", images.shape, render_poses.shape, hwf, args.datadir)

    i_test = [int(x) for x in args.tframe.split(",")]
    i_val = i_test
    i_train = np.array([i for i in range(len(images)) if i not in i_test])

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
    # render-only runs keep the training record in args.txt and config.txt
    args_file = "args_render.txt" if args.render_only else "args.txt"
    f = os.path.join(basedir, expname, args_file)

    with open(f, "w") as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write("{} = {}\n".format(arg, attr))
    if args.config is not None and not args.render_only:
        dest_config = os.path.join(basedir, expname, "config.txt")
        src_config = os.path.abspath(args.config)
        if os.path.abspath(dest_config) != src_config:
            with open(dest_config, "w") as file:
                with open(args.config, "r") as config_file:
                    file.write(config_file.read())

    # Save coordinate system metadata for post-processing scripts
    exp_dir = os.path.join(basedir, expname)
    np.save(os.path.join(exp_dir, "c2w.npy"), c2w_recenter)
    pb = np.load(os.path.join(args.datadir, "poses_bounds.npy"))
    bds_raw = pb[:, -2:]
    sc = 1.0 / (bds_raw.min() * BD_FACTOR)
    np.save(os.path.join(exp_dir, "llff_scale.npy"), np.array([sc]))
    print(f"Saved c2w.npy and llff_scale.npy (sc={sc:.8f}) to {exp_dir}")

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(
        args
    )
    model = render_kwargs_train["network_fn"]
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
                mode = "test"
                images = images[i_test]
                render_masks = masks[i_test] if use_masks else None
            elif args.render_train:
                mode = "train"
                images = images[i_train]
                render_masks = masks[i_train] if use_masks else None
            elif args.render_all:
                mode = "all"
                images = images[i_comp]
                render_masks = masks[i_comp] if use_masks else None

            else:
                # Default is smoother render_poses path, without ground truth or masks
                mode = "path"
                images = None
                render_masks = None

            # start is the global_step stored in the checkpoint, one less than its iteration
            testsavedir = os.path.join(
                basedir, expname, f"renderonly_{mode}_{start + 1:06d}"
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
                print(f"Could not write video.mp4: {e}")

            # the ground truth is full size, so a reduced render gets no error maps
            if images is not None and args.render_factor == 0:
                print(f"Generating error maps for {len(rgbs)} frames...")
                for k in range(len(rgbs)):
                    pred = torch.tensor(rgbs[k])
                    gt = torch.tensor(images[k])
                    err_map = torch.sum((pred - gt) ** 2, dim=-1).cpu().numpy()
                    err_map_colored = (
                        colorize_np(err_map, range=(0.0, 1.0)) * 255
                    ).astype(np.uint8)
                    imageio.imwrite(
                        os.path.join(testsavedir, f"{k:03d}_err.png"),
                        err_map_colored,
                    )
                    if render_masks is not None:
                        masked_err = err_map * render_masks[k]
                        err_map_masked_colored = (
                            colorize_np(masked_err, range=(0.0, 1.0)) * 255
                        ).astype(np.uint8)
                        imageio.imwrite(
                            os.path.join(testsavedir, f"{k:03d}_err_masked.png"),
                            err_map_masked_colored,
                        )

            return

    # Prepare raybatch tensor of random rays
    N_rand = args.N_rand
    print("get rays")
    rays = np.stack(
        [get_rays_np(H, W, K, p) for p in poses[:, :3, :4]], 0
    )  # [N, ro+rd, H, W, 3]
    print("done, concats")
    rays_rgb = np.concatenate([rays, images[:, None]], 1)  # [N, ro+rd+rgb, H, W, 3]
    rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])  # [N, H, W, ro+rd+rgb, 3]
    print("Arranging neighbor rays for all training pixels...")
    filtered_rays_list = []
    water_flags_list = []
    for n in i_train:
        # without masks, follow --water like render_rays does
        mask_n = masks[n] if use_masks else np.full((H, W), float(args.water == 1))
        padded = np.pad(rays_rgb[n], ((1, 1), (1, 1), (0, 0), (0, 0)), mode="edge")
        neighbors = np.stack(
            [
                padded[1 + dh : 1 + dh + H, 1 + dw : 1 + dw + W]
                for dh, dw in NEIGHBOR_OFFSETS
            ],
            axis=2,
        )  # [H, W, 9, ro+rd+rgb, 3]
        filtered_rays_list.append(neighbors.reshape(-1, 9, 3, 3))
        water_flags_list.append((mask_n.reshape(-1) > 0.5).astype(np.float32))
    rays_rgb = np.concatenate(filtered_rays_list, axis=0)  # [N_total, 9, 3, 3]
    rays_water_flags = np.concatenate(water_flags_list, axis=0)  # [N_total]
    n_water = int(np.sum(rays_water_flags > 0.5))
    n_nonwater = len(rays_water_flags) - n_water
    print(
        f"Total training rays: {len(rays_rgb):,} (water: {n_water:,}, non-water: {n_nonwater:,})"
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
    images = torch.Tensor(images).to(device)
    rays_rgb = torch.Tensor(rays_rgb)
    rays_water_flags = torch.Tensor(rays_water_flags)
    poses = torch.Tensor(poses).to(device)

    N_iters = args.N_iter + 1
    print("Begin")
    print("TRAIN views are", i_train)
    print("TEST views are", i_test)
    print("VAL views are", i_val)

    start = start + 1
    for i in trange(start, N_iters):
        # Sample a random ray batch over all images
        batch = rays_rgb[i_batch : i_batch + N_rand].to(device)  # [B, 9, 2+1, 3*?]
        batch_wm = rays_water_flags[i_batch : i_batch + N_rand].to(device)  # [B]
        batch = torch.reshape(batch, [-1, 3, 3])
        batch = torch.transpose(batch, 0, 1)
        batch_rays, target_s = batch[:2], batch[2]

        i_batch += N_rand
        if i_batch >= rays_rgb.shape[0]:
            tqdm.write("Shuffle data after an epoch!")
            rand_idx = torch.randperm(rays_rgb.shape[0])
            rays_rgb = rays_rgb[rand_idx]
            rays_water_flags = rays_water_flags[rand_idx]
            i_batch = 0

        #####  Core optimization loop  #####
        rgb, _, _, Xs, _, _, extras = render(
            H,
            W,
            K,
            chunk=args.chunk,
            rays=batch_rays,
            pixel_water_mask=batch_wm,
            **render_kwargs_train,
        )

        optimizer.zero_grad(set_to_none=True)

        target_s = target_s.view([-1, 9, 3])
        target_s = target_s[:, 0, :].squeeze()

        img_loss = img2mse(rgb, target_s)

        model = render_kwargs_train["network_fn"]
        dD_out = extras["dD"]
        dD_reshaped = dD_out.view(-1, 9)
        water_rays = extras["water_mask"] > 0.5
        smooth_loss = torch.tensor(0.0, device=device)
        if args.water == 1 and args.smooth_weight > 0 and water_rays.any():
            smooth_loss = torch.mean(model.smooth(dD_reshaped[water_rays]))
        effective_smooth_weight = ramp(
            args.smooth_weight, args.smooth_warmup, global_step
        )
        loss = img_loss + effective_smooth_weight * smooth_loss

        flat_loss = torch.tensor(0.0, device=device)
        effective_flat_weight = 0.0
        if args.flatness_weight > 0:
            # the variance of a single value is NaN
            if water_rays.sum() > 1:
                flat_loss = torch.var(Xs[water_rays, 2])
            effective_flat_weight = ramp(
                args.flatness_weight, args.flatness_warmup, global_step
            )
            loss = loss + effective_flat_weight * flat_loss

        # Distortion loss (Mip-NeRF 360)
        dist_loss = torch.tensor(0.0, device=device)
        effective_dist_weight = 0.0
        if args.distortion_weight > 0:
            effective_dist_weight = ramp(
                args.distortion_weight, args.distortion_warmup, global_step
            )
            dist_loss = distortion_loss(extras["weights"], extras["z_vals"])
            loss = loss + effective_dist_weight * dist_loss

        psnr = mse2psnr(img_loss)

        # Multi-view depth consistency loss
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
            effective_consist_weight = ramp(
                args.consistency_weight, args.consistency_warmup, i
            )

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
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                tqdm.write(f"Iter {i:,}: consistency loss skipped (CUDA out of memory)")

        writer.add_scalar("loss/smooth", smooth_loss, global_step=i)
        writer.add_scalar("psnr", psnr, global_step=i, walltime=None)
        writer.add_scalar("loss", loss, global_step=i, walltime=None)
        writer.add_scalar(
            "smooth/effective_weight", effective_smooth_weight, global_step=i
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
            writer.add_scalar(
                "flatness/effective_weight", effective_flat_weight, global_step=i
            )

        if not torch.isfinite(loss):
            tqdm.write(
                f"Iter {i:,}: non-finite loss ({loss.item():.4f}), batch skipped"
            )
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
            tqdm.write(f"Saved checkpoints at {path}")

        if i % args.i_video == 0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_path(
                    render_poses, hwf, K, args.chunk, render_kwargs_test, args=args
                )
            tqdm.write(f"Done, saving {rgbs.shape} {disps.shape}")
            moviebase = os.path.join(
                basedir, expname, "{}_spiral_{:06d}_".format(expname, i)
            )
            imageio.mimwrite(moviebase + "rgb.mp4", to8b(rgbs), fps=30, quality=8)

        if i % args.i_testset == 0 and i > 0:
            testsavedir = os.path.join(basedir, expname, "testset_{:06d}".format(i))
            os.makedirs(testsavedir, exist_ok=True)
            test_masks = masks[i_test] if use_masks else None
            tqdm.write(f"test poses shape {poses[i_test].shape}")
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
            tqdm.write("Saved test set")

        if i % args.i_print == 0:
            dD_info = ""
            if args.water == 1:
                dD_info = f"  dD mean {dD_out.abs().mean().item():.6f} max {dD_out.abs().max().item():.4f}"
            smooth_val = smooth_loss.item()
            consist_info = ""
            if args.consistency_weight > 0:
                consist_val = consistency_loss.item()
                consist_info = (
                    f"  consistency {consist_val:.6f} ({n_consist_pairs:,} pairs)"
                )
            tqdm.write(
                f"Iter {i:,}  loss {loss.item():.6f}  psnr {psnr.item():.3f}  "
                f"smooth {smooth_val:.6f}{dD_info}{consist_info}"
            )

        global_step += 1


if __name__ == "__main__":
    train()
