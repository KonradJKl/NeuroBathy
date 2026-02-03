import os
import sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter
from utils import *
import matplotlib.pyplot as plt
import pandas as pd
from run_nerfrac_helpers import *

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
from load_LINEMOD import load_LINEMOD_data
from collections import OrderedDict


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


DEBUG = False


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn

    def ret(inputs, water):
        outputs_flat = []
        Xs = []
        z_vals = []
        sample_ray = []
        dD_list = []

        for i in range(0, inputs.shape[0], chunk):
            outputs_flat_t, Xs_t, z_vals_t, sample_ray_t, dD_t = fn(inputs[i:i+chunk], water)
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


def run_network(rays_o, viewdirs, water, fn, embed_fn, embeddirs_fn, netchunk=1024, N_samples=None, N_importance=None):
    """Prepares inputs and applies network 'fn'.
    """
    if viewdirs is not None:
        embedded = torch.cat([viewdirs, rays_o], -1)

    embedded = torch.reshape(embedded, [-1, 9, 6])
    outputs_flat, Xs, z_vals, sample_ray, dD = batchify(fn, netchunk)(embedded, water)
    outputs = torch.reshape(outputs_flat, [embedded.shape[0]] + [N_samples+N_importance] + [outputs_flat.shape[-1]])

    return outputs, Xs, z_vals, sample_ray, dD


def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}

    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True, savedir=None,
           near=0., far=1.,
           use_viewdirs=False, c2w_staticcam=None,
           **kwargs):

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
            rays_padded[0:H_img, 0:W_img, :, :],  # top-left
            rays_padded[0:H_img, 1:W_img + 1, :, :],  # top
            rays_padded[0:H_img, 2:W_img + 2, :, :],  # top-right
            rays_padded[1:H_img + 1, 0:W_img, :, :],  # left
            rays_padded[1:H_img + 1, 2:W_img + 2, :, :],  # right
            rays_padded[2:H_img + 2, 0:W_img, :, :],  # bottom-left
            rays_padded[2:H_img + 2, 1:W_img + 1, :, :],  # bottom
            rays_padded[2:H_img + 2, 2:W_img + 2, :, :],  # bottom-right
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

    sh = rays_d.shape
    
    # for forward facing scenes
    if ndc:
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    rays = torch.cat([rays_o, rays_d], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)

    if rays_d.numel() % (9*3) == 0:
        rays_d = rays_d.view([-1, 9, 3])
        rays_d = rays_d[:, 0, :].squeeze()
        sh = rays_d.shape

    for k in all_ret:
        if k == 'Xs':
            continue
        if k == 'sample_ray':
            continue
        if k == 'dD':
            continue

        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map', 'Xs', 'depth_map', 'sample_ray']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0, args=None, masks=None):

    H, W, focal = hwf

    if render_factor != 0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []

    t = time.time()

    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, Xs, depth_map, sample_ray, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3, :4], savedir=savedir, **render_kwargs)
        if savedir is not None:
            np.save(os.path.join(savedir, f"{args.expname}{i:03d}_depth.npy"), depth_map.cpu().numpy().reshape(H, W).astype('float32'))
        rgb = torch.reshape(rgb, [H, W, 3])
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i == 0:
            print(rgb.shape, disp.shape)

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, args.expname + '{:03d}.png'.format(i))
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

                masked_filename = os.path.join(savedir, args.expname + '{:03d}_masked.png'.format(i))
                imageio.imwrite(masked_filename, rgb_masked)

                # save with RGBA
                rgba = np.concatenate([rgb8, (mask_binary * 255).astype(np.uint8)[..., None]], axis=-1)
                rgba_filename = os.path.join(savedir, args.expname + '{:03d}_rgba.png'.format(i))
                imageio.imwrite(rgba_filename, rgba)
            # --- END Save masked ---

            Xs_pts = Xs.cpu().numpy()
            ptsname = os.path.join(savedir, 'Xs_pts{:03d}.npy'.format(i))
            np.save(ptsname, Xs_pts)

            # --- Save camera pose ---
            posename = os.path.join(savedir, 'pose_{:03d}.npy'.format(i))
            np.save(posename, c2w[:3, :4].cpu().numpy())

            # --- Save rays ---
            ray_dir = sample_ray.cpu().numpy()
            np.save(os.path.join(savedir, 'rays_{:03d}.npy'.format(i)), ray_dir)
            
            if gt_imgs is not None:
                gt_img = to8b(gt_imgs[i].cpu().numpy())
                out_img = rgb8
                
                LPIPS = util_of_lpips(net='vgg')
                lpipsout = LPIPS.calc_lpips(gt_img, out_img)
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

                dic = {'PSNR': psnrout, 'SSIM': ssimout, 'LPIPS': lpipsout, 'average': average}
                df = pd.DataFrame([dic])
                df.to_csv(os.path.join(savedir, 'result.txt'), sep='\t', index=False, float_format='%.3f')

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    print('input_ch:', args.N_importance)
    model = NeRF.get_by_name(args.nerf_type, D=args.netdepth, W=args.netwidth,
                             input_ch=input_ch, output_ch=output_ch, skips=skips,
                             input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs,
                             embed_fn=embed_fn, embeddirs_fn=embeddirs_fn, modelargs=args).to(device)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        # fine sampling is used in class NeRFrac forward and not here
        model_fine = NeRF.get_by_name(args.nerf_type, D=args.netdepth_fine, W=args.netwidth_fine,
                                      input_ch=input_ch, output_ch=output_ch, skips=skips,
                                      input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs,
                                      embed_fn=embed_fn, embeddirs_fn=embeddirs_fn, modelargs=args).to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda rays_o, viewdirs, water, network_fn,  N_samples, N_importance: run_network(rays_o, viewdirs, water, network_fn,
                                                                                                         embed_fn=embed_fn,
                                                                                                         embeddirs_fn=embeddirs_fn,
                                                                                                         netchunk=args.netchunk,
                                                                                                         N_samples=N_samples,
                                                                                                         N_importance=N_importance)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)
        if not args.frozen_nerf:
            print('~~~~~~~~~~~~~~~~~~load checkpoint mode~~~~~~~~~~~~~~~~')
            start = ckpt['global_step']
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            model.load_state_dict(ckpt['network_fn_state_dict'])
        else:
            print('~~~~~~~~~~~~~~~~~~~freeze nerf mode~~~~~~~~~~~~~~~~~~~')
            for name in model.state_dict():
                print(name)
            new_state_dict = OrderedDict()

            for k, v in ckpt['network_fn_state_dict'].items():
                k = '_occ.'+k
                if k[0:4] == '_occ':
                    new_state_dict[k] = v

            for k, v in ckpt['network_fine_state_dict'].items():
                k = '_occfine.'+k
                if k[0:4] == '_occ':
                    new_state_dict[k] = v

            start = ckpt['global_step']

            model.load_state_dict(new_state_dict, strict=False)
            for name, value in model.named_parameters():
                if name[:4] == '_occ':
                    value.requires_grad = False

    if args.check_weight == 1:
        for name in model.state_dict():
            print(name)
        print(model.state_dict()['_occ.pts_linears.7.weight'].shape)
        sys.exit()
    ##########################

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'water': args.water,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'network_fine': model_fine,
        'N_samples': args.N_samples,
        'network_fn': model,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):

    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    infinity = torch.tensor([1e10], device=dists.device).expand(dists[..., :1].shape)
    dists = torch.cat([dists, infinity], -1)

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
    rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape, device=raw.device) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
            noise = torch.Tensor(noise).to(raw.device)

    alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    ones_col = torch.ones((alpha.shape[0], 1), device=alpha.device)
    weights = alpha * torch.cumprod(torch.cat([ones_col, 1. - alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                water,
                N_samples,
                N_importance,
                retraw=False,
                lindisp=False,
                perturb=0.,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):

    rays_o, rays_d = ray_batch[:, :, 0:3], ray_batch[:, :, 3:6]
    rays_o = torch.reshape(rays_o, [-1, 3])
    rays_d = torch.reshape(rays_d, [-1, 3])

    N_rays = rays_o.shape[0]

    viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

    raw, Xs, z_vals, sample_ray, dD = network_query_fn(rays_o, viewdirs, water, network_fn, N_samples, N_importance)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, sample_ray, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map, 'Xs': Xs, 'sample_ray': sample_ray, 'dD': dD}
    ret['depth_map'] = depth_map
    if retraw:
        ret['raw'] = raw

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')

    # training options
    parser.add_argument("--nerf_type", type=str, default="original",
                        help='nerf network type, original or nerof')
    parser.add_argument("--N_iter", type=int, default=200000,
                        help='num training iterations')
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*2,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--frozen_nerf", action='store_true',
                        help='freeze nerf for training')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=64,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--render_train", action='store_true',
                        help='Render training views instead of test views')
    parser.add_argument("--render_all", action='store_true',
                        help='Render all views')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')
    parser.add_argument("--use_mask_selection", action='store_true',
                        help='Use water masks for selective ray sampling during training')

    # deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    # blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    # llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--check_weight", type=int, default=0)
    parser.add_argument("--i_print",   type=int, default=1000,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=10000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=400000,
                        help='frequency of render_poses video saving')

    parser.add_argument("--yita",   type=float, default=1.,
                        help='refractive index')

    parser.add_argument("--H",   type=int, default=392)
    parser.add_argument("--W",   type=int, default=392)
    parser.add_argument("--f",   type=float, default=500.,
                        help='camera focal')

    parser.add_argument("--water",   type=int, default=1,
                        help='with water surface')
    parser.add_argument("--initdepth",   type=float, default=-0.6,
                        help='initial depth')
    parser.add_argument("--tframe",   type=str, default=4,
                        help='test frame, 4 is the center camera')

    parser.add_argument("--omultires", type=int, default=0,
                        help='log2 of max freq for positional encoding (refractive field center)')
    parser.add_argument("--refracvmultires", type=int, default=0,
                        help='log2 of max freq for positional encoding (refractive field dir)')

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
    args.expname = args.expname + '_' + args.nerf_type + '_d' + str(args.initdepth)\
                                + '_yita' + str(args.yita) + '_tframe' + str(args.tframe)\
                                + '_factor' + str(args.factor) + 'multires'\
                                + str(args.omultires) + '_' + str(args.refracvmultires)\
                                + '_' + str(args.N_samples)

    if not os.path.exists('./tensorboard/'+args.expname):
        os.makedirs('./tensorboard/'+args.expname)
    writer = SummaryWriter('./tensorboard/'+args.expname)

    # Load data
    K = None
    if args.dataset_type == 'llff':
        images, masks, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                         recenter=True, bd_factor=.75,
                                                                         spherify=args.spherify, load_masks=True)
        use_masks = masks is not None
        if use_masks:
            print(f'Using water masks for ray selection, shape: {masks.shape}')
            masks = np.transpose(masks, [2, 0, 1])  # [N, H, W]
        else:
            print('No masks found, training without mask-based ray selection')

        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]
        i_test = [int(x) for x in args.tframe.split(',')]
        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                            (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
            
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[..., :3]*images[..., -1:] + (1.-images[..., -1:])
        else:
            images = images[..., :3]

    elif args.dataset_type == 'LINEMOD':
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res, args.testskip)
        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if args.white_bkgd:
            images = images[..., :3]*images[..., -1:] + (1.-images[..., -1:])
        else:
            images = images[..., :3]

    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:, :3, -1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    print(H, W)
    hwf = [H, W, focal]

    # 定义内参矩阵
    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])
    print('K:', K)
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
    f = os.path.join(basedir, expname, 'args.txt')

    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        dest_config = os.path.join(basedir, expname, "config.txt")
        src_config = os.path.abspath(args.config)
        if os.path.abspath(dest_config) != src_config:
            with open(dest_config, "w") as file:
                with open(args.config, "r") as config_file:
                    file.write(config_file.read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    if hasattr(torch, "compile"):
        render_kwargs_train["network_fn"] = torch.compile(render_kwargs_train["network_fn"], mode="reduce-overhead")
    global_step = start

    bds_dict = {
        'near': near,
        'far': far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
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

            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)
            rgbs, _ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test,
                                  savedir=testsavedir, render_factor=args.render_factor, args=args, masks=render_masks)
            print('Done rendering', testsavedir)
            try:
                imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'),
                                 to8b(rgbs), fps=30, quality=8)
            except Exception as e:
                print('[WARN] Could not write video with imageio/ffmpeg backend:', e)

            print(f"Generating error maps for {len(rgbs)} frames...")
            for k in range(len(rgbs)):
                pred = torch.tensor(rgbs[k])
                gt = torch.tensor(images[k])
                err_map = torch.sum((pred - gt) ** 2, dim=-1).cpu().numpy()
                err_map_colored = (colorize_np(err_map, range=(0., 1.)) * 255).astype(np.uint8)
                imageio.imwrite(os.path.join(testsavedir, '{:03d}_err.png'.format(k)), err_map_colored)
                if use_masks and args.use_mask_selection:
                    mask = masks[render_idx[k]]
                    masked_err = err_map * mask
                    err_map_masked_colored = (colorize_np(masked_err, range=(0., 1.)) * 255).astype(np.uint8)
                    imageio.imwrite(os.path.join(testsavedir, '{:03d}_err_masked.png'.format(k)), err_map_masked_colored)

            return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:, :3, :4]], 0)  # [N, ro+rd, H, W, 3]
        print('done, concats')
        rays_rgb = np.concatenate([rays, images[:, None]], 1)  # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])  # [N, H, W, ro+rd+rgb, 3]
        # Pre-compute neighbor arrangement with masks
        print('Arranging neighbor rays with mask filtering...')
        filtered_rays_list = []

        for n in range(len(rays_rgb)):
            if n not in i_train:
                continue

            img_rays = rays_rgb[n]  # [H, W, 3, 3] (ro, rd, rgb)
            mask_n = masks[n] if use_masks else np.ones((H, W))

            for h in range(H):
                for w in range(W):
                    # Skip if not water (when using masks)
                    if use_masks and args.use_mask_selection and mask_n[h, w] <= 0.5:
                        continue

                    # Collect center + 8 neighbors
                    ray_group = np.zeros((9, 3, 3))  # [9 neighbors, ro+rd+rgb, 3]
                    ray_group[0] = img_rays[h, w]  # center

                    neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
                    for num, (dh, dw) in enumerate(neighbor_offsets):
                        nh = max(0, min(H - 1, h + dh))
                        nw = max(0, min(W - 1, w + dw))
                        ray_group[num + 1] = img_rays[nh, nw]

                    filtered_rays_list.append(ray_group)

        rays_rgb = np.array(filtered_rays_list)  # [N_water_pixels, 9, 3, 3]
        print(f'Filtered to {len(rays_rgb)} water rays with neighbors')

        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)
        print('done')
        i_batch = 0
        
    # Move training data to GPU
    images = torch.Tensor(images).pin_memory().to(device, non_blocking=True)
    rays_rgb = torch.Tensor(rays_rgb).pin_memory().to(device, non_blocking=True)
    poses = torch.Tensor(poses).pin_memory().to(device, non_blocking=True)

    N_iters = args.N_iter + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)
    
    start = start + 1
    for i in trange(start, N_iters):
        time0 = time.time()

        # Sample random ray batch
        if use_batching:
            # Random over all images
            batch = rays_rgb[i_batch:i_batch+N_rand]  # [B, 9, 2+1, 3*?]
            batch = torch.reshape(batch, [-1, 3, 3])
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0

        else:
            # Random from one image
            img_i = np.random.choice(i_train)
            target = images[img_i]
            target = torch.Tensor(target).to(device)
            pose = poses[img_i, :3, :4]

            if N_rand is not None:
                rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                if i < args.precrop_iters:
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH),
                            torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                        ), -1)
                    if i == start:
                        print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)

                # --- Mask-based ray selection ---
                if use_masks and args.use_mask_selection:
                    current_mask = masks[img_i]  # (H, W)
                    mask_flat = current_mask.reshape(-1)  # (H * W,)
                    water_indices = np.where(mask_flat > 0.5)[0]

                    if len(water_indices) >= N_rand:
                        select_inds = np.random.choice(water_indices, size=[N_rand], replace=False)
                    else:
                        print(f"Warning: Only {len(water_indices)} water pixels, need {N_rand}")
                        remaining = N_rand - len(water_indices)
                        non_water_indices = np.where(mask_flat <= 0.5)[0]
                        extra_inds = np.random.choice(non_water_indices, size=[remaining], replace=False)
                        select_inds = np.concatenate([water_indices, extra_inds])
                        np.random.shuffle(select_inds)
                else:
                    select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)
                # --- END Mask-based ray selection ---

                select_coords = coords[select_inds].long()  # (N_rand, 2)

                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        #####  Core optimization loop  #####
        rgb, disp, acc, Xs, _, _, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                  verbose=i < 10, retraw=True,
                                                  **render_kwargs_train)

        optimizer.zero_grad(set_to_none=True)

        target_s = target_s.view([-1, 9, 3])
        target_s = target_s[:, 0, :].squeeze()

        img_loss = img2mse(rgb, target_s)

        trans = extras['raw'][..., -1]

        loss = img_loss
        model = render_kwargs_train['network_fn']
        dD_out = extras['dD']
        dD_reshaped = dD_out.view(-1, 9)
        smooth_loss = 0
        if args.water == 1:
            smooth_loss = torch.mean(model.smooth(dD_reshaped))

        loss = img_loss + 0.1 * smooth_loss

        psnr = mse2psnr(img_loss)

        writer.add_scalar('loss_smooth', smooth_loss, global_step=i)
        writer.add_scalar('psnr', psnr, global_step=i, walltime=None)
        writer.add_scalar('loss', loss, global_step=i, walltime=None)

        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        writer.add_scalar('lrate', new_lrate, global_step=i, walltime=None)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        dt = time.time()-time0

        # Rest is logging
        if i % args.i_weights == 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                # 'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i % args.i_video == 0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, args=args)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)

        if i % args.i_testset == 0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            test_masks = masks[i_test] if use_masks else None
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk,
                            render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir, args=args, masks=test_masks)
            print('Saved test set')
    
        if i % args.i_print == 0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
          
        global_step += 1


if __name__=='__main__':
    train()
