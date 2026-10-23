"""
The NeRFrac model: NeRFrac's refractive surface field and Snell's law (Zhan et al.,
ICCV 2023) around the NeRF MLP, with the positional encoding and ray helpers of
nerf-pytorch.

NeuroBathy addition: land rays (water_mask <= 0.5) skip the refraction and are sampled
straight from the near plane (sampling_direct, _forward_land); forward routes each ray
by its mask.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Misc
img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = (
    lambda x: -10.0 * torch.log(x) / torch.log(torch.tensor([10.0], device=x.device))
)
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=N_freqs)

        freq_bands = freq_bands.tolist()

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        "include_input": True,
        "input_dims": 3,
        "max_freq_log2": multires - 1,
        "num_freqs": multires,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


def get_embeddero(multires, i=0):
    if i == -1:
        return nn.Identity(), 2

    embed_kwargs = {
        "include_input": True,
        "input_dims": 2,
        "max_freq_log2": multires - 1,
        "num_freqs": multires,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class LeastSquares:
    def lstq(self, A, Y):
        """Batched least-squares solution w of A @ w = Y by QR, for A [B, m, n] with
        full column rank and Y [B, m, 1]."""
        q, r = torch.linalg.qr(A, mode="reduced")
        w = torch.linalg.solve_triangular(r, q.transpose(1, 2) @ Y, upper=True)

        return w


class NeRF:
    @staticmethod
    def get_by_name(type, *args, **kwargs):
        print("NeRF type selected: %s" % type)

        if type == "nerfrac":
            model = NeRFrac(*args, **kwargs)
        else:
            raise ValueError("Type %s not recognized." % type)
        return model


class NeRFrac(nn.Module):
    def __init__(
        self,
        D=8,
        W=256,
        input_ch=3,
        output_ch=4,
        skips=[4],
        input_ch_views=3,
        use_viewdirs=False,
        embed_fn=None,
        embeddirs_fn=None,
        modelargs=None,
    ):
        super(NeRFrac, self).__init__()
        self.D = D
        self.W = W
        self.skips = skips
        self.embed_fn = embed_fn
        self.embeddirs_fn = embeddirs_fn

        self.yita = modelargs.yita
        self.register_buffer("Himg", torch.tensor(float(modelargs.H)))
        self.register_buffer("Wimg", torch.tensor(float(modelargs.W)))
        self.register_buffer("f", torch.tensor(float(modelargs.f)))
        self.initdepth = modelargs.initdepth

        self.i_embed = modelargs.i_embed
        self.omultires = modelargs.omultires
        self.refracvmultires = modelargs.refracvmultires
        self.N_samples = modelargs.N_samples
        self.N_importance = modelargs.N_importance

        # refractive field P.E.
        self.embedo_fn, self.o_ch = get_embeddero(self.omultires, self.i_embed)
        self.embedrefracv_fn, self.refracv_ch = get_embedder(
            self.refracvmultires, self.i_embed
        )

        self._occ = NeRFOriginal(
            D=D,
            W=W,
            input_ch=input_ch,
            output_ch=output_ch,
            skips=skips,
            input_ch_views=input_ch_views,
            use_viewdirs=use_viewdirs,
        )
        self._occfine = NeRFOriginal(
            D=D,
            W=W,
            input_ch=input_ch,
            output_ch=output_ch,
            skips=skips,
            input_ch_views=input_ch_views,
            use_viewdirs=use_viewdirs,
        )

        self._refrac, self._refrac_out = self.create_refrac_net()
        # flat initialization
        nn.init.uniform_(self._refrac_out.weight, a=-1e-5, b=1e-5)

    def create_refrac_net(self):
        layers = [nn.Linear(self.refracv_ch + self.o_ch, self.W)]
        for i in range(self.D - 1):
            layer = nn.Linear
            in_channels = self.W
            if i in self.skips:
                in_channels += self.o_ch

            layers += [layer(in_channels, self.W)]
        return nn.ModuleList(layers), nn.Linear(self.W, 1)

    def query_refrac(self, new_views, new_origin, net, net_final):
        h = torch.cat([new_views, new_origin], dim=-1)
        for i, l in enumerate(net):
            h = net[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([new_origin, h], -1)

        return net_final(h)

    def sampling(self, Xs, new_view):
        N_rays = Xs.shape[0]
        device = Xs.device

        input_dirs = new_view[:, None].expand([N_rays, self.N_samples, 3])

        far = torch.ones([N_rays, 1], dtype=torch.float, device=device)
        dis = far - Xs[..., -1:]
        new_view = new_view * dis / new_view[..., -1:]

        near = torch.zeros([N_rays, 1], dtype=torch.float, device=device)
        far = torch.ones([N_rays, 1], dtype=torch.float, device=device)
        t_vals = torch.linspace(0.0, 1.0, steps=self.N_samples, device=device)
        z_vals = near * (1.0 - t_vals) + far * (t_vals)
        z_vals = z_vals.expand([N_rays, self.N_samples])

        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape, device=device)

        z_vals = lower + (upper - lower) * t_rand
        pts = (
            Xs[..., None, :] + new_view[..., None, :] * z_vals[..., :, None]
        )  # [N_rays, N_samples, 3]
        pts_flat = torch.reshape(pts, [-1, pts.shape[-1]])
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])

        return pts_flat, input_dirs_flat, z_vals, new_view

    def sampling_direct(self, origins, view_dirs):
        """Stratified samples along the unrefracted ray, from the near plane to z = 1
        in NDC. Returns what sampling() returns, except that the view directions are
        the scaled ray, not unit vectors."""
        N_rays = origins.shape[0]
        device = origins.device

        near = torch.zeros([N_rays, 1], dtype=torch.float, device=device)
        far = torch.ones([N_rays, 1], dtype=torch.float, device=device)
        t_vals = torch.linspace(0.0, 1.0, steps=self.N_samples, device=device)
        z_vals = near * (1.0 - t_vals) + far * t_vals
        z_vals = z_vals.expand([N_rays, self.N_samples])

        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        t_rand = torch.rand(z_vals.shape, device=device)
        z_vals = lower + (upper - lower) * t_rand

        # scaled so that z_vals from 0 to 1 run from the origin to the far plane
        sample_ray = view_dirs * (far - origins[..., -1:]) / view_dirs[..., -1:]

        pts = origins[..., None, :] + sample_ray[..., None, :] * z_vals[..., :, None]
        pts_flat = torch.reshape(pts, [-1, pts.shape[-1]])

        input_dirs = sample_ray[:, None].expand([N_rays, self.N_samples, 3])
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])

        return pts_flat, input_dirs_flat, z_vals, sample_ray

    def get_norm(self, Xs_near):
        # given n points in real world, calculate local normal using LeastSquares

        # centering
        Xs_center = torch.mean(Xs_near, dim=1).unsqueeze(dim=1)
        Xs_near = Xs_near - Xs_center

        pts_xy, pts_z = torch.split(Xs_near, [2, 1], dim=-1)

        ls = LeastSquares()
        w = ls.lstq(pts_xy, pts_z)
        w = w.squeeze(-1)
        w1, w2 = torch.split(w, [1, 1], dim=-1)
        norm_near = torch.cat([-w1, -w2, torch.ones_like(w1)], dim=-1)
        norm_near = norm_near / torch.norm(norm_near, dim=-1, keepdim=True)

        return norm_near

    def smooth(self, dD_near):
        N = dD_near.size()[0]

        # dD of the 3x3 neighborhood in run_nerfrac.py's NEIGHBOR_OFFSETS order: 0 is
        # the center, 2 and 7 lie above and below it, 4 and 5 left and right
        dx = torch.abs((dD_near[:, 5] - dD_near[:, 4]) / 2)
        dy = torch.abs((dD_near[:, 7] - dD_near[:, 2]) / 2)

        loss = F.smooth_l1_loss(dx + dy, torch.zeros_like(dx + dy))

        loss = loss.repeat(N)

        return loss

    def ndc2cam(self, ray, o_ndc, W=None, H=None, f=None):
        t1 = -(W * o_ndc[:, 0]) / (2 * f)
        t2 = -(H * o_ndc[:, 1]) / (2 * f)
        t3 = o_ndc[:, 2] - 1

        t1 = t1.unsqueeze(1)
        t2 = t2.unsqueeze(1)
        t3 = t3.unsqueeze(1)

        ndc_dz = -t3
        ndc_ray = ray * ndc_dz / (ray[:, 2].unsqueeze(1))
        dxz = -(W * ndc_ray[:, 0].unsqueeze(1)) / (2 * f) + t1
        dyz = -(H * ndc_ray[:, 1].unsqueeze(1)) / (2 * f) + t2

        z = -torch.ones_like(dxz)
        x = z * dxz
        y = z * dyz
        d_cam = torch.cat([x, y, z], dim=-1)
        d_cam = d_cam / torch.norm(d_cam, dim=-1, keepdim=True)

        return d_cam

    def cam2ndc(self, ray, o_ndc, W=None, H=None, f=None):
        t1 = -(W * o_ndc[:, 0]) / (2 * f)
        t2 = -(H * o_ndc[:, 1]) / (2 * f)
        t3 = o_ndc[:, 2] - 1

        t1 = t1.unsqueeze(1)
        t2 = t2.unsqueeze(1)
        t3 = t3.unsqueeze(1)

        dx = ray[:, 0].unsqueeze(1)
        dy = ray[:, 1].unsqueeze(1)
        dz = ray[:, 2].unsqueeze(1)

        ndc_dx = -(2 * f * (dx / dz - t1)) / W
        ndc_dy = -(2 * f * (dy / dz - t2)) / H
        ndc_dz = -t3

        d_ndc = torch.cat([ndc_dx, ndc_dy, ndc_dz], dim=-1)

        return d_ndc

    def ndcp2camp(self, points, W=None, H=None, f=None):
        z = 2 / (points[:, 2] - 1)
        y = -(H * points[:, 1] * z) / (2 * f)
        x = -(W * points[:, 0] * z) / (2 * f)
        z = z.unsqueeze(1)
        y = y.unsqueeze(1)
        x = x.unsqueeze(1)

        points_camera = torch.cat([x, y, z], dim=-1)

        return points_camera

    def getweight(self, raw, z_vals, rays_d):
        raw2alpha = lambda raw, dists, act_fn=F.relu: 1.0 - torch.exp(
            -act_fn(raw) * dists
        )

        dists = z_vals[..., 1:] - z_vals[..., :-1]
        infinity = torch.tensor([1e10], device=z_vals.device).expand(
            dists[..., :1].shape
        )
        dists = torch.cat([dists, infinity], -1)

        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        alpha = raw2alpha(raw[..., 3], dists)  # [N_rays, N_samples]
        weights = (
            alpha
            * torch.cumprod(
                torch.cat(
                    [
                        torch.ones((alpha.shape[0], 1), device=alpha.device),
                        1.0 - alpha + 1e-10,
                    ],
                    -1,
                ),
                -1,
            )[:, :-1]
        )

        return weights

    def finesampling(self, Xs, sample_ray, z_vals, weights):
        N_rays = Xs.shape[0]

        z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(
            z_vals_mid, weights[..., 1:-1], self.N_importance, det=True
        )
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = Xs[..., None, :] + sample_ray[..., None, :] * z_vals[..., :, None]

        pts_flat = torch.reshape(pts, [-1, pts.shape[-1]])

        sample_ray = sample_ray / torch.norm(sample_ray, dim=-1, keepdim=True)

        input_dirs = sample_ray[:, None].expand(
            [N_rays, self.N_samples + self.N_importance, 3]
        )
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])

        return pts_flat, input_dirs_flat, z_vals

    def _forward_water(self, x_water):
        """The refractive path. The 9 rays of each bundle hit the learned surface, a
        plane fitted to the 9 hits gives the normal, and the center ray is refracted
        at its hit and sampled below the surface."""
        N_w = x_water.shape[0]
        input_views_ndc, input_o_ndc = torch.split(x_water, [3, 3], dim=-1)

        input_views_ndc = torch.reshape(
            input_views_ndc, [-1, input_views_ndc.shape[-1]]
        )
        input_o_ndc = torch.reshape(input_o_ndc, [-1, input_o_ndc.shape[-1]])

        ox, oy, oz = torch.split(input_o_ndc, [1, 1, 1], dim=-1)
        virx, viry, virz = torch.split(input_views_ndc, [1, 1, 1], dim=-1)

        # the origins lie on the near plane (constant oz), so only (ox, oy) is encoded
        input_o = torch.cat([ox, oy], dim=-1)
        refrac_input_o = self.embedo_fn(input_o)
        refrac_input_views = self.embedrefracv_fn(input_views_ndc)

        dD = self.query_refrac(
            refrac_input_views, refrac_input_o, self._refrac, self._refrac_out
        )
        dD = torch.clamp(dD, -1.0, 1.0)  # prevent physically unreasonable displacements
        d = (oz - self.initdepth) / (-virz)
        # calculate surface points Xs
        Xs_ndc = input_o_ndc + (d + dD) * input_views_ndc

        # back to real world for normal calculation
        Xs_real = self.ndcp2camp(Xs_ndc, self.Wimg, self.Himg, self.f)
        Xs_real = Xs_real.view([-1, 9, 3])

        # normal calculation
        norm_real = self.get_norm(Xs_real)

        input_views_ndc = input_views_ndc.view([-1, 9, 3])
        input_views_ndc = input_views_ndc[:, 0, :]
        Xs_ndc = Xs_ndc.view([-1, 9, 3])
        Xs_ndc = Xs_ndc[:, 0, :]

        input_view_real = self.ndc2cam(
            input_views_ndc, Xs_ndc, self.Wimg, self.Himg, self.f
        )

        # vector form of Snell's law; the normal points back towards the camera, so
        # c1 = cos(theta_i) > 0
        c1 = -torch.sum(norm_real * input_view_real, dim=-1)
        mid = torch.ones_like(c1) - (1 / self.yita) * (1 / self.yita) * (
            torch.ones_like(c1) - c1 * c1
        )
        mid = torch.clamp(mid, min=1e-6)
        c2 = torch.sqrt(mid)
        c1 = c1.unsqueeze(-1)
        c2 = c2.unsqueeze(-1)
        f2 = (1 / self.yita) * (c1 * norm_real + input_view_real) - c2 * norm_real
        f2 = f2 / torch.norm(f2, dim=-1, keepdim=True)

        refracted_view_ndc = self.cam2ndc(f2, Xs_ndc, self.Wimg, self.Himg, self.f)
        refracted_view_ndc = refracted_view_ndc / torch.norm(
            refracted_view_ndc, dim=-1, keepdim=True
        )

        input_pts, view_flat, z_vals, sample_ray = self.sampling(
            Xs_ndc, refracted_view_ndc
        )

        input_pts = self.embed_fn(input_pts)
        view_flat = self.embeddirs_fn(view_flat)
        out = self._occ(torch.cat([input_pts, view_flat], dim=-1))
        out = torch.reshape(out, [N_w] + [self.N_samples] + [out.shape[-1]])

        # fine sampling
        weights = self.getweight(out, z_vals, sample_ray)
        input_pts_fine, view_flat_fine, z_vals_fine = self.finesampling(
            Xs_ndc, sample_ray, z_vals, weights
        )
        input_pts_fine = self.embed_fn(input_pts_fine)
        view_flat_fine = self.embeddirs_fn(view_flat_fine)
        out_fine = self._occfine(torch.cat([input_pts_fine, view_flat_fine], dim=-1))

        return out_fine, Xs_ndc, z_vals_fine, sample_ray, dD

    def _forward_land(self, x_land):
        """The land path. The center ray of each bundle is sampled straight from the
        near plane, without refraction."""
        N_land = x_land.shape[0]
        input_views_ndc, input_o_ndc = torch.split(x_land, [3, 3], dim=-1)

        views_center = input_views_ndc[:, 0, :]  # [N_land, 3]
        origins_center = input_o_ndc[:, 0, :]  # [N_land, 3]

        # no surface displacement, one value per neighbor as in the water path
        dD = torch.zeros(N_land * 9, 1, device=x_land.device)

        input_pts, view_flat, z_vals, sample_ray = self.sampling_direct(
            origins_center, views_center
        )

        input_pts = self.embed_fn(input_pts)
        view_flat = self.embeddirs_fn(view_flat)
        out = self._occ(torch.cat([input_pts, view_flat], dim=-1))
        out = torch.reshape(out, [N_land] + [self.N_samples] + [out.shape[-1]])

        # fine sampling
        weights = self.getweight(out, z_vals, sample_ray)
        input_pts_fine, view_flat_fine, z_vals_fine = self.finesampling(
            origins_center, sample_ray, z_vals, weights
        )
        input_pts_fine = self.embed_fn(input_pts_fine)
        view_flat_fine = self.embeddirs_fn(view_flat_fine)
        out_fine = self._occfine(torch.cat([input_pts_fine, view_flat_fine], dim=-1))

        # a land ray has no surface point, so its Xs is the origin on the near plane
        Xs_ndc = origins_center

        return out_fine, Xs_ndc, z_vals_fine, sample_ray, dD

    def forward(self, x, water_mask):
        """Water rays (water_mask > 0.5) take the refractive path, the others the land
        path.

        x is [N_ray, 9, 6]: the NDC view direction and origin of each ray and its 3x3
        neighbors. Returns, in ray order, the fine network output [N_ray * S, C], Xs
        [N_ray, 3] (the origin for land rays), z_vals [N_ray, S], the scaled sample
        rays [N_ray, 3] and dD [N_ray * 9, 1] (0 for land rays), with
        S = N_samples + N_importance.
        """
        N_ray = x.shape[0]
        water_bool = water_mask > 0.5
        N_water = water_bool.sum().item()

        if N_water == N_ray:
            return self._forward_water(x)
        if N_water == 0:
            return self._forward_land(x)

        water_idx = torch.where(water_bool)[0]
        land_idx = torch.where(~water_bool)[0]
        water_out = self._forward_water(x[water_idx])
        land_out = self._forward_land(x[land_idx])

        # both results back into ray order
        S = self.N_samples + self.N_importance
        out_ch = water_out[0].shape[-1]
        merged = [
            torch.empty(N_ray, S, out_ch, device=x.device),
            torch.empty(N_ray, 3, device=x.device),
            torch.empty(N_ray, S, device=x.device),
            torch.empty(N_ray, 3, device=x.device),
            torch.empty(N_ray, 9, 1, device=x.device),
        ]
        for out, water, land in zip(merged, water_out, land_out):
            out[water_idx] = water.view(-1, *out.shape[1:])
            out[land_idx] = land.view(-1, *out.shape[1:])
        out_fine, Xs_ndc, z_vals_fine, sample_ray, dD = merged

        return (
            out_fine.view(-1, out_ch),
            Xs_ndc,
            z_vals_fine,
            sample_ray,
            dD.view(-1, 1),
        )


class NeRFOriginal(nn.Module):
    def __init__(
        self,
        D=8,
        W=256,
        input_ch=3,
        output_ch=4,
        skips=[4],
        input_ch_views=3,
        use_viewdirs=False,
    ):
        super(NeRFOriginal, self).__init__()
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        layers = [nn.Linear(input_ch, W)]
        for i in range(D - 1):
            layer = nn.Linear
            in_channels = W
            if i in self.skips:
                in_channels += input_ch

            layers += [layer(in_channels, W)]

        self.pts_linears = nn.ModuleList(layers)

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(
            x, [self.input_ch, self.input_ch_views], dim=-1
        )
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs


# Ray helpers
def get_rays(H, W, K, c2w):
    device = c2w.device
    i, j = torch.meshgrid(
        torch.linspace(0, W - 1, W, device=device),
        torch.linspace(0, H - 1, H, device=device),
        indexing="ij",
    )  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack(
        [(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], -1
    )
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(
        dirs[..., np.newaxis, :] * c2w[:3, :3], -1
    )  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing="xy"
    )
    dirs = np.stack(
        [(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -np.ones_like(i)], -1
    )
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(
        dirs[..., np.newaxis, :] * c2w[:3, :3], -1
    )  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))

    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1.0 / (W / (2.0 * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1.0 / (H / (2.0 * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1.0 + 2.0 * near / rays_o[..., 2]

    d0 = (
        -1.0
        / (W / (2.0 * focal))
        * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    )
    d1 = (
        -1.0
        / (H / (2.0 * focal))
        * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    )
    d2 = -2.0 * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False):
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    device = weights.device
    if det:
        u = torch.linspace(0.0, 1.0, steps=N_samples, device=device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples
