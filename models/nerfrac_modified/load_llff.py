"""
LLFF scene loading: a slightly modified version of the LLFF data loading code (see
https://github.com/Fyusion/LLFF for the original), as NeRF and NeRFrac use it.

NeuroBathy additions: the water masks of masks_<factor>/ are loaded with the images,
grayscale images are read as RGB, and load_llff_data also returns the recentering
transform, which run_nerfrac.py saves as c2w.npy for the georeferencing.
"""

import os
import shutil
import subprocess

import imageio
import numpy as np

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def _image_files(imgdir):
    """The image file names of a folder, sorted, without hidden files."""
    return [
        f
        for f in sorted(os.listdir(imgdir))
        if not f.startswith(".")
        and f.lower().endswith(IMAGE_EXTENSIONS)
        and os.path.isfile(os.path.join(imgdir, f))
    ]


def _minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, "images_{}".format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, "images_{}x{}".format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return

    imgdir = os.path.join(basedir, "images")
    imgs = _image_files(imgdir)
    if not imgs:
        raise FileNotFoundError(f"No images in {imgdir}")
    imgdir_orig = imgdir

    for r in factors + resolutions:
        if isinstance(r, int):
            name = "images_{}".format(r)
            resizearg = "{}%".format(100.0 / r)
        else:
            name = "images_{}x{}".format(r[1], r[0])
            resizearg = "{}x{}".format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue

        print("Minifying", r, basedir)

        # built without a shell (paths with spaces) in a temporary folder that is
        # renamed when done, so that a failed run leaves no half-built folder
        tmpdir = imgdir + ".tmp"
        shutil.rmtree(tmpdir, ignore_errors=True)
        os.makedirs(tmpdir)
        try:
            for f in imgs:
                shutil.copy(os.path.join(imgdir_orig, f), tmpdir)
            print(f"mogrify -resize {resizearg} -format png ({len(imgs)} images)")
            subprocess.run(
                ["mogrify", "-resize", resizearg, "-format", "png", *imgs],
                cwd=tmpdir,
                check=True,
            )
            for f in imgs:
                if not f.endswith(".png"):
                    os.remove(os.path.join(tmpdir, f))
            os.replace(tmpdir, imgdir)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
        print("Done")


def _load_data(basedir, factor=None, width=None, height=None):
    poses_arr = np.load(os.path.join(basedir, "poses_bounds.npy"))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    bds = poses_arr[:, -2:].transpose([1, 0])

    def full_res_shape():
        imgdir = os.path.join(basedir, "images")
        return imageio.v2.imread(os.path.join(imgdir, _image_files(imgdir)[0])).shape

    sfx = ""

    if factor is not None:
        sfx = "_{}".format(factor)
        _minify(basedir, factors=[factor])
    elif height is not None:
        sh = full_res_shape()
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = "_{}x{}".format(width, height)
    elif width is not None:
        sh = full_res_shape()
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = "_{}x{}".format(width, height)
    else:
        factor = 1

    imgdir = os.path.join(basedir, "images" + sfx)
    if not os.path.exists(imgdir):
        raise FileNotFoundError(f"{imgdir} does not exist")

    imgfiles = [os.path.join(imgdir, f) for f in _image_files(imgdir)]
    if poses.shape[-1] != len(imgfiles):
        raise ValueError(
            f"{len(imgfiles)} images in {imgdir} but {poses.shape[-1]} poses in "
            "poses_bounds.npy"
        )

    sh = imageio.v2.imread(imgfiles[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1.0 / factor

    imgs = []
    for f in imgfiles:
        img = imageio.v2.imread(f)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.ndim == 3 and img.shape[-1] == 1:
            img = np.concatenate([img, img, img], axis=-1)
        elif img.ndim == 3 and img.shape[-1] == 4:
            img = img[..., :3]
        imgs.append(img)
    imgs = np.stack(imgs, -1) / 255.0
    return poses, bds, imgs


def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def poses_avg(poses):
    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

    return c2w


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.0])
    hwf = c2w[:, 4:5]

    for theta in np.linspace(0.0, 2.0 * np.pi * rots, N + 1)[:-1]:
        c = np.dot(
            c2w[:3, :4],
            np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0])
            * rads,
        )
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.0])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses


def recenter_poses(poses):
    """The poses relative to their average pose, and that average pose (c2w).

    The new z axis points against the mean viewing direction (the cameras look along
    -z), not up, so a level water surface is tilted in the NeRF frame by the angle
    between the mean viewing direction and the nadir.
    """
    poses_ = poses + 0
    bottom = np.reshape([0, 0, 0, 1.0], [1, 4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_
    return poses, c2w[:3, :4]


def spherify_poses(poses, bds):
    p34_to_44 = lambda p: np.concatenate(
        [p, np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])], 1
    )

    rays_d = poses[:, :3, 2:3]
    rays_o = poses[:, :3, 3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(
            -np.linalg.inv((np.transpose(A_i, [0, 2, 1]) @ A_i).mean(0)) @ (b_i).mean(0)
        )
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)

    center = pt_mindist
    up = (poses[:, :3, 3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([0.1, 0.2, 0.3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:, :3, :4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))

    sc = 1.0 / rad
    poses_reset[:, :3, 3] *= sc
    bds *= sc
    rad *= sc

    centroid = np.mean(poses_reset[:, :3, 3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad**2 - zh**2)
    new_poses = []

    for th in np.linspace(0.0, 2.0 * np.pi, 120):
        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0, 0, -1.0])

        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        new_poses.append(p)

    new_poses = np.stack(new_poses, 0)

    new_poses = np.concatenate(
        [new_poses, np.broadcast_to(poses[0, :3, -1:], new_poses[:, :3, -1:].shape)], -1
    )
    poses_reset = np.concatenate(
        [
            poses_reset[:, :3, :4],
            np.broadcast_to(poses[0, :3, -1:], poses_reset[:, :3, -1:].shape),
        ],
        -1,
    )

    return poses_reset, new_poses, bds


def _load_masks(basedir, factor=None):
    """Water masks of masks_<factor>/ (or masks/) as float32 0/1, [H, W, N], or None
    without that folder."""
    maskdir = os.path.join(basedir, "masks" if factor is None else f"masks_{factor}")
    if not os.path.exists(maskdir):
        return None
    maskfiles = [
        os.path.join(maskdir, f)
        for f in sorted(os.listdir(maskdir))
        if f.endswith("png") and not f.startswith(".")
    ]
    if not maskfiles:
        raise FileNotFoundError(f"No PNG masks in {maskdir}")

    def imread_mask(f):
        mask = imageio.v2.imread(f)
        if mask.ndim == 3:
            mask = mask[..., 0]
        # 0/255 masks are thresholded, 0/1 masks taken as they are
        if mask.max() > 1:
            return (mask > 127).astype(np.float32)
        return mask.astype(np.float32)

    return np.stack([imread_mask(f) for f in maskfiles], axis=-1)


def load_llff_data(
    basedir, factor=8, recenter=True, bd_factor=0.75, spherify=False, load_masks=True
):
    """Images [N, H, W, 3], masks [H, W, N] (None without a mask folder), poses
    [N, 3, 5] (rotation, position, then H, W, focal), depth bounds [N, 2], the render
    poses (a spiral, or a circle with spherify), the index of the view nearest to the
    mean pose, and the recentering transform c2w [3, 4] (None without recenter).

    The scene is scaled so that the smallest depth bound is 1 / bd_factor.
    """
    poses, bds, imgs = _load_data(basedir, factor=factor)
    print(f"Loaded {basedir}, depth bounds {bds.min():.2f} to {bds.max():.2f}")

    masks = _load_masks(basedir, factor=factor) if load_masks else None
    if masks is not None:
        if masks.shape[-1] != imgs.shape[-1]:
            raise ValueError(
                f"{masks.shape[-1]} masks for {imgs.shape[-1]} images in {basedir}"
            )
        if masks.shape[:2] != imgs.shape[:2]:
            raise ValueError(
                f"Masks are {masks.shape[1]}x{masks.shape[0]} px but images are "
                f"{imgs.shape[1]}x{imgs.shape[0]} px in {basedir}"
            )

    # Correct rotation matrix ordering and move variable dim to axis 0
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    images = np.moveaxis(imgs, -1, 0).astype(np.float32)
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)

    # Rescale if bd_factor is provided
    sc = 1.0 if bd_factor is None else 1.0 / (bds.min() * bd_factor)
    poses[:, :3, 3] *= sc
    bds *= sc

    c2w_recenter = None
    if recenter:
        poses, c2w_recenter = recenter_poses(poses)

    if spherify:
        poses, render_poses, bds = spherify_poses(poses, bds)
    else:
        c2w = poses_avg(poses)

        # spiral around the mean pose, with half the 90th percentile of the camera
        # offsets as its amplitude per axis
        up = normalize(poses[:, :3, 1].sum(0))
        close_depth = bds.min() * 0.9
        zdelta = close_depth * 0.2
        tt = poses[:, :3, 3]
        rads = np.percentile(np.abs(tt), 90, 0)
        N_views = 120
        N_rots = 2
        focal = 1
        rads /= 2
        render_poses = render_path_spiral(
            c2w, up, rads, focal, zdelta, zrate=0.5, rots=N_rots, N=N_views
        )

    render_poses = np.array(render_poses).astype(np.float32)

    c2w = poses_avg(poses)
    dists = np.sum(np.square(c2w[:3, 3] - poses[:, :3, 3]), -1)
    i_test = np.argmin(dists)

    # spherify_poses returns float64 poses
    poses = poses.astype(np.float32)
    return images, masks, poses, bds, render_poses, i_test, c2w_recenter
