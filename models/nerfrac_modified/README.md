# Modified NeRFrac

This folder is a modified copy of
[NeRFrac: Neural Radiance Fields through Refractive Surface](https://github.com/Yifever20002/NeRFrac)
(Zhan, Nobuhara, Nishino, Zheng, ICCV 2023), based on upstream commit `5e2e848`.
The rendering core and the data loaders come from NeRF / nerf-pytorch through NeRFrac.

Changes made for shallow-water bathymetry from drone imagery:

- Per-ray water/land routing: water rays go through the refractive field and
  Snell's law, land rays are rendered along straight lines. Both share the
  radiance MLP and are scattered back into one batch (`run_nerfrac_helpers.py`,
  `NeRFrac.forward`).
- Water masks are loaded with the images (`load_llff.py`) and follow each ray
  through batching and rendering.
- Additional losses: distortion loss (Mip-NeRF 360), cross-view depth
  consistency and an optional water-surface flatness term. The refractive-field
  smoothness term that upstream defines but never calls (`NeRFrac.smooth`) is
  applied to water rays (`--smooth_weight`, default 0.01, with warmup).
- Training safeguards: gradient clipping (`--grad_clip`, default 1.0), batches
  with a NaN or Inf loss are skipped, the refractive offset is clamped to
  [-1, 1] and the Snell's-law radicand is clamped (only an index below 1 could
  make it negative; from air into water it never is).
- Expected depth, mode depth (sample with the highest weight) and accumulated
  opacity are saved for every rendered view; upstream saved only the RGB image
  and the surface points.
- `c2w.npy`, which upstream writes to the working directory, is saved into each
  experiment folder next to a new `llff_scale.npy`, so the postprocessing can
  map renders back to COLMAP and UTM coordinates.
- Vectorized 3x3 neighbour-ray gathering (border pixels use edge replication
  instead of upstream's shifted window), `solve_triangular` for the surface
  normal, registered buffers for H/W/f, and TF32 / `torch.compile` settings.
- Fixed squeeze dimensions for chunks that contain a single water ray.
- SSIM computed with OpenCV instead of scikit-image (`utils.py`).

The original README, demo scenes and synthetic configs are in the upstream
repository. The upstream configs set `llffhold` and `check_weight`, which this
copy no longer accepts, so delete those two lines before using them. Please
cite NeRFrac if you use this code:

```
@inproceedings{zhan2023nerfrac,
  title={Nerfrac: Neural radiance fields through refractive surface},
  author={Zhan, Yifan and Nobuhara, Shohei and Nishino, Ko and Zheng, Yinqiang},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={18402--18412},
  year={2023}
}
```

License: MIT (`LICENSE`). `utils.py` is Apache-2.0 (`LICENSE-APACHE`) and
contains BSD-2-Clause code from NeRF++ (`LICENSE-BSD-NERFPP`).
