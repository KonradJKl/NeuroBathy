# Georeferencing transforms

Umeyama similarity transforms (scale, rotation, translation) from each model frame
to ETRS89 / UTM 33N (EPSG:25833), computed with `postprocessing/01_compute_umeyama.py`
from the reconstructed camera centres and the GPS positions in the EXIF data of the
Pielach images. `02_extract_bathymetry.py`, `07_make_pointcloud.py` and
`08_water_surface_analysis.py` read them with `--umeyama_npz`.

| File | Model frame | Cameras | Scale | Horizontal RMS (m) | Z RMS (m) |
|---|---|---|---|---|---|
| `nerfrac_nadir_umeyama.npz` | COLMAP, nadir | 9 | 6.007 | 0.23 | 0.32 |
| `nerfrac_oblique_umeyama.npz` | COLMAP, oblique | 9 | 5.759 | 0.24 | 0.04 |
| `nerfrac_50m_umeyama.npz` | COLMAP, 50 m | 280 | 10.071 | 0.14 | 0.06 |
| `nerfrac_nadir_rotated_umeyama.npz` | COLMAP, nadir with 3 images rotated by 180° | 9 | 5.983 | 0.22 | 0.32 |
| `nerfrac_perfect_v2_umeyama.npz` | COLMAP, 17-image "perfect" subset | 17 | 22.719 | 0.25 | 0.13 |
| `vggt_nadir_umeyama.npz` | VGGT, nadir | 9 | 73.687 | 0.97 | 0.40 |
| `vggt_oblique_umeyama.npz` | VGGT, oblique | 9 | 70.310 | 18.32 | 2.48 |
| `vggt_50m_v1_umeyama.npz` | VGGT, 50 m | 280 | 76.553 | 22.13 | 5.67 |

The RMS values are those of the horizontal and the vertical residuals of the camera
fit. The large residuals of the VGGT oblique and 50 m transforms come from VGGT's camera
estimates for these image sets.

Derived from the Pielach River dataset (G. Mandlburger, TU Wien, CC BY 4.0,
https://doi.org/10.48436/taz19-r6618), see `data/gt/pielach/README.md`. These files are
licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) and are not covered
by the MIT license of the code. The `vggt_*` files are computed from VGGT-1B camera
estimates. The VGGT-1B weights are licensed CC BY-NC 4.0.
