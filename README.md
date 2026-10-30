# NeuroBathy project page

This is the `gh-pages` branch of [NeuroBathy](https://github.com/KonradJKl/NeuroBathy).
GitHub Pages serves it as it is at https://konradjkl.github.io/NeuroBathy/, and
`.nojekyll` turns Jekyll off.

| Path | Content |
|---|---|
| `index.html`, `assets/style.css` | The page |
| `assets/viewer.js` | The 3D view (three.js), reads `data/` |
| `data/` | Point clouds, depth grids with their errors against the LiDAR and survey points of two models, and the LiDAR riverbed |
| `figures/` | Depth map, LiDAR comparison and the NeRFrac diagram |
| `build/` | The scripts that write `data/` and the two maps |

Browsers do not load `data/` from a `file://` page, so serve the folder to look at it
locally:

```bash
python -m http.server 8000
```

and open http://localhost:8000.

## Rebuilding the data and the figures

The data and the two maps come from the outputs of NeuroBathy v1.0. The 3D view shows
two NeRFrac models, both with distortion loss and 256 samples: the best model on the 9
oblique Pielach images (`oblique9`, the first row of the results table) and the model on
all 280 images within 50 m of a centre point (`all280`, the fourth row).
`build/make_viewer_data.py` writes the files of one model and runs once for each; both
runs write the same LiDAR riverbed. The paths below use `gt=../data/gt/pielach` as in
the README on `main`.

### The best model

Train and render it, then run 02 and 03 as in "Reproducing the best model" in the README
on `main`. In its `postprocessing/` folder:

1. `07_make_pointcloud.py` with `--no_align` writes the scene point cloud.
2. `15_lidar_comparison.py` compares the depth grid of 02 with the bathymetric LiDAR
   point cloud (folder 03 of the dataset).
3. `build/make_viewer_data.py` thins the point cloud, adds the depth grid of 02 with its
   errors from 15, grids the river bottom of the LiDAR point cloud and writes the files
   of the model to `data/`.
4. `build/make_figures.py` writes `figures/depth_map.png` and
   `figures/lidar_comparison.png`.

The calls are at the top of both build scripts. Step 3:

```bash
python /path/to/gh-pages/build/make_viewer_data.py --model oblique9 \
    --pointcloud_dir pointclouds/pielach_oblique \
    --grid_csv bathymetry/pielach_oblique.csv \
    --comparison_csv validation/pielach_oblique/comparison.csv \
    --lidar_dir validation_lidar/pielach_oblique \
    --wsm_path $gt/05_water_surface_model/Pielach_20241024_WSM.tif \
    --lidar_laz /path/to/03_LiDAR_topobathy_classified/Pielach_20241024_topoBathy_LiDAR.laz \
    --postprocessing_dir . \
    --output_dir /path/to/gh-pages/data
```

### The model on 280 images

The renders, camera poses and water masks of this model are not published, so its data
can only be rebuilt from them where they exist. These are the calls, in the
`postprocessing/` folder, with `scene` the folder of the 280 images (`poses_bounds.npy`
and `masks_8/`) and `log` the experiment folder (`c2w.npy`, `llff_scale.npy` and the
renders in `testset_950000/`):

```bash
lidar=/path/to/03_LiDAR_topobathy_classified/Pielach_20241024_topoBathy_LiDAR.laz

python 02_extract_bathymetry.py --mode nerfrac \
    --testset_dir $log/testset_950000 \
    --log_folder $log \
    --data_dir $scene \
    --umeyama_npz ../data/transforms/nerfrac_50m_umeyama.npz \
    --wsm_path $gt/05_water_surface_model/Pielach_20241024_WSM.tif \
    --mask_dir $scene/masks_8 \
    --output_csv bathymetry/pielach_all280.csv

python 03_validate.py \
    --pred_csv bathymetry/pielach_all280.csv \
    --ref_csv $gt/07_underwater_reference_targets/Pielach_20241024_underwater_reference_targets.csv \
    --transect_csv $gt/08_underwater_transect_points/Pielach_20241024_underwater_transect_points.csv \
    --wsm_path $gt/05_water_surface_model/Pielach_20241024_WSM.tif \
    --label "NeRFrac all280" \
    --output_dir validation/pielach_all280

python 15_lidar_comparison.py \
    --pred_csv bathymetry/pielach_all280.csv \
    --lidar_laz "$lidar" \
    --wsm_path $gt/05_water_surface_model/Pielach_20241024_WSM.tif \
    --label "NeRFrac all280" \
    --output_dir validation_lidar/pielach_all280

python /path/to/gh-pages/build/stream_cloud.py \
    --log_folder $log --testset testset_950000 \
    --poses_bounds $scene/poses_bounds.npy --factor 8 \
    --umeyama_npz ../data/transforms/nerfrac_50m_umeyama.npz \
    --mask_dir $scene/masks_8 --postprocessing_dir . \
    --voxel_size 0.4 --output_dir pointclouds/pielach_all280

python /path/to/gh-pages/build/make_viewer_data.py --model all280 \
    --pointcloud_dir pointclouds/pielach_all280 --thinned \
    --preview_voxel_size 0.8 \
    --grid_csv bathymetry/pielach_all280.csv \
    --comparison_csv validation/pielach_all280/comparison.csv \
    --lidar_dir validation_lidar/pielach_all280 \
    --wsm_path $gt/05_water_surface_model/Pielach_20241024_WSM.tif \
    --lidar_laz "$lidar" \
    --postprocessing_dir . \
    --output_dir /path/to/gh-pages/data
```

02 writes 291,866 cells. 03 gives an RMSE of 0.40 m on the targets and 0.58 m on all
points, 15 gives 0.49 m against the LiDAR, the numbers of the table. The data of the
page come from outputs of 02 and 03 with these SHA-256 checksums:

| File | SHA-256 |
|---|---|
| `bathymetry/pielach_all280.csv` | `b6d8cf4789096fef3216f3f36c53f3474731e36f3cd75d63906217687b1b57e5` |
| `validation/pielach_all280/comparison.csv` | `cb558eca569ebc9e3bdb3efcac15682426685935fc7691642de16f66aa94d922` |

07 holds all views in memory at once, and the 280 views do not fit. `stream_cloud.py`
builds the same cloud (07 with `--mask_dir --no_align` and its default mode depth) one
view at a time and thins it as `make_viewer_data.py` does, here with 0.4 m voxels, and
`--thinned` keeps all its points. On the best model, `stream_cloud.py` with
`--voxel_size 0.23` gives exactly the points of 07 thinned by `make_viewer_data.py`.

`figures/nerfrac_original_vs_modified.png` is a copy of
`docs/figures/nerfrac_original_vs_modified.png` on `main`.

## Data format

`data/scene.json` holds the CRS (EPSG:25833), `depth_max`, the models, the LiDAR
riverbed (`lidar`), the attribution and the license. `models` lists the best model
(`oblique9`) and the model on 280 images (`all280`). The first one opens the page, and
the viewer places the objects of each model at its `origin` minus the `origin` of the
first one. Each model has its `id` and `label`, `home` (what its home view frames:
`river`, its water points on the WSM, or `cloud`, its whole cloud), the `origin` and
`step` of the integer coordinates of its levels, the file and point count of each level,
its flag file (`replaced`), its depth grid (`grid`), the errors of the grid against the
LiDAR (`error`, with the RMSE of 15) and its survey file. Level 0 is an even preview of
the cloud, level 1 holds the other points. The cell layers, the grids and `lidar`, each
have a `file`, a `count`, their own `origin` and `step`, and a `cell_size` (0.1 m).

| | `oblique9` | `all280` |
|---|---|---|
| Levels | `scene_0.bin`, `scene_1.bin` | `all280_scene_0.bin`, `all280_scene_1.bin` |
| Flags | `oblique9_replaced.bin` | `all280_replaced.bin` |
| Depth grid | `oblique9_grid.bin` | `all280_grid.bin` |
| Errors | `oblique9_error.bin` | `all280_error.bin` |
| Survey points | `survey_points.json` | `all280_survey_points.json` |

The level files are little-endian and planar: `count * 3` uint16 positions (x, y, z of
each point, position = origin + q * step), then `count * 3` uint8 colours, then `count`
uint8 depth codes. Code 0 is land or water off the WSM; codes 1-255 map linearly to 0 to
`depth_max` m below the WSM, deeper and shallower points clipped.

The cell files are little-endian and planar as well: `count * 3` uint16 positions, then
`count` uint16 depths below the WSM in mm, not clipped at `depth_max`. A grid file holds
every cell of the 02 grid of its model at its `utm_elevation` with its `depth_m`.
`lidar_bed.bin` holds the 0.1 m cells, counted from the lower-left corner of the LiDAR
tile (points on its east and north edges form one more column and row), that contain
river bottom points (class 40): each at the cell centre and the median elevation of its
points, with the depth below the WSM at the cell centre. As in
`15_lidar_comparison.py`, cells off the WSM or with a depth of 0 or less are left out.

An error file holds one little-endian int16 per cell of the grid of its model, in the
order of the grid file: the predicted minus the LiDAR depth in mm (`error_m` of the
`lidar_comparison.csv` of 15), or `-32768` for the cells that 15 does not compare, which
have no river bottom points of the LiDAR or a LiDAR or predicted depth of 0 or less. The
Error view draws the cells of the grid in these colours, the cells without a value in
grey.

A flag file holds one uint8 per point of the levels of its model, in their order. Bit 0
is set for the water points in a 0.1 m cell of the grid, bit 1 for those in a cell of
the LiDAR. The Depth and Error views hide the points with bit 0 and the LiDAR view those
with bit 1, and each draws its cells in their place.

A survey file lists the survey points with `name`, `kind` (`target` or `transect`), `x`,
`y` and `z` (m from the origin of the levels of its model, z is the riverbed),
`depth_gt` and `depth_pred` of the model (m, `null` without a prediction).

## Licenses

- The page and the build scripts (`index.html`, `assets/`, `build/`) are MIT licensed,
  like the code on `main`, see its [`LICENSE`](https://github.com/KonradJKl/NeuroBathy/blob/main/LICENSE).
- `data/`, `figures/depth_map.png` and `figures/lidar_comparison.png` are derived from
  the [Pielach River research dataset October 2024](https://doi.org/10.48436/taz19-r6618)
  by G. Mandlburger, TU Wien, licensed
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/): reconstructed, thinned,
  gridded and re-plotted. They are licensed CC BY 4.0 as well.
- `figures/nerfrac_original_vs_modified.png` comes from `main` and keeps its license
  there (MIT).
- The page loads three.js 0.170.0 and mermaid 11.4.1 from jsDelivr. Both are MIT
  licensed.
