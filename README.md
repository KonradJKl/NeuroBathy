# NeuroBathy project page

This is the `gh-pages` branch of [NeuroBathy](https://github.com/KonradJKl/NeuroBathy).
GitHub Pages serves it as it is at https://konradjkl.github.io/NeuroBathy/, and
`.nojekyll` turns Jekyll off.

| Path | Content |
|---|---|
| `index.html`, `assets/style.css` | The page |
| `assets/viewer.js` | The 3D view (three.js), reads `data/` |
| `data/` | Point cloud of the best model and the 138 survey points |
| `figures/` | Depth map, LiDAR comparison and the NeRFrac diagram |
| `build/` | The scripts that write `data/` and the two maps |

Browsers do not load `data/` from a `file://` page, so serve the folder to look at it
locally:

```bash
python -m http.server 8000
```

and open http://localhost:8000.

## Rebuilding the data and the figures

The data and the two maps come from the outputs of NeuroBathy v1.0 for the best model
(NeRFrac, 9 oblique Pielach images, distortion loss, 256 samples). Train and render it,
then run 02 and 03 as in "Reproducing the best model" in the README on `main`. In its
`postprocessing/` folder:

1. `07_make_pointcloud.py` with `--no_align` writes the scene point cloud.
2. `15_lidar_comparison.py` compares the depth grid of 02 with the bathymetric LiDAR
   point cloud (folder 03 of the dataset).
3. `build/make_viewer_data.py` thins the point cloud and writes `data/`.
4. `build/make_figures.py` writes `figures/depth_map.png` and
   `figures/lidar_comparison.png`.

The calls are at the top of both build scripts. `figures/nerfrac_original_vs_modified.png`
is a copy of `docs/figures/nerfrac_original_vs_modified.png` on `main`.

## Data format

`data/scene.json` holds the CRS (EPSG:25833), the `origin` and `step` of the integer
coordinates, the file and point count of each level, `depth_max`, the name of the survey
file, the attribution and the license. Level 0 is an even preview of the cloud, level 1
holds the other points.

Each `.bin` file is little-endian and planar: `count * 3` uint16 positions (x, y, z of
each point, position = origin + q * step), then `count * 3` uint8 colours, then `count`
uint8 depth codes. Code 0 is land or water off the WSM; codes 1-255 map linearly to 0 to
`depth_max` m below the WSM.

`data/survey_points.json` lists the survey points with `name`, `kind` (`target` or
`transect`), `x`, `y` and `z` (m from the origin, z is the riverbed), `depth_gt` and
`depth_pred` (m, `null` without a prediction).

## Licenses

- The page and the build scripts (`index.html`, `assets/`, `build/`) are MIT licensed,
  like the code on `main`, see its [`LICENSE`](https://github.com/KonradJKl/NeuroBathy/blob/main/LICENSE).
- `data/`, `figures/depth_map.png` and `figures/lidar_comparison.png` are derived from
  the [Pielach River research dataset October 2024](https://doi.org/10.48436/taz19-r6618)
  by G. Mandlburger, TU Wien, licensed
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/): reconstructed, thinned and
  re-plotted. They are licensed CC BY 4.0 as well.
- `figures/nerfrac_original_vs_modified.png` comes from `main` and keeps its license
  there (MIT).
- The page loads three.js 0.170.0 and mermaid 11.4.1 from jsDelivr. Both are MIT
  licensed.
