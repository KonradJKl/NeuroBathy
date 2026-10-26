# Pielach ground truth

Unmodified copies of folders 05-08 of the Pielach River dataset, extracted from
the published archives. Each folder keeps its original `readme.txt`.

| Folder | Content |
|---|---|
| `05_water_surface_model/` | Water surface model, GeoTIFF, 0.5 m grid |
| `06_control_points/` | Ground control points: checkerboard targets and points on saddle-roof control surfaces (total station) |
| `07_underwater_reference_targets/` | Underwater checkerboard targets (total station) |
| `08_underwater_transect_points/` | River cross-section points (total station) |

All coordinates are ETRS89 / UTM 33N (EPSG:25833) with ellipsoidal heights.

**Source:** G. Mandlburger, *Pielach River research dataset October 2024 - Mapping
shallow inland running waters with UAV-borne photo and laser bathymetry*,
TU Wien Research Data, v1.0.0, 2025. https://doi.org/10.48436/taz19-r6618

**License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). These files
are not covered by the MIT license of the code in this repository.

The drone images (folder 09) and the classified topo-bathymetric LiDAR point cloud
(folder 03) used by `postprocessing/15_lidar_comparison.py` are not included; download
them from the DOI above. The dataset is described in Mandlburger et al. (2025),
https://doi.org/10.23784/HN130-06.
