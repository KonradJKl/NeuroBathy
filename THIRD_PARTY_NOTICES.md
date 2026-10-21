# Third-party code and data

NeuroBathy's own code is MIT-licensed (`LICENSE`). The parts below come from
other projects and keep their own licenses and notices.

| Component | Where | License |
|---|---|---|
| [NeRFrac](https://github.com/Yifever20002/NeRFrac) (Zhan et al., ICCV 2023), commit `5e2e848`, modified | `models/nerfrac_modified/` | MIT, `models/nerfrac_modified/LICENSE` |
| [NeRF](https://github.com/bmild/nerf) and [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch), through NeRFrac | rendering core and data loaders in `models/nerfrac_modified/` | MIT, Copyright (c) 2020 bmild, `models/nerfrac_modified/LICENSE` |
| [IBRNet](https://github.com/googleinterns/IBRNet) utilities, through NeRFrac, modified | `models/nerfrac_modified/utils.py` | Apache-2.0, `models/nerfrac_modified/LICENSE-APACHE` |
| [NeRF++](https://github.com/Kai-46/nerfplusplus) constants and image/colorbar helpers, through IBRNet and NeRFrac | `models/nerfrac_modified/utils.py` | BSD-2-Clause, Copyright (c) 2020, the NeRF++ authors, `models/nerfrac_modified/LICENSE-BSD-NERFPP` |
| [LLFF](https://github.com/Fyusion/LLFF) data loading, through NeRF and nerf-pytorch | `models/nerfrac_modified/load_llff.py` | shipped under MIT by NeRF and nerf-pytorch (LLFF itself is GPL-3.0) |
| [COLMAP](https://github.com/colmap/colmap) model reader, adapted | `postprocessing/colmap_io.py` | BSD-3-Clause, notice in the file header |

Not included, installed separately:

- [VGGT](https://github.com/facebookresearch/vggt): VGGT License. The default
  VGGT-1B weights are CC BY-NC 4.0 (non-commercial).
- [SAM 3](https://github.com/facebookresearch/sam3): SAM License, gated
  checkpoint on Hugging Face.
- [COLMAP](https://colmap.github.io/) and LLFF's `imgs2poses.py`, used to
  compute the camera poses.

Data:

- `data/gt/pielach/`: Pielach River dataset, G. Mandlburger, TU Wien,
  CC BY 4.0. See `data/gt/pielach/README.md`.
