# Camera poses

`pielach_oblique/poses_bounds.npy` holds the LLFF camera poses and depth bounds of the 9
oblique Pielach images used by `models/nerfrac_modified/configs/pielach_oblique.txt`.
LLFF's `imgs2poses.py` wrote it from the COLMAP model of my thesis, the same model that
`data/transforms/nerfrac_oblique_umeyama.npz` maps to UTM 33N. A new COLMAP run has its
own coordinate frame and scale, so these poses are needed to reproduce the thesis model
with the shipped transform.

Derived from the Pielach River dataset (G. Mandlburger, TU Wien, CC BY 4.0,
https://doi.org/10.48436/taz19-r6618), see `data/gt/pielach/README.md`. This file is
licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) and is not covered
by the MIT license of the code.
