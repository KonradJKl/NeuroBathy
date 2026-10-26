"""
Make the two result figures of the project page from the outputs of NeuroBathy v1.0
for the best model (NeRFrac, 9 oblique Pielach images, distortion loss, 256 samples).

figures/depth_map.png is the depth grid of 02_extract_bathymetry.py with the 138
survey points of 03_validate.py, coloured by their error. figures/lidar_comparison.png
shows the predicted depth, the LiDAR depth and their difference from
15_lidar_comparison.py. The maps are in metres east and north of a round point in
UTM 33N (EPSG:25833).

Usage example:
    # in NeuroBathy/postprocessing, after 02 and 03 of "Reproducing the best model"
    # in its README
    python 15_lidar_comparison.py \\
        --pred_csv bathymetry/pielach_oblique.csv \\
        --lidar_laz /path/to/03_LiDAR_topobathy_classified/Pielach_20241024_topoBathy_LiDAR.laz \\
        --wsm_path ../data/gt/pielach/05_water_surface_model/Pielach_20241024_WSM.tif \\
        --label "NeRFrac oblique" \\
        --output_dir validation_lidar/pielach_oblique

    # in the root of the page
    python build/make_figures.py \\
        --grid_csv /path/to/NeuroBathy/postprocessing/bathymetry/pielach_oblique.csv \\
        --comparison_csv /path/to/NeuroBathy/postprocessing/validation/pielach_oblique/comparison.csv \\
        --lidar_dir /path/to/NeuroBathy/postprocessing/validation_lidar/pielach_oblique
"""

import argparse
import io
import json
import os

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator

ORIGIN_E, ORIGIN_N = 527600.0, 5340250.0  # local origin of the maps, UTM 33N
X_LABEL = "Metres east of 527 600 E (UTM 33N)"
Y_LABEL = "Metres north of 5 340 250 N"
RESOLUTION = 0.1  # grid cell size of 02 and 15, m
DEPTH_MAX = 2.0  # the depth colours run from 0 to DEPTH_MAX, m
ERROR_MAX = 1.0  # the error colours run from -ERROR_MAX to +ERROR_MAX, m
DEPTH_CMAP = "viridis"
ERROR_CMAP = "coolwarm"

# the PNGs are WIDTH px wide; layout sizes below are in px of the PNG
WIDTH = 1800
DPI = 200
INK = "#1e1e1e"
INK_SECONDARY = "#505050"
FRAME = "#aaaaaa"

plt.rcParams.update(
    {
        "font.size": 15,
        "axes.titlesize": 17,
        "axes.labelsize": 16,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 15,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.edgecolor": FRAME,
        "axes.linewidth": 0.8,
        "xtick.color": FRAME,
        "ytick.color": FRAME,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
    }
)


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="Make the result figures of the project page."
    )
    parser.add_argument(
        "--grid_csv",
        type=str,
        required=True,
        help="gridded bathymetry from 02_extract_bathymetry.py",
    )
    parser.add_argument(
        "--comparison_csv",
        type=str,
        required=True,
        help="comparison.csv from 03_validate.py",
    )
    parser.add_argument(
        "--lidar_dir",
        type=str,
        required=True,
        help="output folder of 15_lidar_comparison.py (underwater mode) with "
        "lidar_comparison.csv and lidar_stats.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(os.path.dirname(here), "figures"),
        help="output folder (default: figures/ of the page)",
    )
    return parser.parse_args()


def signed(value, digits):
    """Number with its sign and a real minus sign, 0 without a sign."""
    if round(value, digits) == 0:
        return f"{0:.{digits}f}"
    return f"{value:+.{digits}f}".replace("-", "\u2212")


def to_raster(e, n, values):
    """Image of the grid cells with centres e, n (north up) and its extent in local
    metres."""
    col = np.rint((e - e.min()) / RESOLUTION).astype(np.int64)
    row = np.rint((n - n.min()) / RESOLUTION).astype(np.int64)
    image = np.full((row.max() + 1, col.max() + 1), np.nan)
    image[row, col] = values
    x0 = e.min() - RESOLUTION / 2 - ORIGIN_E
    y0 = n.min() - RESOLUTION / 2 - ORIGIN_N
    extent = (
        x0, x0 + image.shape[1] * RESOLUTION, y0, y0 + image.shape[0] * RESOLUTION
    )
    return image, extent


def map_limits(extent, margin=1.5):
    x0, x1, y0, y1 = extent
    return (x0 - margin, x1 + margin), (y0 - margin, y1 + margin)


def add_axes(fig, left, bottom, width, height):
    """Axes at a box in px, measured from the lower left corner of the figure."""
    W, H = fig.get_size_inches() * DPI
    return fig.add_axes([left / W, bottom / H, width / W, height / H])


def draw_map(ax, image, extent, xlim, ylim, **style):
    shown = ax.imshow(
        image, extent=extent, origin="lower", interpolation="nearest", **style
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_major_locator(MultipleLocator(20))
    return shown


def depth_colorbar(fig, mappable, cax, label, **kwargs):
    cbar = fig.colorbar(mappable, cax=cax, extend="max", **kwargs)
    cbar.set_ticks(np.arange(0, DEPTH_MAX + 0.01, 0.5))
    cbar.formatter = FuncFormatter(lambda v, _: f"{v:.1f}")
    return style_colorbar(cbar, label)


def error_colorbar(fig, mappable, cax, label, extend, **kwargs):
    cbar = fig.colorbar(mappable, cax=cax, extend=extend, **kwargs)
    cbar.set_ticks(np.linspace(-ERROR_MAX, ERROR_MAX, 5))
    cbar.formatter = FuncFormatter(lambda v, _: signed(v, 1))
    return style_colorbar(cbar, label)


def style_colorbar(cbar, label):
    cbar.update_ticks()
    cbar.set_label(label)
    cbar.outline.set_edgecolor(FRAME)
    cbar.outline.set_linewidth(0.8)
    cbar.ax.tick_params(color=FRAME)
    return cbar


def palette():
    """256 colours: greys for text and lines, and samples of both colour maps."""
    greys = np.repeat(np.arange(0, 256, 5)[:, None] / 255, 3, axis=1)
    depth = plt.get_cmap(DEPTH_CMAP)(np.linspace(0, 1, 102))[:, :3]
    error = plt.get_cmap(ERROR_CMAP)(np.linspace(0, 1, 102))[:, :3]
    return np.rint(np.concatenate([greys, depth, error]) * 255).astype(np.int64)


def nearest_colors(rgb, colors):
    """Index of the nearest palette colour for every pixel."""
    pixels = rgb.reshape(-1, 3).astype(np.int64)
    codes, inverse = np.unique(
        pixels[:, 0] << 16 | pixels[:, 1] << 8 | pixels[:, 2], return_inverse=True
    )
    unique = np.stack([codes >> 16, codes >> 8 & 255, codes & 255], axis=1)
    nearest = np.empty(len(unique), dtype=np.uint8)
    for start in range(0, len(unique), 4096):
        diff = unique[start : start + 4096, None, :] - colors[None]
        nearest[start : start + 4096] = (diff**2).sum(axis=2).argmin(axis=1)
    return nearest[inverse.ravel()].reshape(rgb.shape[:2])


def save_png(fig, path):
    """Save as an 8-bit PNG with a fixed palette, so the file stays small and the
    colour bars keep smooth ramps."""
    buffer = io.BytesIO()
    fig.savefig(buffer, dpi=DPI, facecolor="white")
    plt.close(fig)
    rgb = np.asarray(Image.open(buffer).convert("RGB"))
    colors = palette()
    image = Image.fromarray(nearest_colors(rgb, colors))
    image.putpalette(colors.astype(np.uint8).ravel().tolist())
    image.save(path, optimize=True)
    size = os.path.getsize(path) / 1024
    print(f"Saved: {path} ({image.width} x {image.height} px, {size:.0f} KB)")


def plot_depth_map(grid, points, path):
    image, extent = to_raster(
        grid["utm_easting"].values, grid["utm_northing"].values, grid["depth_m"].values
    )
    xlim, ylim = map_limits(extent)
    left, right, top = 170, 40, 30
    axis_h, gap, bar_block, bottom = 145, 20, 175, 10
    map_w = WIDTH - left - right
    map_h = round(map_w * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0]))
    height = top + map_h + axis_h + gap + bar_block + bottom
    fig = plt.figure(figsize=(WIDTH / DPI, height / DPI))

    ax = add_axes(fig, left, height - top - map_h, map_w, map_h)
    depth = draw_map(
        ax, image, extent, xlim, ylim, cmap=DEPTH_CMAP, vmin=0, vmax=DEPTH_MAX
    )
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)

    x = points["gt_easting"].values - ORIGIN_E
    y = points["gt_northing"].values - ORIGIN_N
    error = points["error_m"].values
    target = points["type"].values == "reference_target"
    style = dict(
        cmap=ERROR_CMAP,
        vmin=-ERROR_MAX,
        vmax=ERROR_MAX,
        edgecolors="black",
        linewidths=0.7,
        zorder=3,
    )
    errors = ax.scatter(x[~target], y[~target], c=error[~target], s=55, **style)
    ax.scatter(x[target], y[target], c=error[target], s=85, marker="s", **style)

    grey = plt.get_cmap(ERROR_CMAP)(0.5)
    handles = [
        Line2D([], [], ls="", marker="o", ms=8, mfc=grey, mec="black", mew=0.7),
        Line2D([], [], ls="", marker="s", ms=9, mfc=grey, mec="black", mew=0.7),
    ]
    labels = [
        f"Cross-section points ({(~target).sum()})",
        f"Underwater reference targets ({target.sum()})",
    ]
    ax.legend(
        handles, labels, loc="upper left", bbox_to_anchor=(0.17, 1.0), frameon=False
    )

    bar_w, bar_gap, bar_h = 660, 180, 34
    bar_x = left + (map_w - 2 * bar_w - bar_gap) / 2
    bar_y = bottom + 75
    depth_ax = add_axes(fig, bar_x, bar_y, bar_w, bar_h)
    error_ax = add_axes(fig, bar_x + bar_w + bar_gap, bar_y, bar_w, bar_h)
    depth_colorbar(
        fig, depth, depth_ax, "Predicted depth (m)", orientation="horizontal"
    )
    error_colorbar(
        fig,
        errors,
        error_ax,
        "Predicted − surveyed (m)",
        "neither",
        orientation="horizontal",
    )
    for bar_ax in (depth_ax, error_ax):
        bar_ax.xaxis.set_label_position("top")
    save_png(fig, path)


def plot_lidar_comparison(lidar, stats, path):
    e, n = lidar["utm_easting"].values, lidar["utm_northing"].values
    error_title = (
        f"Predicted − LiDAR: RMSE {stats['rmse']:.3f} m, "
        f"bias {signed(stats['bias'], 3)} m"
    )
    panels = [
        ("pred_depth", "Predicted depth"),
        ("lidar_depth", "LiDAR depth"),
        ("error_m", error_title),
    ]
    rasters = [to_raster(e, n, lidar[column].values) for column, _ in panels]
    xlim, ylim = map_limits(rasters[0][1])

    left, right, top = 170, 280, 10
    title_h, panel_gap, axis_h, bottom = 95, 15, 145, 10
    map_w = WIDTH - left - right
    map_h = round(map_w * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0]))
    height = top + 3 * (title_h + map_h) + 2 * panel_gap + axis_h + bottom
    fig = plt.figure(figsize=(WIDTH / DPI, height / DPI))

    axes, shown = [], []
    for i, ((image, extent), (_, title)) in enumerate(zip(rasters, panels)):
        map_y = height - top - (i + 1) * (title_h + map_h) - i * panel_gap
        ax = add_axes(fig, left, map_y, map_w, map_h)
        if i < 2:
            style = dict(cmap=DEPTH_CMAP, vmin=0, vmax=DEPTH_MAX)
        else:
            style = dict(cmap=ERROR_CMAP, vmin=-ERROR_MAX, vmax=ERROR_MAX)
        shown.append(draw_map(ax, image, extent, xlim, ylim, **style))
        ax.set_title(title, loc="left", pad=10)
        ax.tick_params(labelbottom=i == 2)
        axes.append(ax)
    axes[2].set_xlabel(X_LABEL)
    y_middle = (axes[0].get_position().y1 + axes[2].get_position().y0) / 2
    fig.text(
        20 / WIDTH,
        y_middle,
        Y_LABEL,
        rotation=90,
        va="center",
        fontsize=plt.rcParams["axes.labelsize"],
    )

    bar_x, bar_w = (left + map_w + 40) / WIDTH, 36 / WIDTH
    top_box, middle_box = axes[0].get_position(), axes[1].get_position()
    depth_ax = fig.add_axes([bar_x, middle_box.y0, bar_w, top_box.y1 - middle_box.y0])
    error_box = axes[2].get_position()
    error_ax = fig.add_axes([bar_x, error_box.y0, bar_w, error_box.height])
    depth_colorbar(fig, shown[0], depth_ax, "Depth below the water surface (m)")
    error_colorbar(fig, shown[2], error_ax, "Predicted − LiDAR (m)", "both")
    save_png(fig, path)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    grid = pd.read_csv(args.grid_csv)
    points = pd.read_csv(args.comparison_csv)
    points = points[points["matched"] == "yes"]
    print(f"Grid: {len(grid):,} cells, {len(points)} matched survey points")
    plot_depth_map(grid, points, os.path.join(args.output_dir, "depth_map.png"))

    lidar = pd.read_csv(os.path.join(args.lidar_dir, "lidar_comparison.csv"))
    with open(os.path.join(args.lidar_dir, "lidar_stats.json")) as f:
        stats = json.load(f)
    print(
        f"LiDAR: {len(lidar):,} cells, RMSE {stats['rmse']:.3f} m, "
        f"bias {stats['bias']:+.3f} m"
    )
    plot_lidar_comparison(
        lidar, stats, os.path.join(args.output_dir, "lidar_comparison.png")
    )


if __name__ == "__main__":
    main()
