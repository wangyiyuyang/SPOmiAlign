#!/usr/bin/env python3
"""Align one h5ad to another, with pre-alignment SSI rendering and timing."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import scanpy as sc
from PIL import Image
from scipy.ndimage import gaussian_filter

try:
    from skimage.metrics import structural_similarity
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "scikit-image is required for SSI/SSIM computation. "
        "Please install scikit-image before running this script."
    ) from exc


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
for import_root in (PROJECT_ROOT, PACKAGE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from data_preprocessing import rasterize_h5ad_to_image, scatter_h5ad_to_image  # noqa: E402
from roma import align_and_process_images  # noqa: E402


DEFAULT_COORDINATE_COLUMN_CANDIDATES = [
    ("Raw_Slideseq_X", "Raw_Slideseq_Y"),
    ("x", "y"),
    ("X", "Y"),
    ("coord_x", "coord_y"),
    ("spatial_x", "spatial_y"),
    ("NisslTiffX", "NisslTiffY"),
    ("CCF_X", "CCF_Y"),
]


def add_render_args(parser: argparse.ArgumentParser, prefix: str, label: str) -> None:
    parser.add_argument(
        f"--{prefix}-render-mode",
        choices=["scatter", "raster"],
        default="scatter",
        help=f"Rendering backend for {label}.",
    )
    parser.add_argument(
        f"--{prefix}-background",
        choices=["white", "black"],
        default="white",
        help=f"Background color for {label} rendering.",
    )
    parser.add_argument(
        f"--{prefix}-display-long-side",
        type=int,
        default=2200,
        help=f"Rendered long side (pixels) for {label}. Use 0 to keep native extent.",
    )
    parser.add_argument(
        f"--{prefix}-padding",
        type=int,
        default=0,
        help=f"Rendered padding (pixels) for {label}.",
    )
    parser.add_argument(
        f"--{prefix}-point-shape",
        choices=["circle", "square"],
        default="square",
        help=f"Point shape for {label} rendering.",
    )
    parser.add_argument(
        f"--{prefix}-point-radius",
        type=int,
        default=5,
        help=f"Point radius for {label} rendering.",
    )
    parser.add_argument(
        f"--{prefix}-intensity-mode",
        choices=["X_sum", "obs_col"],
        default="X_sum",
        help=f"Intensity source for {label} raster rendering.",
    )
    parser.add_argument(
        f"--{prefix}-intensity-obs-col",
        default=None,
        help=f"obs column used when {label} intensity mode is obs_col.",
    )
    parser.add_argument(
        f"--{prefix}-intensity-log-transform",
        action="store_true",
        help=f"Apply log1p to intensity before rendering {label}.",
    )
    parser.add_argument(
        f"--{prefix}-threshold-percentile",
        type=float,
        default=None,
        help=f"Optional percentile threshold for {label} rendering.",
    )
    parser.add_argument(
        f"--{prefix}-enhance",
        action="store_true",
        help=f"Enable enhancement pipeline for {label} raster rendering.",
    )
    parser.add_argument(
        f"--{prefix}-marker-alpha",
        type=float,
        default=0.9,
        help=f"Marker alpha for {label} rendering.",
    )
    parser.add_argument(
        f"--{prefix}-gray-min",
        type=float,
        default=0.35,
        help=f"Minimum grayscale tone for {label} rendering.",
    )
    parser.add_argument(
        f"--{prefix}-gray-max",
        type=float,
        default=0.88,
        help=f"Maximum grayscale tone for {label} rendering.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render SSI maps for two h5ad files, align source h5ad to target h5ad, "
            "then export the same alignment outputs as align_sm_to_st_square.py."
        )
    )
    parser.add_argument(
        "--target-h5ad",
        type=Path,
        required=True,
        help="Target/reference h5ad path (fixed image).",
    )
    parser.add_argument(
        "--source-h5ad",
        type=Path,
        required=True,
        help="Source h5ad path to align (moving image).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "h5ad_to_h5ad_square",
        help="Root directory where sample outputs are written.",
    )
    parser.add_argument(
        "--sample-id",
        default=None,
        help="Optional output subfolder name. Defaults to <target>__vs__<source>.",
    )
    parser.add_argument(
        "--method",
        default="affine+bspline",
        choices=["Rigid", "Affine", "Homography", "bspline", "affine+bspline"],
        help="Alignment method passed to align_and_process_images.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device for alignment, e.g. 'cuda:0', 'cuda:1', or 'cpu'. Defaults to cuda:0 if available.",
    )
    parser.add_argument(
        "--target-rotate",
        type=float,
        default=0.0,
        help="Clockwise pre-rotation used when rendering the target h5ad.",
    )
    parser.add_argument(
        "--source-rotate",
        type=float,
        default=-90.0,
        help="Clockwise pre-rotation used when rendering the source h5ad.",
    )
    parser.add_argument(
        "--target-flip-horizontal",
        action="store_true",
        help="Horizontally flip the rendered target h5ad before alignment.",
    )
    parser.add_argument(
        "--source-flip-horizontal",
        action="store_true",
        help="Horizontally flip the rendered source h5ad before alignment.",
    )
    parser.add_argument(
        "--target-spatial-key",
        default="spatial",
        help="Target coordinate key in obsm when not using obs columns.",
    )
    parser.add_argument(
        "--source-spatial-key",
        default="spatial",
        help="Source coordinate key in obsm when not using obs columns.",
    )
    parser.add_argument(
        "--target-x-obs-col",
        default=None,
        help="Target x-coordinate obs column. If set, target obs columns are used instead of obsm.",
    )
    parser.add_argument(
        "--target-y-obs-col",
        default=None,
        help="Target y-coordinate obs column. If set, target obs columns are used instead of obsm.",
    )
    parser.add_argument(
        "--source-x-obs-col",
        default=None,
        help="Source x-coordinate obs column. If set, source obs columns are used instead of obsm.",
    )
    parser.add_argument(
        "--source-y-obs-col",
        default=None,
        help="Source y-coordinate obs column. If set, source obs columns are used instead of obsm.",
    )
    add_render_args(parser, "target", "target/ST")
    add_render_args(parser, "source", "source/SM")
    parser.add_argument(
        "--reference-blur-sigma",
        type=float,
        default=6.0,
        help="Gaussian blur sigma used to build the SSI self-reference image.",
    )
    parser.add_argument(
        "--ssim-sigma",
        type=float,
        default=1.5,
        help="Gaussian sigma used by skimage.metrics.structural_similarity.",
    )
    return parser.parse_args()


def ensure_h5ad(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Missing {label} h5ad: {resolved}")
    return resolved


def default_sample_id(target_h5ad: Path, source_h5ad: Path) -> str:
    return f"{target_h5ad.stem}__vs__{source_h5ad.stem}"


def get_xy_from_adata(adata, coord_config: dict) -> np.ndarray:
    mode = str(coord_config["mode"])
    if mode == "obs":
        x_col = str(coord_config["x_obs_col"])
        y_col = str(coord_config["y_obs_col"])
        xy = np.column_stack(
            [
                np.asarray(adata.obs[x_col], dtype=np.float64),
                np.asarray(adata.obs[y_col], dtype=np.float64),
            ]
        )
    else:
        spatial_key = str(coord_config["spatial_key"])
        if spatial_key not in adata.obsm:
            raise KeyError(f"adata does not contain obsm['{spatial_key}'].")
        xy = np.asarray(adata.obsm[spatial_key], dtype=np.float64)

    if xy.ndim != 2 or xy.shape[1] < 2:
        raise ValueError(f"Coordinate array must have shape (N, 2+), got {xy.shape}.")
    return xy[:, :2]


def finite_xy_mask(xy: np.ndarray) -> np.ndarray:
    arr = np.asarray(xy, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Coordinate array must have shape (N, 2+), got {arr.shape}.")
    return np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1])


def sanitize_dataframe_for_h5ad_write(df, *, name: str) -> None:
    if "_index" in df.columns:
        print(f"[WARN] Removing reserved '_index' column from {name} before temp h5ad write.")
        df.drop(columns=["_index"], inplace=True)
    if df.index.name == "_index":
        print(f"[WARN] Clearing reserved index name '_index' from {name} before temp h5ad write.")
        df.index.name = None


def sanitize_adata_for_h5ad_write(adata) -> None:
    sanitize_dataframe_for_h5ad_write(adata.obs, name="adata.obs")
    sanitize_dataframe_for_h5ad_write(adata.var, name="adata.var")

    if adata.raw is not None:
        try:
            raw_adata = adata.raw.to_adata()
            sanitize_dataframe_for_h5ad_write(raw_adata.var, name="adata.raw.var")
            adata.raw = raw_adata
        except Exception as exc:
            print(f"[WARN] Could not sanitize adata.raw before temp h5ad write: {exc}")


def resolve_coordinate_config(
    *,
    h5ad_path: Path,
    label: str,
    spatial_key: str,
    x_obs_col: str | None,
    y_obs_col: str | None,
) -> dict:
    adata = sc.read_h5ad(str(h5ad_path))
    if int(adata.n_obs) == 0:
        raise ValueError(f"{label}: {h5ad_path} contains 0 observations and cannot be aligned.")

    if (x_obs_col is None) ^ (y_obs_col is None):
        raise ValueError(f"{label}: x/y obs columns must be provided together.")

    if x_obs_col is not None and y_obs_col is not None:
        missing = [col for col in [x_obs_col, y_obs_col] if col not in adata.obs.columns]
        if missing:
            raise KeyError(f"{label}: missing obs coordinate column(s): {missing}")
        return {
            "mode": "obs",
            "spatial_key": None,
            "x_obs_col": str(x_obs_col),
            "y_obs_col": str(y_obs_col),
        }

    if spatial_key in adata.obsm:
        xy = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
        if xy.ndim == 2 and xy.shape[1] >= 2:
            return {
                "mode": "obsm",
                "spatial_key": str(spatial_key),
                "x_obs_col": None,
                "y_obs_col": None,
            }

    for cand_x, cand_y in DEFAULT_COORDINATE_COLUMN_CANDIDATES:
        if cand_x in adata.obs.columns and cand_y in adata.obs.columns:
            print(f"[OK] {label}: auto-detected obs coordinate columns: {cand_x}, {cand_y}")
            return {
                "mode": "obs",
                "spatial_key": None,
                "x_obs_col": cand_x,
                "y_obs_col": cand_y,
            }

    raise KeyError(
        f"{label}: no usable coordinates found. "
        f"Tried obsm['{spatial_key}'] and known obs column pairs."
    )


def read_spatial_df(h5ad_path: Path, coord_config: dict) -> pd.DataFrame:
    adata = sc.read_h5ad(str(h5ad_path))
    xy = get_xy_from_adata(adata, coord_config)

    if "id" in adata.obs.columns:
        spot_id = adata.obs["id"].astype(str).to_numpy()
    else:
        spot_id = adata.obs_names.astype(str).to_numpy()

    return pd.DataFrame({"spot_id": spot_id, "x": xy[:, 0], "y": xy[:, 1]})


def build_render_meta(
    *,
    input_h5ad: Path,
    coord_config: dict,
    rotate: float,
    scale: float,
    flip_horizontal: bool,
    display_long_side: int,
    padding: int,
    rotate_origin: str = "data",
) -> dict:
    adata = sc.read_h5ad(str(input_h5ad))
    xy = get_xy_from_adata(adata, coord_config)
    keep = finite_xy_mask(xy)
    if not np.any(keep):
        raise ValueError(f"{input_h5ad} has no finite coordinate points.")
    xy_valid = np.asarray(xy[keep], dtype=np.float64)
    x = xy_valid[:, 0]
    y = xy_valid[:, 1]
    if x.size == 0:
        raise ValueError(f"{input_h5ad} has no coordinate points.")

    pts = np.column_stack([x, y]).astype(np.float64)
    if rotate_origin == "data":
        origin = pts.mean(axis=0)
    elif rotate_origin == "center":
        origin = np.array([(np.nanmin(x) + np.nanmax(x)) / 2.0, (np.nanmin(y) + np.nanmax(y)) / 2.0], dtype=np.float64)
    elif rotate_origin == "zero":
        origin = np.array([0.0, 0.0], dtype=np.float64)
    else:
        raise ValueError("rotate_origin must be 'data'|'center'|'zero'")

    x_rot = x.copy()
    y_rot = y.copy()
    if abs(float(rotate)) > 1e-12 or abs(float(scale) - 1.0) > 1e-12:
        th = np.deg2rad(-float(rotate))
        c, s = np.cos(th), np.sin(th)
        matrix = float(scale) * np.array([[c, s], [-s, c]], dtype=np.float64)
        out = (pts - origin.reshape(1, 2)) @ matrix.T + origin.reshape(1, 2)
        x_rot = out[:, 0]
        y_rot = out[:, 1]

    flip_axis_x = None
    if bool(flip_horizontal):
        flip_axis_x = float(np.nanmin(x_rot) + np.nanmax(x_rot))
        x_rot = flip_axis_x - x_rot

    use_display_scaling = int(display_long_side) > 0
    fit_scale = 1.0
    min_x = float(np.nanmin(x_rot))
    min_y = float(np.nanmin(y_rot))

    if use_display_scaling:
        x_shift = x_rot - min_x
        y_shift = y_rot - min_y
        long_side = max(float(np.nanmax(x_shift)), float(np.nanmax(y_shift)), 1.0)
        usable_long_side = max(float(int(display_long_side) - 2 * max(int(padding), 0) - 1), 1.0)
        fit_scale = usable_long_side / long_side
        x_draw = x_shift * fit_scale + float(max(int(padding), 0))
        y_draw = y_shift * fit_scale + float(max(int(padding), 0))
        width = int(np.ceil(np.nanmax(x_draw))) + max(int(padding), 0) + 1
        height = int(np.ceil(np.nanmax(y_draw))) + max(int(padding), 0) + 1
    else:
        width = int(np.ceil(np.nanmax(x_rot))) + 1
        height = int(np.ceil(np.nanmax(y_rot))) + 1

    return {
        "rotate_deg": float(rotate),
        "scale": float(scale),
        "rotation_origin": [float(origin[0]), float(origin[1])],
        "rotate_origin_mode": str(rotate_origin),
        "flip_horizontal": bool(flip_horizontal),
        "flip_axis_x": None if flip_axis_x is None else float(flip_axis_x),
        "use_display_scaling": bool(use_display_scaling),
        "display_long_side": int(display_long_side),
        "padding": int(padding),
        "fit_scale": float(fit_scale),
        "min_x": float(min_x),
        "min_y": float(min_y),
        "render_width": int(width),
        "render_height": int(height),
    }


def render_style_from_args(args: argparse.Namespace, prefix: str) -> dict:
    style = {
        "render_mode": str(getattr(args, f"{prefix}_render_mode")),
        "background": str(getattr(args, f"{prefix}_background")),
        "display_long_side": int(getattr(args, f"{prefix}_display_long_side")),
        "padding": int(getattr(args, f"{prefix}_padding")),
        "point_shape": str(getattr(args, f"{prefix}_point_shape")),
        "point_radius": int(getattr(args, f"{prefix}_point_radius")),
        "intensity_mode": str(getattr(args, f"{prefix}_intensity_mode")),
        "intensity_obs_col": getattr(args, f"{prefix}_intensity_obs_col"),
        "intensity_log_transform": bool(getattr(args, f"{prefix}_intensity_log_transform")),
        "threshold_percentile": getattr(args, f"{prefix}_threshold_percentile"),
        "enhance": bool(getattr(args, f"{prefix}_enhance")),
        "marker_alpha": float(getattr(args, f"{prefix}_marker_alpha")),
        "gray_min": float(getattr(args, f"{prefix}_gray_min")),
        "gray_max": float(getattr(args, f"{prefix}_gray_max")),
        "dpi": 100,
    }
    if style["intensity_mode"] == "obs_col" and not style["intensity_obs_col"]:
        raise ValueError(f"{prefix}: --{prefix}-intensity-obs-col is required when intensity mode is obs_col.")
    return style


def map_points_forward_with_meta(points_xy: np.ndarray, render_meta: dict) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float64).copy()
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError(f"points_xy must be shape (N,2+), got {pts.shape}")

    origin = np.asarray(render_meta.get("rotation_origin", [0.0, 0.0]), dtype=np.float64)
    rotate_deg = float(render_meta.get("rotate_deg", 0.0))
    scale_val = float(render_meta.get("scale", 1.0))
    if abs(rotate_deg) > 1e-12 or abs(scale_val - 1.0) > 1e-12:
        th = np.deg2rad(-rotate_deg)
        c, s = np.cos(th), np.sin(th)
        matrix = scale_val * np.array([[c, s], [-s, c]], dtype=np.float64)
        pts = (pts - origin.reshape(1, 2)) @ matrix.T + origin.reshape(1, 2)

    if bool(render_meta.get("flip_horizontal", False)):
        flip_axis_x = render_meta.get("flip_axis_x", None)
        if flip_axis_x is None:
            raise ValueError("render_meta is missing flip_axis_x while flip_horizontal=True")
        pts[:, 0] = float(flip_axis_x) - pts[:, 0]

    if bool(render_meta.get("use_display_scaling", False)):
        fit_scale = float(render_meta.get("fit_scale", 1.0))
        padding = float(render_meta.get("padding", 0.0))
        min_x = float(render_meta.get("min_x", 0.0))
        min_y = float(render_meta.get("min_y", 0.0))
        pts[:, 0] = (pts[:, 0] - min_x) * fit_scale + padding
        pts[:, 1] = (pts[:, 1] - min_y) * fit_scale + padding

    return pts


def render_transformed_source_in_target_render_space(
    *,
    transformed_h5ad: Path,
    transformed_coord_config: dict,
    output_png: Path,
    target_render_meta: dict,
    source_render_style: dict,
) -> None:
    adata = sc.read_h5ad(str(transformed_h5ad))
    xy = get_xy_from_adata(adata, transformed_coord_config)
    xy_render = map_points_forward_with_meta(xy[:, :2], target_render_meta)
    adata.obsm["spatial"] = xy_render
    sanitize_adata_for_h5ad_write(adata)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        tmp_h5ad = Path(td) / "tmp_transformed_render_space.h5ad"
        adata.write_h5ad(str(tmp_h5ad))
        scatter_h5ad_to_image(
            input_h5ad=str(tmp_h5ad),
            output_png=str(output_png),
            background=str(source_render_style["background"]),
            point_shape=str(source_render_style["point_shape"]),
            radius=int(source_render_style["point_radius"]),
            threshold_percentile=source_render_style["threshold_percentile"],
            intensity_log_transform=bool(source_render_style["intensity_log_transform"]),
            rotate=0.0,
            scale=1.0,
            dpi=int(source_render_style.get("dpi", 100)),
            marker_alpha=float(source_render_style["marker_alpha"]),
            gray_min=float(source_render_style["gray_min"]),
            gray_max=float(source_render_style["gray_max"]),
            display_long_side=0,
            padding=0,
            canvas_size=(
                int(target_render_meta["render_width"]),
                int(target_render_meta["render_height"]),
            ),
        ) if str(source_render_style["render_mode"]) == "scatter" else rasterize_h5ad_to_image(
            input_h5ad=str(tmp_h5ad),
            output_png=str(output_png),
            spatial_key="spatial",
            intensity_mode=str(source_render_style["intensity_mode"]),
            intensity_obs_col=source_render_style["intensity_obs_col"],
            intensity_log_transform=bool(source_render_style["intensity_log_transform"]),
            threshold_percentile=source_render_style["threshold_percentile"],
            background=str(source_render_style["background"]),
            point_shape=str(source_render_style["point_shape"]),
            radius=int(source_render_style["point_radius"]),
            enhance=bool(source_render_style["enhance"]),
            rotate=0.0,
            scale=1.0,
            canvas_size=(
                int(target_render_meta["render_width"]),
                int(target_render_meta["render_height"]),
            ),
        )


def write_render_space_h5ad(
    *,
    input_h5ad: Path,
    xy_render: np.ndarray,
    tmp_h5ad: Path,
) -> None:
    adata = sc.read_h5ad(str(input_h5ad))
    adata.obsm["spatial"] = np.asarray(xy_render, dtype=np.float64)
    sanitize_adata_for_h5ad_write(adata)
    adata.write_h5ad(str(tmp_h5ad))


def render_h5ad(
    *,
    input_h5ad: Path,
    output_png: Path,
    coord_config: dict,
    render_meta: dict,
    render_style: dict,
) -> np.ndarray | None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    adata = sc.read_h5ad(str(input_h5ad))
    xy = get_xy_from_adata(adata, coord_config)
    xy_render = map_points_forward_with_meta(xy[:, :2], render_meta)

    with tempfile.TemporaryDirectory() as td:
        tmp_h5ad = Path(td) / "tmp_render_space.h5ad"
        write_render_space_h5ad(
            input_h5ad=input_h5ad,
            xy_render=xy_render,
            tmp_h5ad=tmp_h5ad,
        )
        scatter_h5ad_to_image(
            input_h5ad=str(tmp_h5ad),
            output_png=str(output_png),
            background=str(render_style["background"]),
            point_shape=str(render_style["point_shape"]),
            radius=int(render_style["point_radius"]),
            threshold_percentile=render_style["threshold_percentile"],
            intensity_log_transform=bool(render_style["intensity_log_transform"]),
            rotate=0.0,
            scale=1.0,
            dpi=int(render_style.get("dpi", 100)),
            marker_alpha=float(render_style["marker_alpha"]),
            gray_min=float(render_style["gray_min"]),
            gray_max=float(render_style["gray_max"]),
            display_long_side=0,
            padding=0,
            canvas_size=(
                int(render_meta["render_width"]),
                int(render_meta["render_height"]),
            ),
        ) if str(render_style["render_mode"]) == "scatter" else rasterize_h5ad_to_image(
            input_h5ad=str(tmp_h5ad),
            output_png=str(output_png),
            spatial_key="spatial",
            intensity_mode=str(render_style["intensity_mode"]),
            intensity_obs_col=render_style["intensity_obs_col"],
            intensity_log_transform=bool(render_style["intensity_log_transform"]),
            threshold_percentile=render_style["threshold_percentile"],
            background=str(render_style["background"]),
            point_shape=str(render_style["point_shape"]),
            radius=int(render_style["point_radius"]),
            enhance=bool(render_style["enhance"]),
            rotate=0.0,
            scale=1.0,
            canvas_size=(
                int(render_meta["render_width"]),
                int(render_meta["render_height"]),
            ),
        )
    return None


def save_overlay(target_png: Path, transformed_source_png: Path, out_path: Path) -> None:
    target_img = cv2.imread(str(target_png), cv2.IMREAD_COLOR)
    source_img = cv2.imread(str(transformed_source_png), cv2.IMREAD_COLOR)
    if target_img is None or source_img is None:
        raise FileNotFoundError(f"Failed to read image(s): {target_png} / {transformed_source_png}")

    th, tw = target_img.shape[:2]
    sh, sw = source_img.shape[:2]
    canvas = np.empty((th, tw, 3), dtype=np.uint8)
    canvas[:] = source_img[0, 0]
    h = min(th, sh)
    w = min(tw, sw)
    canvas[:h, :w] = source_img[:h, :w]

    overlay = cv2.addWeighted(target_img, 0.5, canvas, 0.5, 0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)


def load_grayscale_png(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        gray = image.convert("L")
        return np.asarray(gray, dtype=np.float32) / 255.0


def save_grayscale_png(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def make_self_reference(image: np.ndarray, reference_blur_sigma: float) -> np.ndarray:
    reference = gaussian_filter(image, sigma=max(reference_blur_sigma, 0.0))
    if reference.max() > 0:
        reference = reference / reference.max()
    return reference.astype(np.float32)


def sample_map_at_points(ssi_map: np.ndarray, xy: np.ndarray) -> np.ndarray:
    points = np.rint(xy).astype(np.int32)
    height, width = ssi_map.shape
    sampled = np.empty(points.shape[0], dtype=np.float32)
    for idx, (x, y) in enumerate(points):
        xx = int(np.clip(x, 0, width - 1))
        yy = int(np.clip(y, 0, height - 1))
        sampled[idx] = float(ssi_map[yy, xx])
    return sampled


def write_ssi_outputs(
    *,
    input_h5ad: Path,
    coord_config: dict,
    render_png: Path,
    render_meta: dict,
    output_dir: Path,
    label: str,
    reference_blur_sigma: float,
    ssim_sigma: float,
) -> None:
    adata = sc.read_h5ad(str(input_h5ad))
    xy = get_xy_from_adata(adata, coord_config)
    keep = finite_xy_mask(xy)
    if not np.any(keep):
        raise ValueError(f"{input_h5ad} has no finite coordinate points for SSI output.")
    xy_rendered = map_points_forward_with_meta(xy[keep, :2], render_meta)

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_visualization.png"
    shutil.copy2(render_png, raw_path)

    raw_image = load_grayscale_png(raw_path)
    reference = make_self_reference(raw_image, reference_blur_sigma)
    save_grayscale_png(reference, output_dir / "reference_raster.png")

    ssi_score, ssi_map = structural_similarity(
        raw_image,
        reference,
        data_range=1.0,
        gaussian_weights=True,
        sigma=float(ssim_sigma),
        use_sample_covariance=False,
        full=True,
    )
    ssi_map = np.clip((ssi_map + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)
    save_grayscale_png(ssi_map, output_dir / "ssi_map.png")

    spot_ssi = pd.DataFrame(
        {
            "x": xy_rendered[:, 0],
            "y": xy_rendered[:, 1],
            "ssi": sample_map_at_points(ssi_map, xy_rendered),
        }
    )
    spot_ssi.to_csv(output_dir / "spot_ssi.csv", index=False)

    summary = {
        "label": label,
        "input_h5ad": str(input_h5ad),
        "render_png": str(render_png),
        "n_obs_used": int(xy_rendered.shape[0]),
        "ssi_score": float(ssi_score),
        "reference_blur_sigma": float(reference_blur_sigma),
        "ssim_sigma": float(ssim_sigma),
        "render_width": int(raw_image.shape[1]),
        "render_height": int(raw_image.shape[0]),
        "render_meta": render_meta,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"[OK] {label} SSI saved: {output_dir} (score={ssi_score:.6f})")


def run_pair(args: argparse.Namespace) -> None:
    target_h5ad = ensure_h5ad(args.target_h5ad, "target")
    source_h5ad = ensure_h5ad(args.source_h5ad, "source")
    sample_id = str(args.sample_id or default_sample_id(target_h5ad, source_h5ad))
    target_render_style = render_style_from_args(args, "target")
    source_render_style = render_style_from_args(args, "source")
    target_coord_config = resolve_coordinate_config(
        h5ad_path=target_h5ad,
        label="target",
        spatial_key=str(args.target_spatial_key),
        x_obs_col=args.target_x_obs_col,
        y_obs_col=args.target_y_obs_col,
    )
    source_coord_config = resolve_coordinate_config(
        h5ad_path=source_h5ad,
        label="source",
        spatial_key=str(args.source_spatial_key),
        x_obs_col=args.source_x_obs_col,
        y_obs_col=args.source_y_obs_col,
    )

    sample_out = args.output_dir / sample_id
    render_dir = sample_out / "render"
    ssi_dir = sample_out / "ssi"
    alignment_dir = sample_out / "alignment"
    render_dir.mkdir(parents=True, exist_ok=True)
    ssi_dir.mkdir(parents=True, exist_ok=True)
    alignment_dir.mkdir(parents=True, exist_ok=True)

    target_png = render_dir / "st_scatter.png"
    source_png = render_dir / "sm_scatter_rot_ccw90.png"

    print(f"Sample ID: {sample_id}")
    print(f"[OK] Target h5ad: {target_h5ad}")
    print(f"[OK] Source h5ad: {source_h5ad}")
    print(f"[OK] Target coordinates: {target_coord_config}")
    print(f"[OK] Source coordinates: {source_coord_config}")
    print(f"[OK] Target render style: {target_render_style}")
    print(f"[OK] Source render style: {source_render_style}")

    target_meta = build_render_meta(
        input_h5ad=target_h5ad,
        coord_config=target_coord_config,
        rotate=float(args.target_rotate),
        scale=1.0,
        flip_horizontal=bool(args.target_flip_horizontal),
        display_long_side=int(target_render_style["display_long_side"]),
        padding=int(target_render_style["padding"]),
    )
    source_meta = build_render_meta(
        input_h5ad=source_h5ad,
        coord_config=source_coord_config,
        rotate=float(args.source_rotate),
        scale=1.0,
        flip_horizontal=bool(args.source_flip_horizontal),
        display_long_side=int(source_render_style["display_long_side"]),
        padding=int(source_render_style["padding"]),
    )
    (render_dir / "st_render_meta.json").write_text(json.dumps(target_meta, indent=2), encoding="utf-8")
    (render_dir / "sm_render_meta.json").write_text(json.dumps(source_meta, indent=2), encoding="utf-8")
    (render_dir / "st_render_style.json").write_text(json.dumps(target_render_style, indent=2), encoding="utf-8")
    (render_dir / "sm_render_style.json").write_text(json.dumps(source_render_style, indent=2), encoding="utf-8")

    render_h5ad(
        input_h5ad=target_h5ad,
        output_png=target_png,
        coord_config=target_coord_config,
        render_meta=target_meta,
        render_style=target_render_style,
    )
    source_origin = render_h5ad(
        input_h5ad=source_h5ad,
        output_png=source_png,
        coord_config=source_coord_config,
        render_meta=source_meta,
        render_style=source_render_style,
    )

    write_ssi_outputs(
        input_h5ad=target_h5ad,
        coord_config=target_coord_config,
        render_png=target_png,
        render_meta=target_meta,
        output_dir=ssi_dir / "st",
        label="target",
        reference_blur_sigma=float(args.reference_blur_sigma),
        ssim_sigma=float(args.ssim_sigma),
    )
    write_ssi_outputs(
        input_h5ad=source_h5ad,
        coord_config=source_coord_config,
        render_png=source_png,
        render_meta=source_meta,
        output_dir=ssi_dir / "sm",
        label="source",
        reference_blur_sigma=float(args.reference_blur_sigma),
        ssim_sigma=float(args.ssim_sigma),
    )

    align_start = time.perf_counter()
    align_and_process_images(
        img1_path=str(target_png),
        img2_path=str(source_png),
        h5ad_path=str(source_h5ad),
        method=str(args.method),
        output_dir=str(alignment_dir),
        spatial_key=str(source_coord_config["spatial_key"] or "spatial"),
        x_obs_col=source_coord_config["x_obs_col"],
        y_obs_col=source_coord_config["y_obs_col"],
        rotate=0.0,
        scale=1.0,
        origin=source_origin,
        auto_upscale_reference=False,
        source_render_meta=source_meta,
        target_render_meta=target_meta,
        device=args.device,
    )
    align_elapsed = time.perf_counter() - align_start

    timing_summary = {
        "sample_id": sample_id,
        "target_h5ad": str(target_h5ad),
        "source_h5ad": str(source_h5ad),
        "method": str(args.method),
        "alignment_seconds": float(align_elapsed),
    }
    with (alignment_dir / "alignment_timing.json").open("w", encoding="utf-8") as fh:
        json.dump(timing_summary, fh, indent=2)
    print(f"[OK] Alignment timing saved: {alignment_dir / 'alignment_timing.json'} ({align_elapsed:.4f}s)")

    transformed_h5ad = alignment_dir / "transformed.h5ad"
    if not transformed_h5ad.exists():
        raise FileNotFoundError(f"Transformed source h5ad not found: {transformed_h5ad}")

    transformed_source_png = alignment_dir / "transformed_sm_scatter_for_check.png"
    render_transformed_source_in_target_render_space(
        transformed_h5ad=transformed_h5ad,
        transformed_coord_config=source_coord_config,
        output_png=transformed_source_png,
        target_render_meta=target_meta,
        source_render_style=source_render_style,
    )

    target_coords = read_spatial_df(target_h5ad, target_coord_config)
    source_coords = read_spatial_df(transformed_h5ad, source_coord_config)
    target_csv = alignment_dir / "st_original_coords.csv"
    source_csv = alignment_dir / "sm_aligned_coords.csv"
    target_coords.to_csv(target_csv, index=False)
    source_coords.to_csv(source_csv, index=False)
    print(f"[OK] Target original coordinates saved: {target_csv} (n={len(target_coords)})")
    print(f"[OK] Source aligned coordinates saved: {source_csv} (n={len(source_coords)})")

    overlay_path = alignment_dir / "sm_transformed_vs_st_overlay.png"
    save_overlay(target_png, transformed_source_png, overlay_path)
    print(f"[OK] Validation overlay saved: {overlay_path}")


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {args.output_dir}")
    print(f"Method: {args.method}")
    print(f"Target rotation: {args.target_rotate}")
    print(f"Source rotation: {args.source_rotate}")
    print(f"Target horizontal flip: {bool(args.target_flip_horizontal)}")
    print(f"Source horizontal flip: {bool(args.source_flip_horizontal)}")
    print(
        "Target render: "
        f"long_side={args.target_display_long_side}, padding={args.target_padding}, "
        f"shape={args.target_point_shape}, radius={args.target_point_radius}, "
        f"alpha={args.target_marker_alpha}"
    )
    print(
        "Source render: "
        f"long_side={args.source_display_long_side}, padding={args.source_padding}, "
        f"shape={args.source_point_shape}, radius={args.source_point_radius}, "
        f"alpha={args.source_marker_alpha}"
    )
    print(f"SSI reference blur sigma: {args.reference_blur_sigma}")
    print(f"SSI sigma: {args.ssim_sigma}")

    run_pair(args)
    print("\nWorkflow completed successfully.")


if __name__ == "__main__":
    main()
