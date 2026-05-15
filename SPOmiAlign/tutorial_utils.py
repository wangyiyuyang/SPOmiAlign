from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.spatial import cKDTree


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
TUTORIAL_1_DIR = "Tutorial 1 spatial omics to CCF (omic-to-image)"
TUTORIAL_2_DIR = "Tutorial 2 spatial transcriptomics multimodal alignment(omic-to-omic)"
TUTORIAL_3_DIR = "Tutorial 3 spatial multi-omics alignment without paired images(omic-to-omic)"
TUTORIAL_4_DIR = "Tutorial 4 spatial multi-omics alignment with paired images(image-to-image)"
for import_root in (PROJECT_ROOT, PACKAGE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from align_h5ad_to_h5ad_square import (  # noqa: E402
    build_render_meta,
    render_h5ad,
    render_transformed_source_in_target_render_space,
    resolve_coordinate_config,
    write_ssi_outputs,
)
from data_preprocessing import _apply_rotate_scale_clockwise, scatter_h5ad_to_image  # noqa: E402
from reassignment import (  # noqa: E402
    build_reassigned_h5ad_from_mapping,
    cmap_blue,
    cmap_orange,
    get_spatial_from_adata,
    mean_internal_nn_distance,
    plot_h5ad_umi_squares,
)
from roma import align_and_process_images  # noqa: E402


def find_project_root(start: Path | None = None) -> Path:
    start = Path.cwd() if start is None else Path(start).expanduser()
    for candidate in (start, *start.parents):
        if (candidate / "SPOmiAlign").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate the SPOmiAlign project root.")


def _looks_like_data_root(path: Path) -> bool:
    return any(
        (path / child).exists()
        for child in (
            TUTORIAL_1_DIR,
            TUTORIAL_2_DIR,
            TUTORIAL_3_DIR,
            TUTORIAL_4_DIR,
        )
    )


def resolve_data_root(project_root: Path | None = None) -> Path:
    project_root = PROJECT_ROOT if project_root is None else Path(project_root).expanduser()
    env_path = os.environ.get("SPOMIALIGN_DATA_DIR")
    candidates = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(project_root / "Data")
    candidates.append(project_root.parent / "Data")

    checked = []
    for candidate in candidates:
        data_root = Path(candidate).expanduser()
        checked.append(data_root)
        if data_root.is_dir() and _looks_like_data_root(data_root):
            return data_root.resolve()

    checked_text = "\n".join(f"  - {path}" for path in checked)
    raise FileNotFoundError(
        "Tutorial data were not found. Download Data from the README link and "
        f"extract it to {project_root / 'Data'}.\nChecked:\n{checked_text}"
    )


def resolve_output_root(project_root: Path | None = None) -> Path:
    project_root = PROJECT_ROOT if project_root is None else Path(project_root).expanduser()
    env_path = os.environ.get("SPOMIALIGN_OUTPUT_DIR")
    output_root = Path(env_path).expanduser() if env_path else project_root / "output"
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root.resolve()


def get_tutorial_paths(project_root: Path | None = None) -> tuple[Path, Path]:
    project_root = find_project_root(project_root or Path.cwd())
    data_root = resolve_data_root(project_root)
    output_root = resolve_output_root(project_root)
    print(f"Data directory: {data_root}")
    print(f"Output directory: {output_root}")
    return data_root, output_root


def require_path(path: Path, label: str) -> Path:
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path.resolve()


def first_existing(paths: list[Path | str], label: str) -> Path:
    checked = []
    for path in paths:
        candidate = Path(path).expanduser()
        checked.append(candidate)
        if candidate.exists():
            return candidate.resolve()
    checked_text = "\n".join(f"  - {path}" for path in checked)
    raise FileNotFoundError(f"{label} not found. Checked:\n{checked_text}")


def resolve_h5ad_pair(
    data_root: Path,
    *,
    target_section_path: list[str],
    source_section_path: list[str],
) -> tuple[Path, Path]:
    target = first_existing([data_root / path for path in target_section_path], "Target h5ad")
    source = first_existing([data_root / path for path in source_section_path], "Source h5ad")
    print(f"Target h5ad: {target}")
    print(f"Source h5ad: {source}")
    return target, source


def read_bgr(image_path: Path | str) -> np.ndarray:
    image_path = Path(image_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    return image


def show_images(
    images: list[Path | str | np.ndarray],
    titles: list[str] | None = None,
    *,
    figsize: tuple[float, float] | None = None,
) -> None:
    titles = titles or ["" for _ in images]
    if figsize is None:
        figsize = (5.5 * len(images), 5.5)
    fig, axes = plt.subplots(1, len(images), figsize=figsize)
    if len(images) == 1:
        axes = [axes]
    for ax, image_or_path, title in zip(axes, images, titles):
        image = read_bgr(image_or_path) if isinstance(image_or_path, (str, Path)) else image_or_path
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def _corner_fill_color(image: np.ndarray, fallback: int = 255) -> np.ndarray:
    if image.size == 0:
        return np.array([fallback, fallback, fallback], dtype=np.uint8)
    corner = np.asarray(image[0, 0], dtype=np.uint8)
    if corner.ndim == 0:
        return np.array([corner, corner, corner], dtype=np.uint8)
    return corner[:3]


def _fit_to_reference_canvas(reference_img: np.ndarray, moving_img: np.ndarray) -> np.ndarray:
    ref_h, ref_w = reference_img.shape[:2]
    mov_h, mov_w = moving_img.shape[:2]
    canvas = np.empty((ref_h, ref_w, 3), dtype=np.uint8)
    canvas[:] = _corner_fill_color(moving_img)
    h_limit = min(ref_h, mov_h)
    w_limit = min(ref_w, mov_w)
    canvas[:h_limit, :w_limit] = moving_img[:h_limit, :w_limit]
    return canvas


def make_blend_overlay(
    reference_path: Path | str,
    moving_path: Path | str,
    out_path: Path | str,
    *,
    alpha_reference: float = 0.55,
    alpha_moving: float = 0.45,
) -> Path:
    reference_img = read_bgr(reference_path)
    moving_img = _fit_to_reference_canvas(reference_img, read_bgr(moving_path))
    overlay = cv2.addWeighted(reference_img, alpha_reference, moving_img, alpha_moving, 0)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)
    return out_path.resolve()


def make_h5ad_overlay(
    reference_image_path: Path | str,
    aligned_h5ad_png: Path | str,
    out_path: Path | str,
    *,
    threshold: int = 50,
    alpha_reference: float = 0.45,
    alpha_spots: float = 0.55,
) -> Path:
    reference_img = read_bgr(reference_image_path)
    aligned_gray = cv2.imread(str(aligned_h5ad_png), cv2.IMREAD_GRAYSCALE)
    if aligned_gray is None:
        raise FileNotFoundError(f"Failed to read aligned h5ad image: {aligned_h5ad_png}")

    mask = np.zeros(reference_img.shape[:2], dtype=np.uint8)
    h_limit = min(mask.shape[0], aligned_gray.shape[0])
    w_limit = min(mask.shape[1], aligned_gray.shape[1])
    mask[:h_limit, :w_limit] = aligned_gray[:h_limit, :w_limit]

    spot_mask = mask > int(threshold)
    overlay = reference_img.copy()
    spot_layer = reference_img.copy()
    spot_layer[spot_mask] = (0, 255, 0)
    if np.any(spot_mask):
        overlay[spot_mask] = cv2.addWeighted(
            reference_img[spot_mask],
            alpha_reference,
            spot_layer[spot_mask],
            alpha_spots,
            0,
        )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)
    return out_path.resolve()


def image_size_xy(image_path: Path | str) -> tuple[int, int]:
    image = read_bgr(image_path)
    height, width = image.shape[:2]
    return width, height


def _resolve_alignment_mode(method: str | None = None, Alignment_mode: str | None = None) -> str:
    return str(Alignment_mode if Alignment_mode is not None else (method or "affine+bspline"))


def _resolve_coordinates(
    *,
    x_obs_col: str | None,
    y_obs_col: str | None,
    x_coordinate: str | None,
    y_coordinate: str | None,
) -> tuple[str | None, str | None]:
    return (
        x_coordinate if x_coordinate is not None else x_obs_col,
        y_coordinate if y_coordinate is not None else y_obs_col,
    )


def _resolve_spot_style(
    SPOT: dict | str | None = None,
    *,
    point_shape: str = "circle",
    point_radius: int = 5,
) -> tuple[str, int]:
    if isinstance(SPOT, str):
        point_shape = SPOT
    elif isinstance(SPOT, dict):
        point_shape = str(SPOT.get("shape", SPOT.get("SPOT_shape", point_shape)))
        point_radius = int(SPOT.get("radius", SPOT.get("SPOT_radius", point_radius)))
    return str(point_shape), int(point_radius)


def _resolve_pair_spot_style(
    SPOT: dict | str | None = None,
    *,
    point_shape: str = "square",
    target_radius: int = 5,
    source_radius: int = 5,
) -> tuple[str, int, int]:
    if isinstance(SPOT, str):
        point_shape = SPOT
    elif isinstance(SPOT, dict):
        point_shape = str(SPOT.get("shape", SPOT.get("SPOT_shape", point_shape)))
        shared_radius = SPOT.get("radius", SPOT.get("SPOT_radius", None))
        if shared_radius is not None:
            target_radius = int(shared_radius)
            source_radius = int(shared_radius)
        target_radius = int(SPOT.get("target_radius", target_radius))
        source_radius = int(SPOT.get("source_radius", source_radius))
    return str(point_shape), int(target_radius), int(source_radius)


def run_image_to_image_alignment(
    *,
    data_root: Path,
    output_root: Path,
    sample_id: str,
    target_image_path: str,
    source_image_path: str,
    target_h5ad_path: str | None = None,
    source_h5ad_path: str | None = None,
    target_h5ad_spatial_key: str = "spatial",
    source_h5ad_spatial_key: str = "spatial",
    h5ad_point_shape: str = "square",
    h5ad_point_radius: int = 6,
    h5ad_dpi: int = 150,
    method: str | None = None,
    Alignment_mode: str | None = None,
    device: str | None = None,
) -> dict[str, Path]:
    alignment_method = _resolve_alignment_mode(method, Alignment_mode)
    target_image = require_path(data_root / target_image_path, "Target image")
    source_image = require_path(data_root / source_image_path, "Source image")
    target_h5ad = require_path(data_root / target_h5ad_path, "Target h5ad") if target_h5ad_path is not None else None
    source_h5ad = require_path(data_root / source_h5ad_path, "Source h5ad") if source_h5ad_path is not None else None
    sample_dir = Path(output_root) / "img_2_img" / sample_id
    alignment_dir = sample_dir / "alignment"
    alignment_dir.mkdir(parents=True, exist_ok=True)

    print(f"Target image: {target_image}")
    print(f"Source image: {source_image}")
    if target_h5ad is not None:
        print(f"Target h5ad: {target_h5ad}")
    if source_h5ad is not None:
        print(f"Source h5ad: {source_h5ad}")
    align_and_process_images(
        img1_path=str(target_image),
        img2_path=str(source_image),
        h5ad_path=str(source_h5ad) if source_h5ad is not None else None,
        method=alignment_method,
        output_dir=str(alignment_dir),
        spatial_key=str(source_h5ad_spatial_key),
        rotate=0.0,
        scale=1.0,
        auto_upscale_reference=False,
        device=device,
    )

    aligned_source = require_path(alignment_dir / "aligned_source_img2.png", "Aligned source image")
    before_overlay = alignment_dir / "overlay_before_alignment.png"
    after_overlay = alignment_dir / "overlay_after_alignment.png"
    make_blend_overlay(target_image, source_image, before_overlay)
    make_blend_overlay(target_image, aligned_source, after_overlay)

    result = {
        "target_section_visualization": target_image,
        "source_section_visualization": source_image,
        "aligned_source": aligned_source,
        "before_overlay": before_overlay.resolve(),
        "after_overlay": after_overlay.resolve(),
        "alignment_dir": alignment_dir.resolve(),
    }
    if target_h5ad is not None:
        target_h5ad_png = alignment_dir / "target_h5ad_scatter.png"
        render_h5ad_on_image_canvas(
            h5ad_path=target_h5ad,
            output_png=target_h5ad_png,
            canvas_image_path=target_image,
            spatial_key=target_h5ad_spatial_key,
            point_shape=h5ad_point_shape,
            point_radius=h5ad_point_radius,
            dpi=h5ad_dpi,
        )
        result["target_h5ad"] = target_h5ad.resolve()
        result["target_h5ad_visualization"] = target_h5ad_png.resolve()
    if source_h5ad is not None:
        transformed_h5ad = require_path(alignment_dir / "transformed.h5ad", "Transformed h5ad")
        transformed_h5ad_png = alignment_dir / "transformed_h5ad_scatter.png"
        render_h5ad_on_image_canvas(
            h5ad_path=transformed_h5ad,
            output_png=transformed_h5ad_png,
            canvas_image_path=target_image,
            spatial_key=source_h5ad_spatial_key,
            point_shape=h5ad_point_shape,
            point_radius=h5ad_point_radius,
            dpi=h5ad_dpi,
        )
        h5ad_overlay = make_h5ad_overlay(
            target_image,
            transformed_h5ad_png,
            alignment_dir / "color_alignment_overlay.png",
        )
        result["source_h5ad"] = source_h5ad.resolve()
        result["transformed_h5ad"] = transformed_h5ad.resolve()
        result["transformed_h5ad_visualization"] = transformed_h5ad_png.resolve()
        result["h5ad_overlay"] = h5ad_overlay
    return result


def render_h5ad_on_image_canvas(
    *,
    h5ad_path: Path,
    output_png: Path,
    canvas_image_path: Path | str,
    spatial_key: str = "spatial",
    point_shape: str = "square",
    point_radius: int = 6,
    dpi: int = 150,
    gray_min: float = 0.55,
    gray_max: float = 0.9,
) -> Path:
    scatter_h5ad_to_image(
        input_h5ad=str(h5ad_path),
        output_png=str(output_png),
        spatial_key=str(spatial_key),
        intensity_mode="X_sum",
        intensity_log_transform=True,
        threshold_percentile=None,
        background="black",
        point_shape=str(point_shape),
        radius=int(point_radius),
        rotate=0.0,
        scale=1.0,
        display_long_side=0,
        padding=0,
        canvas_size=image_size_xy(canvas_image_path),
        dpi=int(dpi),
        marker_alpha=1.0,
        gray_min=float(gray_min),
        gray_max=float(gray_max),
        invert_intensity=False,
        enhance=True,
    )
    return Path(output_png).resolve()


def render_spatial_h5ad(
    *,
    h5ad_path: Path,
    output_png: Path,
    x_obs_col: str | None = "Raw_Slideseq_X",
    y_obs_col: str | None = "Raw_Slideseq_Y",
    x_coordinate: str | None = None,
    y_coordinate: str | None = None,
    intensity_obs_col: str | None = "nFeature_Spatial",
    SPOT_UMI: str | None = None,
    threshold_percentile: float | None = None,
    SPOT: dict | str | None = None,
    point_shape: str = "circle",
    point_radius: int = 5,
    rotate: float = 0.0,
    canvas_image_path: Path | None = None,
    dpi: int = 150,
    gray_min: float = 0.55,
    gray_max: float = 0.85,
) -> np.ndarray:
    x_col, y_col = _resolve_coordinates(
        x_obs_col=x_obs_col,
        y_obs_col=y_obs_col,
        x_coordinate=x_coordinate,
        y_coordinate=y_coordinate,
    )
    intensity_col = SPOT_UMI if SPOT_UMI is not None else intensity_obs_col
    point_shape, point_radius = _resolve_spot_style(
        SPOT,
        point_shape=point_shape,
        point_radius=point_radius,
    )
    canvas_size = image_size_xy(canvas_image_path) if canvas_image_path is not None else None
    _, origin = scatter_h5ad_to_image(
        input_h5ad=str(h5ad_path),
        output_png=str(output_png),
        x_obs_col=x_col,
        y_obs_col=y_col,
        intensity_mode="obs_col",
        intensity_obs_col=intensity_col,
        intensity_log_transform=True,
        threshold_percentile=threshold_percentile,
        background="black",
        point_shape=point_shape,
        radius=point_radius,
        rotate=float(rotate),
        scale=1.0,
        display_long_side=0,
        padding=0,
        canvas_size=canvas_size,
        dpi=int(dpi),
        marker_alpha=1.0,
        gray_min=float(gray_min),
        gray_max=float(gray_max),
        invert_intensity=False,
        enhance=True,
    )
    return origin


def generate_spatial_omics_ssi(
    *,
    data_root: Path,
    output_root: Path,
    sample_id: str,
    source_omic_path: str,
    manual_rotate: float,
    SSI_dpi: int = 150,
    x_obs_col: str | None = "Raw_Slideseq_X",
    y_obs_col: str | None = "Raw_Slideseq_Y",
    x_coordinate: str | None = None,
    y_coordinate: str | None = None,
    intensity_obs_col: str | None = "nFeature_Spatial",
    SPOT_UMI: str | None = None,
    threshold_percentile: float | None = None,
    SPOT: dict | str | None = None,
    point_shape: str = "circle",
    point_radius: int = 5,
) -> dict[str, Path | np.ndarray]:
    h5ad_path = require_path(Path(data_root) / source_omic_path, "Input h5ad")
    sample_dir = Path(output_root) / "h5ad_2_img" / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    source_ssi = sample_dir / f"{sample_id}_SSI.png"
    origin = render_spatial_h5ad(
        h5ad_path=h5ad_path,
        output_png=source_ssi,
        x_obs_col=x_obs_col,
        y_obs_col=y_obs_col,
        x_coordinate=x_coordinate,
        y_coordinate=y_coordinate,
        intensity_obs_col=intensity_obs_col,
        SPOT_UMI=SPOT_UMI,
        threshold_percentile=threshold_percentile,
        SPOT=SPOT,
        point_shape=point_shape,
        point_radius=point_radius,
        rotate=manual_rotate,
        dpi=SSI_dpi,
    )
    return {
        "input_h5ad": h5ad_path,
        "source_section_visualization": source_ssi.resolve(),
        "origin": origin,
    }


def run_omic_to_image_alignment(
    *,
    data_root: Path,
    output_root: Path,
    sample_id: str,
    source_omic_path: str,
    target_image_path: str,
    manual_rotate: float,
    SSI_IMAGE_PATH: str | None = None,
    method: str | None = None,
    Alignment_mode: str | None = None,
    x_obs_col: str | None = "Raw_Slideseq_X",
    y_obs_col: str | None = "Raw_Slideseq_Y",
    x_coordinate: str | None = None,
    y_coordinate: str | None = None,
    intensity_obs_col: str | None = "nFeature_Spatial",
    SPOT_UMI: str | None = None,
    threshold_percentile: float | None = None,
    SPOT: dict | str | None = None,
    point_shape: str = "circle",
    point_radius: int = 5,
    SSI_dpi: int = 150,
    gray_min: float = 0.55,
    gray_max: float = 0.85,
    device: str | None = None,
) -> dict[str, Path]:
    alignment_method = _resolve_alignment_mode(method, Alignment_mode)
    x_col, y_col = _resolve_coordinates(
        x_obs_col=x_obs_col,
        y_obs_col=y_obs_col,
        x_coordinate=x_coordinate,
        y_coordinate=y_coordinate,
    )
    intensity_col = SPOT_UMI if SPOT_UMI is not None else intensity_obs_col
    point_shape, point_radius = _resolve_spot_style(
        SPOT,
        point_shape=point_shape,
        point_radius=point_radius,
    )
    h5ad_path = require_path(data_root / source_omic_path, "Input h5ad")
    target_image = require_path(data_root / target_image_path, "Reference image")
    sample_dir = Path(output_root) / "h5ad_2_img" / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    source_ssi = sample_dir / f"{sample_id}_SSI.png"
    origin = render_spatial_h5ad(
        h5ad_path=h5ad_path,
        output_png=source_ssi,
        x_obs_col=x_col,
        y_obs_col=y_col,
        intensity_obs_col=intensity_col,
        threshold_percentile=threshold_percentile,
        point_shape=point_shape,
        point_radius=point_radius,
        rotate=manual_rotate,
        dpi=SSI_dpi,
        gray_min=gray_min,
        gray_max=gray_max,
    )

    if SSI_IMAGE_PATH is not None:
        intermediate_image = require_path(data_root / SSI_IMAGE_PATH, "SSI reference image")
        stage1_dir = sample_dir / "alignment_with_intermediate"
        align_and_process_images(
            img1_path=str(intermediate_image),
            img2_path=str(source_ssi),
            h5ad_path=str(h5ad_path),
            method=alignment_method,
            output_dir=str(stage1_dir),
            x_obs_col=x_col,
            y_obs_col=y_col,
            rotate=float(manual_rotate),
            scale=1.0,
            origin=origin,
            auto_upscale_reference=False,
            device=device,
        )
        stage1_h5ad = require_path(stage1_dir / "transformed.h5ad", "Intermediate transformed h5ad")
        final_dir = sample_dir / "alignment"
        align_and_process_images(
            img1_path=str(target_image),
            img2_path=str(intermediate_image),
            h5ad_path=str(stage1_h5ad),
            method=alignment_method,
            output_dir=str(final_dir),
            x_obs_col=x_col,
            y_obs_col=y_col,
            rotate=0.0,
            scale=1.0,
            origin=origin,
            auto_upscale_reference=False,
            device=device,
        )
    else:
        intermediate_image = None
        final_dir = sample_dir / "alignment"
        align_and_process_images(
            img1_path=str(target_image),
            img2_path=str(source_ssi),
            h5ad_path=str(h5ad_path),
            method=alignment_method,
            output_dir=str(final_dir),
            x_obs_col=x_col,
            y_obs_col=y_col,
            rotate=float(manual_rotate),
            scale=1.0,
            origin=origin,
            auto_upscale_reference=False,
            device=device,
        )

    transformed_h5ad = require_path(final_dir / "transformed.h5ad", "Transformed h5ad")
    transformed_png = final_dir / "transformed_h5ad_scatter.png"
    render_spatial_h5ad(
        h5ad_path=transformed_h5ad,
        output_png=transformed_png,
        x_obs_col=x_col,
        y_obs_col=y_col,
        intensity_obs_col=intensity_col,
        threshold_percentile=threshold_percentile,
        point_shape=point_shape,
        point_radius=point_radius,
        rotate=0.0,
        canvas_image_path=target_image,
        dpi=SSI_dpi,
        gray_min=gray_min,
        gray_max=gray_max,
    )
    overlay = make_h5ad_overlay(target_image, transformed_png, final_dir / "color_alignment_overlay.png")

    result = {
        "input_h5ad": h5ad_path,
        "target_section_visualization": target_image,
        "source_section_visualization": source_ssi.resolve(),
        "transformed_h5ad": transformed_h5ad,
        "transformed_png": transformed_png.resolve(),
        "overlay": overlay,
        "alignment_dir": final_dir.resolve(),
    }
    if intermediate_image is not None:
        result["SSI_image"] = intermediate_image
    return result


def default_h5ad_render_style(
    *,
    display_long_side: int,
    padding: int,
    point_radius: int,
    dpi: int = 100,
    point_shape: str = "square",
    background: str = "white",
) -> dict:
    return {
        "render_mode": "scatter",
        "background": background,
        "display_long_side": int(display_long_side),
        "padding": int(padding),
        "point_shape": str(point_shape),
        "point_radius": int(point_radius),
        "intensity_mode": "X_sum",
        "intensity_obs_col": None,
        "intensity_log_transform": False,
        "threshold_percentile": None,
        "enhance": False,
        "marker_alpha": 0.95,
        "gray_min": 0.35,
        "gray_max": 0.88,
        "dpi": int(dpi),
    }


def _scale_render_meta_for_dpi(render_meta: dict, dpi: int, base_dpi: int = 100) -> dict:
    factor = float(dpi) / float(base_dpi)
    scaled = dict(render_meta)
    if abs(factor - 1.0) <= 1e-12:
        return scaled
    scaled["fit_scale"] = float(render_meta["fit_scale"]) * factor
    scaled["padding"] = int(round(float(render_meta["padding"]) * factor))
    scaled["render_width"] = max(1, int(round(float(render_meta["render_width"]) * factor)))
    scaled["render_height"] = max(1, int(round(float(render_meta["render_height"]) * factor)))
    if int(render_meta.get("display_long_side", 0)) > 0:
        scaled["display_long_side"] = max(1, int(round(float(render_meta["display_long_side"]) * factor)))
    return scaled


def _scale_render_style_for_dpi(render_style: dict, dpi: int, base_dpi: int = 100) -> dict:
    factor = float(dpi) / float(base_dpi)
    scaled = dict(render_style)
    scaled["dpi"] = int(dpi)
    if abs(factor - 1.0) <= 1e-12:
        return scaled
    scaled["point_radius"] = max(1, int(round(float(render_style["point_radius"]) * factor)))
    if int(render_style.get("display_long_side", 0)) > 0:
        scaled["display_long_side"] = max(1, int(round(float(render_style["display_long_side"]) * factor)))
    scaled["padding"] = int(round(float(render_style.get("padding", 0)) * factor))
    return scaled


def _render_demo_style_h5ad(
    *,
    h5ad_path: Path,
    output_png: Path,
    coord_config: dict,
    rotate: float,
    scale: float,
    point_shape: str,
    point_radius: int,
    dpi: int,
    display_long_side: int = 2200,
    padding: int = 32,
    gray_min: float = 0.35,
    gray_max: float = 0.88,
    marker_size: float | None = None,
) -> np.ndarray | None:
    _, origin = scatter_h5ad_to_image(
        input_h5ad=str(h5ad_path),
        output_png=str(output_png),
        spatial_key=str(coord_config["spatial_key"] or "spatial"),
        x_obs_col=coord_config["x_obs_col"],
        y_obs_col=coord_config["y_obs_col"],
        background="white",
        point_shape=str(point_shape),
        radius=int(point_radius),
        threshold_percentile=None,
        intensity_log_transform=False,
        rotate=float(rotate),
        scale=float(scale),
        marker_size=marker_size,
        display_long_side=int(display_long_side),
        padding=int(padding),
        dpi=int(dpi),
        marker_alpha=0.9,
        gray_min=float(gray_min),
        gray_max=float(gray_max),
    )
    return origin


def prepare_directional_inputs(
    *,
    st_h5ad_path: Path,
    sm_h5ad_path: Path,
    aligned_sm_h5ad_path: Path,
    out_dir: Path,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    st_adata = sc.read_h5ad(str(st_h5ad_path))
    st_adata.obsm["spatial_raw"] = np.asarray(st_adata.obsm["spatial"]).copy()
    st_with_keys_path = out_dir / "st_with_reference_keys.h5ad"
    st_adata.write_h5ad(str(st_with_keys_path), compression="gzip")

    sm_raw_adata = sc.read_h5ad(str(sm_h5ad_path))
    sm_aligned_adata = sc.read_h5ad(str(aligned_sm_h5ad_path))
    raw_xy = np.asarray(sm_raw_adata.obsm["spatial"]).copy()
    aligned_xy = np.asarray(sm_aligned_adata.obsm["spatial"]).copy()
    if raw_xy.shape[0] != aligned_xy.shape[0]:
        raise ValueError("The original SM and aligned SM h5ad files have different spot counts.")

    sm_aligned_adata.obsm["spatial_raw"] = raw_xy
    sm_aligned_adata.obsm["spatial_spomialign"] = aligned_xy.copy()
    sm_aligned_adata.obsm["spatial"] = aligned_xy.copy()
    sm_with_keys_path = out_dir / "sm_with_spomialign.h5ad"
    sm_aligned_adata.write_h5ad(str(sm_with_keys_path), compression="gzip")

    return st_with_keys_path.resolve(), sm_with_keys_path.resolve()


def _rotate_spatial_for_plot(
    adata: sc.AnnData,
    *,
    spatial_key: str,
    rotate_deg: float = 0.0,
    scale: float = 1.0,
) -> sc.AnnData:
    if abs(float(rotate_deg)) <= 1e-12 and abs(float(scale) - 1.0) <= 1e-12:
        return adata
    adata_plot = adata.copy()
    xy = np.asarray(adata_plot.obsm[spatial_key]).copy()
    x_rot, y_rot, _ = _apply_rotate_scale_clockwise(
        xy[:, 0],
        xy[:, 1],
        rotate_deg=float(rotate_deg),
        scale=float(scale),
        origin_mode="data",
    )
    adata_plot.obsm[spatial_key] = np.column_stack([x_rot, y_rot])
    return adata_plot


def run_directional_reassignment_fixed_roles(
    *,
    s1_h5ad: Path,
    s2_h5ad: Path,
    direction: str,
    out_dir: Path,
    plot_dir: Path,
    s1_spatial_key: str,
    s2_spatial_key: str,
    d_ref_max: float | None = None,
    distance_multiplier: float = 2.0,
    scale_by_mapping_factor: bool = True,
    plot_s1_rotate: float = 0.0,
    plot_s1_scale: float = 1.0,
    plot_s2_rotate: float = 0.0,
    plot_s2_scale: float = 1.0,
) -> dict[str, Path | dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    adata_s1 = sc.read_h5ad(str(s1_h5ad))
    adata_s2 = sc.read_h5ad(str(s2_h5ad))
    high_xy = get_spatial_from_adata(adata_s1, s1_spatial_key)
    low_xy = get_spatial_from_adata(adata_s2, s2_spatial_key)

    def clean_xy(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mask = np.isfinite(xy).all(axis=1)
        return xy[mask, :], mask

    high_xy_clean, high_mask = clean_xy(high_xy)
    low_xy_clean, low_mask = clean_xy(low_xy)
    if high_xy_clean.shape[0] == 0 or low_xy_clean.shape[0] == 0:
        raise ValueError("S1 or S2 has no valid coordinates after filtering.")

    _, nn_low = mean_internal_nn_distance(low_xy_clean)
    computed_d_ref_max = float(np.max(nn_low)) if nn_low.size > 0 else 0.0
    d_ref_max_value = computed_d_ref_max if d_ref_max is None else float(d_ref_max)
    high_indices_all = np.where(high_mask)[0]
    low_indices_all = np.where(low_mask)[0]

    dist, idx = cKDTree(low_xy_clean).query(high_xy_clean, k=1)
    if d_ref_max_value > 0:
        valid = dist <= float(distance_multiplier) * d_ref_max_value
    else:
        valid = np.ones_like(dist, dtype=bool)
    high_idx_clean = high_indices_all[valid]
    low_idx_clean = low_indices_all[idx[valid]]
    dist_f = dist[valid]

    if direction == "high_to_low":
        mapping = pd.DataFrame(
            {
                "source_index": high_idx_clean,
                "target_index": low_idx_clean,
                "source_x": high_xy_clean[valid][:, 0],
                "source_y": high_xy_clean[valid][:, 1],
                "target_x": low_xy_clean[idx[valid]][:, 0],
                "target_y": low_xy_clean[idx[valid]][:, 1],
                "distance": dist_f,
            }
        )
        out_h5ad = out_dir / "reassigned_high_to_low_on_st.h5ad"
        reassigned_cmap = cmap_orange
    elif direction == "low_to_high":
        mapping = pd.DataFrame(
            {
                "source_index": low_idx_clean,
                "target_index": high_idx_clean,
                "source_x": low_xy_clean[idx[valid]][:, 0],
                "source_y": low_xy_clean[idx[valid]][:, 1],
                "target_x": high_xy_clean[valid][:, 0],
                "target_y": high_xy_clean[valid][:, 1],
                "distance": dist_f,
            }
        )
        out_h5ad = out_dir / "reassigned_low_to_high_on_sm.h5ad"
        reassigned_cmap = cmap_blue
    else:
        raise ValueError("direction must be 'high_to_low' or 'low_to_high'.")

    meta = {
        "low_res_name": "S2",
        "high_res_name": "S1",
        "d_ref_max": d_ref_max_value,
        "computed_d_ref_max": computed_d_ref_max,
        "distance_multiplier": float(distance_multiplier),
        "reassignment_direction": direction,
        "s1_spatial_key": s1_spatial_key,
        "s2_spatial_key": s2_spatial_key,
        "scale_by_mapping_factor": bool(scale_by_mapping_factor),
    }
    map_csv = out_dir / f"{direction}_mapping.csv"
    mapping.to_csv(map_csv, index=False)

    adata_s1_plot = _rotate_spatial_for_plot(
        adata_s1,
        spatial_key=s1_spatial_key,
        rotate_deg=float(plot_s1_rotate),
        scale=float(plot_s1_scale),
    )
    adata_s2_plot = _rotate_spatial_for_plot(
        adata_s2,
        spatial_key=s2_spatial_key,
        rotate_deg=float(plot_s2_rotate),
        scale=float(plot_s2_scale),
    )

    plot_h5ad_umi_squares(
        adata_s1_plot,
        out_png=str(plot_dir / "S1_umi.png"),
        title=f"S1 UMI ({s1_spatial_key})",
        spatial_key=s1_spatial_key,
        cmap=cmap_orange,
    )
    plot_h5ad_umi_squares(
        adata_s2_plot,
        out_png=str(plot_dir / "S2_umi.png"),
        title=f"S2 UMI ({s2_spatial_key})",
        spatial_key=s2_spatial_key,
        cmap=cmap_blue,
    )
    build_reassigned_h5ad_from_mapping(
        mapping=mapping,
        meta=meta,
        adata_s1=adata_s1,
        adata_s2=adata_s2,
        out_h5ad=str(out_h5ad),
        scale_by_mapping_factor=bool(scale_by_mapping_factor),
        reserved_col=None,
    )
    plot_h5ad_umi_squares(
        sc.read_h5ad(str(out_h5ad)),
        out_png=str(plot_dir / f"reassigned_{direction}_umi.png"),
        title=f"Reassigned ({direction}) UMI",
        spatial_key="spatial",
        cmap=reassigned_cmap,
    )

    return {
        "h5ad": out_h5ad.resolve(),
        "mapping_csv": map_csv.resolve(),
        "plot_dir": plot_dir.resolve(),
        "meta": meta,
    }


def run_reassignment(
    *,
    output_root: Path,
    sample_id: str,
    target_h5ad: Path,
    source_h5ad: Path,
    aligned_source_h5ad: Path,
    s1_spatial_key: str = "spatial_spomialign",
    s2_spatial_key: str = "spatial_raw",
    direction: str | tuple[str, ...] | list[str] | None = None,
    plot_s1_rotate: float = 0.0,
    plot_s1_scale: float = 1.0,
    plot_s2_rotate: float = 0.0,
    plot_s2_scale: float = 1.0,
) -> dict[str, dict]:
    directions = (
        ("high_to_low", "low_to_high")
        if direction is None
        else ((direction,) if isinstance(direction, str) else tuple(direction))
    )

    sample_output_dir = Path(output_root) / "h5ad_2_h5ad" / sample_id
    st_with_keys_path, sm_with_keys_path = prepare_directional_inputs(
        st_h5ad_path=target_h5ad,
        sm_h5ad_path=source_h5ad,
        aligned_sm_h5ad_path=aligned_source_h5ad,
        out_dir=sample_output_dir / "directional_inputs",
    )

    outputs: dict[str, dict] = {}
    for direction in directions:
        outputs[direction] = run_directional_reassignment_fixed_roles(
            s1_h5ad=sm_with_keys_path,
            s2_h5ad=st_with_keys_path,
            direction=direction,
            out_dir=sample_output_dir / direction,
            plot_dir=sample_output_dir / direction / "plots",
            s1_spatial_key=s1_spatial_key,
            s2_spatial_key=s2_spatial_key,
            d_ref_max=None,
            distance_multiplier=2.0,
            scale_by_mapping_factor=True,
            plot_s1_rotate=float(plot_s1_rotate),
            plot_s1_scale=float(plot_s1_scale),
            plot_s2_rotate=float(plot_s2_rotate),
            plot_s2_scale=float(plot_s2_scale),
        )
    return outputs


def generate_h5ad_pair_ssi(
    *,
    output_root: Path,
    sample_id: str,
    target_h5ad: Path,
    source_h5ad: Path,
    target_rotate: float = 0.0,
    source_rotate: float = -90.0,
    source_scale: float = 1.0,
    display_long_side: int = 2200,
    padding: int = 32,
    SSI_dpi: int = 200,
    SPOT: dict | str | None = None,
    point_shape: str = "square",
    target_radius: int = 5,
    source_radius: int = 5,
) -> dict:
    point_shape, target_radius, source_radius = _resolve_pair_spot_style(
        SPOT,
        point_shape=point_shape,
        target_radius=target_radius,
        source_radius=source_radius,
    )
    sample_dir = Path(output_root) / "h5ad_2_h5ad" / sample_id
    render_dir = sample_dir / "rendering"
    render_dir.mkdir(parents=True, exist_ok=True)

    target_coord_config = resolve_coordinate_config(
        h5ad_path=target_h5ad,
        label="target/ST",
        spatial_key="spatial",
        x_obs_col=None,
        y_obs_col=None,
    )
    source_coord_config = resolve_coordinate_config(
        h5ad_path=source_h5ad,
        label="source/SM",
        spatial_key="spatial",
        x_obs_col=None,
        y_obs_col=None,
    )
    target_meta = build_render_meta(
        input_h5ad=target_h5ad,
        coord_config=target_coord_config,
        rotate=float(target_rotate),
        scale=1.0,
        flip_horizontal=False,
        display_long_side=int(display_long_side),
        padding=int(padding),
    )
    source_meta = build_render_meta(
        input_h5ad=source_h5ad,
        coord_config=source_coord_config,
        rotate=float(source_rotate),
        scale=float(source_scale),
        flip_horizontal=False,
        display_long_side=int(display_long_side),
        padding=int(padding),
    )
    target_style = default_h5ad_render_style(
        display_long_side=display_long_side,
        padding=padding,
        point_radius=target_radius,
        dpi=SSI_dpi,
        point_shape=point_shape,
    )
    source_style = default_h5ad_render_style(
        display_long_side=display_long_side,
        padding=padding,
        point_radius=source_radius,
        dpi=SSI_dpi,
        point_shape=point_shape,
    )
    target_png = render_dir / "st_section_visualization.png"
    source_png = render_dir / "sm_section_visualization.png"
    _render_demo_style_h5ad(
        h5ad_path=target_h5ad,
        output_png=target_png,
        coord_config=target_coord_config,
        rotate=float(target_rotate),
        scale=1.0,
        point_shape=point_shape,
        point_radius=target_radius,
        dpi=int(SSI_dpi),
        display_long_side=int(display_long_side),
        padding=int(padding),
    )
    source_origin = _render_demo_style_h5ad(
        h5ad_path=source_h5ad,
        output_png=source_png,
        coord_config=source_coord_config,
        rotate=float(source_rotate),
        scale=float(source_scale),
        point_shape=point_shape,
        point_radius=source_radius,
        dpi=int(SSI_dpi),
        display_long_side=int(display_long_side),
        padding=int(padding),
    )

    return {
        "target_section_visualization": target_png.resolve(),
        "source_section_visualization": source_png.resolve(),
        "target_coord_config": target_coord_config,
        "source_coord_config": source_coord_config,
        "target_render_meta": target_meta,
        "source_render_meta": source_meta,
        "target_render_style": target_style,
        "source_render_style": source_style,
        "source_origin": source_origin,
    }


def generate_h5ad_ssi(
    *,
    output_root: Path,
    sample_id: str,
    h5ad_path: Path,
    label: str,
    output_name: str,
    rotate: float = 0.0,
    scale: float = 1.0,
    display_long_side: int = 2200,
    padding: int = 32,
    SSI_dpi: int = 200,
    SPOT: dict | str | None = None,
    point_shape: str = "square",
    point_radius: int = 5,
    marker_size: float | None = None,
) -> dict:
    h5ad_path = require_path(h5ad_path, f"{label} h5ad")
    point_shape, point_radius = _resolve_spot_style(
        SPOT,
        point_shape=point_shape,
        point_radius=point_radius,
    )
    sample_dir = Path(output_root) / "h5ad_2_h5ad" / sample_id
    render_dir = sample_dir / "rendering"
    render_dir.mkdir(parents=True, exist_ok=True)

    coord_config = resolve_coordinate_config(
        h5ad_path=h5ad_path,
        label=label,
        spatial_key="spatial",
        x_obs_col=None,
        y_obs_col=None,
    )
    render_meta = build_render_meta(
        input_h5ad=h5ad_path,
        coord_config=coord_config,
        rotate=float(rotate),
        scale=float(scale),
        flip_horizontal=False,
        display_long_side=int(display_long_side),
        padding=int(padding),
    )
    render_style = default_h5ad_render_style(
        display_long_side=display_long_side,
        padding=padding,
        point_radius=point_radius,
        dpi=SSI_dpi,
        point_shape=point_shape,
    )

    render_png = render_dir / output_name
    origin = _render_demo_style_h5ad(
        h5ad_path=h5ad_path,
        output_png=render_png,
        coord_config=coord_config,
        rotate=float(rotate),
        scale=float(scale),
        point_shape=point_shape,
        point_radius=point_radius,
        dpi=int(SSI_dpi),
        display_long_side=int(display_long_side),
        padding=int(padding),
        marker_size=marker_size,
    )
    return {
        "input_h5ad": h5ad_path,
        "section_visualization": render_png.resolve(),
        "coord_config": coord_config,
        "render_meta": render_meta,
        "render_style": render_style,
        "origin": origin,
    }


def generate_ssi_visualization(
    *,
    output_root: Path,
    sample_id: str,
    data_root: Path | None = None,
    source_omic_path: str | None = None,
    h5ad_path: Path | None = None,
    label: str | None = None,
    output_name: str | None = None,
    target_h5ad: Path | None = None,
    source_h5ad: Path | None = None,
    **kwargs,
) -> dict:
    if source_omic_path is not None:
        if data_root is None:
            raise ValueError("data_root is required when source_omic_path is provided.")
        return generate_spatial_omics_ssi(
            data_root=data_root,
            output_root=output_root,
            sample_id=sample_id,
            source_omic_path=source_omic_path,
            **kwargs,
        )

    if h5ad_path is not None:
        return generate_h5ad_ssi(
            output_root=output_root,
            sample_id=sample_id,
            h5ad_path=h5ad_path,
            label=label or "h5ad",
            output_name=output_name or "section_visualization.png",
            **kwargs,
        )

    if target_h5ad is not None and source_h5ad is not None:
        return generate_h5ad_pair_ssi(
            output_root=output_root,
            sample_id=sample_id,
            target_h5ad=target_h5ad,
            source_h5ad=source_h5ad,
            **kwargs,
        )

    raise ValueError("Provide either source_omic_path or both target_h5ad and source_h5ad.")


def run_omic_to_omic_alignment(
    *,
    output_root: Path,
    sample_id: str,
    target_h5ad: Path,
    source_h5ad: Path,
    target_rotate: float = 0.0,
    source_rotate: float = -90.0,
    source_scale: float = 1.0,
    display_long_side: int = 2200,
    padding: int = 32,
    SSI_dpi: int = 200,
    SPOT: dict | str | None = None,
    point_shape: str = "square",
    target_radius: int = 5,
    source_radius: int = 5,
    method: str | None = None,
    Alignment_mode: str | None = None,
    device: str | None = None,
    run_reassignment: bool = True,
    target_alignment_image: Path | str | None = None,
    source_alignment_image: Path | str | None = None,
) -> dict:
    alignment_method = _resolve_alignment_mode(method, Alignment_mode)
    point_shape, target_radius, source_radius = _resolve_pair_spot_style(
        SPOT,
        point_shape=point_shape,
        target_radius=target_radius,
        source_radius=source_radius,
    )
    sample_dir = Path(output_root) / "h5ad_2_h5ad" / sample_id
    render_dir = sample_dir / "rendering"
    ssi_dir = sample_dir / "ssi"
    alignment_dir = sample_dir / "alignment"
    render_dir.mkdir(parents=True, exist_ok=True)
    ssi_dir.mkdir(parents=True, exist_ok=True)
    alignment_dir.mkdir(parents=True, exist_ok=True)

    target_coord_config = resolve_coordinate_config(
        h5ad_path=target_h5ad,
        label="target/ST",
        spatial_key="spatial",
        x_obs_col=None,
        y_obs_col=None,
    )
    source_coord_config = resolve_coordinate_config(
        h5ad_path=source_h5ad,
        label="source/SM",
        spatial_key="spatial",
        x_obs_col=None,
        y_obs_col=None,
    )
    target_meta = build_render_meta(
        input_h5ad=target_h5ad,
        coord_config=target_coord_config,
        rotate=float(target_rotate),
        scale=1.0,
        flip_horizontal=False,
        display_long_side=int(display_long_side),
        padding=int(padding),
    )
    source_meta = build_render_meta(
        input_h5ad=source_h5ad,
        coord_config=source_coord_config,
        rotate=float(source_rotate),
        scale=float(source_scale),
        flip_horizontal=False,
        display_long_side=int(display_long_side),
        padding=int(padding),
    )

    target_style = default_h5ad_render_style(
        display_long_side=display_long_side,
        padding=padding,
        point_radius=target_radius,
        dpi=SSI_dpi,
        point_shape=point_shape,
    )
    source_style = default_h5ad_render_style(
        display_long_side=display_long_side,
        padding=padding,
        point_radius=source_radius,
        dpi=SSI_dpi,
        point_shape=point_shape,
    )
    target_png = (
        require_path(Path(target_alignment_image), "Target alignment image")
        if target_alignment_image is not None
        else render_dir / "st_section_visualization.png"
    )
    source_png = (
        require_path(Path(source_alignment_image), "Source alignment image")
        if source_alignment_image is not None
        else render_dir / "sm_section_visualization.png"
    )

    if target_alignment_image is None:
        _render_demo_style_h5ad(
            h5ad_path=target_h5ad,
            output_png=target_png,
            coord_config=target_coord_config,
            rotate=float(target_rotate),
            scale=1.0,
            point_shape=point_shape,
            point_radius=target_radius,
            dpi=int(SSI_dpi),
            display_long_side=int(display_long_side),
            padding=int(padding),
        )
        write_ssi_outputs(
            input_h5ad=target_h5ad,
            coord_config=target_coord_config,
            render_png=target_png,
            render_meta=target_meta,
            output_dir=ssi_dir / "target",
            label="target/ST",
            reference_blur_sigma=6.0,
            ssim_sigma=1.5,
        )
        target_alignment_png = require_path(ssi_dir / "target" / "ssi_map.png", "Target alignment image")
    else:
        target_alignment_png = target_png

    if source_alignment_image is None:
        source_origin = _render_demo_style_h5ad(
            h5ad_path=source_h5ad,
            output_png=source_png,
            coord_config=source_coord_config,
            rotate=float(source_rotate),
            scale=float(source_scale),
            point_shape=point_shape,
            point_radius=source_radius,
            dpi=int(SSI_dpi),
            display_long_side=int(display_long_side),
            padding=int(padding),
        )
        write_ssi_outputs(
            input_h5ad=source_h5ad,
            coord_config=source_coord_config,
            render_png=source_png,
            render_meta=source_meta,
            output_dir=ssi_dir / "source",
            label="source/SM",
            reference_blur_sigma=6.0,
            ssim_sigma=1.5,
        )
        source_alignment_png = require_path(ssi_dir / "source" / "ssi_map.png", "Source alignment image")
    else:
        source_origin = None
        source_alignment_png = source_png

    before_overlay = make_blend_overlay(
        target_alignment_png,
        source_alignment_png,
        alignment_dir / "overlay_before_alignment.png",
    )
    print(f"[OK] Alignment target image: {target_alignment_png}")
    print(f"[OK] Alignment source image: {source_alignment_png}")
    align_and_process_images(
        img1_path=str(target_alignment_png),
        img2_path=str(source_alignment_png),
        h5ad_path=str(source_h5ad),
        method=alignment_method,
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
        device=device,
    )

    transformed_h5ad = require_path(alignment_dir / "transformed.h5ad", "Transformed h5ad")
    transformed_png = alignment_dir / "transformed_sm_in_st_frame.png"
    render_transformed_source_in_target_render_space(
        transformed_h5ad=transformed_h5ad,
        transformed_coord_config=source_coord_config,
        output_png=transformed_png,
        target_render_meta=target_meta,
        source_render_style=source_style,
    )
    after_overlay = make_blend_overlay(
        target_png,
        transformed_png,
        alignment_dir / "overlay_after_alignment.png",
    )

    result = {
        "target_h5ad": target_h5ad,
        "source_h5ad": source_h5ad,
        "target_section_visualization": target_png.resolve(),
        "source_section_visualization": source_png.resolve(),
        "target_alignment_image": target_alignment_png.resolve(),
        "source_alignment_image": source_alignment_png.resolve(),
        "target_alignment_ssi": target_alignment_png.resolve(),
        "source_alignment_ssi": source_alignment_png.resolve(),
        "transformed_h5ad": transformed_h5ad,
        "transformed_section_visualization": transformed_png.resolve(),
        "before_overlay": before_overlay,
        "after_overlay": after_overlay,
        "alignment_dir": alignment_dir.resolve(),
    }

    if run_reassignment:
        result.update(
            globals()["run_reassignment"](
                output_root=output_root,
                sample_id=sample_id,
                target_h5ad=target_h5ad,
                source_h5ad=source_h5ad,
                aligned_source_h5ad=transformed_h5ad,
                s1_spatial_key="spatial_spomialign",
                s2_spatial_key="spatial_raw",
            )
        )

    return result
