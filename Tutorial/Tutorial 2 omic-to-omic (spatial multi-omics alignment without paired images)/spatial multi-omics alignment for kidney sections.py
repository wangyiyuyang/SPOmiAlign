# Auto-generated from the matching notebook.


# # spatial multi-omics alignment for kidney sections
#
# Tutorial 3: spatial multi-omics alignment without paired images (omic-to-omic)

# ## 1. Load package and data paths

from pathlib import Path
import sys
import importlib
import matplotlib.pyplot as plt
import cv2
try:
    START_DIR = Path(__file__).resolve().parent
except NameError:
    START_DIR = Path.cwd()

PROJECT_ROOT = next(
    candidate for candidate in (START_DIR, *START_DIR.parents)
    if (candidate / "SPOmiAlign").is_dir()
)
spomialign_path = PROJECT_ROOT / "SPOmiAlign"
if str(spomialign_path) not in sys.path:
    sys.path.insert(0, str(spomialign_path))

import tutorial_utils as tutorial_utils_module
importlib.reload(tutorial_utils_module)

from tutorial_utils import (
    generate_ssi_visualization,
    get_tutorial_paths,
    run_omic_to_omic_alignment,
    run_reassignment,
    read_bgr,
)

DATA_DIR, OUTPUT_ROOT = get_tutorial_paths(PROJECT_ROOT)
# %matplotlib inline

# ## 2. Parameter settings
#
# | Parameter | Meaning |
# | --- | --- |
# | `target_section_path` | reference ST h5ad paths in `Data/`. |
# | `source_section_path` | source SM h5ad paths relative to `Data/`. |
# | `target_rotate` / `source_rotate` | Clockwise manual rotation applied when rendering the target/source h5ad into SSI. |
# | `SPOT` | Spot shape (`square` / `circle`) and visualization radius. |
# | `Alignment_mode` | SPOmiAlign supports three alignment modes: Rigid (`Rigid`), Affine (`Affine`, `Homography`), and Non-Rigid (`bspline`, `affine+bspline`). |
# | `device` | Torch device used by alignment, for example `cuda:0`, `cuda:1`, or `cpu`; `None` keeps automatic selection. |

DATA_PARAMS = {
    "target_section_path": 'Tutorial 3 spatial multi-omics alignment without paired images(omic-to-omic)/st.h5ad',
    "source_section_path": 'Tutorial 3 spatial multi-omics alignment without paired images(omic-to-omic)/sm.h5ad',
}

SSI_PARAMS = {
    "target_rotate": 0.0,
    "source_rotate": -90.0,
    "SPOT": {"shape": "square", "target_radius": 5, "source_radius": 5},
}

ALIGNMENT_PARAMS = {
    "sample_id": "R114",
    "Alignment_mode": "affine+bspline",
    "device": None,
}
DATA_PARAMS, SSI_PARAMS, ALIGNMENT_PARAMS

# ## 3. Generate h5ad render images
#
# The target and source h5ad files are rendered for visual inspection. The alignment step builds SSI maps from these renderings and uses the SSI maps for matching.

target_h5ad = DATA_DIR / DATA_PARAMS["target_section_path"]
source_h5ad = DATA_DIR / DATA_PARAMS["source_section_path"]

target_render = generate_ssi_visualization(
    output_root=OUTPUT_ROOT,
    sample_id=ALIGNMENT_PARAMS["sample_id"],
    h5ad_path=target_h5ad,
    label="target/ST",
    output_name="st_section_visualization.png",
    rotate=SSI_PARAMS["target_rotate"],
    SPOT={"shape": SSI_PARAMS["SPOT"]["shape"], "radius": SSI_PARAMS["SPOT"]["target_radius"]},
)
source_render = generate_ssi_visualization(
    output_root=OUTPUT_ROOT,
    sample_id=ALIGNMENT_PARAMS["sample_id"],
    h5ad_path=source_h5ad,
    label="source/SM",
    output_name="sm_section_visualization.png",
    rotate=SSI_PARAMS["source_rotate"],
    SPOT={"shape": SSI_PARAMS["SPOT"]["shape"], "radius": SSI_PARAMS["SPOT"]["source_radius"]},
)

images = [target_render["section_visualization"], source_render["section_visualization"]]
titles = ["Target h5ad render", "Source h5ad render"]

# Handle stale notebook state where images/titles were accidentally wrapped by a trailing comma.
if len(images) == 1 and isinstance(images[0], (list, tuple)):
    images = list(images[0])
if len(titles) == 1 and isinstance(titles[0], (list, tuple)):
    titles = list(titles[0])

fig, axes = plt.subplots(1, len(images), figsize=(12, 6))
if len(images) == 1:
    axes = [axes]
for ax, image_or_path, title in zip(axes, images, titles):
    image = read_bgr(image_or_path) if isinstance(image_or_path, (str, Path)) else image_or_path
    if image is None:
        raise ValueError(f"Failed to read image: {image_or_path}")
    if image.ndim == 2:
        ax.imshow(image, cmap="gray")
    else:
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.show()

# ## 4. Run SPOmiAlign alignment

result = run_omic_to_omic_alignment(
    output_root=OUTPUT_ROOT,
    target_h5ad=target_h5ad,
    source_h5ad=source_h5ad,
    target_alignment_image=target_render["section_visualization"],
    source_alignment_image=source_render["section_visualization"],
    run_reassignment=False,
    **SSI_PARAMS,
    **ALIGNMENT_PARAMS,
)
result

# ## 5. Reassignment module
#
# | Parameter | Meaning |
# | --- | --- |
# | `direction` | Reassignment direction; default is bidirectional, otherwise along this direction. |
# | `s1_spatial_key` | Source coordinates produced by SPOmiAlign. |
# | `s2_spatial_key` | Target coordinates used as the reference layout. |


REASSIGNMENT_PARAMS = {
    "direction": ("high_to_low","low_to_high"),
    "s1_spatial_key": "spatial_spomialign",
    "s2_spatial_key": "spatial_raw",
}

result.update(
    run_reassignment(
        output_root=OUTPUT_ROOT,
        sample_id=ALIGNMENT_PARAMS["sample_id"],
        target_h5ad=target_h5ad,
        source_h5ad=source_h5ad,
        aligned_source_h5ad=result["transformed_h5ad"],
        **REASSIGNMENT_PARAMS,
    )
)
result["high_to_low"], result["low_to_high"]

# ## 6. Outputs

images=[result["target_alignment_image"], result["source_alignment_image"], result["before_overlay"], result["after_overlay"]]
titles=["Reference ST alignment image", "Source SM alignment image", "overlay before alignment", "overlay after alignment"]
figsize=(22, 5)
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

images=[
        result["high_to_low"]["plot_dir"] / "S1_umi.png",
        result["high_to_low"]["plot_dir"] / "S2_umi.png",
        result["high_to_low"]["plot_dir"] / "reassigned_high_to_low_umi.png",
    ]
titles=["Aligned SM", "Reference ST", "High-to-low reassignment"]
figsize=(18, 5)
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
