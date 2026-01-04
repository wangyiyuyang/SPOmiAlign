#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import scanpy as sc
from PIL import Image
import cv2


# =========================
# 工具：强度处理（可选 log + 1-99 分位裁剪 + 归一化到 [0,1] + 可选阈值筛点）
# =========================
def _prepare_intensity(
    intensity: np.ndarray,
    *,
    intensity_log_transform: bool = False,
    threshold_percentile: float | None = None,
    clip_low: float = 1.0,
    clip_high: float = 99.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    返回：
      intensity_norm: [0,1] 浮点
      keep_mask: bool mask（若 threshold_percentile=None，则全 True）
    """
    v = np.asarray(intensity, dtype=np.float64)

    # 可选 log1p（统一用 log1p，避免 0 的问题）
    if intensity_log_transform:
        v = np.log1p(np.maximum(v, 0.0))

    # 1-99 分位裁剪
    p1 = float(np.nanpercentile(v, clip_low))
    p99 = float(np.nanpercentile(v, clip_high))
    if not np.isfinite(p1) or not np.isfinite(p99) or (p99 <= p1):
        vmin = float(np.nanmin(v)) if np.isfinite(np.nanmin(v)) else 0.0
        vmax = float(np.nanmax(v)) if np.isfinite(np.nanmax(v)) else (vmin + 1e-9)
        p1, p99 = vmin, vmax + 1e-9

    v_clip = np.clip(v, p1, p99)
    v_norm = (v_clip - p1) / (p99 - p1 + 1e-12)
    v_norm = np.clip(v_norm, 0.0, 1.0)

    # 可选阈值（按 v_norm 的分位点）
    if threshold_percentile is None:
        keep = np.ones(v_norm.shape[0], dtype=bool)
    else:
        thr = float(np.nanpercentile(v_norm, float(threshold_percentile)))
        keep = v_norm > thr

    return v_norm.astype(np.float32), keep


# =========================
# 工具：点模板（圆 / 方）
# =========================
def _make_kernel(radius: int, shape: str = "circle") -> np.ndarray:
    r = int(max(0, radius))
    if r == 0:
        return np.ones((1, 1), dtype=np.float32)

    size = 2 * r + 1
    if shape.lower() in ("square", "rect", "box"):
        return np.ones((size, size), dtype=np.float32)

    # circle
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    mask = (x * x + y * y) <= (r * r)
    return mask.astype(np.float32)


# =========================
# 工具：灰度增强（CLAHE + Gamma + Unsharp），用于最终 uint8 灰度图
# =========================
def enhance_gray_uint8(
    gray_uint8: np.ndarray,
    clahe_clip: float = 4.0,
    clahe_grid: tuple[int, int] = (8, 8),
    gamma: float = 0.8,
    unsharp_ksize: tuple[int, int] = (5, 5),
    unsharp_sigma: float = 1.0,
    unsharp_amount: float = 1.5,
) -> np.ndarray:
    """
    输入/输出：uint8 灰度图 (H, W)
    gamma < 1 会变亮；gamma > 1 会变暗
    """
    if gray_uint8.dtype != np.uint8:
        raise ValueError("enhance_gray_uint8 expects uint8 image")

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=tuple(clahe_grid))
    g = clahe.apply(gray_uint8)

    # Gamma（保持与你前面一致：invGamma = 1/gamma）
    inv_gamma = 1.0 / float(gamma)
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
    g = cv2.LUT(g, table)

    # Unsharp
    blurred = cv2.GaussianBlur(g, unsharp_ksize, unsharp_sigma)
    g = cv2.addWeighted(g, float(unsharp_amount), blurred, -float(unsharp_amount - 1.0), 0)

    return g


# =========================
# 工具：顺时针旋转 + 等比缩放（数学坐标系），可选
# =========================
def _apply_rotate_scale_clockwise(
    x: np.ndarray,
    y: np.ndarray,
    *,
    rotate_deg: float = 0.0,
    scale: float = 1.0,
    origin_mode: str = "data",  # 'data'（质心）|'center'（包围盒中心）|'zero'
) -> tuple[np.ndarray, np.ndarray]:
    pts = np.vstack([x, y]).T.astype(np.float64)

    if origin_mode == "data":
        origin = pts.mean(axis=0)
    elif origin_mode == "center":
        origin = np.array([(pts[:, 0].min() + pts[:, 0].max()) / 2.0,
                           (pts[:, 1].min() + pts[:, 1].max()) / 2.0], dtype=np.float64)
    elif origin_mode == "zero":
        origin = np.array([0.0, 0.0], dtype=np.float64)
    else:
        raise ValueError("origin_mode must be 'data'|'center'|'zero'")
    # 顺时针：θ -> -θ
    th = np.deg2rad(-float(rotate_deg))
    c, s = np.cos(th), np.sin(th)
    R_cw = np.array([[c,  s],
                     [-s, c]], dtype=np.float64)

    A = float(scale) * R_cw
    out = (pts - origin.reshape(1, 2)) @ A.T + origin.reshape(1, 2)
    return out[:, 0], out[:, 1], origin


# =========================
# 工具：自动 scale 到 (1152, 864) + 负值平移到 >=0（始终执行、无需参数）
# =========================
def _auto_scale_to_canvas_and_shift_nonnegative(
    x: np.ndarray,
    y: np.ndarray,
    *,
    target_w: float = 1152.0,
    target_h: float = 864.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    自动坐标处理（无参数暴露）：

    A) 等比放大规则（同一个 scale 同时乘到 x 和 y）：
       1) 若 xmax > target_w 且 ymax > target_h：不做 scale
       2) 若仅 xmax <= target_w：scale = target_w / xmax
       3) 若仅 ymax <= target_h：scale = target_h / ymax
       4) 若 xmax <= target_w 且 ymax <= target_h：scale = max(target_w/xmax, target_h/ymax)

    B) 若存在负值：整体平移到 >=0
       x += -min(x) if min(x)<0
       y += -min(y) if min(y)<0

    注意：该过程不会强制输出画布尺寸=1152×864；PNG 尺寸仍由变换后的坐标最大值决定。
    """
    x = np.asarray(x, dtype=np.float64).copy()
    y = np.asarray(y, dtype=np.float64).copy()
    if x.size == 0:
        return x, y

    eps = 1e-12
    xmax = float(np.nanmax(x))
    ymax = float(np.nanmax(y))

    need_scale_x = (xmax <= target_w)
    need_scale_y = (ymax <= target_h)

    if need_scale_x or need_scale_y:
        sx = target_w / max(xmax, eps)
        sy = target_h / max(ymax, eps)
        if need_scale_x and need_scale_y:
            s = max(sx, sy)
        elif need_scale_x:
            s = sx
        else:
            s = sy
        x *= s
        y *= s

    xmin = float(np.nanmin(x))
    ymin = float(np.nanmin(y))
    if xmin < 0:
        x += (-xmin)
    if ymin < 0:
        y += (-ymin)

    return x, y
def _only_shift_nonnegative(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    仅负值平移到 >=0（无参数暴露）：
       x += -min(x) if min(x)<0
       y += -min(y) if min(y)<0
    """
    x = np.asarray(x, dtype=np.float64).copy()
    y = np.asarray(y, dtype=np.float64).copy()
    if x.size == 0:
        return x, y

    xmin = float(np.nanmin(x))
    ymin = float(np.nanmin(y))
    if xmin < 0:
        x += (-xmin)
        print("x add offset")
    if ymin < 0:
        y += (-ymin)
        print("y add offset")

    return x, y


# =========================
# 核心：h5ad -> 像素级栅格化灰度图（可选增强）
# =========================
def rasterize_h5ad_to_image(
    *,
    input_h5ad: str,
    output_png: str,

    # 坐标来源：默认使用 obsm['spatial'] 的两列
    spatial_key: str = "spatial",
    x_obs_col: str | None = None,
    y_obs_col: str | None = None,

    # 强度：默认 X.sum(axis=1)；若要用 obs 列，则 intensity_mode="obs_col" 且传 intensity_obs_col
    intensity_mode: str = "X_sum",       # "X_sum" | "obs_col"
    intensity_obs_col: str | None = None,

    # 强度统一：可选 log1p；统一 1-99 分位裁剪 + 归一化；可选阈值筛点
    intensity_log_transform: bool = False,
    threshold_percentile: float | None = None,  # 例如 80；None 表示不阈值

    # 叠加规则（默认白底黑点）
    background: str = "white",  # "white" | "black"

    # 点形状与大小
    point_shape: str = "circle",  # "circle" | "square"
    radius: int = 5,

    # 可选：输出前增强（CLAHE+Gamma+Unsharp）
    enhance: bool = False,
    clahe_clip: float = 4.0,
    clahe_grid: tuple[int, int] = (8, 8),
    gamma: float = 0.8,
    unsharp_ksize: tuple[int, int] = (5, 5),
    unsharp_sigma: float = 1.0,
    unsharp_amount: float = 1.5,

    # 可选：用户指定的 rotate/scale（不传则默认不做）
    rotate: float = 0.0,            # 顺时针角度
    scale: float = 1.0,             # 等比缩放
    rotate_origin: str = "data",    # 'data'|'center'|'zero'

    canvas_size: tuple[int, int] | None = None,  # 新增：(width, height)
):
    """
    输出：
      - output_png：像素级栅格化灰度图（默认白底黑点）
    """
    os.makedirs(os.path.dirname(output_png), exist_ok=True)

    adata = sc.read_h5ad(input_h5ad)

    # ===== 取坐标 =====
    if x_obs_col is not None and y_obs_col is not None:
        x = np.asarray(adata.obs[x_obs_col], dtype=np.float64)
        y = np.asarray(adata.obs[y_obs_col], dtype=np.float64)
    else:
        if spatial_key not in adata.obsm:
            raise KeyError(f"obsm['{spatial_key}'] 不存在；请传 x_obs_col/y_obs_col 指定坐标列")
        xy = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
        if xy.ndim != 2 or xy.shape[1] < 2:
            raise ValueError(f"obsm['{spatial_key}'] 形状应为 (N,2+)；实际 {xy.shape}")
        x, y = xy[:, 0], xy[:, 1]

    # ===== 取强度（raw）=====
    if intensity_mode == "X_sum":
        # 兼容稀疏矩阵
        try:
            intensity_raw = np.array(adata.X.sum(axis=1)).reshape(-1)
        except Exception:
            intensity_raw = np.asarray(adata.X.sum(axis=1)).ravel()
    elif intensity_mode == "obs_col":
        if not intensity_obs_col:
            raise ValueError("intensity_mode='obs_col' 时必须提供 intensity_obs_col")
        intensity_raw = np.asarray(adata.obs[intensity_obs_col], dtype=np.float64)
    else:
        raise ValueError("intensity_mode must be 'X_sum' or 'obs_col'")

    # ===== 统一强度处理：可选log + 1-99 clip + 归一化 + 可选阈值 =====
    intensity_norm, keep_mask = _prepare_intensity(
        intensity_raw,
        intensity_log_transform=bool(intensity_log_transform),
        threshold_percentile=threshold_percentile,
        clip_low=1.0,
        clip_high=99.0,
    )

    # 可选：保留在 adata.obs 里（供调试/后续使用），虽然不再写出 h5ad
    adata.obs["render_intensity_raw"] = intensity_raw
    adata.obs["render_intensity_norm"] = intensity_norm
    adata.obs["render_keep"] = keep_mask.astype(np.int8)

    # ===== 过滤有效点（坐标/强度有效 且 keep）=====
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(intensity_norm) & keep_mask
    x = x[valid]
    y = y[valid]
    v = intensity_norm[valid]

    if x.size == 0:
        raise ValueError("有效点数为 0：请检查坐标/强度列，或 threshold_percentile 是否过高")
    origin=None
    # ===== 坐标：可选（用户指定）的顺时针旋转 + 缩放 =====
    if (abs(float(rotate)) > 1e-12) or (abs(float(scale) - 1.0) > 1e-12):
        x, y,origin = _apply_rotate_scale_clockwise(
            x, y,
            rotate_deg=float(rotate),
            scale=float(scale),
            origin_mode=str(rotate_origin),
        )

    # ===== 坐标：自动 scale 到 1152×864 + 负值平移到 >=0（始终执行，无需参数）=====
    # x, y = _auto_scale_to_canvas_and_shift_nonnegative(x, y)
    

    # ===== 像素化坐标（四舍五入到像素）=====
    x_pix = np.rint(x).astype(np.int32)
    y_pix = np.rint(y).astype(np.int32)

    if canvas_size is not None:
        # 用户指定了分辨率 (W, H)
        W, H = canvas_size
    else:
        # 自动计算分辨率
        W = int(x_pix.max()) + 1
        H = int(y_pix.max()) + 1

    if W <= 0 or H <= 0:
        raise ValueError(f"非法尺寸：W={W}, H={H}")

    # ===== 背景与“点更暗/更亮”的规则 =====
    # 统一使用 v in [0,1]：
    # - 白底黑点：背景=1.0，点值越大越黑 => val = 1 - v
    # - 黑底白点：背景=0.0，点值越大越白 => val = v
    bg = background.lower()
    if bg not in ("white", "black"):
        raise ValueError("background must be 'white' or 'black'")

    if bg == "white":
        img = np.ones((H, W), dtype=np.float32)
        vals = 1.0 - v
        take_darker = True   # 用 min 叠加（更黑）
    else:
        img = np.zeros((H, W), dtype=np.float32)
        vals = v
        take_darker = False  # 用 max 叠加（更白）

    # ===== 点模板（圆/方）=====
    kernel = _make_kernel(radius=int(radius), shape=str(point_shape))
    r = int(max(0, radius))

    # ===== 栅格化叠加（像素级盖印）=====
    for xv, yv, val in zip(x_pix, y_pix, vals):
        # patch bbox
        r0 = yv - r
        c0 = xv - r
        r1 = yv + r + 1
        c1 = xv + r + 1

        rr0 = max(0, r0)
        cc0 = max(0, c0)
        rr1 = min(H, r1)
        cc1 = min(W, c1)

        if rr0 >= rr1 or cc0 >= cc1:
            continue

        kr0 = rr0 - r0
        kc0 = cc0 - c0
        kr1 = kr0 + (rr1 - rr0)
        kc1 = kc0 + (cc1 - cc0)

        patch = float(val) * kernel[kr0:kr1, kc0:kc1]

        if take_darker:
            # 白底黑点：用 min 让更黑覆盖
            img[rr0:rr1, cc0:cc1] = np.minimum(img[rr0:rr1, cc0:cc1], patch)
        else:
            # 黑底白点：用 max 让更白覆盖
            img[rr0:rr1, cc0:cc1] = np.maximum(img[rr0:rr1, cc0:cc1], patch)

    # ===== 保存 PNG =====
    img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)

    if enhance:
        img_u8 = enhance_gray_uint8(
            img_u8,
            clahe_clip=clahe_clip,
            clahe_grid=clahe_grid,
            gamma=gamma,
            unsharp_ksize=unsharp_ksize,
            unsharp_sigma=unsharp_sigma,
            unsharp_amount=unsharp_amount,
        )

    Image.fromarray(img_u8, mode="L").save(output_png)
    print(f"✅ PNG 已保存：{output_png}（{W}×{H}，background={background}，shape={point_shape}，radius={radius}）")

    return output_png,origin

# =========================
# 调用示例
# =========================
if __name__ == "__main__":
    # 例1：MERFISH（默认 spatial_key='spatial'，默认 intensity_mode='X_sum'）
    rasterize_h5ad_to_image(
        input_h5ad="/mnt/A3/ivy/register_data/SPOmiAlign_Repro/data_preprocessing/Zhuang-ABCA-3-log2-metadata_08.h5ad",
        output_png="/mnt/A3/ivy/register_data/SPOmiAlign_Repro/output_image/merfish_008.png",
        background="black",
        point_shape="circle",
        radius=1,
        threshold_percentile=None,      # 不阈值
        intensity_log_transform=False,  # 你的 X 已 log2，通常不再 log
        enhance=True,
        rotate=0.0,
        scale=1.0,
    )

    # 例2：Slide-seq 43（用 obs 列作为强度，且 80 分位筛点）
    rasterize_h5ad_to_image(
        input_h5ad="/mnt/A3/ivy/register_data/SPOmiAlign_Repro/data_preprocessing/Puck_Num_43.h5ad",
        output_png="/mnt/A3/ivy/register_data/SPOmiAlign_Repro/output_image/Puck_Num_43.png",
        x_obs_col="Raw_Slideseq_X",
        y_obs_col="Raw_Slideseq_Y",
        # intensity_mode="obs_col",
        intensity_obs_col="nFeature_Spatial",
        intensity_log_transform=True,   # nFeature 计数建议 log1p
        threshold_percentile=80,        # 80 分位筛点
        background="black",
        point_shape="circle",
        radius=5,
        enhance=True,                  # 你前面那套增强
        rotate=90,
        scale=1.0,
    )

    # 例2：Slide-seq 29（用 obs 列作为强度，且 80 分位筛点）
    rasterize_h5ad_to_image(
        input_h5ad="/mnt/A3/ivy/register_data/SPOmiAlign_Repro/data_preprocessing/Puck_Num_29.h5ad",
        output_png="/mnt/A3/ivy/register_data/SPOmiAlign_Repro/output_image/Puck_Num_29.png",
        x_obs_col="Raw_Slideseq_X",
        y_obs_col="Raw_Slideseq_Y",
        # intensity_mode="obs_col",
        intensity_obs_col="nFeature_Spatial",
        intensity_log_transform=True,   # nFeature 计数建议 log1p
        threshold_percentile=80,        # 80 分位筛点
        background="black",
        point_shape="circle",
        radius=5,
        enhance=True,                  # 你前面那套增强
        rotate=180,
        scale=1.0,
    )
    #例3:sm、st、sp
    rasterize_h5ad_to_image(
        input_h5ad="/mnt/A3/ivy/register_data/3omics/Cerebellum-PLATO.h5ad",
        output_png="/mnt/A3/ivy/register_data/SPOmiAlign_Repro/output_image/sp.png",
        background="white",
        point_shape="square",
        radius=15,
        threshold_percentile=None,      # 不阈值
        intensity_log_transform=False,  # 你的 X 已 log2，通常不再 log
        enhance=False,
        rotate=0.0,
        scale=1.0,
    )

    rasterize_h5ad_to_image(
        input_h5ad="/mnt/A3/ivy/register_data/3omics/Cerebellum-MAGIC-seq.h5ad",
        output_png="/mnt/A3/ivy/register_data/SPOmiAlign_Repro/output_image/st.png",
        background="white",
        point_shape="square",
        radius=15,
        threshold_percentile=None,      # 不阈值
        intensity_log_transform=False,  # 你的 X 已 log2，通常不再 log
        enhance=False,
        rotate=0.0,
        scale=1.0,
    )

    rasterize_h5ad_to_image(
        input_h5ad="/mnt/A3/ivy/register_data/3omics/Cerebellum-MALDI-MSI.h5ad",
        output_png="/mnt/A3/ivy/register_data/SPOmiAlign_Repro/output_image/sm.png",
        background="white",
        point_shape="square",
        radius=12,
        threshold_percentile=None,      # 不阈值
        intensity_log_transform=False,  # 你的 X 已 log2，通常不再 log
        enhance=False,
        rotate=60.0,
        scale=0.6,
    )

