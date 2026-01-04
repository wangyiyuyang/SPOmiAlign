import os
import sys
import cv2
import time
import torch
import numpy as np
import pandas as pd
import scanpy as sc
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
import SimpleITK as sitk

# ==========================================
# 0. [关键修改] 导入 data_process
# ==========================================
# 将当前目录加入系统路径，确保能找到同目录下的 data_process.py
sys.path.append(os.getcwd())

try:
    from data_preprocessing import rasterize_h5ad_to_image
    print("✅ Successfully imported 'rasterize_h5ad_to_image' from data_process.py")
except ImportError:
    print("❌ Error: Could not import 'data_process.py'. Please ensure it is in the same directory.")
    sys.exit(1)

# 确保安装了 romatch
from romatch import roma_outdoor

# [配置] 防止 HDF5 锁文件问题
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def debug_step_visualization(adata, bg_bgr, step_name, output_dir, x_col, y_col, rotate, radius):
    """
    极简染色法可视化 (白底 + 红色组织 + 蓝色位点覆盖)
    """
    # --- 1. 数据清洗 (防止 h5ad 保存报错) ---
    def sanitize_dataframe(df):
        if '_index' in df.columns:
            df.drop(columns=['_index'], inplace=True)
        if df.index.name == '_index':
            df.index.name = None

    # 深度拷贝一份，避免修改原对象
    tmp_adata = adata.copy()
    sanitize_dataframe(tmp_adata.obs)
    sanitize_dataframe(tmp_adata.var)
    if tmp_adata.raw is not None:
        try:
            raw_adata = tmp_adata.raw.to_adata()
            sanitize_dataframe(raw_adata.var)
            tmp_adata.raw = raw_adata
        except: pass

    # 保存临时 h5ad 以便调用栅格化函数
    temp_h5ad = os.path.join(output_dir, f"tmp_{step_name}.h5ad")
    tmp_adata.write(temp_h5ad)
    
    # --- 2. 获取尺寸并生成位点图 ---
    h, w = bg_bgr.shape[:2]
    out_png = os.path.join(output_dir, f"tmp_pts_{step_name}.png")
    
    from data_preprocessing import rasterize_h5ad_to_image
    rasterize_h5ad_to_image(
        input_h5ad=temp_h5ad,
        output_png=out_png,
        x_obs_col=x_col,
        y_obs_col=y_col,
        intensity_obs_col="nFeature_Spatial", # 使用这个作为强度参考
        intensity_log_transform=True,
        threshold_percentile=80,
        background="black",  # 生成黑底白点图，方便做二值化
        point_shape="circle",
        radius=radius,
        enhance=True,
        rotate=rotate,
        scale=1.0,
        canvas_size=(w, h)
    )

    # --- 3. 读取图像并二值化提取掩模 (Mask) ---
    # 背景图掩模
    bg_gray = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2GRAY) if len(bg_bgr.shape)==3 else bg_bgr
    _, bg_mask = cv2.threshold(bg_gray, 5, 255, cv2.THRESH_BINARY)
    
    # 位点图掩模
    pts_gray = cv2.imread(out_png, cv2.IMREAD_GRAYSCALE)
    if pts_gray is None:
        print(f"❌ 错误: 无法读取栅格化后的图片 {out_png}")
        return
    pts_gray = cv2.resize(pts_gray, (w, h))
    _, pts_mask = cv2.threshold(pts_gray, 5, 255, cv2.THRESH_BINARY)

    # --- 4. 染色并合成 (白底风格) ---
    # 创建纯白画布
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)

    # 先给背景组织区域刷上红色 (BGR: 0, 0, 255)
    canvas[bg_mask > 0] = [0, 0, 255]

    # 再给位点区域刷上蓝色 (BGR: 255, 0, 0)，直接覆盖红色
    canvas[pts_mask > 0] = [255, 0, 0]

    # --- 5. 保存并清理 ---
    save_path = os.path.join(output_dir, f"DEBUG_{step_name}_Overlay.jpg")
    cv2.imwrite(save_path, canvas)
    
    # 清理临时文件
    if os.path.exists(temp_h5ad): os.remove(temp_h5ad)
    if os.path.exists(out_png): os.remove(out_png)

    print(f"✅ 诊断图已生成: {save_path} (红色:组织, 蓝色:位点)")

# ==========================================
# 1. 核心数学与变换函数 (保持不变)
# ==========================================

def estimate_rigid_transform_svd(src_pts, dst_pts):
    """SVD 刚体变换估计 (Kabsch算法)"""
    if len(src_pts) < 3:
        raise ValueError("Rigid transform requires at least 3 points.")
    centroid_src = np.mean(src_pts, axis=0)
    centroid_dst = np.mean(dst_pts, axis=0)
    src_centered = src_pts - centroid_src
    dst_centered = dst_pts - centroid_dst
    H = src_centered.T @ dst_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = Vt.T @ U.T
    t = centroid_dst.T - R @ centroid_src.T
    M = np.hstack([R, t.reshape(2, 1)])
    return M

def warp_image_bspline(moving_img_np: np.ndarray, tx: sitk.Transform, out_size_xy=None):
    """SimpleITK B-spline 图像变换"""
    if moving_img_np.ndim == 2:
        moving = sitk.GetImageFromArray(moving_img_np.astype(np.float32))
    else:
        moving = sitk.GetImageFromArray(moving_img_np.astype(np.float32), isVector=True)
    H, W = moving_img_np.shape[:2]
    if out_size_xy is None: out_size_xy = (W, H)
    out_size_xy = [int(s) for s in out_size_xy]
    ref = sitk.Image(out_size_xy, moving.GetPixelID(), moving.GetNumberOfComponentsPerPixel())
    ref.SetSpacing(moving.GetSpacing())
    ref.SetOrigin(moving.GetOrigin())
    ref.SetDirection(moving.GetDirection())
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref)
    resampler.SetTransform(tx)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    warped = resampler.Execute(moving)
    return sitk.GetArrayFromImage(warped)

def fit_bspline_transform(fixed_kpts, moving_kpts, H, W, mesh_size=None, max_iter=None, inpaint_radius=3, smooth_sigma=15.0):
    """构造 B-spline 变换"""
    dx = np.full((H, W), np.nan, dtype=np.float32)
    dy = np.full((H, W), np.nan, dtype=np.float32)
    cnt = np.zeros((H, W), dtype=np.int32)
    for (mx, my), (fx, fy) in zip(moving_kpts, fixed_kpts):
        x, y = int(round(float(fx))), int(round(float(fy)))
        if 0 <= x < W and 0 <= y < H:
            ddx, ddy = float(mx) - float(fx), float(my) - float(fy)
            if np.isnan(dx[y, x]):
                dx[y, x], dy[y, x] = ddx, ddy
                cnt[y, x] = 1
            else:
                c = cnt[y, x]
                dx[y, x] = (dx[y, x] * c + ddx) / (c + 1)
                dy[y, x] = (dy[y, x] * c + ddy) / (c + 1)
                cnt[y, x] = c + 1
    known = ~np.isnan(dx)
    if not np.any(known):
        return sitk.Transform(2, sitk.sitkIdentity)
    mask_unknown = (~known).astype(np.uint8) * 255
    dx0, dy0 = np.nan_to_num(dx, nan=0.0).astype(np.float32), np.nan_to_num(dy, nan=0.0).astype(np.float32)
    dx_filled = cv2.inpaint(dx0, mask_unknown, inpaint_radius, cv2.INPAINT_TELEA)
    dy_filled = cv2.inpaint(dy0, mask_unknown, inpaint_radius, cv2.INPAINT_TELEA)
    k = int(max(3, (smooth_sigma * 6) // 2 * 2 + 1))
    disp_np = np.stack([cv2.GaussianBlur(dx_filled, (k, k), smooth_sigma), cv2.GaussianBlur(dy_filled, (k, k), smooth_sigma)], axis=-1).astype(np.float64)
    disp_img = sitk.GetImageFromArray(disp_np, isVector=True)
    disp_img.SetSpacing([1.0, 1.0])
    try:
        return sitk.DisplacementFieldTransform(disp_img)
    except:
        return sitk.Transform(2, sitk.sitkIdentity)

# ==========================================
# 2. RoMa 辅助函数 (保持不变)
# ==========================================

def compute_edge_weight(img_path, H, W, device, sigma=2.0, decay=5.0):
    pil_img = Image.open(img_path).convert('RGB').resize((W, H))
    img_tensor = (torch.tensor(np.array(pil_img)) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    gray_tensor = 0.299 * img_tensor[:, 0] + 0.587 * img_tensor[:, 1] + 0.114 * img_tensor[:, 2]
    fft_im = torch.fft.fft2(gray_tensor.unsqueeze(1))
    fft_shifted = torch.fft.fftshift(fft_im)
    h, w = H, W
    y, x = torch.arange(h, device=device) - h // 2, torch.arange(w, device=device) - w // 2
    Y, X = torch.meshgrid(y, x, indexing='ij')
    dist_sq = X**2 + Y**2
    sigma_val = sigma * min(h, w) / 100.0
    weight = (1.0 - torch.exp(-dist_sq / (2 * (sigma_val**2) + 1e-6))) * torch.exp(-torch.sqrt(dist_sq) / (decay * min(h, w) + 1e-6))
    img_filtered = torch.fft.ifft2(torch.fft.ifftshift(fft_shifted * weight)).real
    edge_map = torch.abs(img_filtered.squeeze())
    edge_map = (edge_map - edge_map.min()) / (edge_map.max() - edge_map.min() + 1e-8)
    _, edge_map_bin = cv2.threshold((torch.pow(edge_map, 1.2).cpu().numpy() * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return torch.from_numpy(edge_map_bin).float().to(device) / 255.0

def apply_grid_nms(certainty_tensor, kernel_size=3):
    x = certainty_tensor.unsqueeze(0).unsqueeze(0) if certainty_tensor.dim() == 2 else certainty_tensor.unsqueeze(1)
    pad = kernel_size // 2
    max_val = F.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=pad)
    return (x * ((x == max_val) & (x > 1e-6))).squeeze()

def draw_matches_visualization(im1, im2, kpts1, kpts2, radius=6):
    """绘制颜色编码的匹配点"""
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]
    vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    vis[:h1, :w1] = im1
    vis[:h2, w1:w1+w2] = im2
    vis = (vis.astype(np.float32) * 0.5).astype(np.uint8)
    num_matches = len(kpts1)
    for i in range(num_matches):
        pt1 = (int(kpts1[i, 0]), int(kpts1[i, 1]))
        pt2 = (int(kpts2[i, 0]) + w1, int(kpts2[i, 1]))
        hue = int(170 * (i / num_matches))
        color_bgr = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0].tolist()
        cv2.circle(vis, pt1, radius + 2, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(vis, pt2, radius + 2, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(vis, pt1, radius, color_bgr, -1, cv2.LINE_AA)
        cv2.circle(vis, pt2, radius, color_bgr, -1, cv2.LINE_AA)
    return vis

def show_image_in_jupyter(img, title="Image", figsize=(20, 10), dpi=100):
    if img is None: return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img
    plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(img_rgb)
    plt.title(title, fontsize=15)
    plt.axis('off')
    plt.show()

# ==========================================
# 3. 主流程函数
# ==========================================

def align_and_process_images(
    img1_path, 
    img2_path, 
    h5ad_path, 
    h5ad_path_img1=None, 
    method='Affine', 
    output_dir='output',
    spatial_key='spatial',     
    x_obs_col=None,            
    y_obs_col=None,
    # === 新增参数 ===
    rotate: float = 0.0,
    scale: float = 1.0,
    rotate_origin: str = "data",
    origin=None
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # [内部辅助函数：复用 data_processing 的逻辑]
    def apply_manual_transform(
        pts: np.ndarray, 
        deg: float, 
        s: float, 
        origin
    ) -> np.ndarray:
        """
        基于画布尺寸 (target_size) 使用矩阵运算实现旋转/缩放。
        重构版本：内部逻辑已改为线性代数矩阵变换，支持任意角度。
        
        参数:
            pts: (N, 2) 坐标点数组。
            deg: 旋转角度 (顺时针为正，支持任意浮点数)。
            s: 缩放比例 (会在旋转前应用，以 (0,0) 为缩放中心)。
            target_size: (width, height) 目标图像的宽高，用于计算旋转中心。
            
        返回:
            (N, 2) 变换后的坐标，float32 类型。
        """
        if pts.size == 0:
            return pts

        # 1. 预处理数据
        # 保持原函数逻辑：先基于 (0,0) 进行全局缩放
        pts_f64 = pts.astype(np.float64) * float(s)
        
        # 2. 确定旋转中心 (Origin)
        # 原函数基于 target_size 的翻转本质上是围绕图像中心旋转
        # W, H = float(target_size[0]), float(target_size[1])
        # 3. 构建旋转矩阵 (参考 _apply_rotate_scale_clockwise 逻辑)
        # 顺时针：θ -> -θ
        th = np.deg2rad(-float(deg))
        c, sin_t = np.cos(th), np.sin(th)
        
        # 旋转矩阵 R
        R_cw = np.array([[c,  sin_t],
                        [-sin_t, c]], dtype=np.float64)

        # 4. 应用仿射变换
        # 公式: out = (pts - origin) @ R.T + origin
        # 注意：这里 scale 设为 1.0，因为步骤1中已经处理了缩放 s
        pts_centered = pts_f64 - origin.reshape(1, 2)
        out = pts_centered @ R_cw.T + origin.reshape(1, 2)

        # 5. 格式恢复 (N, 2) float32
        return out.astype(np.float32)
    
    # 1. Initialize & Load
    H, W = 864, 1152
    roma_model = roma_outdoor(device=device, coarse_res=560, upsample_res=(H, W))
    
    im1_pil_raw = Image.open(img1_path).convert('RGB')
    im2_pil_raw = Image.open(img2_path).convert('RGB')
    orig_W, orig_H = im1_pil_raw.size
    orig_W2, orig_H2 = im2_pil_raw.size
    
    im1_bgr_orig = cv2.cvtColor(np.array(im1_pil_raw), cv2.COLOR_RGB2BGR)
    im2_bgr_orig = cv2.cvtColor(np.array(im2_pil_raw), cv2.COLOR_RGB2BGR)
    
    start_time = time.time()
    
    # 2. RoMa Logic
    print("Running RoMa matching...")
    warp, certainty = roma_model.match(img1_path, img2_path, device=device)
    edge_weight = compute_edge_weight(img2_path, H, W, device)
    edge_full = torch.cat((torch.zeros_like(edge_weight), edge_weight), dim=1)
    
    if certainty.dim() == 3: certainty = certainty.squeeze(0)
    certainty_weighted = certainty * (edge_full * 0.9 + 0.1)
    certainty_weighted[:, :W] = 0 
    
    certainty_filtered = apply_grid_nms(certainty_weighted, kernel_size=3)
    kpts1_ts, kpts2_ts, _ = roma_model.get_sorted_matches(warp, certainty_filtered, H, W)
    num_points = min(len(kpts1_ts), 5000)
    kpts1, kpts2 = kpts1_ts[:num_points].cpu().numpy(), kpts2_ts[:num_points].cpu().numpy()
    
    # 3. Transform Estimation
    print(f"Estimating {method} transform...")
    M, tx_csv, tx_bspline = None, None, None
    
    if method.lower() == 'homography':
        M, _ = cv2.findHomography(kpts2, kpts1, cv2.RANSAC)
    elif method.lower() == 'bspline':
        M = fit_bspline_transform(kpts1, kpts2, H, W)
        tx_csv = fit_bspline_transform(kpts2, kpts1, H, W)
    elif method.lower() == 'affine+bspline':
        if len(kpts2) >= 3:
            A = np.hstack([kpts2, np.ones((len(kpts2), 1))])
            X, _, _, _ = np.linalg.lstsq(A, kpts1, rcond=None)
            M = X.T
        else: M = np.eye(2, 3, dtype=np.float32)
        kpts2_affine = (M @ np.hstack([kpts2, np.ones((len(kpts2), 1))]).T).T
        tx_bspline = fit_bspline_transform(kpts1, kpts2_affine, H, W)
        tx_csv = fit_bspline_transform(kpts2_affine, kpts1, H, W)
    else: # Default Affine/Rigid
        if len(kpts2) >= 3:
            A = np.hstack([kpts2, np.ones((len(kpts2), 1))])
            X, _, _, _ = np.linalg.lstsq(A, kpts1, rcond=None)
            M = X.T
        else: M = np.eye(2, 3, dtype=np.float32)

    # 4. Image Warping
    print("Warping images...")
    fill_color = im2_bgr_orig[0, 0].tolist()
    im2_resized = cv2.cvtColor(np.array(im2_pil_raw.resize((W, H))), cv2.COLOR_RGB2BGR)
    im1_resized = cv2.cvtColor(np.array(im1_pil_raw.resize((W, H))), cv2.COLOR_RGB2BGR)

    if method.lower() == 'homography':
        warped_small = cv2.warpPerspective(im2_resized, M, (W, H), borderValue=fill_color)
    elif method.lower() == 'bspline':
        warped_small = warp_image_bspline(im2_resized, M, (W, H)).astype(np.uint8)
    elif method.lower() == 'affine+bspline':
        im2_aff = cv2.warpAffine(im2_resized, M, (W, H), borderValue=fill_color)
        warped_small = warp_image_bspline(im2_aff, tx_bspline, (W, H)).astype(np.uint8)
    else:
        warped_small = cv2.warpAffine(im2_resized, M, (W, H), borderValue=fill_color)
        
    warped_im2_orig = cv2.resize(warped_small, (orig_W, orig_H), interpolation=cv2.INTER_LINEAR)

    # 5. H5AD Processing
    print("Processing H5AD coordinates...")
    h5ad_save_path = os.path.join(output_dir, "transformed.h5ad")
    
    def get_coords(adata_obj):
        if x_obs_col and y_obs_col and x_obs_col in adata_obj.obs:
            return np.column_stack((adata_obj.obs[x_obs_col], adata_obj.obs[y_obs_col])).astype(np.float32)
        return adata_obj.obsm[spatial_key].copy().astype(np.float32)

    if os.path.exists(h5ad_path):
        adata = sc.read_h5ad(h5ad_path)
        # debug_step_visualization(adata, im2_bgr_orig, "0_Raw_Source", output_dir, x_obs_col, y_obs_col,rotate=180.0,radius=3)
        pts = get_coords(adata)
        if (abs(rotate) > 1e-12) or (abs(scale - 1.0) > 1e-12):
            print(f"🔄 Applying manual pre-transform: rotate={rotate}°, scale={scale}")
            h, w = im2_bgr_orig.shape[:2]
            pts = apply_manual_transform(pts, rotate, scale,origin=origin)
            # 更新临时坐标回 adata 供可视化
            adata.obs[x_obs_col] = pts[:, 0]
            adata.obs[y_obs_col] = pts[:, 1]
            
            # === [调试点 1]：手动变换后 vs 原始 Source Image ===
            # 💡 注意：如果你在这里旋转了点，但没有旋转 im2_bgr_orig，这里一定会对不上！
            # debug_step_visualization(adata, im2_bgr_orig, "1_After_Manual", output_dir, x_obs_col, y_obs_col,rotate=0.0,radius=3)
        
        # Scaling
        scale_x, scale_y = W / orig_W2, H / orig_H2
        pts[:, 0] *= scale_x
        pts[:, 1] *= scale_y

        adata.obs[x_obs_col] = pts[:, 0]
        adata.obs[y_obs_col] = pts[:, 1]
        # === [调试点 2]：缩放后 vs RoMa 的中间图 (im2_resized) ===
        # debug_step_visualization(adata, im2_resized, "2_RoMa_Resolution", output_dir, x_obs_col, y_obs_col,rotate=0.0,radius=2)
        
        # Transform
        if method.lower() == 'bspline':
            pts_t = np.array([tx_csv.TransformPoint((float(p[0]), float(p[1]))) for p in pts])
        elif method.lower() == 'affine+bspline':
            pts_aff = cv2.transform(pts.reshape(-1, 1, 2), M).reshape(-1, 2)
            pts_t = np.array([tx_csv.TransformPoint((float(p[0]), float(p[1]))) for p in pts_aff])
        elif method.lower() == 'homography':
            pts_t = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), M).reshape(-1, 2)
        else:
            pts_t = cv2.transform(pts.reshape(-1, 1, 2), M).reshape(-1, 2)
        
        adata.obs[x_obs_col] = pts_t[:, 0]
        adata.obs[y_obs_col] = pts_t[:, 1]
        # === [调试点 3]：对齐变换后 vs 目标图 (im1_resized) ===
        # 💡 此时点应该已经映射到了 Target 图的 1152 空间
        # debug_step_visualization(adata, warped_small, "3_After_Alignment", output_dir, x_obs_col, y_obs_col,rotate=0.0,radius=2)
            
        # Rescale back
        scale_x_back, scale_y_back = orig_W / W, orig_H / H
        pts_transformed_orig = pts_t.copy()
        pts_transformed_orig[:, 0] *= scale_x_back
        pts_transformed_orig[:, 1] *= scale_y_back
        
        # Update Data
        adata.obsm[spatial_key] = pts_transformed_orig
        if x_obs_col: adata.obs[x_obs_col] = pts_transformed_orig[:, 0]
        if y_obs_col: adata.obs[y_obs_col] = pts_transformed_orig[:, 1]
        # === [调试点 4]：最终结果 vs 原始 Target Image ===
        # debug_step_visualization(adata, warped_im2_orig, "4_Final_Result", output_dir, x_obs_col, y_obs_col,rotate=0.0,radius=2)
        
        # === [Sanitize] Clean _index issues ===
        def sanitize_dataframe(df, name="df"):
            if '_index' in df.columns:
                print(f"⚠️ Removing '_index' column from {name}...")
                df.drop(columns=['_index'], inplace=True)
            if df.index.name == '_index':
                print(f"⚠️ Renaming index from '_index' to None in {name}...")
                df.index.name = None

        sanitize_dataframe(adata.obs, "adata.obs")
        sanitize_dataframe(adata.var, "adata.var")

        if adata.raw is not None:
            try:
                raw_adata = adata.raw.to_adata()
                if '_index' in raw_adata.var.columns or raw_adata.var.index.name == '_index':
                    print("⚠️ Found '_index' issue in adata.raw! Re-generating raw...")
                    sanitize_dataframe(raw_adata.var, "adata.raw.var")
                    adata.raw = raw_adata
            except Exception as e:
                print(f"⚠️ Warning: Could not sanitize adata.raw: {e}")

        # Save
        try:
            adata.write(h5ad_save_path)
            print(f"✅ Saved transformed H5AD to {h5ad_save_path}")
        except Exception as e:
            print(f"❌ Failed to save H5AD: {e}")
            h5ad_save_path = None
    else:
        h5ad_save_path = None

    # 6. Basic Visualization (Save to disk only)
    kpts1_orig = kpts1.copy()
    kpts1_orig[:, 0] *= (orig_W / W)
    kpts1_orig[:, 1] *= (orig_H / H)
    kpts2_orig = kpts2.copy()
    kpts2_orig[:, 0] *= (orig_W2 / W)
    kpts2_orig[:, 1] *= (orig_H2 / H)
    
    vis_matches = draw_matches_visualization(im1_resized, im2_resized, kpts1, kpts2, radius=3)
    cv2.imwrite(os.path.join(output_dir, "1_matches_color_coded.jpg"), vis_matches)

    h1, w1 = im1_bgr_orig.shape[:2]
    h2, w2 = warped_im2_orig.shape[:2]
    if h1 != h2: warped_im2_orig = cv2.resize(warped_im2_orig, (w2, h1))
    vis_compare = np.concatenate((im1_bgr_orig, warped_im2_orig), axis=1)
    cv2.imwrite(os.path.join(output_dir, "2_alignment_compare.jpg"), vis_compare)
    
    vis_img_overlay = cv2.addWeighted(im1_bgr_orig, 0.4, warped_im2_orig, 0.6, 0)
    cv2.imwrite(os.path.join(output_dir, "3_alignment_overlay.jpg"), vis_img_overlay)

    # show_image_in_jupyter(vis_matches,"RoMa Keypoint Matches")
    # show_image_in_jupyter(vis_compare,"Target vs. Aligned Source")
    # show_image_in_jupyter(vis_img_overlay,"Overlay")

    end_time = time.time()
    print(f"Alignment runtime: {end_time - start_time:.4f} seconds")
    print(f"Done. Results saved to {output_dir}")

# =========================
# Usage
# =========================
if __name__ == "__main__":
    img1 = "/mnt/A3/ivy/register_data/SPOmiAlign_Repro/output_image/Puck_Num_43.png"
    img2 = "/mnt/A3/ivy/register_data/SPOmiAlign_Repro/output_image/Puck_Num_29.png"
    h5ad_source = "/mnt/A3/ivy/register_data/SPOmiAlign_Repro/data_preprocessing/Puck_Num_29.h5ad"
    h5ad_target = "/mnt/A3/ivy/register_data/SPOmiAlign_Repro/data_preprocessing/Puck_Num_43.h5ad"

    if os.path.exists(img1) and os.path.exists(img2):
        align_and_process_images(
            img1, 
            img2, 
            h5ad_source, 
            h5ad_path_img1=h5ad_target,
            method="affine+bspline",
            output_dir='./Result/PUCK29_Slideseq_Final_Raster',
            x_obs_col="Raw_Slideseq_X",
            y_obs_col="Raw_Slideseq_Y"
        )