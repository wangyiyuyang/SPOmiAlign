import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

# --- 核心修改：使用与 demo_match_aligen.py 一致的模型接口 ---
from romatch import roma_outdoor

def evaluate_snr(certainty_map, img_path):
    """
    计算信噪比 (SNR)：边缘区域平均置信度 / 平坦区域平均置信度
    certainty_map: (H, W) numpy array, range [0, 1]
    """
    # 读取原图生成参考边缘 (Ground Truth)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: 
        print(f"无法读取图片: {img_path}")
        return 0, 0, 0
    
    # 调整大小以匹配 certainty map (因为模型可能会resize输入)
    h, w = certainty_map.shape
    img_resized = cv2.resize(img, (w, h))
    
    # 使用 Canny 算子提取物理边缘
    edges = cv2.Canny(img_resized, 100, 200)
    mask_edge = edges > 127
    mask_flat = ~mask_edge
    
    # 计算区域均值
    mean_edge = certainty_map[mask_edge].mean() if mask_edge.any() else 0
    mean_flat = certainty_map[mask_flat].mean() if mask_flat.any() else 0
    
    # SNR: 边缘响应相对于平坦区域的倍数
    snr = mean_edge / (mean_flat + 1e-6)
    return mean_edge, mean_flat, snr

def main():
    # 1. 配置参数
    # 请替换为您实际的测试图片路径
    imA_path = "/data/Newdisk/Bigmodel/zxm/Match/RoMa-main/RoMa-main/Dataset/simulated_MISAR/E15_5-S1-HE.jpg"
    imB_path = "/data/Newdisk/Bigmodel/zxm/Match/RoMa-main/RoMa-main/Dataset/simulated_MISAR/new/E15_5-S2-HE_warped_rt15.png"
    
    out_dir = "outputs_evaluation_roma_outdoor"
    os.makedirs(out_dir, exist_ok=True)

    # 傅里叶边缘加权参数
    edge_params = {
        "edge_sigma": 25,    # 控制边缘提取的频率范围
        "edge_decay": 15.0,  # 控制权重随距离衰减的速度
        "edge_power": 2.0    # 权重幂次，>0 开启加权
    }

    # 2. 设备与模型加载 (参考 demo_match_aligen.py)
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading RoMa outdoor model...")
    # 使用高分辨率设置，与 demo 保持一致
    roma_model = roma_outdoor(device=device, coarse_res=560, upsample_res=(864, 1152))
    
    # 获取模型预期的输出分辨率
    H_out, W_out = roma_model.get_output_resolution()
    print(f"Model output resolution: {H_out}x{W_out}")

    # ---------------------------------------------------------
    # 3. 基准推理 (Baseline): 关闭边缘加权
    # ---------------------------------------------------------
    print("\nRunning Baseline Inference (edge_power=0.0)...")
    with torch.no_grad():
        # 直接传入路径，模型内部会处理读取和resize
        # 传入 edge_power=0.0 以禁用加权逻辑
        warp_base, cert_base = roma_model.match(
            imA_path, imB_path, 
            device=device, 
            edge_power=0.0 
        )
        
        # --- 新增：采样匹配点 (Baseline) ---
        # 采样 5000 个点
        matches_base, _ = roma_model.sample(warp_base, cert_base, num=5000)
        # 获取像素坐标 (注意：这里需要知道模型内部resize后的尺寸，通常是 H_out, W_out)
        # roma_model.match 内部会将图片 resize 到 (H_out, W_out)
        kptsA_base, kptsB_base = roma_model.to_pixel_coordinates(matches_base, H_out, W_out, H_out, W_out)
        kptsA_base = kptsA_base.cpu().numpy()
        kptsB_base = kptsB_base.cpu().numpy()
        # -----------------------------------

        # 处理输出形状
        # roma_outdoor 默认 symmetric=True，certainty shape 通常为 (B, 1, H, 2*W) 或 (H, 2*W)
        cert_base = cert_base.squeeze() # -> (H, 2*W)
        
        # 提取左半部分 (对应 imA 的置信度图)
        if cert_base.shape[-1] == 2 * W_out:
            cert_base_np = cert_base[:, :W_out].cpu().numpy()
        else:
            # 如果模型配置改变导致非对称输出
            cert_base_np = cert_base.cpu().numpy()

    # ---------------------------------------------------------
    # 4. 加权推理 (Weighted): 开启边缘加权
    # ---------------------------------------------------------
    print(f"Running Weighted Inference (edge_power={edge_params['edge_power']})...")
    with torch.no_grad():
        # 传入我们在 matcher.py 中新增的参数
        warp_weighted, cert_weighted = roma_model.match(
            imA_path, imB_path, 
            device=device, 
            edge_power=edge_params['edge_power'],
            edge_sigma=edge_params['edge_sigma'],
            edge_decay=edge_params['edge_decay']
        )
        
        # --- 新增：采样匹配点 (Weighted) ---
        matches_weighted, _ = roma_model.sample(warp_weighted, cert_weighted, num=5000)
        kptsA_weighted, kptsB_weighted = roma_model.to_pixel_coordinates(matches_weighted, H_out, W_out, H_out, W_out)
        kptsA_weighted = kptsA_weighted.cpu().numpy()
        kptsB_weighted = kptsB_weighted.cpu().numpy()
        # -----------------------------------
        
        cert_weighted = cert_weighted.squeeze()
        
        # 同样提取左半部分
        if cert_weighted.shape[-1] == 2 * W_out:
            cert_weighted_np = cert_weighted[:, :W_out].cpu().numpy()
        else:
            cert_weighted_np = cert_weighted.cpu().numpy()

    # ---------------------------------------------------------
    # 5. 评估指标计算 (SNR)
    # ---------------------------------------------------------
    print("\n--- Evaluation Results ---")
    
    # 评估基准
    m_edge_b, m_flat_b, snr_b = evaluate_snr(cert_base_np, imA_path)
    print(f"[Baseline] Edge Mean: {m_edge_b:.4f}, Flat Mean: {m_flat_b:.4f}")
    print(f"[Baseline] SNR (Edge/Flat): {snr_b:.4f}")
    
    # 评估加权
    m_edge_w, m_flat_w, snr_w = evaluate_snr(cert_weighted_np, imA_path)
    print(f"[Weighted] Edge Mean: {m_edge_w:.4f}, Flat Mean: {m_flat_w:.4f}")
    print(f"[Weighted] SNR (Edge/Flat): {snr_w:.4f}")
    
    # 计算提升百分比
    if snr_b > 0:
        improvement = (snr_w - snr_b) / snr_b * 100
        print(f"SNR Improvement: {improvement:.2f}%")
    else:
        print("SNR Improvement: N/A (Baseline SNR is 0)")

    # ---------------------------------------------------------
    # 6. 可视化保存
    # ---------------------------------------------------------
    print("Generating visualizations...")
    
    # 读取原图用于叠加 (转为灰度再转RGB，保持背景素雅)
    img_A = cv2.imread(imA_path)
    img_B = cv2.imread(imB_path) # 读取图B用于画连线
    
    if img_A is None:
        img_A = np.zeros((cert_base_np.shape[0], cert_base_np.shape[1], 3), dtype=np.uint8)
    if img_B is None:
        img_B = np.zeros((cert_base_np.shape[0], cert_base_np.shape[1], 3), dtype=np.uint8)
    
    # 调整原图大小以匹配输出 (H_out, W_out)
    # 注意：cert_base_np 的尺寸就是 (H_out, W_out)
    h, w = cert_base_np.shape
    img_A_resized = cv2.resize(img_A, (w, h))
    img_B_resized = cv2.resize(img_B, (w, h)) # 同样调整图B
    
    # 转为灰度背景，以免原图色彩干扰热力图
    img_A_gray = cv2.cvtColor(img_A_resized, cv2.COLOR_BGR2GRAY)
    img_A_bgr = cv2.cvtColor(img_A_gray, cv2.COLOR_GRAY2BGR)

    def create_overlay(certainty_map, bg_img):
        # 1. 动态 Min-Max 归一化 (增强对比度)
        c_min = certainty_map.min()
        c_max = certainty_map.max()
        
        # 防止除以零
        if c_max - c_min < 1e-6:
            norm = np.zeros_like(certainty_map)
        else:
            norm = (certainty_map - c_min) / (c_max - c_min)
            
        # 2. 再次截断以防万一
        norm = np.clip(norm, 0, 1)
        
        # 3. 转为 uint8
        norm_uint8 = (norm * 255).astype(np.uint8)
        
        # 4. 生成热力图 (JET: 蓝=低, 红=高)
        heatmap = cv2.applyColorMap(norm_uint8, cv2.COLORMAP_JET)
        
        # 5. 叠加: 0.6 * 热力图 + 0.4 * 原图
        overlay = cv2.addWeighted(heatmap, 0.6, bg_img, 0.4, 0)
        
        # 打印统计信息以便调试
        print(f"  [Map Stats] Min: {c_min:.4f}, Max: {c_max:.4f} -> Normalized to [0, 1]")
        
        return overlay, heatmap

    # --- 新增：绘制匹配连线函数 ---
    def draw_matches_side_by_side(img1, img2, kpts1, kpts2, name="matches"):
        """
        img1, img2: (H, W, 3) BGR images
        kpts1, kpts2: (N, 2) numpy arrays, coordinates in (x, y)
        """
        h, w = img1.shape[:2]
        # 创建并排画布
        vis = np.zeros((h, w * 2, 3), dtype=np.uint8)
        vis[:, :w] = img1
        vis[:, w:] = img2
        
        # 随机颜色
        # 绘制连线
        # 为了避免太乱，只画前 200 个点或者随机采样
        num_draw = min(len(kpts1), 200)
        indices = np.random.choice(len(kpts1), num_draw, replace=False)
        
        for idx in indices:
            pt1 = (int(kpts1[idx, 0]), int(kpts1[idx, 1]))
            pt2 = (int(kpts2[idx, 0] + w), int(kpts2[idx, 1])) # 右图x坐标偏移 w
            
            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.line(vis, pt1, pt2, color, 1, cv2.LINE_AA)
            cv2.circle(vis, pt1, 2, color, -1)
            cv2.circle(vis, pt2, 2, color, -1)
            
        return vis

    # --- 新增：绘制密集匹配点分布 (不画连线) ---
    def draw_dense_matches(img, kpts, color=(0, 255, 0), radius=1, alpha=0.4):
        """
        在单张图上绘制所有匹配点，使用半透明叠加显示密度
        img: (H, W, 3) BGR
        kpts: (N, 2)
        """
        # 创建一个覆盖层
        overlay = img.copy()
        print(len(kpts))
        # 批量绘制点 (OpenCV 的 circle 不支持批量，只能循环，但对于几千个点很快)
        # 为了速度和效果，我们直接操作像素或使用 matplotlib，这里用 cv2 模拟
        
        for pt in kpts:
            cv2.circle(overlay, (int(pt[0]), int(pt[1])), radius, color, -1)
            
        # 混合原图和覆盖层，产生透明效果
        return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    # --- 新增：颜色编码对应关系可视化 ---
    def draw_color_coded_matches(img1, img2, kpts1, kpts2, radius=2):
        """
        使用位置编码颜色来展示对应关系。
        1. 根据 kpts1 的位置生成颜色 (HSV 空间: x->Hue, y->Saturation/Value)
        2. 用相同的颜色绘制 kpts2
        这样可以通过颜色直观判断匹配点来自图1的哪个区域。
        """
        h, w = img1.shape[:2]
        
        # 创建画布
        vis1 = img1.copy()
        vis2 = img2.copy()
        
        # 归一化坐标用于生成颜色
        # x_norm: 0~1, y_norm: 0~1
        x_norm = kpts1[:, 0] / w
        y_norm = kpts1[:, 1] / h
        
        # 生成颜色 (HSV -> BGR)
        # Hue (色相): 由 x 坐标决定 (0~179)
        # Saturation (饱和度): 设为 255 (鲜艳)
        # Value (亮度): 由 y 坐标决定 (100~255, 避免太黑)
        
        # 为了计算快，我们用 numpy 批量操作，然后转 uint8
        hue = (x_norm * 179).astype(np.uint8)
        sat = np.ones_like(hue) * 255
        val = ((1 - y_norm) * 155 + 100).astype(np.uint8) # y越大越暗，或者反过来
        
        # 组合成 HSV 图像 (N, 1, 3)
        hsv_pixels = np.stack([hue, sat, val], axis=1).reshape(-1, 1, 3)
        # 转为 BGR
        bgr_pixels = cv2.cvtColor(hsv_pixels, cv2.COLOR_HSV2BGR).reshape(-1, 3)
        
        # 绘制点
        for i, (pt1, pt2) in enumerate(zip(kpts1, kpts2)):
            color = bgr_pixels[i].tolist()
            # 实心圆
            cv2.circle(vis1, (int(pt1[0]), int(pt1[1])), radius, color, -1)
            cv2.circle(vis2, (int(pt2[0]), int(pt2[1])), radius, color, -1)
            
        # 拼接
        return np.hstack((vis1, vis2))

    # --- 新增：绘制 Top-K 高置信度匹配点 ---
    def draw_topk_matches(img1, img2, kpts1, kpts2, scores, k=5000, radius=1):
        """
        根据 scores 筛选前 k 个点进行绘制 (颜色编码)
        """
        if len(scores) > k:
            # 获取前 k 个最高分的索引
            # argsort 是升序，取最后 k 个并反转
            top_indices = np.argsort(scores)[-k:][::-1]
            kpts1_top = kpts1[top_indices]
            kpts2_top = kpts2[top_indices]
        else:
            kpts1_top = kpts1
            kpts2_top = kpts2
            
        # 复用之前的颜色编码绘制函数
        return draw_color_coded_matches(img1, img2, kpts1_top, kpts2_top, radius=radius)

    # 1. 获取 Baseline 的所有采样点分数 (需要重新采样或修改 sample 接口，
    # 但 roma_model.sample 返回的 matches 已经是采样后的结果，
    # 我们需要获取这些点对应的 certainty 值)
    
    # 为了准确获取 Top-K，我们直接从原始 certainty map 中提取
    # 注意：这里我们简单地复用之前采样出的 5000 个点，
    # 因为 roma_model.sample 已经做了加权采样，这里我们假设要看这 5000 个点里
    # 哪些是相对更高分的，或者直接画出这 5000 个点（因为它们本身就是高分代表）。
    
    # 如果您是想从全图中严格筛选 Top 5000 (而不是概率采样)，
    # 需要手动操作 certainty map。以下是实现严格 Top-K 的逻辑：

    def get_strict_topk_kpts(warp, certainty, H, W, k=5000):
        """
        从全图中严格筛选 certainty 最高的 k 个点
        """
        # 展平 certainty
        cert_flat = certainty.flatten() # (H*W,)
        
        # 获取 Top-K 索引
        if cert_flat.numel() > k:
            topk_vals, topk_indices = torch.topk(cert_flat, k)
        else:
            topk_vals = cert_flat
            topk_indices = torch.arange(cert_flat.numel(), device=cert_flat.device)
            
        # 将索引转回 (y, x) 坐标
        # 注意：certainty shape 是 (B, 1, H, W) 或 (H, W)
        # 这里假设输入已经是 (H, W)
        h_map, w_map = certainty.shape[-2:]
        
        # 计算采样点在 feature map 上的坐标
        y_norm = (topk_indices // w_map).float() / (h_map - 1)
        x_norm = (topk_indices % w_map).float() / (w_map - 1)
        
        # 对应的 warp 值 (目标图坐标)
        # warp shape: (B, 2, H, W) -> (2, H, W)
        warp_flat = warp.reshape(2, -1) # (2, H*W)
        warp_topk = warp_flat[:, topk_indices] # (2, k)
        
        # 源图坐标 (归一化 -1~1 -> 0~1)
        # 注意 RoMa 的 warp 是 -1~1，我们需要转换
        # 但这里我们直接用网格坐标作为源图坐标
        kpts1_norm = torch.stack([x_norm, y_norm], dim=1) # (k, 2) range [0, 1]
        
        # 目标图坐标 (warp 值是 -1~1)
        kpts2_norm = (warp_topk.permute(1, 0) + 1) / 2 # (k, 2) range [0, 1]
        
        # 转为像素坐标
        kpts1_pixel = kpts1_norm * torch.tensor([W, H], device=kpts1_norm.device)
        kpts2_pixel = kpts2_norm * torch.tensor([W, H], device=kpts2_norm.device)
        
        return kpts1_pixel.cpu().numpy(), kpts2_pixel.cpu().numpy()

    print("Extracting Strict Top-5000 Matches by Certainty...")
    
    # Baseline Top-5000
    # 注意：cert_base 之前被 squeeze 成了 (H, 2*W)，我们需要切分回 (H, W)
    # 且 warp_base 需要对应
    # 重新获取一下原始输出比较安全，或者复用之前的逻辑
    # 假设 cert_base_np 是 numpy (H, W)，我们需要 torch tensor
    
    # 简便起见，我们直接利用之前计算好的 cert_base (Tensor) 和 warp_base
    # warp_base: (B, 2, H, W)
    # cert_base: (B, 1, H, W) 或 (H, W)
    
    # 确保维度正确 (取 Batch 0)
    if len(warp_base.shape) == 4:
        w_b = warp_base[0]
        c_b = cert_base if len(cert_base.shape)==3 else cert_base.unsqueeze(0) # 确保有 channel
        if c_b.shape[0] != 1: c_b = c_b.unsqueeze(0) # (1, H, W)
    else:
        w_b = warp_base
        c_b = cert_base
        
    # 截取左半部分 (因为 RoMa 输出是拼接的)
    h_map, w_map = w_b.shape[-2:]
    w_half = w_map // 2
    
    # 提取左图对应的 warp 和 certainty
    w_b_left = w_b[:, :, :w_half]
    c_b_left = c_b[:, :w_half] if c_b.dim() == 2 else c_b[:, :, :w_half]
    
    kptsA_top_base, kptsB_top_base = get_strict_topk_kpts(w_b_left, c_b_left, H_out, W_out, k=5000)
    
    # Weighted Top-5000
    if len(warp_weighted.shape) == 4:
        w_w = warp_weighted[0]
        c_w = cert_weighted if len(cert_weighted.shape)==3 else cert_weighted.unsqueeze(0)
    else:
        w_w = warp_weighted
        
    w_w_left = w_w[:, :, :w_half]
    c_w_left = c_w[:, :w_half] if c_w.dim() == 2 else c_w[:, :, :w_half]
    
    kptsA_top_weighted, kptsB_top_weighted = get_strict_topk_kpts(w_w_left, c_w_left, H_out, W_out, k=5000)

    # --- 修改：基于已采样点的 Top-K 筛选 ---
    def get_topk_from_sampled(kpts1, kpts2, certainty_map, k=5000):
        """
        从已有的采样点中，根据 certainty_map 的值筛选 Top-K
        kpts1: (N, 2) 像素坐标
        certainty_map: (H, W) numpy array
        """
        h, w = certainty_map.shape
        scores = []
        valid_indices = []
        
        for i, pt in enumerate(kpts1):
            x, y = int(pt[0]), int(pt[1])
            # 边界检查
            if 0 <= x < w and 0 <= y < h:
                score = certainty_map[y, x]
                scores.append(score)
                valid_indices.append(i)
                
        scores = np.array(scores)
        valid_indices = np.array(valid_indices)
        
        # 筛选 Top-K
        if len(scores) > k:
            # argsort 是升序，取最后 k 个并反转 -> 降序
            top_k_idx_in_scores = np.argsort(scores)[-k:][::-1]
            final_indices = valid_indices[top_k_idx_in_scores]
        else:
            final_indices = valid_indices
            
        return kpts1[final_indices], kpts2[final_indices]

    print("Extracting Top-5000 Matches from Sampled Points...")
    
    # Baseline Top-5000 (从之前 sample 的点中筛选)
    # 注意：之前的 sample 已经是 5000 个了，如果想看更“精”的，可以设 k=2000
    # 或者如果之前 sample 了更多点（比如 10000），这里选 Top 5000
    # 这里演示从已有的 kptsA_base 中按分数排序（虽然 sample 已经是加权采样，但不是严格 Top-K）
    
    kptsA_top_base, kptsB_top_base = get_topk_from_sampled(kptsA_base, kptsB_base, cert_base_np, k=5000)
    
    # Weighted Top-5000
    kptsA_top_weighted, kptsB_top_weighted = get_topk_from_sampled(kptsA_weighted, kptsB_weighted, cert_weighted_np, k=5000)

    # 绘制 Top-K 颜色编码图
    print("Drawing Top-5000 Sorted Matches...")
    vis_topk_base = draw_color_coded_matches(img_A_resized, img_B_resized, kptsA_top_base, kptsB_top_base)
    cv2.imwrite(os.path.join(out_dir, "matches_top5000_baseline.png"), vis_topk_base)
    
    vis_topk_weighted = draw_color_coded_matches(img_A_resized, img_B_resized, kptsA_top_weighted, kptsB_top_weighted)
    cv2.imwrite(os.path.join(out_dir, "matches_top5000_weighted.png"), vis_topk_weighted)
    
    print(f"  - matches_top5000_*.png: 基于采样点筛选的高置信度匹配")

    # 绘制 Top-K 颜色编码图
    print("Drawing Top-5000 Strict Certainty Matches...")
    vis_topk_base = draw_color_coded_matches(img_A_resized, img_B_resized, kptsA_top_base, kptsB_top_base)
    cv2.imwrite(os.path.join(out_dir, "matches_top5000_baseline.png"), vis_topk_base)
    
    vis_topk_weighted = draw_color_coded_matches(img_A_resized, img_B_resized, kptsA_top_weighted, kptsB_top_weighted)
    cv2.imwrite(os.path.join(out_dir, "matches_top5000_weighted.png"), vis_topk_weighted)
    
    print(f"  - matches_top5000_*.png: 严格筛选置信度最高的5000个点")

    # 绘制 Baseline 匹配连线
    print("Drawing Baseline Matches...")
    vis_matches_base = draw_matches_side_by_side(img_A_resized, img_B_resized, kptsA_base, kptsB_base)
    cv2.imwrite(os.path.join(out_dir, "matches_baseline.png"), vis_matches_base)
    
    # 绘制 Weighted 匹配连线
    print("Drawing Weighted Matches...")
    vis_matches_weighted = draw_matches_side_by_side(img_A_resized, img_B_resized, kptsA_weighted, kptsB_weighted)
    cv2.imwrite(os.path.join(out_dir, "matches_weighted.png"), vis_matches_weighted)
    # -----------------------------

    # 绘制 Baseline 密集点分布
    print("Drawing Baseline Dense Points...")
    # 图A上的点 (绿色)
    vis_kptsA_base = draw_dense_matches(img_A_resized, kptsA_base, color=(0, 255, 0))
    # 图B上的点 (红色)
    vis_kptsB_base = draw_dense_matches(img_B_resized, kptsB_base, color=(0, 0, 255))
    
    # 拼接显示
    vis_dense_base = np.hstack((vis_kptsA_base, vis_kptsB_base))
    cv2.imwrite(os.path.join(out_dir, "dense_points_baseline.png"), vis_dense_base)
    
    # 绘制 Weighted 密集点分布
    print("Drawing Weighted Dense Points...")
    vis_kptsA_weighted = draw_dense_matches(img_A_resized, kptsA_weighted, color=(0, 255, 0))
    vis_kptsB_weighted = draw_dense_matches(img_B_resized, kptsB_weighted, color=(0, 0, 255))
    
    vis_dense_weighted = np.hstack((vis_kptsA_weighted, vis_kptsB_weighted))
    cv2.imwrite(os.path.join(out_dir, "dense_points_weighted.png"), vis_dense_weighted)
    # -----------------------------

    # 绘制 Baseline 颜色编码图
    print("Drawing Baseline Color-Coded Matches...")
    vis_color_base = draw_color_coded_matches(img_A_resized, img_B_resized, kptsA_base, kptsB_base)
    cv2.imwrite(os.path.join(out_dir, "matches_color_coded_baseline.png"), vis_color_base)
    
    # 绘制 Weighted 颜色编码图
    print("Drawing Weighted Color-Coded Matches...")
    vis_color_weighted = draw_color_coded_matches(img_A_resized, img_B_resized, kptsA_weighted, kptsB_weighted)
    cv2.imwrite(os.path.join(out_dir, "matches_color_coded_weighted.png"), vis_color_weighted)

    print("Processing Baseline Visualization...")
    overlay_base, heatmap_base = create_overlay(cert_base_np, img_A_bgr)
    
    print("Processing Weighted Visualization...")
    overlay_weighted, heatmap_weighted = create_overlay(cert_weighted_np, img_A_bgr)
    
    # 计算差异图 (使用热力图表示差异大小)
    diff = np.abs(cert_weighted_np - cert_base_np)
    # 放大差异以便观察 (x5) 并截断
    diff_norm = np.clip(diff * 5, 0, 1)
    diff_uint8 = (diff_norm * 255).astype(np.uint8)
    diff_heatmap = cv2.applyColorMap(diff_uint8, cv2.COLORMAP_JET)

    # 保存图片
    cv2.imwrite(os.path.join(out_dir, "overlay_baseline.png"), overlay_base)
    cv2.imwrite(os.path.join(out_dir, "overlay_weighted.png"), overlay_weighted)
    cv2.imwrite(os.path.join(out_dir, "heatmap_baseline.png"), heatmap_base)
    cv2.imwrite(os.path.join(out_dir, "heatmap_weighted.png"), heatmap_weighted)
    cv2.imwrite(os.path.join(out_dir, "diff_heatmap_x5.png"), diff_heatmap)
    
    print(f"\nVisualizations saved to directory: {out_dir}")
    print(f"  - overlay_*.png: 热力图叠加在原图上")
    print(f"  - heatmap_*.png: 纯热力图")
    print(f"  - diff_heatmap_x5.png: 差异热力图")
    print(f"  - matches_*.png: 匹配点连线图")
    print(f"  - dense_points_*.png: 密集匹配点分布图 (左绿右红)")
    print(f"  - matches_color_coded_*.png: 颜色编码对应关系图 (推荐查看)")

if __name__ == "__main__":
    main()