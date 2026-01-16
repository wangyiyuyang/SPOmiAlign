import os 
import cv2 
import sys 
from pathlib import Path
import numpy as np
parent_dir = str(Path(__file__).resolve().parent.parent.parent)
sys.path.append("../../SPOmiAlign")
from roma import align_and_process_images
from data_preprocessing import rasterize_h5ad_to_image

DATA_DIR="../../SPOmiAlign_Repro"
h5ad_path=os.path.join(DATA_DIR,"data_preprocessing","Puck_Num_43.h5ad")
target_image_path=os.path.join(DATA_DIR,"output_reference","CCF_100048576_273.png")
niss_image_path=os.path.join(DATA_DIR,"output_niss","niss_40.png")
SAVE_DIR="../../output"
SAVE_PATH=os.path.join(SAVE_DIR,"h5ad_2_img","PUCK43")
os.makedirs(SAVE_PATH, exist_ok=True)

os.makedirs(SAVE_PATH, exist_ok=True)
print(f"目录已准备就绪: {os.path.abspath(SAVE_PATH)}")

# =========================
# 文件存在性验证
# =========================
files_to_check = {
    "H5AD 数据": h5ad_path,
    "目标参考图像": target_image_path
}

print("\n🔍 正在检查输入文件...")
missing_files = []

for name, path in files_to_check.items():
    if os.path.exists(path):
        # 进一步检查文件是否为空（可选）
        file_size = os.path.getsize(path) / (1024 * 1024)  # MB
        print(f"✅ {name} 已找到: {os.path.basename(path)} ({file_size:.2f} MB)")
    else:
        print(f"❌ {name} 不存在: {os.path.abspath(path)}")
        missing_files.append(path)

if missing_files:
    print("\n程序终止：请确保所有输入文件路径正确。")
    sys.exit(1) # 强行退出，防止后续报错
else:
    print("🚀 所有文件准备就绪，准备开始处理。\n" + "-"*30)
print("🚀 步骤一 H5ad转化成图像。\n" + "-"*30)
Gen_img_path=os.path.join(SAVE_PATH,"Gen_img_PUCK43.png")
_,origin=rasterize_h5ad_to_image(
    input_h5ad=h5ad_path,
    output_png=Gen_img_path,
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

print("🚀 步骤二 将h5ad图像和niss图像对齐。\n" + "-"*30)
save_path_alignment_with_niss=os.path.join(SAVE_PATH,"alignment_with_niss")
transformed_with_niss_h5ad_path=os.path.join(save_path_alignment_with_niss,"transformed.h5ad")
transformed_with_niss_h5ad_img_path=os.path.join(save_path_alignment_with_niss,"transformed_h5ad.png")
img2_warpped_with_niss=os.path.join(save_path_alignment_with_niss,"aligned_source_img2.png")
align_and_process_images(
    img1_path=niss_image_path, 
    img2_path=Gen_img_path, 
    h5ad_path=h5ad_path,  
    method='affine+bspline', 
    output_dir=save_path_alignment_with_niss,     
    x_obs_col="Raw_Slideseq_X",            
    y_obs_col="Raw_Slideseq_Y",
    # === 新增参数 ===
    rotate=90.0,
    scale=1.0,
    origin=origin
)


rasterize_h5ad_to_image(
    input_h5ad=transformed_with_niss_h5ad_path,
    output_png=transformed_with_niss_h5ad_img_path,
    x_obs_col="Raw_Slideseq_X",
    y_obs_col="Raw_Slideseq_Y",
    # intensity_mode="obs_col",
    intensity_obs_col="nFeature_Spatial",
    intensity_log_transform=True,   # nFeature 计数建议 log1p
    threshold_percentile=80,        # 80 分位筛点
    background="black",
    point_shape="circle",
    radius=2,
    enhance=True,                  # 你前面那套增强
    rotate=0.0,
    scale=1.0,

)

print("🚀 步骤三 将h5ad图像和ccf(target)图像对齐。\n" + "-"*30)
save_path_alignment_with_ccf=os.path.join(SAVE_PATH,"alignment_with_ccf")
transformed_with_ccf_h5ad_path=os.path.join(save_path_alignment_with_ccf,"transformed.h5ad")
transformed_with_ccf_h5ad_img_path=os.path.join(save_path_alignment_with_ccf,"transformed_h5ad.png")
align_and_process_images(
    img1_path=target_image_path, 
    img2_path=niss_image_path, 
    h5ad_path=transformed_with_niss_h5ad_path,  
    method='affine+bspline', 
    output_dir=save_path_alignment_with_ccf,     
    x_obs_col="Raw_Slideseq_X",            
    y_obs_col="Raw_Slideseq_Y",
    # === 新增参数 ===
    rotate=0.0,
    scale=1.0,
    origin=origin
)


rasterize_h5ad_to_image(
    input_h5ad=transformed_with_ccf_h5ad_path,
    output_png=transformed_with_ccf_h5ad_img_path,
    x_obs_col="Raw_Slideseq_X",
    y_obs_col="Raw_Slideseq_Y",
    # intensity_mode="obs_col",
    intensity_obs_col="nFeature_Spatial",
    intensity_log_transform=True,   # nFeature 计数建议 log1p
    threshold_percentile=80,        # 80 分位筛点
    background="black",
    point_shape="circle",
    radius=2,
    enhance=True,                  # 你前面那套增强
    rotate=0.0,
    scale=1.0,

)


# =========================
# 步骤四：读取结果，并与彩色 Target Image 叠加预览
# =========================
print("\n🚀 步骤四 正在生成彩色叠加对比图...")
# 1. 读取原图（不带 GRAYSCALE 参数，默认读取 BGR 3通道）
target_img=cv2.imread(target_image_path)
# 对齐后的点位图依然读取为灰度，作为掩膜(Mask)使用
aligned_h5ad_img = cv2.imread(transformed_with_ccf_h5ad_img_path, cv2.IMREAD_GRAYSCALE)

if target_img is None or aligned_h5ad_img is None:
    print("❌ 图像读取失败，请检查路径。")
else:
    # 获取尺寸
    t_h, t_w = target_img.shape[:2]
    a_h, a_w = aligned_h5ad_img.shape[:2]
    print(f"📏 Target 尺寸: {t_w}x{t_h} (彩色) | Aligned H5AD 尺寸: {a_w}x{a_h}")

    # 2. 创建匹配 Target 尺寸的掩膜画布
    # 确保即使 H5AD 图像尺寸小，也能准确对齐到左上角
    h5ad_mask_full = np.zeros((t_h, t_w), dtype=np.uint8)
    h_limit = min(t_h, a_h)
    w_limit = min(t_w, a_w)
    h5ad_mask_full[:h_limit, :w_limit] = aligned_h5ad_img[:h_limit, :w_limit]

    # 3. 叠加预览：在原色图上涂色
    # 我们创建一个副本，不破坏原始 target_img
    overlay_img = target_img.copy()

    # 设定一个阈值，h5ad_mask_full 中大于此值的像素被视为“有点”
    # 如果你 background="black"，则点是亮的（值大）
    point_threshold = 50 
    mask = h5ad_mask_full > point_threshold

    # 将有像素点的地方设为亮绿色 [B, G, R] -> [0, 255, 0]
    # 如果你想用红色，就改用 [0, 0, 255]
    overlay_img[mask] = [0, 255, 0] 

    # 4. 进阶：半透明叠加（可选，如果你想同时看到点下面的原图纹理）
    # alpha 是原图权重，beta 是点位图权重
    # alpha_img = cv2.addWeighted(target_img, 0.7, overlay_img, 0.3, 0)

    # 5. 保存结果
    overlay_save_path = os.path.join(save_path_alignment_with_ccf, "color_alignment_overlay.png")
    cv2.imwrite(overlay_save_path, overlay_img)
    
    # 同时保存一份原图和叠加图的左右对比
    comparison = np.hstack((target_img, overlay_img))
    cv2.imwrite(os.path.join(save_path_alignment_with_ccf, "color_side_by_side.png"), comparison)

    print(f"✅ 彩色叠加图已保存：{overlay_save_path}")
print("\n✨✨✨ 所有流程执行完毕！ ✨✨✨")