import os 
import cv2 
import sys 
import numpy as np
from pathlib import Path

# --- 添加路径以便导入模块 ---
# 如果你的 roma.py, data_preprocessing.py 和 reassignment.py 都在同级目录或 ../../SPOmiAlign
sys.path.append("../../SPOmiAlign")

from roma import align_and_process_images
from data_preprocessing import rasterize_h5ad_to_image
# ✅ 新增：从 reassignment.py 导入核心处理函数
from reassignment import spomialign_reassignment

# =========================
# 配置路径
# =========================
DATA_DIR = "../../SPOmiAlign_Repro"
h5ad_img1_path = os.path.join(DATA_DIR, "output_h5ad", "st_withIntensity.h5ad") # Target (ST)
h5ad_img2_path = os.path.join(DATA_DIR, "output_h5ad", "sm_withIntensity.h5ad") # Source (SM)

SAVE_DIR = "../../output"
SAVE_PATH = os.path.join(SAVE_DIR, "h5ad_2_h5ad", "sm2st")
os.makedirs(SAVE_PATH, exist_ok=True)
print(f"目录已准备就绪: {os.path.abspath(SAVE_PATH)}")

# =========================
# 文件检查
# =========================
files_to_check = {
    "H5AD 目标数据": h5ad_img1_path,
    "H5AD 源数据": h5ad_img2_path
}

print("\n🔍 正在检查输入文件...")
missing_files = []
for name, path in files_to_check.items():
    if os.path.exists(path):
        file_size = os.path.getsize(path) / (1024 * 1024)
        print(f"✅ {name} 已找到: {os.path.basename(path)} ({file_size:.2f} MB)")
    else:
        print(f"❌ {name} 不存在: {os.path.abspath(path)}")
        missing_files.append(path)

if missing_files:
    sys.exit("❌ 程序终止：缺失必要输入文件。")
else:
    print("🚀 所有文件准备就绪，准备开始处理。\n" + "-"*30)

# =========================
# 步骤一：H5AD 转化成图像
# =========================
print("🚀 步骤一 H5ad转化成图像。\n" + "-"*30)
Gen_img1_path = os.path.join(SAVE_PATH, "st.png")
Gen_img2_path = os.path.join(SAVE_PATH, "sm.png")

# 生成 Target 图像
rasterize_h5ad_to_image(
    input_h5ad=h5ad_img1_path,
    output_png=Gen_img1_path,
    background="white",
    point_shape="square",
    radius=15,
    threshold_percentile=None,
    intensity_log_transform=False,
    enhance=False,
    rotate=0.0,
    scale=1.0,
)

# 生成 Source 图像
rasterize_h5ad_to_image(
    input_h5ad=h5ad_img2_path,
    output_png=Gen_img2_path,
    background="white",
    point_shape="square",
    radius=12,
    threshold_percentile=None,
    intensity_log_transform=False,
    enhance=False,
    rotate=60.0,
    scale=0.6,
)

# =========================
# 步骤二：图像对齐 (Alignment)
# =========================
print("🚀 步骤二 将h5ad图像和target图像对齐。\n" + "-"*30)
save_path_alignment = os.path.join(SAVE_PATH, "alignment")
transformed_h5ad_path = os.path.join(save_path_alignment, "transformed.h5ad") # 这是对齐后的中间结果
transformed_h5ad_img_path = os.path.join(save_path_alignment, "transformed_h5ad.png")

align_and_process_images(
    img1_path=Gen_img1_path, 
    img2_path=Gen_img2_path, 
    h5ad_path=h5ad_img2_path,  
    method='affine+bspline', 
    output_dir=save_path_alignment,
    rotate=0.0, 
    scale=1.0,
)

# 生成对齐后的预览图
rasterize_h5ad_to_image(
    input_h5ad=transformed_h5ad_path,
    output_png=transformed_h5ad_img_path,
    background="white",
    point_shape="square",
    radius=12,
    threshold_percentile=None,
    intensity_log_transform=False,
    enhance=False,
    rotate=0.0,
    scale=1.0,
)

# =========================
# 步骤三：叠加预览 (Visualization)
# =========================
print("\n🚀 步骤三 正在生成直接叠加对比图...")

target_img = cv2.imread(Gen_img1_path)
aligned_h5ad_gray = cv2.imread(transformed_h5ad_img_path, cv2.IMREAD_GRAYSCALE)

if target_img is None or aligned_h5ad_gray is None:
    print("❌ 图像读取失败，跳过预览生成。")
else:
    aligned_h5ad_bgr = cv2.cvtColor(aligned_h5ad_gray, cv2.COLOR_GRAY2BGR)
    t_h, t_w = target_img.shape[:2]
    a_h, a_w = aligned_h5ad_bgr.shape[:2]

    # 白色背景画布
    h5ad_full_canvas = np.full((t_h, t_w, 3), 255, dtype=np.uint8)
    h_limit = min(t_h, a_h)
    w_limit = min(t_w, a_w)
    h5ad_full_canvas[:h_limit, :w_limit] = aligned_h5ad_bgr[:h_limit, :w_limit]

    # 叠加
    overlay_img = cv2.addWeighted(target_img, 0.5, h5ad_full_canvas, 0.5, 0)
    
    # 保存
    overlay_save_path = os.path.join(save_path_alignment, "h5ad_alignment_overlay.png")
    cv2.imwrite(overlay_save_path, overlay_img)
    
    comparison = np.hstack((target_img, overlay_img))
    cv2.imwrite(os.path.join(save_path_alignment, "h5ad_side_by_side.png"), comparison)
    print(f"✅ 叠加预览图已保存：{overlay_save_path}")

# =========================
# 步骤四：Reassignment (数据重分配)
# =========================
print("\n🚀 步骤四 执行 Reassignment...")
print("-" * 30)

# 定义最终输出路径
final_reassigned_h5ad = os.path.join(SAVE_PATH, "sm2st_final_reassigned.h5ad")
mapping_csv_path = os.path.join(SAVE_PATH, "sm2st_mapping_table.csv")

# 调用导入的函数
# s1_h5ad = 目标 (Target/ST)
# s2_h5ad = 对齐后的源数据 (Transformed Source/SM) -> 这一步很关键，必须用 transformed_h5ad_path
spomialign_reassignment(
    s1_h5ad=h5ad_img1_path,         
    s2_h5ad=transformed_h5ad_path,  
    out_h5ad=final_reassigned_h5ad,
    map_csv=mapping_csv_path,
    id_col="id",                    # 根据你数据中的 ID 列名调整
    cluster_col="cluster",          # 想要从 ST 继承的列
    s2_cluster_col=["s2_cluster_col"], # 想要从 SM 继承的列
    scale_by_mapping_factor=True    # 开启密度归一化
)

print("\n✨✨✨ 所有流程执行完毕！ ✨✨✨")