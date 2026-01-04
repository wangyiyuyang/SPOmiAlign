import os 
import cv2 
import sys 
import numpy as np
from pathlib import Path

# --- 添加路径以便导入模块 ---
# 如果你的 roma.py, data_preprocessing.py 和 reassignment.py 都在同级目录或 ../../SPOmiAlign
sys.path.append("../../SPOmiAlign")

from roma import align_and_process_images


# =========================
# 配置路径
# =========================
DATA_DIR = "../../SPOmiAlign_Repro"
img1_path = os.path.join(DATA_DIR, "output_image", "E15_5-S1-HE.jpg") # Target (ST)
img2_path = os.path.join(DATA_DIR, "output_image", "E15_5-S2-HE_warped_rt15.png") # Source (SM)

SAVE_DIR = "../../output"
SAVE_PATH = os.path.join(SAVE_DIR, "img_2_img", "S2toS1")
os.makedirs(SAVE_PATH, exist_ok=True)
print(f"目录已准备就绪: {os.path.abspath(SAVE_PATH)}")

# =========================
# 文件检查
# =========================
files_to_check = {
    "目标图片数据": img1_path,
    "源图片数据": img2_path
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
# 步骤一：图像对齐 (Alignment)
# =========================
print("🚀 步骤一 将source图像和target图像对齐。\n" + "-"*30)
save_path_alignment = os.path.join(SAVE_PATH, "alignment")


align_and_process_images(
    img1_path=img1_path, 
    img2_path=img2_path,  
    method='affine+bspline', 
    output_dir=save_path_alignment,
    rotate=0.0, 
    scale=1.0,
)

print("\n✨✨✨ 所有流程执行完毕！ ✨✨✨")