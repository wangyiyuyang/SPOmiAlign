import pandas as pd
import scanpy as sc
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# ===== 输入路径 =====
CSV_PATH = "/public/home/shenninggroup/ivy/py_spatial/3omics/csv_ours/sm_transformed_coordinates_spomialign.csv"
H5AD_PATH = "/public/home/shenninggroup/ivy/py_spatial/3omics/h5ad_ours/adata_SM_with_spatialaligned.h5ad"
OUTPUT_H5AD = "/public/home/shenninggroup/ivy/py_spatial/3omics/h5ad_ours/adata_SM_with_spatialaligned.h5ad"

# 1. 读取 CSV
df = pd.read_csv(CSV_PATH)

# 只取 x_transformed, y_transformed
coords = df[["x_transformed", "y_transformed"]].to_numpy()

# 2. 读取已有的 h5ad
adata = sc.read_h5ad(H5AD_PATH)

# 3. 把坐标存入 obsm['spatial']
adata.obsm["spatial"] = coords

# 4. 保存新的 h5ad
adata.write(OUTPUT_H5AD)

print(f"已写入 obsm['spatial']，保存到 {OUTPUT_H5AD}")
