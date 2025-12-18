# #!/usr/bin/env python
# # -*- coding: utf-8 -*-

# import os
# import numpy as np
# import pandas as pd
# import scanpy as sc
# from anndata import AnnData
# from scipy import sparse
# from scipy.spatial import cKDTree
# import argparse


# # =========================
# # 工具函数：计算内部最近邻距离均值
# # =========================
# def mean_internal_nn_distance(xy):
#     """
#     给定 N×2 坐标矩阵 xy，计算每个点到其最近邻（排除自身）的距离，
#     返回 (均值, 所有最近邻距离向量)。
#     """
#     if xy.shape[0] < 2:
#         # 只有一个点时，没有“最近邻”，这里统一返回 0 和全 0 向量
#         return 0.0, np.zeros(xy.shape[0], dtype=float)

#     tree = cKDTree(xy)
#     # k=2，因为第一个是自己，第二个才是最近邻
#     dist, _ = tree.query(xy, k=2)
#     nn = dist[:, 1]
#     return float(np.mean(nn)), nn


# # =========================
# # 最近邻映射：根据分辨率自动决定方向 + 距离过滤
# # =========================
# def compute_nn_mapping_auto(
#     s1_csv,
#     s2_csv,
#     out_csv,
#     s1_id_col="id",
#     s1_x_col="x",
#     s1_y_col="y",
#     s2_id_col="id",
#     s2_x_col="x_transformed",
#     s2_y_col="y_transformed",
# ):
#     """
#     读入 S1.csv 和 S2.csv，自动判断哪张切片是高分辨率 / 低分辨率：

#       - 计算 S1 / S2 各自内部最近邻距离的均值
#       - 均值大的判定为“低分辨率”，均值小的是“高分辨率”
#       - KNN 始终是：高分辨率点 -> 低分辨率点 找最近邻
#       - 过滤规则：若某高分辨率点与其最近邻低分辨率点距离 > 2 * d_ref_max
#         （d_ref_max = 低分辨率切片内部最近邻最大距离），则删除该高分辨率点（不匹配）

#     输出 mapping CSV，列包括：
#       - high_id        : 高分辨率点的 id
#       - low_id         : 匹配到的低分辨率点 id
#       - high_x, high_y : 高分辨率点的坐标
#       - low_x, low_y   : 匹配到的低分辨率点的坐标
#       - distance       : 高->低 的最近邻距离

#     同时返回 (mapping_df, meta_dict)，其中 meta_dict 记录：
#       - low_res_name   : "S1" 或 "S2"
#       - high_res_name  : "S1" 或 "S2"
#       - d_ref_max      : 低分辨率内部 NN 最大距离
#     """
#     print(f"读取 S1: {s1_csv}")
#     s1 = pd.read_csv(s1_csv)
#     print(f"读取 S2: {s2_csv}")
#     s2 = pd.read_csv(s2_csv)

#     need_cols_s1 = [s1_id_col, s1_x_col, s1_y_col]
#     need_cols_s2 = [s2_id_col, s2_x_col, s2_y_col]

#     miss_s1 = [c for c in need_cols_s1 if c not in s1.columns]
#     miss_s2 = [c for c in need_cols_s2 if c not in s2.columns]
#     if miss_s1:
#         raise ValueError(f"S1 缺少列: {miss_s1}，需要: {need_cols_s1}")
#     if miss_s2:
#         raise ValueError(f"S2 缺少列: {miss_s2}，需要: {need_cols_s2}")

#     # ---- 提取坐标 ----
#     s1_xy = s1[[s1_x_col, s1_y_col]].to_numpy(dtype=float)
#     s2_xy = s2[[s2_x_col, s2_y_col]].to_numpy(dtype=float)

#     def finite_row(m):
#         return np.all(np.isfinite(m), axis=1)

#     s1_ok = finite_row(s1_xy)
#     s2_ok = finite_row(s2_xy)

#     if not np.all(s1_ok):
#         n_bad = np.sum(~s1_ok)
#         print(f"⚠️ S1 有 {n_bad} 行坐标非法(NA/Inf)，已剔除后再匹配。")
#         s1 = s1.loc[s1_ok].copy()
#         s1_xy = s1_xy[s1_ok, :]

#     if not np.all(s2_ok):
#         n_bad = np.sum(~s2_ok)
#         print(f"⚠️ S2 有 {n_bad} 行坐标非法(NA/Inf)，这些行的内部 NN 和 KNN 将忽略这些点。")
#         s2 = s2.loc[s2_ok].copy()
#         s2_xy = s2_xy[s2_ok, :]

#     if s1_xy.shape[0] == 0 or s2_xy.shape[0] == 0:
#         raise ValueError("S1 或 S2 有效坐标为空。")

#     # ---- 计算各自内部最近邻均值 ----
#     mean_s1, nn_s1 = mean_internal_nn_distance(s1_xy)
#     mean_s2, nn_s2 = mean_internal_nn_distance(s2_xy)

#     print(f"S1 内部最近邻距离均值: {mean_s1:.4f}")
#     print(f"S2 内部最近邻距离均值: {mean_s2:.4f}")

#     # 距离大的为低分辨率
#     if mean_s1 > mean_s2:
#         low_res_name = "S1"
#         high_res_name = "S2"
#         low_df, low_xy = s1, s1_xy
#         high_df, high_xy = s2, s2_xy
#         low_id_col, high_id_col = s1_id_col, s2_id_col
#         low_x_col, low_y_col = s1_x_col, s1_y_col
#         high_x_col, high_y_col = s2_x_col, s2_y_col
#         nn_low = nn_s1
#     else:
#         low_res_name = "S2"
#         high_res_name = "S1"
#         low_df, low_xy = s2, s2_xy
#         high_df, high_xy = s1, s1_xy
#         low_id_col, high_id_col = s2_id_col, s1_id_col
#         low_x_col, low_y_col = s2_x_col, s2_y_col
#         high_x_col, high_y_col = s1_x_col, s1_y_col
#         nn_low = nn_s2

#     print(f"\n自动判断：{low_res_name} = 低分辨率，{high_res_name} = 高分辨率")

#     # ---- 低分辨率内部最大最近邻距离 ----
#     d_ref_max = float(np.max(nn_low)) if nn_low.size > 0 else 0.0
#     print(f"低分辨率切片内部最近邻最大距离 d_ref_max = {d_ref_max:.4f}")

#     # ---- 高分辨率 -> 低分辨率 最近邻 ----
#     print("\n--- 高分辨率 → 低分辨率 最近邻搜索 ---")
#     tree = cKDTree(low_xy)
#     dist, idx = tree.query(high_xy, k=1)  # 每个高分辨率点找到最近低分辨率点

#     # ---- 距离过滤：> 2 * d_ref_max 的高分辨率点不匹配 ----
#     if d_ref_max > 0:
#         valid = dist <= 2.0 * d_ref_max
#     else:
#         # 若 d_ref_max = 0（例如只有 1 个点），就不做过滤
#         valid = np.ones_like(dist, dtype=bool)

#     n_drop = np.sum(~valid)
#     print(f"距离过滤：删除 {n_drop} 个高分辨率点（dist > 2 * d_ref_max）。")

#     dist_f = dist[valid]
#     idx_f = idx[valid]
#     high_valid_df = high_df.loc[valid].copy()

#     # ---- 构造映射表：高分辨率 id -> 低分辨率 id ----
#     mapping = pd.DataFrame(
#         {
#             "high_id": high_valid_df[high_id_col].astype(str).values,
#             "low_id": low_df[low_id_col].astype(str).values[idx_f],
#             "high_x": high_valid_df[high_x_col].to_numpy(),
#             "high_y": high_valid_df[high_y_col].to_numpy(),
#             "low_x": low_xy[idx_f, 0],
#             "low_y": low_xy[idx_f, 1],
#             "distance": dist_f,
#         }
#     )

#     print(f"\n映射表前几行：")
#     print(mapping.head())

#     os.makedirs(os.path.dirname(out_csv), exist_ok=True)
#     mapping.to_csv(out_csv, index=False)
#     print(f"\n✅ 映射表已保存：{out_csv}")

#     meta = {
#         "low_res_name": low_res_name,
#         "high_res_name": high_res_name,
#         "d_ref_max": d_ref_max,
#     }
#     return mapping, meta


# # =========================
# # 根据 mapping + 低 / 高 分辨率 h5ad 构建新 h5ad
# # =========================
# def build_reassigned_h5ad(
#     mapping_csv,
#     s1_h5ad,
#     s2_h5ad,
#     out_h5ad,
#     meta,
#     id_col_in_obs="id",
#     cluster_col="cluster",
#     scale_by_mapping_factor=True,
# ):
#     """
#     根据 compute_nn_mapping_auto 得到的 mapping + meta，
#     决定表达来自哪个 h5ad（低分辨率），空间坐标来自哪个 h5ad（高分辨率），
#     构建新的 AnnData：

#       - 低分辨率 h5ad 提供表达矩阵 X_low
#       - 高分辨率点（mapping 中的 high_x, high_y）作为新空间坐标
#       - 每个高分辨率点的表达 = 对应最近的低分辨率 spot 的表达（可选 1/k 缩放）
#       - cluster 从低分辨率 h5ad 映射
#       - 高分辨率 h5ad 的 obs 信息以 high_* 前缀附加到新 obs 中
#     """
#     print(f"\n读取映射表: {mapping_csv}")
#     m = pd.read_csv(mapping_csv)

#     needed_cols = ["high_id", "low_id", "high_x", "high_y"]
#     miss = [c for c in needed_cols if c not in m.columns]
#     if miss:
#         raise ValueError(f"mapping 缺少列: {miss}，必须包含 {needed_cols}")

#     # ---- 读入两个 h5ad ----
#     print(f"读取 S1 h5ad: {s1_h5ad}")
#     adata_s1 = sc.read_h5ad(s1_h5ad)
#     print(f"读取 S2 h5ad: {s2_h5ad}")
#     adata_s2 = sc.read_h5ad(s2_h5ad)

#     # 确定谁是低分辨率 / 高分辨率（和 NN 那一步保持一致）
#     if meta["low_res_name"] == "S1":
#         adata_low = adata_s1
#         adata_high = adata_s2
#         low_name = "S1"
#         high_name = "S2"
#     else:
#         adata_low = adata_s2
#         adata_high = adata_s1
#         low_name = "S2"
#         high_name = "S1"

#     print(f"\n在 h5ad 中：{low_name} 用作低分辨率表达提供者，{high_name} 用作高分辨率空间参考。")

#     # ---- 在低分辨率 h5ad 中，把 obs_names 或 obs[id_col_in_obs] 映射到行索引 ----
#     key_low = pd.Series(np.arange(adata_low.n_obs), index=adata_low.obs_names.astype(str))
#     use_obs_id_low = False
#     if not set(m["low_id"].astype(str)).issubset(set(key_low.index)):
#         if id_col_in_obs in adata_low.obs.columns:
#             key_low = pd.Series(
#                 np.arange(adata_low.n_obs),
#                 index=adata_low.obs[id_col_in_obs].astype(str),
#             )
#             use_obs_id_low = True
#             print(f"低分辨率：使用 adata_low.obs['{id_col_in_obs}'] 进行 id 匹配。")
#         else:
#             miss_ids = set(m["low_id"].astype(str)) - set(adata_low.obs_names.astype(str))
#             sample = list(miss_ids)[:10]
#             raise KeyError(
#                 "mapping 中的 low_id 无法在低分辨率 h5ad 的 obs_names 或 "
#                 f"obs['{id_col_in_obs}'] 中找到。\n"
#                 f"样例未命中的 id（最多10个）: {sample}"
#             )

#     low_idx = m["low_id"].astype(str).map(key_low)
#     n_not_found = low_idx.isna().sum()
#     if n_not_found:
#         print(f"⚠️ 低分辨率中有 {n_not_found} 个 low_id 找不到，将删除这些映射。")
#         keep = low_idx.notna()
#         m = m.loc[keep].copy()
#         low_idx = low_idx.loc[keep]
#     low_idx = low_idx.astype(int).to_numpy()

#     # ---- 在高分辨率 h5ad 中，准备映射 high_id → 行索引，以便拷贝其 obs 信息 ----
#     key_high = pd.Series(np.arange(adata_high.n_obs), index=adata_high.obs_names.astype(str))
#     use_obs_id_high = False
#     if not set(m["high_id"].astype(str)).issubset(set(key_high.index)):
#         if id_col_in_obs in adata_high.obs.columns:
#             key_high = pd.Series(
#                 np.arange(adata_high.n_obs),
#                 index=adata_high.obs[id_col_in_obs].astype(str),
#             )
#             use_obs_id_high = True
#             print(f"高分辨率：使用 adata_high.obs['{id_col_in_obs}'] 进行 id 匹配。")
#         else:
#             miss_ids = set(m["high_id"].astype(str)) - set(
#                 adata_high.obs_names.astype(str)
#             )
#             sample = list(miss_ids)[:10]
#             raise KeyError(
#                 "mapping 中的 high_id 无法在高分辨率 h5ad 的 obs_names 或 "
#                 f"obs['{id_col_in_obs}'] 中找到。\n"
#                 f"样例未命中的 id（最多10个）: {sample}"
#             )

#     high_idx = m["high_id"].astype(str).map(key_high)
#     n_not_found_high = high_idx.isna().sum()
#     if n_not_found_high:
#         print(f"⚠️ 高分辨率中有 {n_not_found_high} 个 high_id 找不到，将删除这些映射。")
#         keep = high_idx.notna()
#         m = m.loc[keep].copy()
#         low_idx = low_idx[keep.to_numpy()]
#         high_idx = high_idx.loc[keep]
#     high_idx = high_idx.astype(int).to_numpy()

#     # ---- 从低分辨率 h5ad 中抽表达矩阵 ----
#     X_low = adata_low.X
#     if sparse.isspmatrix_coo(X_low):
#         print("ℹ️ 检测到低分辨率 X 为 coo_matrix，转换为 csr_matrix。")
#         X_low = X_low.tocsr()

#     if sparse.issparse(X_low):
#         X_new = X_low[low_idx, :]
#     else:
#         X_new = np.asarray(X_low)[low_idx, :]

#     # ---- 1/k 缩放：保持总表达不膨胀 ----
#     if scale_by_mapping_factor:
#         count_map = m["low_id"].astype(str).value_counts()
#         mapping_factor = m["low_id"].astype(str).map(count_map).to_numpy()
#         if np.any(mapping_factor <= 0):
#             raise ValueError("检测到 mapping_factor <= 0，映射计数异常，请检查 mapping。")
#         scale = 1.0 / mapping_factor

#         if sparse.issparse(X_new):
#             X_new = X_new.multiply(scale[:, None])
#             if sparse.isspmatrix_coo(X_new):
#                 print("ℹ️ X_new 为 coo_matrix，转换为 csr_matrix 以便写入 h5ad。")
#                 X_new = X_new.tocsr()
#         else:
#             X_new = X_new * scale[:, None]

#     # ---- 组装新的 obs ----
#     n_obs_new = m.shape[0]
#     obs_names = [f"reassign_{i}" for i in range(n_obs_new)]
#     obs = pd.DataFrame(index=pd.Index(obs_names, name=None))

#     obs["low_id"] = m["low_id"].astype(str).values
#     obs["high_id"] = m["high_id"].astype(str).values

#     # cluster 映射：来自低分辨率 h5ad
#     if cluster_col in adata_low.obs.columns:
#         low_id_index = (
#             adata_low.obs[id_col_in_obs].astype(str)
#             if use_obs_id_low
#             else adata_low.obs_names.astype(str)
#         )
#         col_src = adata_low.obs[cluster_col]
#         col_src_str = col_src.astype(str)
#         id_to_cluster_str = pd.Series(col_src_str.values, index=low_id_index)
#         mapped_cluster_str = m["low_id"].astype(str).map(id_to_cluster_str)
#         n_na = int(mapped_cluster_str.isna().sum())
#         if n_na > 0:
#             print(f"⚠️ cluster 映射出现 {n_na} 个 NA，将以 'Unknown' 填充。")
#             mapped_cluster_str = mapped_cluster_str.fillna("Unknown")

#         cats = pd.unique(col_src_str)
#         obs["cluster"] = pd.Categorical(
#             mapped_cluster_str.values, categories=cats, ordered=False
#         )
#     else:
#         print(f"ℹ️ 低分辨率 h5ad 中无 obs['{cluster_col}']，跳过 cluster 映射。")

#     # 将高分辨率 h5ad 的 obs 信息带过来（加 high_ 前缀）
#     high_obs_selected = adata_high.obs.iloc[high_idx].copy()
#     # 如果用 obs[id] 匹配，则把原来的 obs_names 也保存一下
#     high_obs_selected = high_obs_selected.reset_index().rename(columns={"index": "obs_name"})
#     # 为避免冲突，统一加前缀 high_
#     high_obs_prefixed = high_obs_selected.add_prefix("high_")
#     # 与当前 obs 对齐
#     obs = pd.concat([obs.reset_index(drop=True), high_obs_prefixed.reset_index(drop=True)], axis=1)
#     obs.index = pd.Index(obs_names, name=None)

#     # ---- var 继承自低分辨率 h5ad ----
#     var = adata_low.var.copy()
#     var_names = adata_low.var_names.copy()

#     adata_new = AnnData(X=X_new, obs=obs, var=var)
#     adata_new.var_names = var_names

#     # ---- 空间坐标：使用高分辨率的坐标 ----
#     adata_new.obsm["spatial"] = m[["high_x", "high_y"]].to_numpy(dtype=float)

#     # 记录距离信息
#     adata_new.obs["knn_dist"] = m["distance"].values

#     # ---- 保存 ----
#     os.makedirs(os.path.dirname(out_h5ad), exist_ok=True)
#     adata_new.write_h5ad(out_h5ad, compression="gzip")
#     print(f"\n✅ 新 h5ad 已保存：{out_h5ad}")
#     print(f"   形状：{adata_new.n_obs} × {adata_new.n_vars}")
#     print("   obs 列包含：", list(adata_new.obs.columns))
#     print("   obsm 包含：", list(adata_new.obsm.keys()))
#     if "cluster" in adata_new.obs:
#         print("   cluster dtype:", adata_new.obs["cluster"].dtype)
#     else:
#         print("   cluster dtype: N/A")

#     return adata_new


# # =========================
# # 总管函数：spomialign_reassignment
# # =========================
# def spomialign_reassignment(
#     s1_csv,
#     s2_csv,
#     s1_h5ad,
#     s2_h5ad,
#     out_h5ad,
#     map_csv=None,
#     id_col="id",
#     cluster_col="cluster",
#     scale_by_mapping_factor=True,
# ):
#     """
#     SPOmiAlign reassignment pipeline：

#     输入：
#       - s1_csv : S1 的坐标/ID 表（至少包含 id,x,y 或 x_transformed,y_transformed）
#       - s2_csv : S2 的坐标/ID 表
#       - s1_h5ad: S1 的表达矩阵 h5ad
#       - s2_h5ad: S2 的表达矩阵 h5ad
#       - out_h5ad: 输出的 h5ad（高分辨率空间 + 低分辨率表达）

#     步骤：
#       1) 自动判断 S1 / S2 谁是高分辨率、谁是低分辨率（内部最近邻距离均值，大者为低分辨率）
#       2) 高分辨率点在低分辨率点上做最近邻
#       3) 若高分辨率点的 KNN 距离 > 2 * (低分辨率内部 NN 最大值)，删除该点
#       4) 从低分辨率 h5ad 拷贝表达矩阵到高分辨率点上（可选 1/k 缩放）
#       5) 低分辨率 h5ad 的 cluster 映射到新 obs['cluster']
#       6) 高分辨率 h5ad 的 obs 信息以 high_ 前缀附加到新 obs
#     """
#     if map_csv is None:
#         out_dir = os.path.dirname(out_h5ad)
#         os.makedirs(out_dir, exist_ok=True)
#         base = os.path.splitext(os.path.basename(s2_csv))[0]
#         map_csv = os.path.join(out_dir, f"{base}_nn_mapping.csv")

#     # ① NN + 分辨率判断 + 距离过滤
#     mapping, meta = compute_nn_mapping_auto(
#         s1_csv=s1_csv,
#         s2_csv=s2_csv,
#         out_csv=map_csv,
#         s1_id_col=id_col,
#         s1_x_col="x",
#         s1_y_col="y",
#         s2_id_col=id_col,
#         s2_x_col="x_transformed",
#         s2_y_col="y_transformed",
#     )

#     # ② 构建新 h5ad
#     adata_new = build_reassigned_h5ad(
#         mapping_csv=map_csv,
#         s1_h5ad=s1_h5ad,
#         s2_h5ad=s2_h5ad,
#         out_h5ad=out_h5ad,
#         meta=meta,
#         id_col_in_obs=id_col,
#         cluster_col=cluster_col,
#         scale_by_mapping_factor=scale_by_mapping_factor,
#     )
#     return adata_new


# # =========================
# # 命令行入口（可选）
# # =========================
# def main():
#     parser = argparse.ArgumentParser(
#         description=(
#             "SPOmiAlign reassignment：\n"
#             "自动判断 S1/S2 哪个是高分辨率/低分辨率，高分辨率点在低分辨率上找最近邻，"
#             "并构建带有低分辨率表达的新 h5ad。"
#         )
#     )
#     parser.add_argument("--s1_csv", "-s1", required=True, help="S1 CSV 路径（含 id,x,y）")
#     parser.add_argument(
#         "--s2_csv",
#         "-s2",
#         required=True,
#         help="S2 CSV 路径（含 id,x_transformed,y_transformed）",
#     )
#     parser.add_argument("--s1_h5ad", "-h1", required=True, help="S1 h5ad 路径")
#     parser.add_argument("--s2_h5ad", "-h2", required=True, help="S2 h5ad 路径")
#     parser.add_argument("--out_h5ad", "-o", required=True, help="输出 h5ad 路径")
#     parser.add_argument(
#         "--map_csv",
#         "-m",
#         default=None,
#         help="中间映射表 CSV 输出路径（默认与 out_h5ad 同目录自动命名）",
#     )
#     parser.add_argument(
#         "--no_scale",
#         action="store_true",
#         help="关闭 1/k 缩放（不做表达量密度归一化）",
#     )
#     args = parser.parse_args()

#     spomialign_reassignment(
#         s1_csv=args.s1_csv,
#         s2_csv=args.s2_csv,
#         s1_h5ad=args.s1_h5ad,
#         s2_h5ad=args.s2_h5ad,
#         out_h5ad=args.out_h5ad,
#         map_csv=args.map_csv,
#         id_col="id",
#         cluster_col="cluster",
#         scale_by_mapping_factor=not args.no_scale,
#     )


# if __name__ == "__main__":
#     main()

# #!/usr/bin/env python
# # -*- coding: utf-8 -*-

# import os
# import numpy as np
# import pandas as pd
# import scanpy as sc
# from anndata import AnnData
# from scipy import sparse
# from scipy.spatial import cKDTree
# import argparse


# # =========================
# # 工具函数：计算内部最近邻距离均值
# # =========================
# def mean_internal_nn_distance(xy: np.ndarray):
#     """
#     给定 N×2 坐标矩阵 xy，计算每个点到其最近邻（排除自身）的距离，
#     返回 (均值, 所有最近邻距离向量)。
#     """
#     if xy.shape[0] < 2:
#         # 只有一个点时，没有“最近邻”，这里统一返回 0 和全 0 向量
#         return 0.0, np.zeros(xy.shape[0], dtype=float)

#     tree = cKDTree(xy)
#     # k=2，因为第一个是自己，第二个才是最近邻
#     dist, _ = tree.query(xy, k=2)
#     nn = dist[:, 1]
#     return float(np.mean(nn)), nn


# # =========================
# # 从两个 h5ad 的 obsm['spatial'] 自动判断分辨率 + NN 映射
# # =========================
# def compute_nn_mapping_from_h5ads(
#     adata_s1: AnnData,
#     adata_s2: AnnData,
#     id_col: str = "id",
# ):
#     """
#     使用两个 h5ad 中的 obsm['spatial'] 坐标：

#       1) 从 adata_s1, adata_s2 的 obsm['spatial'] 中读取 xy 坐标（取前两列）
#       2) 分别计算 S1 / S2 内部最近邻距离均值
#       3) 均值大的判定为“低分辨率”，均值小的为“高分辨率”
#       4) 用 高分辨率点 -> 低分辨率点 做最近邻（cKDTree）
#       5) 过滤：若 dist > 2 * d_ref_max（d_ref_max = 低分辨率内部最近邻最大值），删除该高分辨率点

#     返回：
#       mapping_df: DataFrame，列包括：
#         - high_id, low_id
#         - high_x, high_y
#         - low_x, low_y
#         - distance
#         - high_index, low_index   （分别是对高/低分辨率 h5ad 的行索引）

#       meta: dict，包含：
#         - low_res_name   : "S1" or "S2"
#         - high_res_name  : "S1" or "S2"
#         - d_ref_max      : float
#     """
#     # ---- 取 S1 / S2 坐标 ----
#     if "spatial" not in adata_s1.obsm_keys():
#         raise KeyError("adata_s1.obsm 中没有 'spatial' 键。")
#     if "spatial" not in adata_s2.obsm_keys():
#         raise KeyError("adata_s2.obsm 中没有 'spatial' 键。")

#     xy1 = np.asarray(adata_s1.obsm["spatial"])
#     xy2 = np.asarray(adata_s2.obsm["spatial"])

#     if xy1.shape[1] < 2 or xy2.shape[1] < 2:
#         raise ValueError("obsm['spatial'] 至少需要两列坐标（x,y）。")

#     xy1 = xy1[:, :2]
#     xy2 = xy2[:, :2]

#     # 处理坐标中的 NaN / Inf
#     def clean_xy(xy):
#         mask = np.isfinite(xy).all(axis=1)
#         return xy[mask, :], mask

#     xy1_clean, mask1 = clean_xy(xy1)
#     xy2_clean, mask2 = clean_xy(xy2)

#     if xy1_clean.shape[0] == 0 or xy2_clean.shape[0] == 0:
#         raise ValueError("S1 或 S2 有效坐标为空（全是 NA/Inf？）。")

#     print(f"S1 有效坐标点数: {xy1_clean.shape[0]} / {xy1.shape[0]}")
#     print(f"S2 有效坐标点数: {xy2_clean.shape[0]} / {xy2.shape[0]}")

#     # ---- 计算内部最近邻均值 ----
#     mean_s1, nn_s1 = mean_internal_nn_distance(xy1_clean)
#     mean_s2, nn_s2 = mean_internal_nn_distance(xy2_clean)

#     print(f"S1 内部最近邻距离均值: {mean_s1:.4f}")
#     print(f"S2 内部最近邻距离均值: {mean_s2:.4f}")

#     # 距离大的为低分辨率
#     if mean_s1 > mean_s2:
#         low_res_name = "S1"
#         high_res_name = "S2"
#         low_xy, low_mask = xy1_clean, mask1
#         high_xy, high_mask = xy2_clean, mask2
#         nn_low = nn_s1
#         adata_low, adata_high = adata_s1, adata_s2
#     else:
#         low_res_name = "S2"
#         high_res_name = "S1"
#         low_xy, low_mask = xy2_clean, mask2
#         high_xy, high_mask = xy1_clean, mask1
#         nn_low = nn_s2
#         adata_low, adata_high = adata_s2, adata_s1

#     print(f"\n自动判断：{low_res_name} = 低分辨率，{high_res_name} = 高分辨率")

#     # ---- 低分辨率内部最大最近邻距离 ----
#     d_ref_max = float(np.max(nn_low)) if nn_low.size > 0 else 0.0
#     print(f"低分辨率切片内部最近邻最大距离 d_ref_max = {d_ref_max:.4f}")

#     # ---- 高分辨率中有效点的原始索引 ----
#     high_indices_all = np.where(high_mask)[0]
#     low_indices_all = np.where(low_mask)[0]

#     # ---- 高分辨率 → 低分辨率 最近邻 ----
#     print("\n--- 高分辨率 → 低分辨率 最近邻搜索 ---")
#     tree = cKDTree(low_xy)
#     dist, idx = tree.query(high_xy, k=1)  # 每个高分辨率点找到最近低分辨率点

#     # ---- 距离过滤：> 2 * d_ref_max 的高分辨率点不匹配 ----
#     if d_ref_max > 0:
#         valid = dist <= 2.0 * d_ref_max
#     else:
#         # 若 d_ref_max = 0（例如只有 1 个点），就不做过滤
#         valid = np.ones_like(dist, dtype=bool)

#     n_drop = np.sum(~valid)
#     print(f"距离过滤：删除 {n_drop} 个高分辨率点（dist > 2 * d_ref_max）。")

#     dist_f = dist[valid]
#     idx_f = idx[valid]
#     high_idx_clean = high_indices_all[valid]      # 在 adata_high 中的行索引
#     low_idx_clean = low_indices_all[idx_f]        # 在 adata_low 中的行索引

#     # ---- 生成 id ----
#     def get_ids(adata, id_col_name):
#         if id_col_name in adata.obs.columns:
#             return adata.obs[id_col_name].astype(str).to_numpy()
#         else:
#             # 用 obs_names 作为 id
#             return adata.obs_names.astype(str).to_numpy()

#     low_ids_all = get_ids(adata_low, id_col)
#     high_ids_all = get_ids(adata_high, id_col)

#     mapping = pd.DataFrame(
#         {
#             "high_id": high_ids_all[high_idx_clean],
#             "low_id": low_ids_all[low_idx_clean],
#             "high_x": adata_high.obsm["spatial"][high_idx_clean, 0],
#             "high_y": adata_high.obsm["spatial"][high_idx_clean, 1],
#             "low_x": adata_low.obsm["spatial"][low_idx_clean, 0],
#             "low_y": adata_low.obsm["spatial"][low_idx_clean, 1],
#             "distance": dist_f,
#             "high_index": high_idx_clean,
#             "low_index": low_idx_clean,
#         }
#     )

#     print("\n映射表前几行：")
#     print(mapping.head())

#     meta = {
#         "low_res_name": low_res_name,
#         "high_res_name": high_res_name,
#         "d_ref_max": d_ref_max,
#     }
#     return mapping, meta


# # =========================
# # 根据 mapping + 两个 h5ad 构建新的 h5ad
# # =========================
# def build_reassigned_h5ad_from_mapping(
#     mapping: pd.DataFrame,
#     meta: dict,
#     adata_s1: AnnData,
#     adata_s2: AnnData,
#     out_h5ad: str,
#     id_col: str = "id",
#     cluster_col: str = "cluster",
#     scale_by_mapping_factor: bool = True,
# ):
#     """
#     根据 compute_nn_mapping_from_h5ads 的 mapping 和 meta，
#     构建新 h5ad：

#       - 低分辨率 h5ad 提供表达矩阵 X_low
#       - 高分辨率点坐标（mapping['high_x','high_y']）作为新 obsm['spatial']
#       - 每个新点的表达 = 对应 low_index 行的表达（可选 1/k 缩放）
#       - cluster 从低分辨率 h5ad 映射
#       - 高分辨率 h5ad 的 obs 信息以 high_ 前缀附加到新 obs
#       - meta 中的信息写入 adata_new.uns['reassignment_meta']
#     """
#     if meta["low_res_name"] == "S1":
#         adata_low = adata_s1
#         adata_high = adata_s2
#         low_name = "S1"
#         high_name = "S2"
#     else:
#         adata_low = adata_s2
#         adata_high = adata_s1
#         low_name = "S2"
#         high_name = "S1"

#     print(f"\n在 h5ad 中：{low_name} 用作低分辨率表达提供者，{high_name} 用作高分辨率空间参考。")

#     if mapping.shape[0] == 0:
#         raise ValueError("mapping 为空，没有任何匹配点。")

#     low_idx = mapping["low_index"].to_numpy(dtype=int)
#     high_idx = mapping["high_index"].to_numpy(dtype=int)

#     # ---- 从低分辨率 h5ad 抽表达矩阵 ----
#     X_low = adata_low.X
#     if sparse.isspmatrix_coo(X_low):
#         print("ℹ️ 检测到低分辨率 X 为 coo_matrix，转换为 csr_matrix。")
#         X_low = X_low.tocsr()

#     if sparse.issparse(X_low):
#         X_new = X_low[low_idx, :]
#     else:
#         X_new = np.asarray(X_low)[low_idx, :]

#     # ---- 1/k 缩放（避免同一 low spot 映射给很多高点时表达膨胀）----
#     if scale_by_mapping_factor:
#         count_map = pd.Series(low_idx).value_counts()
#         mapping_factor = pd.Series(low_idx).map(count_map).to_numpy()
#         if np.any(mapping_factor <= 0):
#             raise ValueError("检测到 mapping_factor <= 0，映射计数异常。")
#         scale = 1.0 / mapping_factor

#         if sparse.issparse(X_new):
#             X_new = X_new.multiply(scale[:, None])
#             if sparse.isspmatrix_coo(X_new):
#                 print("ℹ️ X_new 为 coo_matrix，转换为 csr_matrix 以便写入 h5ad。")
#                 X_new = X_new.tocsr()
#         else:
#             X_new = X_new * scale[:, None]

#     # ---- 组装新的 obs ----
#     n_obs_new = mapping.shape[0]
#     obs_names = [f"reassign_{i}" for i in range(n_obs_new)]
#     obs = pd.DataFrame(index=pd.Index(obs_names, name=None))

#     obs["low_id"] = mapping["low_id"].astype(str).values
#     obs["high_id"] = mapping["high_id"].astype(str).values

#     # cluster 映射：来自低分辨率 h5ad
#     if cluster_col in adata_low.obs.columns:
#         col_src = adata_low.obs[cluster_col]
#         col_src_str = col_src.astype(str)
#         cats = pd.unique(col_src_str)
#         mapped_cluster_str = col_src_str.iloc[low_idx].reset_index(drop=True)
#         obs["cluster"] = pd.Categorical(
#             mapped_cluster_str.values, categories=cats, ordered=False
#         )
#     else:
#         print(f"ℹ️ 低分辨率 h5ad 中无 obs['{cluster_col}']，跳过 cluster 映射。")

#     # 高分辨率 h5ad 的 obs 信息带过来（加 high_ 前缀）
#     high_obs_sel = adata_high.obs.iloc[high_idx].copy()
#     high_obs_sel = high_obs_sel.reset_index().rename(columns={"index": "obs_name"})
#     high_obs_prefixed = high_obs_sel.add_prefix("high_")
#     obs = pd.concat([obs.reset_index(drop=True), high_obs_prefixed.reset_index(drop=True)], axis=1)
#     obs.index = pd.Index(obs_names, name=None)

#     # ---- var 继承自低分辨率 h5ad ----
#     var = adata_low.var.copy()
#     var_names = adata_low.var_names.copy()

#     adata_new = AnnData(X=X_new, obs=obs, var=var)
#     adata_new.var_names = var_names

#     # ---- 空间坐标：使用高分辨率的坐标 ----
#     adata_new.obsm["spatial"] = mapping[["high_x", "high_y"]].to_numpy(dtype=float)

#     # 记录距离信息
#     adata_new.obs["knn_dist"] = mapping["distance"].values

#     # 记录 meta 信息
#     adata_new.uns["reassignment_meta"] = {
#         "low_res_name": meta["low_res_name"],
#         "high_res_name": meta["high_res_name"],
#         "d_ref_max": float(meta["d_ref_max"]),
#     }

#     # ---- 保存 ----
#     os.makedirs(os.path.dirname(out_h5ad), exist_ok=True)
#     adata_new.write_h5ad(out_h5ad, compression="gzip")
#     print(f"\n✅ 新 h5ad 已保存：{out_h5ad}")
#     print(f"   形状：{adata_new.n_obs} × {adata_new.n_vars}")
#     print("   obs 列包含：", list(adata_new.obs.columns))
#     print("   obsm 包含：", list(adata_new.obsm.keys()))
#     if "cluster" in adata_new.obs:
#         print("   cluster dtype:", adata_new.obs["cluster"].dtype)
#     else:
#         print("   cluster dtype: N/A")

#     return adata_new


# # =========================
# # 总管函数：spomialign_reassignment
# # =========================
# def spomialign_reassignment(
#     s1_h5ad: str,
#     s2_h5ad: str,
#     out_h5ad: str,
#     map_csv: str | None = None,
#     id_col: str = "id",
#     cluster_col: str = "cluster",
#     scale_by_mapping_factor: bool = True,
# ):
#     """
#     SPOmiAlign reassignment pipeline（纯 h5ad 版本）：

#     输入：
#       - s1_h5ad: S1 的 h5ad（含 obsm['spatial']）
#       - s2_h5ad: S2 的 h5ad（含 obsm['spatial']）
#       - out_h5ad: 输出的 h5ad（高分辨率空间 + 低分辨率表达）
#       - map_csv: （可选）保存中间 mapping 的 CSV 路径
#       - id_col: 若 obs 中有该列，用作 id；否则使用 obs_names
#       - cluster_col: 从低分辨率 h5ad 中拷贝 cluster 注释

#     步骤：
#       1) 从两个 h5ad 的 obsm['spatial'] 读取坐标
#       2) 自动判断谁是高分辨率、谁是低分辨率（内部最近邻距离均值）
#       3) 高分辨率点在低分辨率上做最近邻（cKDTree）
#       4) 若 dist > 2 * (低分辨率内部 NN 最大值)，过滤掉这些高分辨率点
#       5) 用低分辨率 h5ad 的表达矩阵构建新 AnnData（支持 1/k 缩放）
#       6) cluster 从低分辨率 h5ad 映射；高分辨率 obs 信息以 high_ 前缀附加
#       7) meta 信息写入 adata_new.uns['reassignment_meta']
#     """
#     print(f"读取 S1 h5ad: {s1_h5ad}")
#     adata_s1 = sc.read_h5ad(s1_h5ad)
#     print(f"读取 S2 h5ad: {s2_h5ad}")
#     adata_s2 = sc.read_h5ad(s2_h5ad)

#     # ① NN + 分辨率判断 + 距离过滤
#     mapping, meta = compute_nn_mapping_from_h5ads(
#         adata_s1=adata_s1,
#         adata_s2=adata_s2,
#         id_col=id_col,
#     )

#     # 可选：保存 mapping CSV
#     if map_csv is not None:
#         os.makedirs(os.path.dirname(map_csv), exist_ok=True)
#         mapping.to_csv(map_csv, index=False)
#         print(f"\n中间映射表已保存：{map_csv}")

#     # ② 构建新 h5ad
#     adata_new = build_reassigned_h5ad_from_mapping(
#         mapping=mapping,
#         meta=meta,
#         adata_s1=adata_s1,
#         adata_s2=adata_s2,
#         out_h5ad=out_h5ad,
#         id_col=id_col,
#         cluster_col=cluster_col,
#         scale_by_mapping_factor=scale_by_mapping_factor,
#     )
#     return adata_new


# # =========================
# # 命令行入口
# # =========================
# def main():
#     parser = argparse.ArgumentParser(
#         description=(
#             "SPOmiAlign reassignment（纯 h5ad 版本）：\n"
#             "自动判断 S1/S2 哪个是高分辨率/低分辨率，高分辨率点在低分辨率上找最近邻，"
#             "并构建带有低分辨率表达的新 h5ad。"
#         )
#     )
#     parser.add_argument("--s1_h5ad", "-h1", required=True, help="S1 h5ad 路径（含 obsm['spatial']）")
#     parser.add_argument("--s2_h5ad", "-h2", required=True, help="S2 h5ad 路径（含 obsm['spatial']）")
#     parser.add_argument("--out_h5ad", "-o", required=True, help="输出 h5ad 路径")
#     parser.add_argument(
#         "--map_csv",
#         "-m",
#         default=None,
#         help="中间映射表 CSV 输出路径（可选）",
#     )
#     parser.add_argument(
#         "--id_col",
#         default="id",
#         help="obs 中用作 id 的列名（若不存在则使用 obs_names）",
#     )
#     parser.add_argument(
#         "--cluster_col",
#         default="cluster",
#         help="从低分辨率 h5ad 中拷贝 cluster 的列名",
#     )
#     parser.add_argument(
#         "--no_scale",
#         action="store_true",
#         help="关闭 1/k 缩放（不做表达量密度归一化）",
#     )
#     args = parser.parse_args()

#     spomialign_reassignment(
#         s1_h5ad=args.s1_h5ad,
#         s2_h5ad=args.s2_h5ad,
#         out_h5ad=args.out_h5ad,
#         map_csv=args.map_csv,
#         id_col=args.id_col,
#         cluster_col=args.cluster_col,
#         scale_by_mapping_factor=not args.no_scale,
#     )


# if __name__ == "__main__":
#     main()

# #!/usr/bin/env python
# # -*- coding: utf-8 -*-

# import os
# import numpy as np
# import pandas as pd
# import scanpy as sc
# from anndata import AnnData
# from scipy import sparse
# from scipy.spatial import cKDTree
# import argparse


# # =========================
# # 工具函数：计算内部最近邻距离均值
# # =========================
# def mean_internal_nn_distance(xy: np.ndarray):
#     """
#     给定 N×2 坐标矩阵 xy，计算每个点到其最近邻（排除自身）的距离，
#     返回 (均值, 所有最近邻距离向量)。
#     """
#     if xy.shape[0] < 2:
#         return 0.0, np.zeros(xy.shape[0], dtype=float)

#     tree = cKDTree(xy)
#     dist, _ = tree.query(xy, k=2)  # 第1个是自己，第2个是最近邻
#     nn = dist[:, 1]
#     return float(np.mean(nn)), nn


# # =========================
# # 从两个 h5ad 的 obsm['spatial'] 自动判断分辨率 + NN 映射
# # =========================
# def compute_nn_mapping_from_h5ads(
#     adata_s1: AnnData,
#     adata_s2: AnnData,
#     id_col: str = "id",
# ):
#     """
#     使用两个 h5ad 中的 obsm['spatial'] 坐标：

#       1) 从 adata_s1, adata_s2 的 obsm['spatial'] 中读取 xy 坐标（取前两列）
#       2) 分别计算 S1 / S2 内部最近邻距离均值
#       3) 均值大的判定为“低分辨率”，均值小的为“高分辨率”
#       4) 用 高分辨率点 -> 低分辨率点 做最近邻（cKDTree）
#       5) 过滤：若 dist > 2 * d_ref_max（d_ref_max = 低分辨率内部最近邻最大值），删除该高分辨率点

#     返回：
#       mapping_df: DataFrame，列包括：
#         - high_id, low_id
#         - high_x, high_y
#         - low_x, low_y
#         - distance
#         - high_index, low_index   （分别是对高/低分辨率 h5ad 的行索引）

#       meta: dict，包含：
#         - low_res_name   : "S1" or "S2"
#         - high_res_name  : "S1" or "S2"
#         - d_ref_max      : float
#     """
#     if "spatial" not in adata_s1.obsm_keys():
#         raise KeyError("adata_s1.obsm 中没有 'spatial' 键。")
#     if "spatial" not in adata_s2.obsm_keys():
#         raise KeyError("adata_s2.obsm 中没有 'spatial' 键。")

#     xy1 = np.asarray(adata_s1.obsm["spatial"])
#     xy2 = np.asarray(adata_s2.obsm["spatial"])

#     if xy1.shape[1] < 2 or xy2.shape[1] < 2:
#         raise ValueError("obsm['spatial'] 至少需要两列坐标（x,y）。")

#     xy1 = xy1[:, :2]
#     xy2 = xy2[:, :2]

#     # 清理 NA/Inf
#     def clean_xy(xy):
#         mask = np.isfinite(xy).all(axis=1)
#         return xy[mask, :], mask

#     xy1_clean, mask1 = clean_xy(xy1)
#     xy2_clean, mask2 = clean_xy(xy2)

#     if xy1_clean.shape[0] == 0 or xy2_clean.shape[0] == 0:
#         raise ValueError("S1 或 S2 有效坐标为空（全是 NA/Inf？）。")

#     print(f"S1 有效坐标点数: {xy1_clean.shape[0]} / {xy1.shape[0]}")
#     print(f"S2 有效坐标点数: {xy2_clean.shape[0]} / {xy2.shape[0]}")

#     # 内部最近邻均值（均值越大 -> 越稀疏 -> 低分辨率）
#     mean_s1, nn_s1 = mean_internal_nn_distance(xy1_clean)
#     mean_s2, nn_s2 = mean_internal_nn_distance(xy2_clean)

#     print(f"S1 内部最近邻距离均值: {mean_s1:.4f}")
#     print(f"S2 内部最近邻距离均值: {mean_s2:.4f}")

#     if mean_s1 > mean_s2:
#         low_res_name = "S1"
#         high_res_name = "S2"
#         low_xy, low_mask = xy1_clean, mask1
#         high_xy, high_mask = xy2_clean, mask2
#         nn_low = nn_s1
#         adata_low, adata_high = adata_s1, adata_s2
#     else:
#         low_res_name = "S2"
#         high_res_name = "S1"
#         low_xy, low_mask = xy2_clean, mask2
#         high_xy, high_mask = xy1_clean, mask1
#         nn_low = nn_s2
#         adata_low, adata_high = adata_s2, adata_s1

#     print(f"\n自动判断：{low_res_name} = 低分辨率，{high_res_name} = 高分辨率")

#     d_ref_max = float(np.max(nn_low)) if nn_low.size > 0 else 0.0
#     print(f"低分辨率切片内部最近邻最大距离 d_ref_max = {d_ref_max:.4f}")

#     # 有效点的原始索引
#     high_indices_all = np.where(high_mask)[0]
#     low_indices_all = np.where(low_mask)[0]

#     # 高分辨率 -> 低分辨率 最近邻
#     print("\n--- 高分辨率 → 低分辨率 最近邻搜索 ---")
#     tree = cKDTree(low_xy)
#     dist, idx = tree.query(high_xy, k=1)

#     # 距离过滤
#     if d_ref_max > 0:
#         valid = dist <= 2.0 * d_ref_max
#     else:
#         valid = np.ones_like(dist, dtype=bool)

#     n_drop = int(np.sum(~valid))
#     print(f"距离过滤：删除 {n_drop} 个高分辨率点（dist > 2 * d_ref_max）。")

#     dist_f = dist[valid]
#     idx_f = idx[valid]
#     high_idx_clean = high_indices_all[valid]
#     low_idx_clean = low_indices_all[idx_f]

#     # 生成 id
#     def get_ids(adata, id_col_name):
#         if id_col_name in adata.obs.columns:
#             return adata.obs[id_col_name].astype(str).to_numpy()
#         return adata.obs_names.astype(str).to_numpy()

#     low_ids_all = get_ids(adata_low, id_col)
#     high_ids_all = get_ids(adata_high, id_col)

#     mapping = pd.DataFrame(
#         {
#             "high_id": high_ids_all[high_idx_clean],
#             "low_id": low_ids_all[low_idx_clean],
#             "high_x": adata_high.obsm["spatial"][high_idx_clean, 0],
#             "high_y": adata_high.obsm["spatial"][high_idx_clean, 1],
#             "low_x": adata_low.obsm["spatial"][low_idx_clean, 0],
#             "low_y": adata_low.obsm["spatial"][low_idx_clean, 1],
#             "distance": dist_f,
#             "high_index": high_idx_clean,
#             "low_index": low_idx_clean,
#         }
#     )

#     print("\n映射表前几行：")
#     print(mapping.head())

#     meta = {
#         "low_res_name": low_res_name,
#         "high_res_name": high_res_name,
#         "d_ref_max": d_ref_max,
#     }
#     return mapping, meta


# # =========================
# # 根据 mapping + 两个 h5ad 构建新的 h5ad
# # =========================
# def build_reassigned_h5ad_from_mapping(
#     mapping: pd.DataFrame,
#     meta: dict,
#     adata_s1: AnnData,
#     adata_s2: AnnData,
#     out_h5ad: str,
#     id_col: str = "id",
#     cluster_col: str = "cluster",
#     scale_by_mapping_factor: bool = True,
# ):
#     """
#     构建新 h5ad：

#       - 低分辨率 h5ad 提供表达矩阵
#       - 新 obsm['spatial'] 用高分辨率点坐标
#       - 每个新点的表达 = 对应 low_index 的表达（可选 1/k 缩放）
#       - cluster 从低分辨率 h5ad 映射（若存在）
#       - 高分辨率 h5ad 的 obs 信息以 high_ 前缀附加到新 obs
#       - 额外：从低分辨率的注释写入 knn_Manual_annotation / knn_Combined_Clusters_annotation
#     """
#     if meta["low_res_name"] == "S1":
#         adata_low = adata_s1
#         adata_high = adata_s2
#         low_name = "S1"
#         high_name = "S2"
#     else:
#         adata_low = adata_s2
#         adata_high = adata_s1
#         low_name = "S2"
#         high_name = "S1"

#     print(f"\n在 h5ad 中：{low_name} 用作低分辨率表达提供者，{high_name} 用作高分辨率空间参考。")

#     if mapping.shape[0] == 0:
#         raise ValueError("mapping 为空，没有任何匹配点。")

#     low_idx = mapping["low_index"].to_numpy(dtype=int)
#     high_idx = mapping["high_index"].to_numpy(dtype=int)

#     # ---- 抽取低分辨率表达矩阵 ----
#     X_low = adata_low.X
#     if sparse.isspmatrix_coo(X_low):
#         print("ℹ️ 检测到低分辨率 X 为 coo_matrix，转换为 csr_matrix。")
#         X_low = X_low.tocsr()

#     if sparse.issparse(X_low):
#         X_new = X_low[low_idx, :]
#     else:
#         X_new = np.asarray(X_low)[low_idx, :]

#     # ---- 1/k 缩放 ----
#     if scale_by_mapping_factor:
#         count_map = pd.Series(low_idx).value_counts()
#         mapping_factor = pd.Series(low_idx).map(count_map).to_numpy()
#         if np.any(mapping_factor <= 0):
#             raise ValueError("检测到 mapping_factor <= 0，映射计数异常。")
#         scale = 1.0 / mapping_factor

#         if sparse.issparse(X_new):
#             X_new = X_new.multiply(scale[:, None])
#             if sparse.isspmatrix_coo(X_new):
#                 print("ℹ️ X_new 为 coo_matrix，转换为 csr_matrix 以便写入 h5ad。")
#                 X_new = X_new.tocsr()
#         else:
#             X_new = X_new * scale[:, None]

#     # ---- 组装 obs ----
#     n_obs_new = mapping.shape[0]
#     obs_names = [f"reassign_{i}" for i in range(n_obs_new)]
#     obs = pd.DataFrame(index=pd.Index(obs_names, name=None))

#     obs["low_id"] = mapping["low_id"].astype(str).values
#     obs["high_id"] = mapping["high_id"].astype(str).values

#     # cluster 映射（来自低分辨率）
#     if cluster_col in adata_low.obs.columns:
#         col_src = adata_low.obs[cluster_col]
#         col_src_str = col_src.astype(str)
#         cats = pd.unique(col_src_str)
#         mapped_cluster_str = col_src_str.iloc[low_idx].reset_index(drop=True)
#         obs["cluster"] = pd.Categorical(mapped_cluster_str.values, categories=cats, ordered=False)
#     else:
#         print(f"ℹ️ 低分辨率 h5ad 中无 obs['{cluster_col}']，跳过 cluster 映射。")

#     # ---- 高分辨率 obs 信息加 high_ 前缀带入 ----
#     high_obs_sel = adata_high.obs.iloc[high_idx].copy()
#     high_obs_sel = high_obs_sel.reset_index().rename(columns={"index": "obs_name"})
#     high_obs_prefixed = high_obs_sel.add_prefix("high_")

#     obs = pd.concat([obs.reset_index(drop=True), high_obs_prefixed.reset_index(drop=True)], axis=1)

#     # ---- 额外：从低分辨率切片带出注释，写成 knn_ 前缀列 ----
#     low_obs_sel = adata_low.obs.iloc[low_idx].copy().reset_index(drop=True)
#     for col in ["Manual_annotation", "Combined_Clusters_annotation"]:
#         if col in low_obs_sel.columns:
#             obs[f"knn_{col}"] = low_obs_sel[col].astype(str).values
#         else:
#             print(f"ℹ️ 低分辨率 h5ad 中无 obs['{col}']，跳过写入 knn_{col}。")

#     # 恢复索引
#     obs.index = pd.Index(obs_names, name=None)

#     # ---- var 继承自低分辨率 ----
#     var = adata_low.var.copy()
#     var_names = adata_low.var_names.copy()

#     adata_new = AnnData(X=X_new, obs=obs, var=var)
#     adata_new.var_names = var_names

#     # ---- 空间坐标：用高分辨率坐标 ----
#     adata_new.obsm["spatial"] = mapping[["high_x", "high_y"]].to_numpy(dtype=float)

#     # 距离信息
#     adata_new.obs["knn_dist"] = mapping["distance"].values

#     # meta
#     adata_new.uns["reassignment_meta"] = {
#         "low_res_name": meta["low_res_name"],
#         "high_res_name": meta["high_res_name"],
#         "d_ref_max": float(meta["d_ref_max"]),
#     }

#     # ---- 保存 ----
#     os.makedirs(os.path.dirname(out_h5ad), exist_ok=True)
#     adata_new.write_h5ad(out_h5ad, compression="gzip")
#     print(f"\n✅ 新 h5ad 已保存：{out_h5ad}")
#     print(f"   形状：{adata_new.n_obs} × {adata_new.n_vars}")
#     print("   obs 列包含：", list(adata_new.obs.columns))
#     print("   obsm 包含：", list(adata_new.obsm.keys()))
#     if "cluster" in adata_new.obs:
#         print("   cluster dtype:", adata_new.obs["cluster"].dtype)
#     else:
#         print("   cluster dtype: N/A")

#     return adata_new


# # =========================
# # 总管函数：spomialign_reassignment
# # =========================
# def spomialign_reassignment(
#     s1_h5ad: str,
#     s2_h5ad: str,
#     out_h5ad: str,
#     map_csv: str | None = None,
#     id_col: str = "id",
#     cluster_col: str = "cluster",
#     scale_by_mapping_factor: bool = True,
# ):
#     """
#     SPOmiAlign reassignment pipeline（纯 h5ad 版本）
#     """
#     print(f"读取 S1 h5ad: {s1_h5ad}")
#     adata_s1 = sc.read_h5ad(s1_h5ad)
#     print(f"读取 S2 h5ad: {s2_h5ad}")
#     adata_s2 = sc.read_h5ad(s2_h5ad)

#     mapping, meta = compute_nn_mapping_from_h5ads(
#         adata_s1=adata_s1,
#         adata_s2=adata_s2,
#         id_col=id_col,
#     )

#     if map_csv is not None:
#         os.makedirs(os.path.dirname(map_csv), exist_ok=True)
#         mapping.to_csv(map_csv, index=False)
#         print(f"\n中间映射表已保存：{map_csv}")

#     adata_new = build_reassigned_h5ad_from_mapping(
#         mapping=mapping,
#         meta=meta,
#         adata_s1=adata_s1,
#         adata_s2=adata_s2,
#         out_h5ad=out_h5ad,
#         id_col=id_col,
#         cluster_col=cluster_col,
#         scale_by_mapping_factor=scale_by_mapping_factor,
#     )
#     return adata_new


# # =========================
# # 命令行入口
# # =========================
# def main():
#     parser = argparse.ArgumentParser(
#         description=(
#             "SPOmiAlign reassignment（纯 h5ad 版本）：\n"
#             "自动判断 S1/S2 哪个是高分辨率/低分辨率，高分辨率点在低分辨率上找最近邻，"
#             "并构建带有低分辨率表达的新 h5ad。"
#         )
#     )
#     parser.add_argument("--s1_h5ad", "-h1", required=True, help="S1 h5ad 路径（含 obsm['spatial']）")
#     parser.add_argument("--s2_h5ad", "-h2", required=True, help="S2 h5ad 路径（含 obsm['spatial']）")
#     parser.add_argument("--out_h5ad", "-o", required=True, help="输出 h5ad 路径")
#     parser.add_argument("--map_csv", "-m", default=None, help="中间映射表 CSV 输出路径（可选）")
#     parser.add_argument("--id_col", default="id", help="obs 中用作 id 的列名（若不存在则使用 obs_names）")
#     parser.add_argument("--cluster_col", default="cluster", help="从低分辨率 h5ad 中拷贝 cluster 的列名")
#     parser.add_argument("--no_scale", action="store_true", help="关闭 1/k 缩放（不做表达量密度归一化）")
#     args = parser.parse_args()

#     spomialign_reassignment(
#         s1_h5ad=args.s1_h5ad,
#         s2_h5ad=args.s2_h5ad,
#         out_h5ad=args.out_h5ad,
#         map_csv=args.map_csv,
#         id_col=args.id_col,
#         cluster_col=args.cluster_col,
#         scale_by_mapping_factor=not args.no_scale,
#     )


# if __name__ == "__main__":
#     main()

# #!/usr/bin/env python
# # -*- coding: utf-8 -*-

# import os
# import numpy as np
# import pandas as pd
# import scanpy as sc
# from anndata import AnnData
# from scipy import sparse
# from scipy.spatial import cKDTree
# import argparse


# # =========================
# # 工具函数：计算内部最近邻距离均值
# # =========================
# def mean_internal_nn_distance(xy: np.ndarray):
#     """
#     给定 N×2 坐标矩阵 xy，计算每个点到其最近邻（排除自身）的距离，
#     返回 (均值, 所有最近邻距离向量)。
#     """
#     if xy.shape[0] < 2:
#         return 0.0, np.zeros(xy.shape[0], dtype=float)

#     tree = cKDTree(xy)
#     dist, _ = tree.query(xy, k=2)  # 第1个是自己，第2个是最近邻
#     nn = dist[:, 1]
#     return float(np.mean(nn)), nn


# # =========================
# # 从两个 h5ad 的 obsm['spatial'] 自动判断分辨率 + NN 映射
# # =========================
# def compute_nn_mapping_from_h5ads(
#     adata_s1: AnnData,
#     adata_s2: AnnData,
#     id_col: str = "id",
# ):
#     """
#     使用两个 h5ad 中的 obsm['spatial'] 坐标：

#       1) 从 adata_s1, adata_s2 的 obsm['spatial'] 中读取 xy 坐标（取前两列）
#       2) 分别计算 S1 / S2 内部最近邻距离均值
#       3) 均值大的判定为“低分辨率”，均值小的为“高分辨率”
#       4) 用 高分辨率点 -> 低分辨率点 做最近邻（cKDTree）
#       5) 过滤：若 dist > 2 * d_ref_max（d_ref_max = 低分辨率内部最近邻最大值），删除该高分辨率点

#     返回：
#       mapping_df: DataFrame，列包括：
#         - high_id, low_id
#         - high_x, high_y
#         - low_x, low_y
#         - distance
#         - high_index, low_index

#       meta: dict，包含：
#         - low_res_name   : "S1" or "S2"
#         - high_res_name  : "S1" or "S2"
#         - d_ref_max      : float
#     """
#     if "spatial" not in adata_s1.obsm_keys():
#         raise KeyError("adata_s1.obsm 中没有 'spatial' 键。")
#     if "spatial" not in adata_s2.obsm_keys():
#         raise KeyError("adata_s2.obsm 中没有 'spatial' 键。")

#     xy1 = np.asarray(adata_s1.obsm["spatial"])
#     xy2 = np.asarray(adata_s2.obsm["spatial"])

#     if xy1.shape[1] < 2 or xy2.shape[1] < 2:
#         raise ValueError("obsm['spatial'] 至少需要两列坐标（x,y）。")

#     xy1 = xy1[:, :2]
#     xy2 = xy2[:, :2]

#     # 清理 NA/Inf
#     def clean_xy(xy):
#         mask = np.isfinite(xy).all(axis=1)
#         return xy[mask, :], mask

#     xy1_clean, mask1 = clean_xy(xy1)
#     xy2_clean, mask2 = clean_xy(xy2)

#     if xy1_clean.shape[0] == 0 or xy2_clean.shape[0] == 0:
#         raise ValueError("S1 或 S2 有效坐标为空（全是 NA/Inf？）。")

#     print(f"S1 有效坐标点数: {xy1_clean.shape[0]} / {xy1.shape[0]}")
#     print(f"S2 有效坐标点数: {xy2_clean.shape[0]} / {xy2.shape[0]}")

#     # 内部最近邻均值（均值越大 -> 越稀疏 -> 低分辨率）
#     mean_s1, nn_s1 = mean_internal_nn_distance(xy1_clean)
#     mean_s2, nn_s2 = mean_internal_nn_distance(xy2_clean)

#     print(f"S1 内部最近邻距离均值: {mean_s1:.4f}")
#     print(f"S2 内部最近邻距离均值: {mean_s2:.4f}")

#     if mean_s1 > mean_s2:
#         low_res_name = "S1"
#         high_res_name = "S2"
#         low_xy, low_mask = xy1_clean, mask1
#         high_xy, high_mask = xy2_clean, mask2
#         nn_low = nn_s1
#         adata_low, adata_high = adata_s1, adata_s2
#     else:
#         low_res_name = "S2"
#         high_res_name = "S1"
#         low_xy, low_mask = xy2_clean, mask2
#         high_xy, high_mask = xy1_clean, mask1
#         nn_low = nn_s2
#         adata_low, adata_high = adata_s2, adata_s1

#     print(f"\n自动判断：{low_res_name} = 低分辨率，{high_res_name} = 高分辨率")

#     d_ref_max = float(np.max(nn_low)) if nn_low.size > 0 else 0.0
#     print(f"低分辨率切片内部最近邻最大距离 d_ref_max = {d_ref_max:.4f}")

#     # 有效点的原始索引
#     high_indices_all = np.where(high_mask)[0]
#     low_indices_all = np.where(low_mask)[0]

#     # 高分辨率 -> 低分辨率 最近邻
#     print("\n--- 高分辨率 → 低分辨率 最近邻搜索 ---")
#     tree = cKDTree(low_xy)
#     dist, idx = tree.query(high_xy, k=1)

#     # 距离过滤
#     if d_ref_max > 0:
#         valid = dist <= 2.0 * d_ref_max
#     else:
#         valid = np.ones_like(dist, dtype=bool)

#     n_drop = int(np.sum(~valid))
#     print(f"距离过滤：删除 {n_drop} 个高分辨率点（dist > 2 * d_ref_max）。")

#     dist_f = dist[valid]
#     idx_f = idx[valid]
#     high_idx_clean = high_indices_all[valid]
#     low_idx_clean = low_indices_all[idx_f]

#     # 生成 id
#     def get_ids(adata, id_col_name):
#         if id_col_name in adata.obs.columns:
#             return adata.obs[id_col_name].astype(str).to_numpy()
#         return adata.obs_names.astype(str).to_numpy()

#     low_ids_all = get_ids(adata_low, id_col)
#     high_ids_all = get_ids(adata_high, id_col)

#     mapping = pd.DataFrame(
#         {
#             "high_id": high_ids_all[high_idx_clean],
#             "low_id": low_ids_all[low_idx_clean],
#             "high_x": adata_high.obsm["spatial"][high_idx_clean, 0],
#             "high_y": adata_high.obsm["spatial"][high_idx_clean, 1],
#             "low_x": adata_low.obsm["spatial"][low_idx_clean, 0],
#             "low_y": adata_low.obsm["spatial"][low_idx_clean, 1],
#             "distance": dist_f,
#             "high_index": high_idx_clean,
#             "low_index": low_idx_clean,
#         }
#     )

#     print("\n映射表前几行：")
#     print(mapping.head())

#     meta = {
#         "low_res_name": low_res_name,
#         "high_res_name": high_res_name,
#         "d_ref_max": d_ref_max,
#     }
#     return mapping, meta


# # =========================
# # 根据 mapping + 两个 h5ad 构建新的 h5ad
# # =========================
# def build_reassigned_h5ad_from_mapping(
#     mapping: pd.DataFrame,
#     meta: dict,
#     adata_s1: AnnData,
#     adata_s2: AnnData,
#     out_h5ad: str,
#     id_col: str = "id",
#     cluster_col: str = "cluster",
#     s2_cluster_col: str = "Manual_annotation",
#     scale_by_mapping_factor: bool = True,
# ):
#     """
#     构建新 h5ad：

#       - 低分辨率 h5ad 提供表达矩阵
#       - 新 obsm['spatial'] 用高分辨率点坐标
#       - 每个新点的表达 = 对应 low_index 的表达（可选 1/k 缩放）
#       - cluster 从低分辨率 h5ad 映射（若存在）
#       - 保留低分辨率切片的 Manual_annotation，列名仍为 Manual_annotation
#       - 仅额外新增一列：高分辨率切片的 s2_cluster_col（例如 Manual_annotation），
#         写入列名为 s2_Manual_annotation（或 s1_Manual_annotation，取决于 high 是谁）
#     """
#     if meta["low_res_name"] == "S1":
#         adata_low = adata_s1
#         adata_high = adata_s2
#         low_name = "S1"
#         high_name = "S2"
#     else:
#         adata_low = adata_s2
#         adata_high = adata_s1
#         low_name = "S2"
#         high_name = "S1"

#     print(f"\n在 h5ad 中：{low_name} 用作低分辨率表达提供者，{high_name} 用作高分辨率空间参考。")

#     if mapping.shape[0] == 0:
#         raise ValueError("mapping 为空，没有任何匹配点。")

#     low_idx = mapping["low_index"].to_numpy(dtype=int)
#     high_idx = mapping["high_index"].to_numpy(dtype=int)

#     # ---- 抽取低分辨率表达矩阵 ----
#     X_low = adata_low.X
#     if sparse.isspmatrix_coo(X_low):
#         print("ℹ️ 检测到低分辨率 X 为 coo_matrix，转换为 csr_matrix。")
#         X_low = X_low.tocsr()

#     if sparse.issparse(X_low):
#         X_new = X_low[low_idx, :]
#     else:
#         X_new = np.asarray(X_low)[low_idx, :]

#     # ---- 1/k 缩放 ----
#     if scale_by_mapping_factor:
#         count_map = pd.Series(low_idx).value_counts()
#         mapping_factor = pd.Series(low_idx).map(count_map).to_numpy()
#         if np.any(mapping_factor <= 0):
#             raise ValueError("检测到 mapping_factor <= 0，映射计数异常。")
#         scale = 1.0 / mapping_factor

#         if sparse.issparse(X_new):
#             X_new = X_new.multiply(scale[:, None])
#             if sparse.isspmatrix_coo(X_new):
#                 print("ℹ️ X_new 为 coo_matrix，转换为 csr_matrix 以便写入 h5ad。")
#                 X_new = X_new.tocsr()
#         else:
#             X_new = X_new * scale[:, None]

#     # ---- 组装 obs ----
#     n_obs_new = mapping.shape[0]
#     obs_names = [f"reassign_{i}" for i in range(n_obs_new)]
#     obs = pd.DataFrame(index=pd.Index(obs_names, name=None))

#     obs["low_id"] = mapping["low_id"].astype(str).values
#     obs["high_id"] = mapping["high_id"].astype(str).values

#     # cluster 映射（来自低分辨率）
#     if cluster_col in adata_low.obs.columns:
#         col_src = adata_low.obs[cluster_col]
#         col_src_str = col_src.astype(str)
#         cats = pd.unique(col_src_str)
#         mapped_cluster_str = col_src_str.iloc[low_idx].reset_index(drop=True)
#         obs["cluster"] = pd.Categorical(mapped_cluster_str.values, categories=cats, ordered=False)
#     else:
#         print(f"ℹ️ 低分辨率 h5ad 中无 obs['{cluster_col}']，跳过 cluster 映射。")

#     # ---- 保留低分辨率的 Manual_annotation（列名仍叫 Manual_annotation）----
#     low_obs_sel = adata_low.obs.iloc[low_idx].copy().reset_index(drop=True)
#     if "Manual_annotation" in low_obs_sel.columns:
#         obs["Manual_annotation"] = low_obs_sel["Manual_annotation"].astype(str).values
#     else:
#         print("ℹ️ 低分辨率 h5ad 中无 obs['Manual_annotation']，跳过写入 Manual_annotation。")

#     # ---- 仅新增：高分辨率切片指定列（默认 Manual_annotation）→ s2_Manual_annotation ----
#     # 自动生成列名：s1_xxx 或 s2_xxx（取决于 high_name 是谁）
#     new_col_name = f"{high_name.lower()}_{s2_cluster_col}"
#     high_obs_sel = adata_high.obs.iloc[high_idx].copy().reset_index(drop=True)
#     if s2_cluster_col in high_obs_sel.columns:
#         obs[new_col_name] = high_obs_sel[s2_cluster_col].astype(str).values
#     else:
#         raise KeyError(
#             f"高分辨率({high_name}) h5ad 的 obs 中找不到列 '{s2_cluster_col}'。"
#             f"\n可用列示例：{list(adata_high.obs.columns)[:50]}"
#         )

#     # ---- var 继承自低分辨率 ----
#     var = adata_low.var.copy()
#     var_names = adata_low.var_names.copy()

#     adata_new = AnnData(X=X_new, obs=obs, var=var)
#     adata_new.var_names = var_names

#     # ---- 空间坐标：用高分辨率坐标 ----
#     adata_new.obsm["spatial"] = mapping[["high_x", "high_y"]].to_numpy(dtype=float)

#     # 距离信息（不是 annotation 列）
#     adata_new.obs["knn_dist"] = mapping["distance"].values

#     # meta
#     adata_new.uns["reassignment_meta"] = {
#         "low_res_name": meta["low_res_name"],
#         "high_res_name": meta["high_res_name"],
#         "d_ref_max": float(meta["d_ref_max"]),
#         "mapped_high_obs_col": s2_cluster_col,
#         "mapped_high_obs_col_outname": new_col_name,
#     }

#     # ---- 保存 ----
#     out_dir = os.path.dirname(out_h5ad)
#     if out_dir:
#         os.makedirs(out_dir, exist_ok=True)
#     adata_new.write_h5ad(out_h5ad, compression="gzip")
#     print(f"\n✅ 新 h5ad 已保存：{out_h5ad}")
#     print(f"   形状：{adata_new.n_obs} × {adata_new.n_vars}")
#     print("   obs 列包含：", list(adata_new.obs.columns))
#     print("   obsm 包含：", list(adata_new.obsm.keys()))
#     if "cluster" in adata_new.obs:
#         print("   cluster dtype:", adata_new.obs["cluster"].dtype)
#     else:
#         print("   cluster dtype: N/A")

#     return adata_new


# # =========================
# # 总管函数：spomialign_reassignment
# # =========================
# def spomialign_reassignment(
#     s1_h5ad: str,
#     s2_h5ad: str,
#     out_h5ad: str,
#     map_csv: str | None = None,
#     id_col: str = "id",
#     cluster_col: str = "cluster",
#     s2_cluster_col: str = "Manual_annotation",
#     scale_by_mapping_factor: bool = True,
# ):
#     """
#     SPOmiAlign reassignment pipeline（纯 h5ad 版本）
#     """
#     print(f"读取 S1 h5ad: {s1_h5ad}")
#     adata_s1 = sc.read_h5ad(s1_h5ad)
#     print(f"读取 S2 h5ad: {s2_h5ad}")
#     adata_s2 = sc.read_h5ad(s2_h5ad)

#     mapping, meta = compute_nn_mapping_from_h5ads(
#         adata_s1=adata_s1,
#         adata_s2=adata_s2,
#         id_col=id_col,
#     )

#     if map_csv is not None:
#         out_dir = os.path.dirname(map_csv)
#         if out_dir:
#             os.makedirs(out_dir, exist_ok=True)
#         mapping.to_csv(map_csv, index=False)
#         print(f"\n中间映射表已保存：{map_csv}")

#     adata_new = build_reassigned_h5ad_from_mapping(
#         mapping=mapping,
#         meta=meta,
#         adata_s1=adata_s1,
#         adata_s2=adata_s2,
#         out_h5ad=out_h5ad,
#         id_col=id_col,
#         cluster_col=cluster_col,
#         s2_cluster_col=s2_cluster_col,
#         scale_by_mapping_factor=scale_by_mapping_factor,
#     )
#     return adata_new


# # =========================
# # 命令行入口
# # =========================
# def main():
#     parser = argparse.ArgumentParser(
#         description=(
#             "SPOmiAlign reassignment（纯 h5ad 版本）：\n"
#             "自动判断 S1/S2 哪个是高分辨率/低分辨率，高分辨率点在低分辨率上找最近邻，"
#             "并构建带有低分辨率表达的新 h5ad。\n"
#             "新 h5ad 保留低分辨率 Manual_annotation（列名仍为 Manual_annotation），"
#             "并仅额外新增一列：{high}_{s2_cluster_col}（例如 s2_Manual_annotation）。"
#         )
#     )
#     parser.add_argument("--s1_h5ad", "-h1", required=True, help="S1 h5ad 路径（含 obsm['spatial']）")
#     parser.add_argument("--s2_h5ad", "-h2", required=True, help="S2 h5ad 路径（含 obsm['spatial']）")
#     parser.add_argument("--out_h5ad", "-o", required=True, help="输出 h5ad 路径")
#     parser.add_argument("--map_csv", "-m", default=None, help="中间映射表 CSV 输出路径（可选）")
#     parser.add_argument("--id_col", default="id", help="obs 中用作 id 的列名（若不存在则使用 obs_names）")
#     parser.add_argument("--cluster_col", default="cluster", help="从低分辨率 h5ad 中拷贝 cluster 的列名")
#     # parser.add_argument(
#     #     "--s2_cluster_col",
#     #     default="Manual_annotation",
#     #     help="从高分辨率切片（通常是S2）映射到新h5ad的obs列名，例如 Manual_annotation",
#     # )
#     parser.add_argument(
#     "--s2_cluster_col",
#     nargs="+",
#     default=["Manual_annotation"],
#     help=(
#         "从高分辨率切片（high-res; 通常是S2）拷贝到新h5ad的obs列名，可写多个。"
#         "例如: --s2_cluster_col cluster barcode_S2"
#     ),
#     )
#     parser.add_argument("--no_scale", action="store_true", help="关闭 1/k 缩放（不做表达量密度归一化）")
#     args = parser.parse_args()

#     spomialign_reassignment(
#         s1_h5ad=args.s1_h5ad,
#         s2_h5ad=args.s2_h5ad,
#         out_h5ad=args.out_h5ad,
#         map_csv=args.map_csv,
#         id_col=args.id_col,
#         cluster_col=args.cluster_col,
#         s2_cluster_col=args.s2_cluster_col,
#         scale_by_mapping_factor=not args.no_scale,
#     )


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy import sparse
from scipy.spatial import cKDTree
import argparse
from typing import Optional, Union, List


# =========================
# 工具函数：计算内部最近邻距离均值
# =========================
def mean_internal_nn_distance(xy: np.ndarray):
    """
    给定 N×2 坐标矩阵 xy，计算每个点到其最近邻（排除自身）的距离，
    返回 (均值, 所有最近邻距离向量)。
    """
    if xy.shape[0] < 2:
        return 0.0, np.zeros(xy.shape[0], dtype=float)

    tree = cKDTree(xy)
    dist, _ = tree.query(xy, k=2)  # 第1个是自己，第2个是最近邻
    nn = dist[:, 1]
    return float(np.mean(nn)), nn


# =========================
# 从两个 h5ad 的 obsm['spatial'] 自动判断分辨率 + NN 映射
# =========================
def compute_nn_mapping_from_h5ads(
    adata_s1: AnnData,
    adata_s2: AnnData,
    id_col: str = "id",
):
    """
    使用两个 h5ad 中的 obsm['spatial'] 坐标：

      1) 从 adata_s1, adata_s2 的 obsm['spatial'] 中读取 xy 坐标（取前两列）
      2) 分别计算 S1 / S2 内部最近邻距离均值
      3) 均值大的判定为“低分辨率”，均值小的为“高分辨率”
      4) 用 高分辨率点 -> 低分辨率点 做最近邻（cKDTree）
      5) 过滤：若 dist > 2 * d_ref_max（d_ref_max = 低分辨率内部最近邻最大值），删除该高分辨率点

    返回：
      mapping_df: DataFrame，列包括：
        - high_id, low_id
        - high_x, high_y
        - low_x, low_y
        - distance
        - high_index, low_index

      meta: dict，包含：
        - low_res_name   : "S1" or "S2"
        - high_res_name  : "S1" or "S2"
        - d_ref_max      : float
    """
    if "spatial" not in adata_s1.obsm_keys():
        raise KeyError("adata_s1.obsm 中没有 'spatial' 键。")
    if "spatial" not in adata_s2.obsm_keys():
        raise KeyError("adata_s2.obsm 中没有 'spatial' 键。")

    xy1 = np.asarray(adata_s1.obsm["spatial"])
    xy2 = np.asarray(adata_s2.obsm["spatial"])

    if xy1.shape[1] < 2 or xy2.shape[1] < 2:
        raise ValueError("obsm['spatial'] 至少需要两列坐标（x,y）。")

    xy1 = xy1[:, :2]
    xy2 = xy2[:, :2]

    # 清理 NA/Inf
    def clean_xy(xy):
        mask = np.isfinite(xy).all(axis=1)
        return xy[mask, :], mask

    xy1_clean, mask1 = clean_xy(xy1)
    xy2_clean, mask2 = clean_xy(xy2)

    if xy1_clean.shape[0] == 0 or xy2_clean.shape[0] == 0:
        raise ValueError("S1 或 S2 有效坐标为空（全是 NA/Inf？）。")

    print(f"S1 有效坐标点数: {xy1_clean.shape[0]} / {xy1.shape[0]}")
    print(f"S2 有效坐标点数: {xy2_clean.shape[0]} / {xy2.shape[0]}")

    # 内部最近邻均值（均值越大 -> 越稀疏 -> 低分辨率）
    mean_s1, nn_s1 = mean_internal_nn_distance(xy1_clean)
    mean_s2, nn_s2 = mean_internal_nn_distance(xy2_clean)

    print(f"S1 内部最近邻距离均值: {mean_s1:.4f}")
    print(f"S2 内部最近邻距离均值: {mean_s2:.4f}")

    if mean_s1 > mean_s2:
        low_res_name = "S1"
        high_res_name = "S2"
        low_xy, low_mask = xy1_clean, mask1
        high_xy, high_mask = xy2_clean, mask2
        nn_low = nn_s1
        adata_low, adata_high = adata_s1, adata_s2
    else:
        low_res_name = "S2"
        high_res_name = "S1"
        low_xy, low_mask = xy2_clean, mask2
        high_xy, high_mask = xy1_clean, mask1
        nn_low = nn_s2
        adata_low, adata_high = adata_s2, adata_s1

    print(f"\n自动判断：{low_res_name} = 低分辨率，{high_res_name} = 高分辨率")

    d_ref_max = float(np.max(nn_low)) if nn_low.size > 0 else 0.0
    print(f"低分辨率切片内部最近邻最大距离 d_ref_max = {d_ref_max:.4f}")

    # 有效点的原始索引
    high_indices_all = np.where(high_mask)[0]
    low_indices_all = np.where(low_mask)[0]

    # 高分辨率 -> 低分辨率 最近邻
    print("\n--- 高分辨率 → 低分辨率 最近邻搜索 ---")
    tree = cKDTree(low_xy)
    dist, idx = tree.query(high_xy, k=1)

    # 距离过滤
    if d_ref_max > 0:
        valid = dist <= 2.0 * d_ref_max
    else:
        valid = np.ones_like(dist, dtype=bool)

    n_drop = int(np.sum(~valid))
    print(f"距离过滤：删除 {n_drop} 个高分辨率点（dist > 2 * d_ref_max）。")

    dist_f = dist[valid]
    idx_f = idx[valid]
    high_idx_clean = high_indices_all[valid]
    low_idx_clean = low_indices_all[idx_f]

    # 生成 id
    def get_ids(adata, id_col_name):
        if id_col_name in adata.obs.columns:
            return adata.obs[id_col_name].astype(str).to_numpy()
        return adata.obs_names.astype(str).to_numpy()

    low_ids_all = get_ids(adata_low, id_col)
    high_ids_all = get_ids(adata_high, id_col)

    mapping = pd.DataFrame(
        {
            "high_id": high_ids_all[high_idx_clean],
            "low_id": low_ids_all[low_idx_clean],
            "high_x": adata_high.obsm["spatial"][high_idx_clean, 0],
            "high_y": adata_high.obsm["spatial"][high_idx_clean, 1],
            "low_x": adata_low.obsm["spatial"][low_idx_clean, 0],
            "low_y": adata_low.obsm["spatial"][low_idx_clean, 1],
            "distance": dist_f,
            "high_index": high_idx_clean,
            "low_index": low_idx_clean,
        }
    )

    print("\n映射表前几行：")
    print(mapping.head())

    meta = {
        "low_res_name": low_res_name,
        "high_res_name": high_res_name,
        "d_ref_max": d_ref_max,
    }
    return mapping, meta


# =========================
# 根据 mapping + 两个 h5ad 构建新的 h5ad
# =========================
def build_reassigned_h5ad_from_mapping(
    mapping: pd.DataFrame,
    meta: dict,
    adata_s1: AnnData,
    adata_s2: AnnData,
    out_h5ad: str,
    id_col: str = "id",
    cluster_col: str = "cluster",
    s2_cluster_col: Union[str, List[str]] = "Manual_annotation",
    scale_by_mapping_factor: bool = True,
):
    """
    构建新 h5ad：

      - 低分辨率 h5ad 提供表达矩阵
      - 新 obsm['spatial'] 用高分辨率点坐标
      - 每个新点的表达 = 对应 low_index 的表达（可选 1/k 缩放）
      - cluster 从低分辨率 h5ad 映射（若存在）
      - （可选）保留低分辨率切片的 Manual_annotation（如果存在）
      - 额外新增：从高分辨率切片拷贝指定的多个 obs 列
        写入列名为 {high}_{col}（例如 s2_cluster, s2_barcode_S2）
    """
    if meta["low_res_name"] == "S1":
        adata_low = adata_s1
        adata_high = adata_s2
        low_name = "S1"
        high_name = "S2"
    else:
        adata_low = adata_s2
        adata_high = adata_s1
        low_name = "S2"
        high_name = "S1"

    print(f"\n在 h5ad 中：{low_name} 用作低分辨率表达提供者，{high_name} 用作高分辨率空间参考。")

    if mapping.shape[0] == 0:
        raise ValueError("mapping 为空，没有任何匹配点。")

    low_idx = mapping["low_index"].to_numpy(dtype=int)
    high_idx = mapping["high_index"].to_numpy(dtype=int)

    # ---- 抽取低分辨率表达矩阵 ----
    X_low = adata_low.X
    if sparse.isspmatrix_coo(X_low):
        print("ℹ️ 检测到低分辨率 X 为 coo_matrix，转换为 csr_matrix。")
        X_low = X_low.tocsr()

    if sparse.issparse(X_low):
        X_new = X_low[low_idx, :]
    else:
        X_new = np.asarray(X_low)[low_idx, :]

    # ---- 1/k 缩放 ----
    if scale_by_mapping_factor:
        count_map = pd.Series(low_idx).value_counts()
        mapping_factor = pd.Series(low_idx).map(count_map).to_numpy()
        if np.any(mapping_factor <= 0):
            raise ValueError("检测到 mapping_factor <= 0，映射计数异常。")
        scale = 1.0 / mapping_factor

        if sparse.issparse(X_new):
            X_new = X_new.multiply(scale[:, None])
            if sparse.isspmatrix_coo(X_new):
                print("ℹ️ X_new 为 coo_matrix，转换为 csr_matrix 以便写入 h5ad。")
                X_new = X_new.tocsr()
        else:
            X_new = X_new * scale[:, None]

    # ---- 组装 obs ----
    n_obs_new = mapping.shape[0]
    obs_names = [f"reassign_{i}" for i in range(n_obs_new)]
    obs = pd.DataFrame(index=pd.Index(obs_names, name=None))

    obs["low_id"] = mapping["low_id"].astype(str).values
    obs["high_id"] = mapping["high_id"].astype(str).values

    # cluster 映射（来自低分辨率）
    if cluster_col in adata_low.obs.columns:
        col_src = adata_low.obs[cluster_col]
        col_src_str = col_src.astype(str)
        cats = pd.unique(col_src_str)
        mapped_cluster_str = col_src_str.iloc[low_idx].reset_index(drop=True)
        obs["cluster"] = pd.Categorical(mapped_cluster_str.values, categories=cats, ordered=False)
    else:
        print(f"ℹ️ 低分辨率 h5ad 中无 obs['{cluster_col}']，跳过 cluster 映射。")

    # ---- 保留低分辨率的 Manual_annotation（如果存在）----
    low_obs_sel = adata_low.obs.iloc[low_idx].copy().reset_index(drop=True)
    if "Manual_annotation" in low_obs_sel.columns:
        obs["Manual_annotation"] = low_obs_sel["Manual_annotation"].astype(str).values
    else:
        print("ℹ️ 低分辨率 h5ad 中无 obs['Manual_annotation']，跳过写入 Manual_annotation。")

    # ---- 新增：高分辨率切片指定列（支持多个；找不到则跳过）----
    high_obs_sel = adata_high.obs.iloc[high_idx].copy().reset_index(drop=True)

    # 统一成 list
    if isinstance(s2_cluster_col, str):
        s2_cols = [s2_cluster_col]
    else:
        s2_cols = list(s2_cluster_col)

    written_cols = []
    # for col in s2_cols:
    #     new_col_name = f"{high_name.lower()}_{col}"
    #     if col in high_obs_sel.columns:
    #         obs[new_col_name] = high_obs_sel[col].astype(str).values
    #         written_cols.append(new_col_name)
    #     else:
    #         print(f"⚠️ 高分辨率({high_name}) h5ad 中无 obs['{col}']，跳过该列。")
    for col in s2_cols:
        if col not in high_obs_sel.columns:
            print(f"⚠️ 高分辨率({high_name}) h5ad 中无 obs['{col}']，跳过该列。")
            continue

        # 默认：不加前缀
        out_name = col

        # 如果新 h5ad（obs）里已存在同名列，则不覆盖，改用前缀名写入
        if out_name in obs.columns:
            out_name = f"{high_name.lower()}_{col}"
            print(
                f"⚠️ 新 h5ad 中已存在 obs['{col}']，不覆盖；"
                f"高分辨率({high_name}) 的列将写入 obs['{out_name}']。"
            )

        obs[out_name] = high_obs_sel[col].astype(str).values


    # ---- var 继承自低分辨率 ----
    var = adata_low.var.copy()
    var_names = adata_low.var_names.copy()

    adata_new = AnnData(X=X_new, obs=obs, var=var)
    adata_new.var_names = var_names

    # ---- 空间坐标：用高分辨率坐标 ----
    adata_new.obsm["spatial"] = mapping[["high_x", "high_y"]].to_numpy(dtype=float)

    # 距离信息（不是 annotation 列）
    adata_new.obs["knn_dist"] = mapping["distance"].values

    # meta
    adata_new.uns["reassignment_meta"] = {
        "low_res_name": meta["low_res_name"],
        "high_res_name": meta["high_res_name"],
        "d_ref_max": float(meta["d_ref_max"]),
        "requested_high_obs_cols": s2_cols,
        "written_high_obs_cols": written_cols,
    }

    # ---- 保存 ----
    out_dir = os.path.dirname(out_h5ad)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    adata_new.write_h5ad(out_h5ad, compression="gzip")
    print(f"\n✅ 新 h5ad 已保存：{out_h5ad}")
    print(f"   形状：{adata_new.n_obs} × {adata_new.n_vars}")
    print("   obs 列包含：", list(adata_new.obs.columns))
    print("   obsm 包含：", list(adata_new.obsm.keys()))
    if "cluster" in adata_new.obs:
        print("   cluster dtype:", adata_new.obs["cluster"].dtype)
    else:
        print("   cluster dtype: N/A")

    return adata_new


# =========================
# 总管函数：spomialign_reassignment
# =========================
def spomialign_reassignment(
    s1_h5ad: str,
    s2_h5ad: str,
    out_h5ad: str,
    map_csv: Optional[str] = None,
    id_col: str = "id",
    cluster_col: str = "cluster",
    s2_cluster_col: Union[str, List[str]] = "Manual_annotation",
    scale_by_mapping_factor: bool = True,
):
    """
    SPOmiAlign reassignment pipeline（纯 h5ad 版本）
    """
    print(f"读取 S1 h5ad: {s1_h5ad}")
    adata_s1 = sc.read_h5ad(s1_h5ad)
    print(f"读取 S2 h5ad: {s2_h5ad}")
    adata_s2 = sc.read_h5ad(s2_h5ad)

    mapping, meta = compute_nn_mapping_from_h5ads(
        adata_s1=adata_s1,
        adata_s2=adata_s2,
        id_col=id_col,
    )

    if map_csv is not None:
        out_dir = os.path.dirname(map_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        mapping.to_csv(map_csv, index=False)
        print(f"\n中间映射表已保存：{map_csv}")

    adata_new = build_reassigned_h5ad_from_mapping(
        mapping=mapping,
        meta=meta,
        adata_s1=adata_s1,
        adata_s2=adata_s2,
        out_h5ad=out_h5ad,
        id_col=id_col,
        cluster_col=cluster_col,
        s2_cluster_col=s2_cluster_col,
        scale_by_mapping_factor=scale_by_mapping_factor,
    )
    return adata_new


# =========================
# 命令行入口
# =========================
def main():
    parser = argparse.ArgumentParser(
        description=(
            "SPOmiAlign reassignment（纯 h5ad 版本）：\n"
            "自动判断 S1/S2 哪个是高分辨率/低分辨率，高分辨率点在低分辨率上找最近邻，"
            "并构建带有低分辨率表达的新 h5ad。\n"
            "新 h5ad 会（若存在）保留低分辨率 Manual_annotation，"
            "并额外新增多个高分辨率 obs 列：{high}_{col}（例如 s2_cluster）。"
        )
    )
    parser.add_argument("--s1_h5ad", "-h1", required=True, help="S1 h5ad 路径（含 obsm['spatial']）")
    parser.add_argument("--s2_h5ad", "-h2", required=True, help="S2 h5ad 路径（含 obsm['spatial']）")
    parser.add_argument("--out_h5ad", "-o", required=True, help="输出 h5ad 路径")
    parser.add_argument("--map_csv", "-m", default=None, help="中间映射表 CSV 输出路径（可选）")
    parser.add_argument("--id_col", default="id", help="obs 中用作 id 的列名（若不存在则使用 obs_names）")
    parser.add_argument("--cluster_col", default="cluster", help="从低分辨率 h5ad 中拷贝 cluster 的列名")

    parser.add_argument(
        "--s2_cluster_col",
        nargs="+",
        default=["Manual_annotation"],
        help=(
            "从高分辨率切片（high-res）拷贝到新h5ad的obs列名，可写多个。"
            "例如: --s2_cluster_col cluster barcode_S2"
        ),
    )
    parser.add_argument("--no_scale", action="store_true", help="关闭 1/k 缩放（不做表达量密度归一化）")
    args = parser.parse_args()

    spomialign_reassignment(
        s1_h5ad=args.s1_h5ad,
        s2_h5ad=args.s2_h5ad,
        out_h5ad=args.out_h5ad,
        map_csv=args.map_csv,
        id_col=args.id_col,
        cluster_col=args.cluster_col,
        s2_cluster_col=args.s2_cluster_col,
        scale_by_mapping_factor=not args.no_scale,
    )


if __name__ == "__main__":
    main()
