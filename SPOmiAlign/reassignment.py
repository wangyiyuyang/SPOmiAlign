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
