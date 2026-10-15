# -*- coding: utf-8 -*-
"""
步骤 1 / Step 1 —— 数据准备
====================================================================
读取 Data_Path 下的 39 个地化栅格 -> 对齐校验 -> 有效像元掩膜 ->
(可选 ilr 处理) -> 按论文 Eq.(13) 归一化到 [0,1] -> 生成样本矩阵。

产出（均落在 OUTPUT_DIR 下）：
  cache/dataset.npz            样本矩阵 X、有效像元掩膜、栅格元数据
  tables/feature_stats.csv     每个地化变量的统计量
  tables/step01_report.json    数据准备报告
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from src.common import (apply_minmax, feature_name_from_path, fit_minmax,
                        list_geochem_rasters, locate_geochem_dir, locate_raster,
                        log, read_aligned_stack, save_json)


# ---------------------------------------------------------------------
def _ilr(X: np.ndarray) -> np.ndarray:
    """等距对数比变换 (Egozcue et al., 2003)，处理闭合效应。"""
    X = np.asarray(X, dtype=np.float64)
    Z = X / X.sum(axis=1, keepdims=True)
    logs = np.log(Z)
    D = Z.shape[1]
    # 标准 ilr 基 (Helmert)
    H = np.zeros((D, D - 1))
    for i in range(D - 1):
        H[:i + 1, i] = 1.0 / np.sqrt((i + 1) * (i + 2))
        H[i + 1, i] = -(i + 1) / np.sqrt((i + 1) * (i + 2))
    return logs @ H


# ---------------------------------------------------------------------
def run() -> dict:
    t0 = time.time()
    log.info("=" * 68)
    log.info("STEP 1 | 数据准备 (Data preparation)")
    log.info("=" * 68)

    data_path = Path(C.Data_Path)
    log.info("Data_Path = %s", data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"数据目录不存在：{data_path}\n"
            f"请设置环境变量 LALA_DATA_PATH，或修改 config.py 中的 Data_Path。")

    # --- 定位地化栅格 -------------------------------------------------
    geo_dir = locate_geochem_dir(data_path, C.GEOCHEM_SUBDIR,
                                 C.GEOCHEM_GLOB, C.AUTO_LOCATE_GEOCHEM)
    files = list_geochem_rasters(geo_dir, C.GEOCHEM_GLOB)
    if not files:
        raise FileNotFoundError(f"{geo_dir} 下没有匹配 {C.GEOCHEM_GLOB} 的栅格。")
    feature_names = [feature_name_from_path(p) for p in files]
    log.info("地化目录: %s", geo_dir)
    log.info("地化变量 (%d): %s", len(files), ", ".join(feature_names))

    # --- 读取并校验对齐 -----------------------------------------------
    cube, meta = read_aligned_stack(files)
    n_band, height, width = cube.shape
    if n_band != len(files):
        raise ValueError("读取的波段数与文件数不一致。")
    log.info("栅格网格: %d 行 x %d 列, %d 个变量", height, width, n_band)

    # --- 展平为样本矩阵 -----------------------------------------------
    X_all = cube.reshape(n_band, -1).T          # (H*W, F)
    valid_mask_flat = np.isfinite(X_all).all(axis=1)
    n_total = X_all.shape[0]
    n_valid = int(valid_mask_flat.sum())
    log.info("有效像元: %d / %d (%.2f%%)", n_valid, n_total, 100.0 * n_valid / n_total)

    # --- 可选研究区掩膜 -----------------------------------------------
    if C.MASK_RASTER:
        mask_path = locate_raster(data_path, C.MASK_RASTER, C.POSITIVE_SEARCH_PARENTS)
        import rasterio
        with rasterio.open(mask_path) as ds:
            mask = (ds.read(1) != 0)
        if mask.shape != (height, width):
            raise ValueError(f"掩膜 {mask_path.name} 与研究区网格不一致。")
        valid_mask_flat &= mask.ravel()
        log.info("叠加掩膜后有效像元: %d", int(valid_mask_flat.sum()))

    X = X_all[valid_mask_flat].astype(np.float64)

    # --- ilr（仅在原始浓度数据时）-------------------------------------
    if C.RAW_COMPOSITION:
        log.info("检测到 RAW_COMPOSITION=True -> 执行 ilr 变换 (Eq. 处理闭合效应)")
        X = _ilr(X)
        feature_names = [f"ilr{i + 1}" for i in range(X.shape[1])]
    else:
        log.info("数据判定为 CLR 结果，跳过 ilr（闭合效应已处理）")

    # --- 统计 ---------------------------------------------------------
    stats = pd.DataFrame({
        "feature": feature_names,
        "min": X.min(axis=0), "max": X.max(axis=0),
        "mean": X.mean(axis=0), "std": X.std(axis=0),
        "p05": np.quantile(X, 0.05, axis=0), "p50": np.quantile(X, 0.50, axis=0),
        "p95": np.quantile(X, 0.95, axis=0),
    })
    stats.to_csv(C.TABLE_DIR / "feature_stats.csv", index=False, encoding="utf-8-sig")

    # --- 归一化到 [0,1] (Eq.13) ---------------------------------------
    norm_params = fit_minmax(X, *C.NORMALIZE_RANGE)
    Xn = apply_minmax(X, norm_params)
    log.info("归一化范围 [%.0f, %.0f]；归一化后逐变量落在 [%.3f, %.3f]",
             C.NORMALIZE_RANGE[0], C.NORMALIZE_RANGE[1], Xn.min(), Xn.max())

    # --- 保存缓存 -----------------------------------------------------
    np.savez_compressed(
        C.CACHE_DIR / "dataset.npz",
        X_raw=X.astype(np.float32),
        X_norm=Xn.astype(np.float32),
        valid_mask=valid_mask_flat.reshape(height, width),
        feature_names=np.array(feature_names, dtype=object),
        transform=np.array(meta["transform"]).astype(np.float64),
        crs=np.array(meta["crs"]),
        height=height, width=width,
        norm_min=norm_params[0], norm_xmax=norm_params[1],
        norm_lo=norm_params[2], norm_hi=norm_params[3],
    )

    report = {
        "data_path": str(data_path),
        "geochem_dir": str(geo_dir),
        "n_features": int(X.shape[1]),
        "n_pixels_total": int(n_total),
        "n_pixels_valid": int(n_valid),
        "valid_fraction": float(n_valid / n_total),
        "raw_composition": bool(C.RAW_COMPOSITION),
        "normalize_range": list(C.NORMALIZE_RANGE),
        "grid": {"height": height, "width": width},
        "crs": meta["crs"],
        "features": feature_names,
        "elapsed_sec": round(time.time() - t0, 2),
        "outputs": {
            "dataset": str(C.CACHE_DIR / "dataset.npz"),
            "feature_stats": str(C.TABLE_DIR / "feature_stats.csv"),
        },
    }
    save_json(C.TABLE_DIR / "step01_report.json", report)
    log.info("STEP 1 完成，用时 %.1f s", time.time() - t0)
    return report


if __name__ == "__main__":
    run()