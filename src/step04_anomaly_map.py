# -*- coding: utf-8 -*-
"""
步骤 4 / Step 4 —— 重构误差 -> 多元地化异常图
====================================================================
论文 Eq.(1)：把每个像元的多变量向量输入训练好的自编码器，用
       RE(i) = (1/D) * sum_j ( I_ij - R_ij )^2
作为该像元的异常得分（重构误差）。小概率的异常样本重构误差高。
得分归一化到 [0,1] 后写回栅格，得到多元地化异常图。

当 RANDOM_SEEDS 含多个种子时，对每个种子的归一化得分取平均作为集成异常图
（模型的稳定性与论文一致，集成可进一步降低随机性影响）。

产出：
  cache/anomaly_scores.npz                  有效像元的原始/归一化异常得分
  cache/anomaly_scores_seed{seed}.npy       各种子的得分（若多种子）
  anomaly_map.tif / anomaly_map_normalized.tif
  figures/step04_anomaly_map.png
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import rasterio

import config as C
from src.autoencoder import DeepAutoencoder
from src.common import (limit_threads, log, resolve_device, save_json, set_seed,
                        write_raster_like)


def _load_model(seed: int, device):
    import torch
    ckpt = torch.load(C.MODEL_DIR / f"deep_autoencoder_seed{seed}.pt",
                      map_location=device, weights_only=False)
    model = DeepAutoencoder(ckpt["dims"], ckpt.get("activation", "sigmoid"),
                            ckpt.get("out_activation", "sigmoid"))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def reconstruction_error(model, X: np.ndarray, device, batch: int = 4096) -> np.ndarray:
    """按 Eq.(1) 计算每个样本的重构误差（对 D 个变量取平均）。"""
    import torch
    out = np.empty(X.shape[0], dtype=np.float64)
    with torch.no_grad():
        for s in range(0, X.shape[0], batch):
            e = min(s + batch, X.shape[0])
            xb = torch.as_tensor(X[s:e], dtype=torch.float32, device=device)
            rb = model(xb)
            err = ((xb - rb) ** 2).mean(dim=1)
            out[s:e] = err.detach().cpu().numpy()
    return out


def _minmax(s: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(s)), float(np.max(s))
    return (s - lo) / (hi - lo) if hi > lo else np.zeros_like(s)


def run(seeds=None) -> dict:
    t0 = time.time()
    seeds = list(seeds) if seeds is not None else list(C.RANDOM_SEEDS)
    limit_threads(C.NUM_THREADS)
    device = resolve_device(C.DEVICE)

    log.info("=" * 68)
    log.info("STEP 4 | 生成多元地化异常图 (seeds=%s)", seeds)
    log.info("=" * 68)

    ds = np.load(C.CACHE_DIR / "dataset.npz", allow_pickle=True)
    X = ds["X_norm"].astype(np.float32)
    valid_mask = ds["valid_mask"].astype(bool)
    height, width = int(ds["height"]), int(ds["width"])
    log.info("输入变量数: %d；计算 %d 个像元的重构误差 ...",
             ds["feature_names"].shape[0], X.shape[0])

    per_seed_raw, per_seed_norm = {}, {}
    for seed in seeds:
        model = _load_model(seed, device)
        raw = reconstruction_error(model, X, device)
        per_seed_raw[seed] = raw
        per_seed_norm[seed] = _minmax(raw)
        log.info("  seed=%d  异常得分范围 %.6g ~ %.6g", seed, raw.min(), raw.max())
        if len(seeds) > 1:
            np.save(C.CACHE_DIR / f"anomaly_scores_seed{seed}.npy", raw)

    if len(seeds) == 1:
        score_raw = per_seed_raw[seeds[0]]
        score_norm = per_seed_norm[seeds[0]]
    else:
        score_norm = np.mean([per_seed_norm[s] for s in seeds], axis=0)
        # 集成原始得分用拼合后再归一化，保持与单种子口径一致
        score_raw = np.mean([per_seed_raw[s] for s in seeds], axis=0)
        log.info("已对 %d 个种子取平均得到集成异常图", len(seeds))

    # 写回栅格（用一张地化栅格作为空间模板）
    with rasterio.open(_find_reference_raster()) as ref:
        profile = ref.profile
    grid_raw = np.full(height * width, np.nan, dtype=np.float64)
    grid_norm = np.full(height * width, np.nan, dtype=np.float64)
    grid_raw[valid_mask.ravel()] = score_raw
    grid_norm[valid_mask.ravel()] = score_norm
    write_raster_like(C.OUTPUT_DIR / "anomaly_map.tif",
                      grid_raw.reshape(height, width), {"profile": profile}, nodata=-9999.0)
    write_raster_like(C.OUTPUT_DIR / "anomaly_map_normalized.tif",
                      grid_norm.reshape(height, width), {"profile": profile}, nodata=-9999.0)

    np.savez_compressed(C.CACHE_DIR / "anomaly_scores.npz",
                        score_raw=score_raw, score_norm=score_norm,
                        valid_mask=valid_mask, height=height, width=width,
                        feature_names=ds["feature_names"], seeds=np.array(seeds))

    _plot(grid_norm.reshape(height, width), seeds)

    report = {
        "seeds": seeds,
        "score_min": float(score_raw.min()), "score_max": float(score_raw.max()),
        "score_mean": float(score_raw.mean()), "score_std": float(score_raw.std()),
        "n_pixels": int(X.shape[0]),
        "per_seed_mean": {int(s): float(per_seed_raw[s].mean()) for s in seeds},
        "elapsed_sec": round(time.time() - t0, 2),
        "outputs": {
            "anomaly_map": str(C.OUTPUT_DIR / "anomaly_map.tif"),
            "anomaly_map_normalized": str(C.OUTPUT_DIR / "anomaly_map_normalized.tif"),
            "scores": str(C.CACHE_DIR / "anomaly_scores.npz"),
            "figure": str(C.FIG_DIR / "step04_anomaly_map.png"),
        },
    }
    save_json(C.TABLE_DIR / f"step04_report_seed{seeds[0]}.json", report)
    log.info("STEP 4 完成，用时 %.1f s", time.time() - t0)
    return report


def _find_reference_raster():
    """取一个地化栅格作为输出空间模板。"""
    from src.common import list_geochem_rasters, locate_geochem_dir
    geo = locate_geochem_dir(Path(C.Data_Path), C.GEOCHEM_SUBDIR,
                             C.GEOCHEM_GLOB, C.AUTO_LOCATE_GEOCHEM)
    return list_geochem_rasters(geo, C.GEOCHEM_GLOB)[0]


def _plot(grid: np.ndarray, seeds) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=150)
    im = ax.imshow(grid, cmap="turbo", interpolation="nearest")
    tag = f"seed={seeds[0]}" if len(seeds) == 1 else f"ensemble of {len(seeds)} seeds"
    ax.set_title(f"Multivariate geochemical anomaly map (reconstruction error, {tag})")
    ax.set_xlabel("Column"); ax.set_ylabel("Row")
    cb = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.02)
    cb.set_label("Normalized anomaly score")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "step04_anomaly_map.png")
    plt.close(fig)


if __name__ == "__main__":
    run()