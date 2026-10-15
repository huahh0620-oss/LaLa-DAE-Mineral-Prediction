# -*- coding: utf-8 -*-
"""
步骤 5 / Step 5 —— 重构误差 -> 多元地化异常图
====================================================================
论文 Eq.(1)：把每个像元的多变量向量输入训练好的自编码器，用

        RE(i) = (1/D) * sum_j ( I_ij - R_ij )^2

作为该像元的异常得分（重构误差）。小概率的异常样本重构误差高。

★ 改进点 ③：加权重构误差
   直接对 D 个变量取平均时，判别力弱的变量会稀释异常信号。本步骤提供
   两种得分模式（config.SCORE_MODE）：

     "mse"      : 论文原式 RE = mean_j (x_j - r_j)^2          （严格复现）
     "weighted" : RE_w = sum_j  w_j (x_j - r_j)^2             （默认）
                  权重 w_j 由每个元素的单元素判别力(AUC-0.5)归一化得到，
                  让异常得分集中在铜金矿化指示元素上。

   为便于对比，两种模式的结果都会写出栅格。

产出：
  cache/anomaly_scores.npz                  异常得分（选定模式 + 两种模式）
  anomaly_map.tif / anomaly_map_normalized.tif      选定模式的异常图
  anomaly_map_mse.tif / anomaly_map_weighted.tif    两种模式的异常图
  figures/step05_anomaly_map.png
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import rasterio

import config as C
from src.autoencoder import DeepAutoencoder
from src.common import (limit_threads, locate_eval_raster, log,
                        read_deposit_aligned, read_deposit_grid, resolve_device,
                        save_json, set_seed, write_raster_like)


def _load_model(seed: int, device):
    import torch
    ckpt = torch.load(C.MODEL_DIR / f"deep_autoencoder_seed{seed}.pt",
                      map_location=device, weights_only=False)
    model = DeepAutoencoder(ckpt["dims"], ckpt.get("activation", "sigmoid"),
                            ckpt.get("out_activation", "sigmoid"))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def reconstruct(model, X: np.ndarray, device, batch: int = 4096) -> np.ndarray:
    """返回重构矩阵 R（与输入同形）。"""
    import torch
    R = np.empty_like(X, dtype=np.float64)
    with torch.no_grad():
        for s in range(0, X.shape[0], batch):
            e = min(s + batch, X.shape[0])
            xb = torch.as_tensor(X[s:e], dtype=torch.float32, device=device)
            R[s:e] = model(xb).detach().cpu().numpy()
    return R


def reconstruction_error(model, X: np.ndarray, device, batch: int = 4096) -> np.ndarray:
    """论文 Eq.(1)：不加权的重构误差。"""
    R = reconstruct(model, X, device, batch)
    return ((X - R) ** 2).mean(axis=1)


def pathfinder_weights(X: np.ndarray, deposit: np.ndarray) -> np.ndarray:
    """按单元素判别力(AUC-0.5)构造权重，归一化到和为 1。"""
    from sklearn.metrics import roc_auc_score
    w = np.array([max(0.0, roc_auc_score(deposit, X[:, j]) - 0.5)
                  for j in range(X.shape[1])])
    if w.sum() <= 0:
        w = np.ones_like(w)
    return w / w.sum()


def weighted_error(model, X, device, weights, batch: int = 4096) -> np.ndarray:
    """改进点 ③：加权重构误差 sum_j w_j (x_j - r_j)^2。"""
    import torch
    out = np.empty(X.shape[0], dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    with torch.no_grad():
        for s in range(0, X.shape[0], batch):
            e = min(s + batch, X.shape[0])
            xb = torch.as_tensor(X[s:e], dtype=torch.float32, device=device)
            rb = model(xb)
            res2 = ((xb - rb) ** 2).detach().cpu().numpy()
            out[s:e] = (res2 * w).sum(axis=1)
    return out


def _minmax(s: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(s)), float(np.max(s))
    return (s - lo) / (hi - lo) if hi > lo else np.zeros_like(s)


def score_mode() -> str:
    mode = str(getattr(C, "SCORE_MODE", "weighted")).lower()
    return mode if mode in ("mse", "weighted") else "weighted"


def run(seeds=None) -> dict:
    t0 = time.time()
    seeds = list(seeds) if seeds is not None else list(C.RANDOM_SEEDS)
    limit_threads(C.NUM_THREADS)
    device = resolve_device(C.DEVICE)
    mode = score_mode()

    log.info("=" * 68)
    log.info("STEP 5 | 生成多元地化异常图 (seeds=%s, score=%s)", seeds, mode)
    log.info("=" * 68)

    ds = np.load(C.CACHE_DIR / "dataset.npz", allow_pickle=True)
    valid_mask = ds["valid_mask"].astype(bool)
    height, width = int(ds["height"]), int(ds["width"])

    from src.step03_pretrain_crbm import load_input_matrix
    X, sel_names, _ = load_input_matrix()
    X = X.astype(np.float64)
    log.info("输入变量 (%d): %s", len(sel_names), ", ".join(sel_names))

    pos_path = locate_eval_raster(C.Data_Path, C.EVAL_RASTER, C.POSITIVE_RASTER,
                                  C.POSITIVE_SEARCH_PARENTS)
    deposit = read_deposit_aligned(pos_path, valid_mask, (height, width))
    deposit_grid = read_deposit_grid(pos_path, (height, width))

    # 改进点 ③ 的权重（只用于 weighted 模式）
    weights = pathfinder_weights(X, deposit)
    log.info("指示元素权重: %s",
             ", ".join(f"{n}={w:.3f}" for n, w in zip(sel_names, weights)))

    per_seed = {}
    for seed in seeds:
        model = _load_model(seed, device)
        e_mse = reconstruction_error(model, X, device)
        e_w = weighted_error(model, X, device, weights)
        per_seed[seed] = {"mse": e_mse, "weighted": e_w}
        log.info("  seed=%d  mse范围 %.3g~%.3g | weighted范围 %.3g~%.3g",
                 seed, e_mse.min(), e_mse.max(), e_w.min(), e_w.max())

    def _agg(key):
        return np.mean([_minmax(per_seed[s][key]) for s in seeds], axis=0) if len(seeds) > 1 \
            else _minmax(per_seed[seeds[0]][key])

    norm_mse, norm_w = _agg("mse"), _agg("weighted")
    raw_sel = np.mean([per_seed[s][mode] for s in seeds], axis=0)
    norm_sel = norm_w if mode == "weighted" else norm_mse

    with rasterio.open(_find_reference_raster()) as ref:
        profile = ref.profile

    def _to_grid(v):
        g = np.full(height * width, np.nan)
        g[valid_mask.ravel()] = v
        return g.reshape(height, width)

    grids = {"mse": _to_grid(norm_mse), "weighted": _to_grid(norm_w)}
    write_raster_like(C.OUTPUT_DIR / "anomaly_map_mse.tif", grids["mse"],
                      {"profile": profile}, nodata=-9999.0)
    write_raster_like(C.OUTPUT_DIR / "anomaly_map_weighted.tif", grids["weighted"],
                      {"profile": profile}, nodata=-9999.0)
    write_raster_like(C.OUTPUT_DIR / "anomaly_map.tif", grids[mode],
                      {"profile": profile}, nodata=-9999.0)
    write_raster_like(C.OUTPUT_DIR / "anomaly_map_normalized.tif", grids[mode],
                      {"profile": profile}, nodata=-9999.0)

    np.savez_compressed(C.CACHE_DIR / "anomaly_scores.npz",
                        score_raw=raw_sel, score_norm=norm_sel,
                        score_norm_mse=norm_mse, score_norm_weighted=norm_w,
                        mode=mode, weights=weights, feature_names=np.array(sel_names, dtype=object),
                        valid_mask=valid_mask, height=height, width=width,
                        seeds=np.array(seeds))

    _plot(grids[mode], deposit_grid, norm_sel, deposit, seeds, pos_path.name, mode)

    report = {
        "seeds": seeds, "score_mode": mode, "input_features": sel_names,
        "pathfinder_weights": {n: float(w) for n, w in zip(sel_names, weights)},
        "positive_raster": str(pos_path),
        "score_min": float(raw_sel.min()), "score_max": float(raw_sel.max()),
        "score_mean": float(raw_sel.mean()), "score_std": float(raw_sel.std()),
        "n_pixels": int(X.shape[0]),
        "elapsed_sec": round(time.time() - t0, 2),
        "outputs": {
            "anomaly_map": str(C.OUTPUT_DIR / "anomaly_map.tif"),
            "anomaly_map_mse": str(C.OUTPUT_DIR / "anomaly_map_mse.tif"),
            "anomaly_map_weighted": str(C.OUTPUT_DIR / "anomaly_map_weighted.tif"),
            "scores": str(C.CACHE_DIR / "anomaly_scores.npz"),
            "figure": str(C.FIG_DIR / "step05_anomaly_map.png"),
        },
    }
    save_json(C.TABLE_DIR / f"step05_report_seed{seeds[0]}.json", report)
    log.info("STEP 5 完成，用时 %.1f s", time.time() - t0)
    return report


def _find_reference_raster():
    from src.common import list_geochem_rasters, locate_geochem_dir
    geo = locate_geochem_dir(Path(C.Data_Path), C.GEOCHEM_SUBDIR,
                             C.GEOCHEM_GLOB, C.AUTO_LOCATE_GEOCHEM)
    return list_geochem_rasters(geo, C.GEOCHEM_GLOB)[0]


def _plot(grid, deposit_grid, norm_sel, deposit, seeds, pos_name, mode) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = np.where(np.isfinite(grid), grid, np.nan)
    yy, xx = np.where(deposit_grid)
    tag = f"seed={seeds[0]}" if len(seeds) == 1 else f"ensemble of {len(seeds)} seeds"

    fig, axes = plt.subplots(1, 2, figsize=(15.6, 5.4), dpi=C.FIGURE_DPI,
                             gridspec_kw={"width_ratios": [1.5, 1]})
    im = axes[0].imshow(g, cmap="turbo", interpolation="nearest")
    axes[0].scatter(xx, yy, marker="*", s=C.STAR_SIZE * 1.9, c="none",
                    edgecolors="white", linewidths=1.9, zorder=5)
    axes[0].scatter(xx, yy, marker="*", s=C.STAR_SIZE, c="k", edgecolors="none", zorder=6,
                    label=f"Known deposits (n={len(yy)})")
    axes[0].set_title(f"Multivariate geochemical anomaly map\n(reconstruction error, {mode}, {tag})")
    axes[0].set_xlabel("Column"); axes[0].set_ylabel("Row")
    axes[0].legend(fontsize=8.5, loc="lower right", framealpha=0.85)
    cb = fig.colorbar(im, ax=axes[0], fraction=0.036, pad=0.02)
    cb.set_label("Normalized anomaly score")

    axes[1].hist(norm_sel, bins=60, color="#9ecae1", edgecolor="none", density=True)
    axes[1].axvline(np.median(norm_sel[deposit]), color="#d7191c", lw=2,
                    label=f"Deposit median = {np.median(norm_sel[deposit]):.2f}")
    axes[1].axvline(np.median(norm_sel), color="k", ls="--", lw=1.4,
                    label=f"All-pixel median = {np.median(norm_sel):.2f}")
    axes[1].set_xlabel("Normalized anomaly score"); axes[1].set_ylabel("Density")
    axes[1].set_title(f"Anomaly score distribution\n({pos_name}, deposits n={int(deposit.sum())})")
    axes[1].legend(fontsize=8.5); axes[1].grid(alpha=0.3, ls=":")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "step05_anomaly_map.png")
    plt.close(fig)


if __name__ == "__main__":
    run()