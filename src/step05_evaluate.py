# -*- coding: utf-8 -*-
"""
步骤 5 / Step 5 —— 空间相关性评价与异常分级
====================================================================
论文 4.2 节：
  * 用 Student's t 值 (Agterberg et al., 1990; Bonham-Carter, 1994) 度量
    异常图与已知矿点的空间相关性；t > 1.96 表示显著。
  * 以 t 值极大值、t=1.96 两个阈值把异常分为 高 / 中 / 低 三级。
  * 另给出 ROC-AUC、Top-X% 面积矿点捕获率、成功度曲线等定量指标。

产出：
  tables/step05_threshold_scan.csv   阈值扫描的 t 值/捕获率
  tables/step05_metrics.csv          综合定量指标
  tables/step05_anomaly_classes.csv  三级异常的面积与矿点占比
  anomaly_classes.tif                高/中/低分级栅格 (3/2/1)
  figures/step05_tvalue.png, step05_anomaly_classes.png, step05_roc_capture.png
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

import config as C
from src.common import (limit_threads, locate_raster, log, save_json,
                        write_raster_like)


# ---------------------------------------------------------------------
def student_t_value(score: np.ndarray, deposit: np.ndarray, threshold: float):
    """Bonham-Carter (1994) 二值证据图 vs 矿点的 studentized t 值。

    返回 (t, n_anomaly, n_deposit_total, n_deposit_in_anomaly)。
    """
    anom = score >= threshold
    n_a = int(anom.sum())
    n_total = score.size
    n_d = int(deposit.sum())
    n_da = int((deposit & anom).sum())
    if n_a == 0 or n_a == n_total or n_d == 0:
        return np.nan, n_a, n_d, n_da
    p = n_a / n_total
    denom = np.sqrt(n_d * p * (1.0 - p))
    if denom == 0:
        return np.nan, n_a, n_d, n_da
    t = (n_da - n_d * p) / denom
    return float(t), n_a, n_d, n_da


def scan_thresholds(score: np.ndarray, deposit: np.ndarray, n: int):
    qs = np.linspace(0.01, 0.99, n)
    thrs = np.unique(np.quantile(score, qs))
    rows = []
    for thr in thrs:
        t, n_a, n_d, n_da = student_t_value(score, deposit, thr)
        rows.append({
            "threshold": float(thr),
            "area_fraction": n_a / score.size,
            "n_deposit_total": n_d,
            "n_deposit_in_anomaly": n_da,
            "capture_rate": (n_da / n_d) if n_d else np.nan,
            "enrichment": (n_da / n_d) / (n_a / score.size) if n_a > 0 and n_d else np.nan,
            "t_value": t,
        })
    return pd.DataFrame(rows)


def _roc_capture(score, deposit):
    order = np.argsort(-score)
    y = deposit[order].astype(np.int64)
    n_p = int(y.sum()); n_n = int(y.size - n_p)
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    tpr = tp / max(n_p, 1)
    fpr = fp / max(n_n, 1)
    # 面积占比轴下的捕获率（成功率曲线）
    area = np.arange(1, y.size + 1) / y.size
    from sklearn.metrics import roc_auc_score, average_precision_score
    auc = float(roc_auc_score(deposit, score))
    ap = float(average_precision_score(deposit, score))
    return dict(tpr=tpr, fpr=fpr, area=area, capture=tp / max(n_p, 1),
                auc=auc, ap=ap, n_pos=n_p, n_neg=n_n)


# ---------------------------------------------------------------------
def run(seed: int | None = None) -> dict:
    t0 = time.time()
    seed = int(seed if seed is not None else C.RANDOM_SEEDS[0])
    limit_threads(C.NUM_THREADS)

    log.info("=" * 68)
    log.info("STEP 5 | 空间相关性评价与异常分级 (seed=%d)", seed)
    log.info("=" * 68)

    d = np.load(C.CACHE_DIR / "anomaly_scores.npz", allow_pickle=True)
    score = d["score_norm"].astype(np.float64)
    valid_mask = d["valid_mask"].astype(bool)
    height, width = int(d["height"]), int(d["width"])

    # --- 读取已知矿点栅格，并对齐有效像元 -----------------------------
    pos_path = locate_raster(Path(C.Data_Path), C.POSITIVE_RASTER,
                             C.POSITIVE_SEARCH_PARENTS)
    with rasterio.open(pos_path) as ds:
        pos_grid = ds.read(1)
        if pos_grid.shape != (height, width):
            raise ValueError(f"矿点栅格 {pos_path.name} 与地化网格不一致。")
        nod = ds.nodata
    if nod is not None:
        pos_grid = np.where(pos_grid == nod, 0, pos_grid)
    deposit = (pos_grid.ravel()[valid_mask.ravel()] == 1)
    log.info("已知矿点像元: %d (来源 %s)", int(deposit.sum()), pos_path.name)

    # --- 阈值扫描 + t 值 ---------------------------------------------
    scan = scan_thresholds(score, deposit, C.N_THRESHOLD_CANDIDATES)
    scan.to_csv(C.TABLE_DIR / "step05_threshold_scan.csv", index=False,
                encoding="utf-8-sig")
    i_max = int(np.nanargmax(scan["t_value"].values))
    thr_max = float(scan["threshold"].iloc[i_max])
    t_max = float(scan["t_value"].iloc[i_max])
    # t=1.96 的最低阈值（显著性门槛）
    sig = scan[scan["t_value"] >= C.T_CRITICAL]
    thr_sig = float(sig["threshold"].min()) if len(sig) else np.nan
    log.info("t 值极大值 = %.3f (阈值 %.4f)", t_max, thr_max)
    log.info("t >= %.2f 的最低阈值 = %.4f", C.T_CRITICAL, thr_sig)

    # --- 三级分级 -----------------------------------------------------
    cls = np.full(score.shape, np.nan)
    cls[score < (thr_sig if np.isfinite(thr_sig) else thr_max)] = 1.0        # 低
    if np.isfinite(thr_sig):
        cls[(score >= thr_sig) & (score < thr_max)] = 2.0                    # 中
    cls[score >= thr_max] = 3.0                                              # 高

    rows = []
    for code, name in [(3.0, "high"), (2.0, "moderate"), (1.0, "low")]:
        m = (cls == code)
        n_a = int(m.sum())
        n_da = int((deposit & m).sum())
        rows.append({
            "class": name, "code": int(code),
            "n_pixels": n_a, "area_fraction": n_a / score.size,
            "n_deposit": n_da,
            "deposit_fraction_of_total": n_da / max(int(deposit.sum()), 1),
            "deposit_density": n_da / max(n_a, 1),
        })
    cls_tbl = pd.DataFrame(rows)
    cls_tbl.to_csv(C.TABLE_DIR / "step05_anomaly_classes.csv", index=False,
                   encoding="utf-8-sig")
    log.info("高异常占面积 %.2f%%，含 %.1f%% 已知矿点",
             100 * cls_tbl.loc[0, "area_fraction"], 100 * cls_tbl.loc[0, "deposit_fraction_of_total"])

    # --- 分级栅格 -----------------------------------------------------
    with rasterio.open(_ref_raster_path()) as ref:
        profile = ref.profile
    grid = np.full(height * width, np.nan)
    grid[valid_mask.ravel()] = cls
    write_raster_like(C.OUTPUT_DIR / "anomaly_classes.tif",
                      grid.reshape(height, width), {"profile": profile}, nodata=-9999.0)

    # --- 综合指标 -----------------------------------------------------
    rc = _roc_capture(score, deposit)
    metrics = {
        "seed": seed,
        "n_pixels": int(score.size),
        "n_deposit": int(deposit.sum()),
        "roc_auc": rc["auc"],
        "average_precision": rc["ap"],
        "t_max": t_max,
        "threshold_t_max": thr_max,
        "threshold_t_critical": thr_sig,
        "high_class_area_fraction": float(cls_tbl.loc[0, "area_fraction"]),
        "high_class_deposit_fraction": float(cls_tbl.loc[0, "deposit_fraction_of_total"]),
    }
    for pct in C.CAPTURE_PERCENTS:
        k = max(1, int(round(score.size * pct / 100.0)))
        idx = np.argsort(-score)[:k]
        metrics[f"capture_top{pct}pct"] = float(deposit[idx].sum() / max(int(deposit.sum()), 1))
    pd.DataFrame([metrics]).to_csv(C.TABLE_DIR / "step05_metrics.csv", index=False,
                                   encoding="utf-8-sig")
    log.info("ROC-AUC=%.3f  AP=%.3f  Top10%%捕获=%.1f%%", rc["auc"], rc["ap"],
             100 * metrics.get("capture_top10pct", np.nan))

    # --- 图件 ---------------------------------------------------------
    _plot_tvalue(scan, t_max, thr_max, thr_sig)
    _plot_classes(grid.reshape(height, width), pos_grid, t_max, thr_max)
    _plot_roc_capture(rc, metrics)

    report = {
        "seed": seed,
        "positive_raster": str(pos_path),
        "t_max": t_max, "threshold_t_max": thr_max,
        "threshold_t_critical": thr_sig, "t_critical": C.T_CRITICAL,
        "metrics": metrics,
        "elapsed_sec": round(time.time() - t0, 2),
        "outputs": {
            "threshold_scan": str(C.TABLE_DIR / "step05_threshold_scan.csv"),
            "metrics": str(C.TABLE_DIR / "step05_metrics.csv"),
            "classes": str(C.TABLE_DIR / "step05_anomaly_classes.csv"),
            "class_raster": str(C.OUTPUT_DIR / "anomaly_classes.tif"),
            "figures": [str(C.FIG_DIR / f) for f in
                        ("step05_tvalue.png", "step05_anomaly_classes.png",
                         "step05_roc_capture.png")],
        },
    }
    save_json(C.TABLE_DIR / f"step05_report_seed{seed}.json", report)
    log.info("STEP 5 完成，用时 %.1f s", time.time() - t0)
    return report


def _ref_raster_path():
    from src.common import list_geochem_rasters, locate_geochem_dir
    geo = locate_geochem_dir(Path(C.Data_Path), C.GEOCHEM_SUBDIR,
                             C.GEOCHEM_GLOB, C.AUTO_LOCATE_GEOCHEM)
    return list_geochem_rasters(geo, C.GEOCHEM_GLOB)[0]


# ---------------------------------------------------------------------
def _plot_tvalue(scan, t_max, thr_max, thr_sig):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 4.6), dpi=150)
    ax.plot(scan["threshold"], scan["t_value"], color="#1f77b4", lw=1.8)
    ax.axhline(C.T_CRITICAL, color="#d62728", ls="--", lw=1.2,
               label=f"t = {C.T_CRITICAL}")
    ax.axvline(thr_max, color="#2ca02c", ls=":", lw=1.2,
               label=f"t-max threshold = {thr_max:.3f}")
    if np.isfinite(thr_sig):
        ax.axvline(thr_sig, color="#ff7f0e", ls=":", lw=1.2,
                   label=f"t=1.96 threshold = {thr_sig:.3f}")
    ax.scatter([thr_max], [t_max], color="#2ca02c", zorder=5)
    ax.set_xlabel("Anomaly score threshold")
    ax.set_ylabel("Student's t-value")
    ax.set_title("Spatial association between anomaly map and known deposits")
    ax.grid(alpha=0.3, ls=":")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "step05_tvalue.png")
    plt.close(fig)


def _plot_classes(grid, pos_grid, t_max, thr_max):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import matplotlib.patches as mpatches

    cmap = ListedColormap(["#2c7bb6", "#fdae61", "#d7191c"])
    fig, ax = plt.subplots(figsize=(9.2, 4.8), dpi=150)
    ax.imshow(np.where(np.isfinite(grid), grid, np.nan), cmap=cmap, vmin=1, vmax=3,
              interpolation="nearest")
    yy, xx = np.where(pos_grid == 1)
    ax.scatter(xx, yy, marker="*", s=42, c="k", edgecolors="w", linewidths=0.4,
               label="Known deposits", zorder=6)
    ax.set_title("Classified geochemical anomalies (low / moderate / high)")
    ax.set_xlabel("Column"); ax.set_ylabel("Row")
    handles = [mpatches.Patch(color="#2c7bb6", label="Low"),
               mpatches.Patch(color="#fdae61", label="Moderate"),
               mpatches.Patch(color="#d7191c", label="High")]
    handles.append(mpatches.Patch(facecolor="none", edgecolor="none",
                                  label=f"t-max={t_max:.2f} @ {thr_max:.3f}"))
    ax.legend(handles=handles, fontsize=8, loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "step05_anomaly_classes.png")
    plt.close(fig)


def _plot_roc_capture(rc, metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6), dpi=150)
    axes[0].plot(rc["fpr"], rc["tpr"], color="#1f77b4", lw=1.8)
    axes[0].plot([0, 1], [0, 1], ls="--", color="grey", lw=1)
    axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate")
    axes[0].set_title(f"ROC curve (AUC={rc['auc']:.3f})")
    axes[0].grid(alpha=0.3, ls=":")

    axes[1].plot(100 * rc["area"], 100 * rc["capture"], color="#d62728", lw=1.8)
    axes[1].plot([0, 100], [0, 100], ls=":", color="grey", lw=1)
    for pct in C.CAPTURE_PERCENTS:
        key = f"capture_top{pct}pct"
        if key in metrics:
            axes[1].scatter([pct], [100 * metrics[key]], zorder=5, color="#2ca02c")
            axes[1].annotate(f"{100 * metrics[key]:.0f}%", (pct, 100 * metrics[key]),
                             textcoords="offset points", xytext=(4, 4), fontsize=8)
    axes[1].set_xlim(0, 100); axes[1].set_ylim(0, 100)
    axes[1].set_xlabel("Study area, cumulative (%)")
    axes[1].set_ylabel("Deposits captured (%)")
    axes[1].set_title("Prediction-area success curve")
    axes[1].grid(alpha=0.3, ls=":")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "step05_roc_capture.png")
    plt.close(fig)


if __name__ == "__main__":
    run()