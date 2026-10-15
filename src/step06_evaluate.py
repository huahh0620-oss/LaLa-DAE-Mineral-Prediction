# -*- coding: utf-8 -*-
"""
步骤 6 / Step 6 —— 空间相关性评价与异常分级
====================================================================
论文 4.2 节：
  * 用 Student's t 值 (Agterberg et al., 1990; Bonham-Carter, 1994) 度量
    异常图与已知矿点的空间相关性；t > 1.96 表示显著。
  * 以 t 值极大值、t=1.96 两个阈值把异常分为 高 / 中 / 低 三级。
  * 另给出 ROC-AUC、Top-X% 面积矿点捕获率、成功度曲线等定量指标。

本步骤同时输出"每个已知矿点落在哪一级异常"的明细表，便于检查
五角星与异常区的对应关系。

产出：
  tables/step06_threshold_scan.csv     阈值扫描的 t 值/捕获率
  tables/step06_metrics.csv            综合定量指标
  tables/step06_anomaly_classes.csv    三级异常的面积与矿点占比
  tables/step06_deposit_pixels.csv     每个矿点像元的得分与所属级别
  anomaly_classes.tif                  高/中/低分级栅格 (3/2/1)
  figures/step06_tvalue.png, step06_anomaly_classes.png, step06_roc_capture.png
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

import config as C
from src.common import (limit_threads, locate_eval_raster, log,
                        read_deposit_aligned, read_deposit_grid, save_json,
                        write_raster_like)


def student_t_value(score, deposit, threshold):
    """Bonham-Carter (1994) 二值证据图 vs 矿点的 studentized t 值。"""
    anom = score >= threshold
    n_a = int(anom.sum()); n_total = score.size; n_d = int(deposit.sum())
    n_da = int((deposit & anom).sum())
    if n_a == 0 or n_a == n_total or n_d == 0:
        return np.nan, n_a, n_d, n_da
    p = n_a / n_total
    denom = np.sqrt(n_d * p * (1.0 - p))
    if denom == 0:
        return np.nan, n_a, n_d, n_da
    return float((n_da - n_d * p) / denom), n_a, n_d, n_da


def scan_thresholds(score, deposit, n):
    thrs = np.unique(np.quantile(score, np.linspace(0.01, 0.99, n)))
    rows = []
    for thr in thrs:
        t, n_a, n_d, n_da = student_t_value(score, deposit, thr)
        rows.append({"threshold": float(thr), "area_fraction": n_a / score.size,
                     "n_deposit_total": n_d, "n_deposit_in_anomaly": n_da,
                     "capture_rate": (n_da / n_d) if n_d else np.nan,
                     "enrichment": (n_da / n_d) / (n_a / score.size)
                     if n_a > 0 and n_d else np.nan, "t_value": t})
    return pd.DataFrame(rows)


def _roc_capture(score, deposit):
    from sklearn.metrics import average_precision_score, roc_auc_score
    order = np.argsort(-score)
    y = deposit[order].astype(np.int64)
    n_p = int(y.sum()); n_n = int(y.size - n_p)
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    return dict(tpr=tp / max(n_p, 1), fpr=fp / max(n_n, 1),
                area=np.arange(1, y.size + 1) / y.size,
                capture=tp / max(n_p, 1),
                auc=float(roc_auc_score(deposit, score)),
                ap=float(average_precision_score(deposit, score)),
                n_pos=n_p, n_neg=n_n)


def run(seed: int | None = None) -> dict:
    t0 = time.time()
    seed = int(seed if seed is not None else C.RANDOM_SEEDS[0])
    limit_threads(C.NUM_THREADS)

    log.info("=" * 68)
    log.info("STEP 6 | 空间相关性评价与异常分级 (seed=%d)", seed)
    log.info("=" * 68)

    d = np.load(C.CACHE_DIR / "anomaly_scores.npz", allow_pickle=True)
    score = d["score_norm"].astype(np.float64)
    valid_mask = d["valid_mask"].astype(bool)
    height, width = int(d["height"]), int(d["width"])

    pos_path = locate_eval_raster(C.Data_Path, C.EVAL_RASTER, C.POSITIVE_RASTER,
                                  C.POSITIVE_SEARCH_PARENTS)
    deposit = read_deposit_aligned(pos_path, valid_mask, (height, width))
    deposit_grid = read_deposit_grid(pos_path, (height, width))
    log.info("已知矿点像元: %d (来源 %s)", int(deposit.sum()), pos_path.name)

    # --- 阈值扫描 + t 值 ---------------------------------------------
    scan = scan_thresholds(score, deposit, C.N_THRESHOLD_CANDIDATES)
    scan.to_csv(C.TABLE_DIR / "step06_threshold_scan.csv", index=False, encoding="utf-8-sig")
    i_max = int(np.nanargmax(scan["t_value"].values))
    thr_max = float(scan["threshold"].iloc[i_max]); t_max = float(scan["t_value"].iloc[i_max])
    sig = scan[scan["t_value"] >= C.T_CRITICAL]
    thr_sig = float(sig["threshold"].min()) if len(sig) else np.nan

    # 中异常下界：富集比 >= MODERATE_ENRICHMENT_MIN 的最高阈值
    enr = scan[(scan["enrichment"] >= C.MODERATE_ENRICHMENT_MIN) &
               (scan["threshold"] <= thr_max)]
    thr_mod = float(enr["threshold"].min()) if len(enr) else (
        thr_sig if np.isfinite(thr_sig) else thr_max)
    thr_mod = min(thr_mod, thr_max)

    log.info("t 值极大值 = %.3f (高异常阈值 thr_max = %.4f)", t_max, thr_max)
    log.info("参考：t >= %.2f 的最低阈值 = %.4f（论文口径，本区偏宽松）",
             C.T_CRITICAL, thr_sig)
    log.info("中异常阈值 thr_mod = %.4f（富集比 >= %.1f）", thr_mod, C.MODERATE_ENRICHMENT_MIN)

    # --- 三级分级 -----------------------------------------------------
    cls = np.where(score >= thr_max, 3.0, np.where(score >= thr_mod, 2.0, 1.0))

    rows = []
    for code, name in [(3.0, "high"), (2.0, "moderate"), (1.0, "low")]:
        m = cls == code
        n_a = int(m.sum()); n_da = int((deposit & m).sum())
        rows.append({"class": name, "code": int(code), "n_pixels": n_a,
                     "area_fraction": n_a / score.size, "n_deposit": n_da,
                     "deposit_fraction_of_total": n_da / max(int(deposit.sum()), 1),
                     "deposit_density": n_da / max(n_a, 1)})
    cls_tbl = pd.DataFrame(rows)
    cls_tbl.to_csv(C.TABLE_DIR / "step06_anomaly_classes.csv", index=False, encoding="utf-8-sig")
    log.info("高异常占面积 %.2f%%，含 %.1f%% 已知矿点",
             100 * cls_tbl.loc[0, "area_fraction"],
             100 * cls_tbl.loc[0, "deposit_fraction_of_total"])

    # --- 分级栅格 -----------------------------------------------------
    with rasterio.open(_ref_raster_path()) as ref:
        profile = ref.profile
    grid = np.full(height * width, np.nan)
    grid[valid_mask.ravel()] = cls
    grid2d = grid.reshape(height, width)
    write_raster_like(C.OUTPUT_DIR / "anomaly_classes.tif", grid2d,
                      {"profile": profile}, nodata=-9999.0)

    # --- 矿点逐一明细 -------------------------------------------------
    dy, dx = np.where(deposit_grid)
    dep_tbl = pd.DataFrame({
        "deposit_id": np.arange(1, len(dy) + 1),
        "row": dy, "col": dx,
        "anomaly_score": score[deposit],
        "percentile": [float((score < v).mean()) for v in score[deposit]],
        "class_code": cls[deposit].astype(int),
    })
    dep_tbl["class"] = dep_tbl["class_code"].map({3: "high", 2: "moderate", 1: "low"})
    dep_tbl.to_csv(C.TABLE_DIR / "step06_deposit_pixels.csv", index=False, encoding="utf-8-sig")

    # --- 综合指标 -----------------------------------------------------
    rc = _roc_capture(score, deposit)
    metrics = {
        "seed": seed, "positive_raster": pos_path.name,
        "n_features_used": int(np.asarray(d["feature_names"]).shape[0]),
        "n_pixels": int(score.size), "n_deposit": int(deposit.sum()),
        "roc_auc": rc["auc"], "average_precision": rc["ap"],
        "t_max": t_max, "threshold_t_max": thr_max,
        "threshold_t_critical": thr_sig,
        "high_class_area_fraction": float(cls_tbl.loc[0, "area_fraction"]),
        "high_class_deposit_fraction": float(cls_tbl.loc[0, "deposit_fraction_of_total"]),
        "moderate_or_high_deposit_fraction": float(
            cls_tbl.loc[0, "n_deposit"] + cls_tbl.loc[1, "n_deposit"]) / max(int(deposit.sum()), 1),
        "median_deposit_percentile": float(np.median(
            [(score < v).mean() for v in score[deposit]])),
        "deposit_mean_score_ratio": float(score[deposit].mean() / score[~deposit].mean()),
    }
    for pct in C.CAPTURE_PERCENTS:
        k = max(1, int(round(score.size * pct / 100.0)))
        metrics[f"capture_top{pct}pct"] = float(
            deposit[np.argsort(-score)[:k]].sum() / max(int(deposit.sum()), 1))
    pd.DataFrame([metrics]).to_csv(C.TABLE_DIR / "step06_metrics.csv", index=False, encoding="utf-8-sig")
    log.info("ROC-AUC=%.3f  AP=%.3f  Top10%%捕获=%.1f%%  矿点中位分位=%.2f",
             rc["auc"], rc["ap"], 100 * metrics["capture_top10pct"],
             metrics["median_deposit_percentile"])

    _plot_tvalue(scan, t_max, thr_max, thr_sig)
    _plot_classes(grid2d, deposit_grid, cls_tbl, t_max, thr_max, thr_sig)
    _plot_roc_capture(rc, metrics)

    report = {
        "seed": seed, "positive_raster": str(pos_path),
        "t_max": t_max, "threshold_t_max": thr_max,
        "threshold_t_critical": thr_sig,
        "threshold_moderate": thr_mod,
        "t_critical": C.T_CRITICAL,
        "moderate_enrichment_min": C.MODERATE_ENRICHMENT_MIN,
        "metrics": metrics,
        "elapsed_sec": round(time.time() - t0, 2),
        "outputs": {
            "threshold_scan": str(C.TABLE_DIR / "step06_threshold_scan.csv"),
            "metrics": str(C.TABLE_DIR / "step06_metrics.csv"),
            "classes": str(C.TABLE_DIR / "step06_anomaly_classes.csv"),
            "deposit_pixels": str(C.TABLE_DIR / "step06_deposit_pixels.csv"),
            "class_raster": str(C.OUTPUT_DIR / "anomaly_classes.tif"),
            "figures": [str(C.FIG_DIR / f) for f in
                        ("step06_tvalue.png", "step06_anomaly_classes.png",
                         "step06_roc_capture.png")],
        },
    }
    save_json(C.TABLE_DIR / f"step06_report_seed{seed}.json", report)
    log.info("STEP 6 完成，用时 %.1f s", time.time() - t0)
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

    fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=C.FIGURE_DPI)
    ax.plot(scan["threshold"], scan["t_value"], color="#1f77b4", lw=1.8)
    ax.axhline(C.T_CRITICAL, color="#d62728", ls="--", lw=1.2, label=f"t = {C.T_CRITICAL}")
    ax.axvline(thr_max, color="#2ca02c", ls=":", lw=1.4, label=f"t-max = {thr_max:.3f}")
    if np.isfinite(thr_sig):
        ax.axvline(thr_sig, color="#ff7f0e", ls=":", lw=1.4, label=f"t=1.96 = {thr_sig:.3f}")
    ax.scatter([thr_max], [t_max], color="#2ca02c", zorder=5, s=42)
    ax.set_xlabel("Anomaly score threshold"); ax.set_ylabel("Student's t-value")
    ax.set_title("Spatial association between anomalies and known deposits")
    ax.grid(alpha=0.3, ls=":"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "step06_tvalue.png"); plt.close(fig)


def _plot_classes(grid2d, pos_grid, cls_tbl, t_max, thr_max, thr_sig):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(["#cfe3f5", "#fdae61", "#d7191c"])
    yy, xx = np.where(pos_grid == 1)

    fig, axes = plt.subplots(1, 2, figsize=(16.0, 5.4), dpi=C.FIGURE_DPI,
                             gridspec_kw={"width_ratios": [1.55, 1]})

    g = np.where(np.isfinite(grid2d), grid2d, np.nan)
    axes[0].imshow(g, cmap=cmap, vmin=1, vmax=3, interpolation="nearest")
    axes[0].scatter(xx, yy, marker="*", s=C.STAR_SIZE * 1.25, c="none",
                    edgecolors="white", linewidths=1.6, zorder=5)
    axes[0].scatter(xx, yy, marker="*", s=C.STAR_SIZE * 0.7, c="k",
                    edgecolors="none", zorder=6)
    axes[0].set_title("Classified geochemical anomalies with known deposits")
    axes[0].set_xlabel("Column"); axes[0].set_ylabel("Row")
    handles = [mpatches.Patch(color="#d7191c", label="High anomaly"),
               mpatches.Patch(color="#fdae61", label="Moderate anomaly"),
               mpatches.Patch(color="#cfe3f5", label="Low anomaly"),
               plt.Line2D([], [], marker="*", color="none", markerfacecolor="k",
                          markeredgecolor="white", markersize=13,
                          label=f"Known deposits (n={len(yy)})")]
    axes[0].legend(handles=handles, fontsize=8, loc="lower right", framealpha=0.9)

    # 右图：面积占比 vs 矿点占比
    labels = cls_tbl["class"].tolist()
    x = np.arange(len(labels)); w = 0.38
    axes[1].bar(x - w / 2, 100 * cls_tbl["area_fraction"], w, label="Area share",
                color="#9ecae1")
    axes[1].bar(x + w / 2, 100 * cls_tbl["deposit_fraction_of_total"], w,
                label="Deposit share", color="#d7191c")
    for i, (a, dd) in enumerate(zip(100 * cls_tbl["area_fraction"],
                                    100 * cls_tbl["deposit_fraction_of_total"])):
        axes[1].text(i - w / 2, a + 1, f"{a:.1f}%", ha="center", fontsize=8)
        axes[1].text(i + w / 2, dd + 1, f"{dd:.1f}%", ha="center", fontsize=8)
    axes[1].set_xticks(x); axes[1].set_xticklabels([s.capitalize() for s in labels])
    axes[1].set_ylabel("Percentage (%)")
    axes[1].set_title("Anomaly class vs known deposits")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3, axis="y", ls=":")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "step06_anomaly_classes.png"); plt.close(fig)


def _plot_roc_capture(rc, metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7), dpi=C.FIGURE_DPI)
    axes[0].plot(rc["fpr"], rc["tpr"], color="#1f77b4", lw=1.9)
    axes[0].plot([0, 1], [0, 1], ls="--", color="grey", lw=1)
    axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate")
    axes[0].set_title(f"ROC curve (AUC={rc['auc']:.3f})")
    axes[0].grid(alpha=0.3, ls=":")

    axes[1].plot(100 * rc["area"], 100 * rc["capture"], color="#d62728", lw=1.9)
    axes[1].plot([0, 100], [0, 100], ls=":", color="grey", lw=1)
    for pct in C.CAPTURE_PERCENTS:
        key = f"capture_top{pct}pct"
        if key in metrics:
            axes[1].scatter([pct], [100 * metrics[key]], zorder=5, color="#2ca02c", s=32)
            axes[1].annotate(f"{100 * metrics[key]:.0f}%", (pct, 100 * metrics[key]),
                             textcoords="offset points", xytext=(5, 3), fontsize=8)
    axes[1].set_xlim(0, 100); axes[1].set_ylim(0, 100)
    axes[1].set_xlabel("Study area, cumulative (%)")
    axes[1].set_ylabel("Deposits captured (%)")
    axes[1].set_title("Prediction-area success curve")
    axes[1].grid(alpha=0.3, ls=":")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "step06_roc_capture.png"); plt.close(fig)


if __name__ == "__main__":
    run()