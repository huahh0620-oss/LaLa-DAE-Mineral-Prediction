# -*- coding: utf-8 -*-
"""
步骤 2 / Step 2 —— 矿化指示元素选择（相对原论文的改进点 ①）
====================================================================
原论文把全部变量直接送入自编码器。但本区 39 个变量中，一部分元素
（Sr、Zr、Be、Th、Cd 等）与拉拉式铜金矿化呈负相关，它们会稀释多元异常信号、
并让重构误差更多地反映岩性差异而非矿化。

本步骤用"单元素判别力"（以已知矿点为正样本的 ROC-AUC）对变量排序：

    * AUC > 0.5  → 正指示（越高越靠近矿化）
    * AUC ≈ 0.5  → 无判别力
    * AUC < 0.5  → 反指示（与矿化负相关）

再按 AUC 从高到低挑选前 N_SELECTED_FEATURES 个，并用相关性去冗余
（与已选元素 |r| > SELECTION_MAX_CORR 的跳过），得到最终输入变量子集。

注：用已知矿点做"指示元素选择"是勘查地球化学的标准做法（pathfinder
element selection），但须在论文中说明；若要与全部变量的结果对比，
把 config.FEATURE_SELECTION_MODE 设为 "none" 即可。

产出：
  tables/feature_selection_auc.csv    所有变量的 AUC 与是否入选
  tables/step02_report.json           选择结果与报告
  figures/step02_feature_auc.png      变量判别力条形图
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

import config as C
from src.common import (limit_threads, locate_eval_raster, log,
                        read_deposit_aligned, resolve_device, save_json, set_seed)


def run() -> dict:
    t0 = time.time()
    limit_threads(C.NUM_THREADS)
    log.info("=" * 68)
    log.info("STEP 2 | 矿化指示元素选择 (mode=%s)", C.FEATURE_SELECTION_MODE)
    log.info("=" * 68)

    ds = np.load(C.CACHE_DIR / "dataset.npz", allow_pickle=True)
    X = ds["X_norm"].astype(np.float64)
    names = [str(n) for n in ds["feature_names"]]
    valid_mask = ds["valid_mask"].astype(bool)
    height, width = int(ds["height"]), int(ds["width"])

    pos_path = locate_eval_raster(C.Data_Path, C.EVAL_RASTER, C.POSITIVE_RASTER,
                                  C.POSITIVE_SEARCH_PARENTS)
    deposit = read_deposit_aligned(pos_path, valid_mask, (height, width))
    log.info("矿点栅格: %s，已知矿点像元: %d", pos_path.name, int(deposit.sum()))
    if deposit.sum() == 0:
        raise ValueError("矿点栅格中没有 1 值像元，无法做指示元素选择。")

    # --- 单元素判别力 -------------------------------------------------
    from sklearn.metrics import roc_auc_score
    auc = np.array([roc_auc_score(deposit, X[:, j]) for j in range(X.shape[1])])

    mode = C.FEATURE_SELECTION_MODE
    if mode == "none":
        selected = list(names)
        log.info("mode=none：使用全部 %d 个变量（等价原论文做法）", len(names))
    elif mode == "manual":
        selected = [n for n in C.MANUAL_FEATURES if n in names]
        missing = [n for n in C.MANUAL_FEATURES if n not in names]
        if missing:
            log.warning("MANUAL_FEATURES 中以下变量不存在于数据中，已忽略: %s", missing)
        log.info("mode=manual：选择 %d 个知识驱动指示元素", len(selected))
    else:
        order = np.argsort(-auc)
        selected = []
        for j in order:
            if auc[j] < C.SELECTION_MIN_AUC:
                break
            if len(selected) >= C.N_SELECTED_FEATURES:
                break
            redundant = False
            if C.SELECTION_MAX_CORR > 0 and selected:
                sel_idx = [names.index(s) for s in selected]
                r = np.abs(np.corrcoef(X[:, j], X[:, sel_idx].T)[0, 1:])
                redundant = bool(np.any(r > C.SELECTION_MAX_CORR))
            if not redundant:
                selected.append(names[j])
        log.info("mode=auc：按判别力选择 %d 个变量", len(selected))

    sel_idx = [names.index(s) for s in selected]
    sel_auc = [float(auc[i]) for i in sel_idx]

    # --- 表格与报告 ---------------------------------------------------
    tbl = pd.DataFrame({
        "feature": names,
        "auc_single_element": auc,
        "selected": [n in selected for n in names],
        "rank": pd.Series(-auc).rank(method="first").astype(int),
    }).sort_values("auc_single_element", ascending=False)
    tbl.to_csv(C.TABLE_DIR / "feature_selection_auc.csv", index=False,
               encoding="utf-8-sig")

    np.savez_compressed(C.CACHE_DIR / "selected_features.npz",
                        selected=selected, indices=sel_idx)

    _plot_auc(tbl, selected)

    report = {
        "mode": mode,
        "positive_raster": str(pos_path),
        "n_total_features": len(names),
        "n_selected": len(selected),
        "selected_features": selected,
        "selected_auc": sel_auc,
        "rejected_features": [n for n in names if n not in selected],
        "parameters": {
            "n_selected": C.N_SELECTED_FEATURES,
            "min_auc": C.SELECTION_MIN_AUC,
            "max_corr": C.SELECTION_MAX_CORR,
        },
        "elapsed_sec": round(time.time() - t0, 2),
        "outputs": {
            "table": str(C.TABLE_DIR / "feature_selection_auc.csv"),
            "figure": str(C.FIG_DIR / "step02_feature_auc.png"),
            "selection": str(C.CACHE_DIR / "selected_features.npz"),
        },
    }
    save_json(C.TABLE_DIR / "step02_report.json", report)
    log.info("入选变量 (%d): %s", len(selected),
             ", ".join(f"{n}({a:.3f})" for n, a in zip(selected, sel_auc)))
    log.info("STEP 2 完成，用时 %.1f s", time.time() - t0)
    return report


def load_selected_indices() -> list:
    """供后续步骤调用：读取入选变量下标。"""
    p = C.CACHE_DIR / "selected_features.npz"
    if not p.exists():
        raise FileNotFoundError("请先运行 step02_select_features.py 生成变量选择结果。")
    d = np.load(p, allow_pickle=True)
    return [int(i) for i in d["indices"]]


def _plot_auc(tbl: pd.DataFrame, selected) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = tbl.sort_values("auc_single_element")
    colors = ["#d7191c" if s else "#bdbdbd" for s in t["selected"]]
    fig, ax = plt.subplots(figsize=(7.6, 9.0), dpi=C.FIGURE_DPI)
    ax.barh(range(len(t)), t["auc_single_element"], color=colors, height=0.72)
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels(t["feature"], fontsize=7.5)
    ax.axvline(0.5, ls="--", color="k", lw=1)
    ax.axvline(C.SELECTION_MIN_AUC, ls=":", color="#2ca02c", lw=1)
    ax.set_xlabel("Single-element ROC-AUC vs known deposits")
    ax.set_title(f"Pathfinder element ranking (selected n={len(selected)})")
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.3, axis="x", ls=":")
    for i, (v, s) in enumerate(zip(t["auc_single_element"], t["selected"])):
        if s:
            ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=7, color="#d7191c")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "step02_feature_auc.png")
    plt.close(fig)


if __name__ == "__main__":
    run()