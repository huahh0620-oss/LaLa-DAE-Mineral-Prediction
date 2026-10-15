# -*- coding: utf-8 -*-
"""
步骤 6 / Step 6 —— 基准测试
====================================================================
把本方法（CRBM 预训练 + 微调的深度自编码器）与多种常用方法在同一套
39 维地化数据、同一评价协议（ROC-AUC、t 值、Top-X% 捕获率）下对比：

    1. deep_autoencoder   本方法（论文方法）
    2. shallow_ae         浅层自编码器（单隐层）
    3. pca                PCA 重构误差
    4. isolation_forest   孤立森林
    5. ocsvm              单类支持向量机
    6. single_element_cu  单元素 Cu 阈值（传统单变量方法）
    7. mahalanobis        马氏距离

产出：
  tables/step06_benchmark.csv       各方法定量指标
  figures/step06_benchmark.png      ROC 对比 + AUC 柱状图
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

import config as C
from src.common import (limit_threads, locate_raster, log, save_json, set_seed)
from src.step05_evaluate import _roc_capture, scan_thresholds, student_t_value


# ---------------------------------------------------------------------
# 各基准方法的异常得分（分值越大越异常）
# ---------------------------------------------------------------------
def score_pca(X, n_components=5, seed=0):
    from sklearn.decomposition import PCA
    p = PCA(n_components=n_components, random_state=seed).fit(X)
    R = p.inverse_transform(p.transform(X))
    return ((X - R) ** 2).mean(axis=1)


def score_mahalanobis(X):
    mu = X.mean(axis=0)
    cov = np.cov(X, rowvar=False) + np.eye(X.shape[1]) * 1e-6
    inv = np.linalg.pinv(cov)
    d = X - mu
    return np.einsum("ij,jk,ik->i", d, inv, d)


def score_isolation_forest(X, seed=0):
    from sklearn.ensemble import IsolationForest
    m = IsolationForest(n_estimators=300, random_state=seed, n_jobs=-1)
    m.fit(X)
    return -m.score_samples(X)          # 越负越异常 -> 取负


def score_ocsvm(X, seed=0, max_fit=5000):
    from sklearn.svm import OneClassSVM
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=min(max_fit, X.shape[0]), replace=False)
    m = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
    m.fit(X[idx])
    return -m.decision_function(X)      # decision 越小越异常 -> 取负


def score_shallow_ae(X, bottleneck=5, epochs=200, lr=0.8, seed=0, device=None):
    import torch
    import torch.nn as nn
    from src.common import resolve_device
    device = device or resolve_device(C.DEVICE)
    set_seed(seed)
    d = X.shape[1]
    net = nn.Sequential(nn.Linear(d, bottleneck), nn.Sigmoid(),
                        nn.Linear(bottleneck, d), nn.Sigmoid()).to(device)
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=C.FINETUNE_MOMENTUM,
                          weight_decay=C.FINETUNE_WEIGHT_DECAY)
    lossf = nn.MSELoss()
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xt), batch_size=C.FINETUNE_BATCH_SIZE, shuffle=True)
    net.train()
    for _ in range(epochs):
        for (b,) in loader:
            opt.zero_grad(); l = lossf(net(b), b); l.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        R = net(Xt).cpu().numpy()
    return ((X - R) ** 2).mean(axis=1)


def score_deep_autoencoder(X, seed):
    """优先复用 step02-04 的结果；否则现场训练。"""
    ap = C.CACHE_DIR / "anomaly_scores.npz"
    if int(seed) == int(C.RANDOM_SEEDS[0]) and ap.exists():
        d = np.load(ap, allow_pickle=True)
        return d["score_raw"].astype(np.float64)
    from src import step02_pretrain_crbm, step03_finetune
    from src.step04_anomaly_map import reconstruction_error, _load_model
    from src.common import resolve_device
    step02_pretrain_crbm.run(seed)
    step03_finetune.run(seed)
    device = resolve_device(C.DEVICE)
    model = _load_model(seed, device)
    return reconstruction_error(model, X.astype(np.float32), device)


def score_single_element(Xn, feature_names, element="Cu"):
    names = list(feature_names)
    j = names.index(element) if element in names else 0
    return Xn[:, j].astype(np.float64)


# ---------------------------------------------------------------------
METHOD_FUNCS = {
    "pca": lambda X, Xn, names, seed: score_pca(Xn, 5, seed),
    "mahalanobis": lambda X, Xn, names, seed: score_mahalanobis(Xn),
    "isolation_forest": lambda X, Xn, names, seed: score_isolation_forest(Xn, seed),
    "ocsvm": lambda X, Xn, names, seed: score_ocsvm(Xn, seed),
    "shallow_ae": lambda X, Xn, names, seed: score_shallow_ae(Xn, 5, C.FINETUNE_EPOCHS,
                                                              C.FINETUNE_LR, seed),
    "single_element_cu": lambda X, Xn, names, seed: score_single_element(Xn, names, "Cu"),
    "deep_autoencoder": lambda X, Xn, names, seed: score_deep_autoencoder(Xn, seed),
}


def evaluate_score(score, deposit):
    order = np.argsort(-score)
    n = score.size
    metrics = {}
    t, n_a, n_d, n_da = student_t_value(_minmax(score), deposit, np.quantile(_minmax(score), 0.9))
    scan = scan_thresholds(_minmax(score), deposit, 80)
    metrics["t_max"] = float(np.nanmax(scan["t_value"].values))
    metrics["threshold_t_max"] = float(scan["threshold"].iloc[int(np.nanargmax(scan["t_value"].values))])
    metrics["roc_auc"] = _roc_capture(score, deposit)["auc"]
    metrics["average_precision"] = _roc_capture(score, deposit)["ap"]
    for pct in C.CAPTURE_PERCENTS:
        k = max(1, int(round(n * pct / 100.0)))
        metrics[f"capture_top{pct}pct"] = float(deposit[order[:k]].sum() / max(int(deposit.sum()), 1))
    metrics["deposit_mean_score_ratio"] = float(
        (score[deposit].mean() / score[~deposit].mean()) if (~deposit).any() else np.nan)
    return metrics


def _minmax(s):
    lo, hi = np.nanmin(s), np.nanmax(s)
    return (s - lo) / (hi - lo) if hi > lo else np.zeros_like(s)


# ---------------------------------------------------------------------
def run() -> dict:
    t0 = time.time()
    limit_threads(C.NUM_THREADS)
    log.info("=" * 68)
    log.info("STEP 6 | 基准测试 (Benchmark)")
    log.info("=" * 68)

    ds = np.load(C.CACHE_DIR / "dataset.npz", allow_pickle=True)
    Xn = ds["X_norm"].astype(np.float64)
    X = ds["X_raw"].astype(np.float64)
    names = list(ds["feature_names"])
    valid_mask = ds["valid_mask"].astype(bool)
    height, width = int(ds["height"]), int(ds["width"])

    pos_path = locate_raster(Path(C.Data_Path),
                             C.POSITIVE_RASTER, C.POSITIVE_SEARCH_PARENTS)
    with rasterio.open(pos_path) as s:
        pg = s.read(1)
        if s.nodata is not None:
            pg = np.where(pg == s.nodata, 0, pg)
    deposit = pg.ravel()[valid_mask.ravel()] == 1
    log.info("正样本(矿点)像元: %d；负/背景像元: %d", int(deposit.sum()), int((~deposit).sum()))

    rows = []
    roc_store = {}
    for method in C.BENCHMARK_METHODS:
        for seed in C.BENCHMARK_SEEDS:
            log.info("-" * 60)
            log.info("方法 %s (seed=%d) ...", method, seed)
            try:
                sc = METHOD_FUNCS[method](X, Xn, names, seed)
            except Exception as e:  # 保证流程不中断
                log.warning("方法 %s 失败: %s", method, e)
                continue
            m = evaluate_score(np.asarray(sc, dtype=np.float64), deposit)
            m.update({"method": method, "seed": int(seed)})
            rows.append(m)
            if method == "deep_autoencoder":
                roc_store[method] = _roc_capture(np.asarray(sc, dtype=np.float64), deposit)
            else:
                roc_store[method] = _roc_capture(np.asarray(sc, dtype=np.float64), deposit)
            log.info("    AUC=%.3f  t_max=%.2f  capture@10%%=%.1f%%",
                     m["roc_auc"], m["t_max"], 100 * m["capture_top10pct"])

    df = pd.DataFrame(rows)
    cols = ["method", "seed", "roc_auc", "average_precision", "t_max",
            "threshold_t_max", "capture_top1pct", "capture_top5pct",
            "capture_top10pct", "capture_top20pct", "deposit_mean_score_ratio"]
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(C.TABLE_DIR / "step06_benchmark.csv", index=False, encoding="utf-8-sig")
    log.info("基准结果已写入 %s", C.TABLE_DIR / "step06_benchmark.csv")

    _plot(df, roc_store, deposit)

    report = {
        "methods": C.BENCHMARK_METHODS, "seeds": C.BENCHMARK_SEEDS,
        "results": df.to_dict(orient="records"),
        "elapsed_sec": round(time.time() - t0, 2),
        "outputs": {
            "benchmark_csv": str(C.TABLE_DIR / "step06_benchmark.csv"),
            "figure": str(C.FIG_DIR / "step06_benchmark.png"),
        },
    }
    save_json(C.TABLE_DIR / "step06_report.json", report)
    log.info("STEP 6 完成，用时 %.1f s", time.time() - t0)
    return report


def _plot(df, roc_store, deposit):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), dpi=150)
    for method, rc in roc_store.items():
        axes[0].plot(rc["fpr"], rc["tpr"], lw=1.6,
                     label=f"{method} (AUC={rc['auc']:.3f})")
    axes[0].plot([0, 1], [0, 1], ls="--", color="grey", lw=1)
    axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC comparison of anomaly-detection methods")
    axes[0].grid(alpha=0.3, ls=":"); axes[0].legend(fontsize=7.5)

    agg = df.groupby("method")["roc_auc"].mean().sort_values()
    colors = ["#d7191c" if m == "deep_autoencoder" else "#9ecae1" for m in agg.index]
    axes[1].barh(range(len(agg)), agg.values, color=colors)
    axes[1].set_yticks(range(len(agg))); axes[1].set_yticklabels(agg.index, fontsize=8.5)
    axes[1].set_xlabel("ROC-AUC (higher is better)")
    axes[1].set_xlim(0, 1)
    axes[1].set_title("Benchmark: ROC-AUC")
    for i, v in enumerate(agg.values):
        axes[1].text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8)
    axes[1].grid(alpha=0.3, axis="x", ls=":")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "step06_benchmark.png")
    plt.close(fig)


if __name__ == "__main__":
    run()