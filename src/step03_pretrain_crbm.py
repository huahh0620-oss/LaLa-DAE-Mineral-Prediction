# -*- coding: utf-8 -*-
"""
步骤 3 / Step 3 —— CRBM 逐层无监督预训练
====================================================================
论文 2.1 节：把若干个 CRBM 逐层堆叠，每层用 CD-1 在上一层的隐层输出上训练，
为深度自编码器提供初始权重（贪心逐层预训练）。

输入是 step02 选出的矿化指示元素子集（FEATURE_SELECTION_MODE="none"
时为全部变量）。网络结构由 config.HIDDEN_SIZES="auto" 按输入维度折半生成。

产出：
  models/pretrained_crbms_seed{seed}.npz   各层 CRBM 权重
  models/pretrain_curves_seed{seed}.csv    各层逐 epoch 重构误差
  figures/step03_pretrain_curves.png       预训练学习曲线
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

import config as C
from src.common import (limit_threads, log, resolve_hidden_sizes, save_json, set_seed)
from src.crbm import CRBM, CRBMConfig
from src.step02_select_features import load_selected_indices


def load_input_matrix():
    """读取归一化样本矩阵，并按 step02 的选择结果取子集。"""
    ds = np.load(C.CACHE_DIR / "dataset.npz", allow_pickle=True)
    X = ds["X_norm"].astype(np.float64)
    names = [str(n) for n in ds["feature_names"]]
    if C.FEATURE_SELECTION_MODE == "none":
        idx = list(range(len(names)))
    else:
        idx = load_selected_indices()
    return X[:, idx], [names[i] for i in idx], idx


def run(seed: int | None = None) -> dict:
    t0 = time.time()
    seed = int(seed if seed is not None else C.RANDOM_SEEDS[0])
    set_seed(seed)
    limit_threads(C.NUM_THREADS)

    log.info("=" * 68)
    log.info("STEP 3 | CRBM 逐层预训练 (seed=%d)", seed)
    log.info("=" * 68)

    X, names, idx = load_input_matrix()
    input_dim = X.shape[1]
    log.info("样本矩阵: %d 个像元 x %d 个变量", *X.shape)
    log.info("输入变量: %s", ", ".join(names))

    hidden_sizes = resolve_hidden_sizes(input_dim, C.HIDDEN_SIZES)
    log.info("网络结构: %d - %s - %d", input_dim,
             " - ".join(map(str, hidden_sizes)), input_dim)

    reps = X.copy()
    layers, curves = [], {}
    layer_dims = [input_dim] + hidden_sizes

    for li, (n_in, n_out) in enumerate(zip(layer_dims[:-1], layer_dims[1:]), start=1):
        log.info("-" * 60)
        log.info("预训练第 %d 层 CRBM: %d -> %d  (epochs=%d, lr=%.2f, sigma=%.2f)",
                 li, n_in, n_out, C.CRBM_EPOCHS, C.CRBM_LR, C.CRBM_SIGMA)
        cfg = CRBMConfig(
            n_visible=n_in, n_hidden=n_out,
            sigma=C.CRBM_SIGMA, lr=C.CRBM_LR, lr_noise=C.CRBM_LR_NOISE,
            momentum=C.CRBM_MOMENTUM, weight_decay=C.CRBM_WEIGHT_DECAY,
            theta_l=C.CRBM_THETA_L, theta_h=C.CRBM_THETA_H,
            weight_init_std=C.CRBM_WEIGHT_INIT_STD,
            learn_a=C.CRBM_LEARN_A, seed=seed + li)
        m = CRBM(cfg)
        hist = m.fit(reps, epochs=C.CRBM_EPOCHS, batch_size=C.CRBM_BATCH_SIZE,
                     patience=C.CRBM_PRETRAIN_PATIENCE, verbose=True)
        curves[f"layer{li}_{n_in}to{n_out}"] = hist
        layers.append(m)
        reps = m.transform(reps)
        log.info("第 %d 层完成，隐层表示维度 %s", li, reps.shape)

    payload = {"n_layers": len(layers)}
    for li, m in enumerate(layers, start=1):
        p = m.get_params()
        for k, v in p.items():
            payload[f"layer{li}_{k}"] = np.array(v) if not isinstance(v, (int, float)) else v
    payload["input_dim"] = input_dim
    payload["hidden_sizes"] = np.array(hidden_sizes)
    np.savez_compressed(C.MODEL_DIR / f"pretrained_crbms_seed{seed}.npz", **payload)

    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in curves.items()]))
    df.index.name = "epoch"
    df.to_csv(C.MODEL_DIR / f"pretrain_curves_seed{seed}.csv", encoding="utf-8-sig")

    _plot_curves(curves, seed)

    report = {
        "seed": seed,
        "input_dim": int(input_dim),
        "input_features": names,
        "hidden_sizes": hidden_sizes,
        "crbm": {"epochs": C.CRBM_EPOCHS, "lr": C.CRBM_LR,
                 "momentum": C.CRBM_MOMENTUM, "sigma": C.CRBM_SIGMA,
                 "learn_a": bool(C.CRBM_LEARN_A)},
        "final_recon_mse": {k: float(v[-1]) for k, v in curves.items()},
        "elapsed_sec": round(time.time() - t0, 2),
        "outputs": {
            "weights": str(C.MODEL_DIR / f"pretrained_crbms_seed{seed}.npz"),
            "curves_csv": str(C.MODEL_DIR / f"pretrain_curves_seed{seed}.csv"),
            "figure": str(C.FIG_DIR / "step03_pretrain_curves.png"),
        },
    }
    save_json(C.TABLE_DIR / f"step03_report_seed{seed}.json", report)
    log.info("STEP 3 完成，用时 %.1f s", time.time() - t0)
    return report


def _plot_curves(curves: dict, seed: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=C.FIGURE_DPI)
    for name, hist in curves.items():
        ax.plot(range(1, len(hist) + 1), hist, lw=1.6, label=name.replace("_", " "))
    ax.set_xlabel("Iteration (epoch)")
    ax.set_ylabel("Reconstruction MSE")
    ax.set_title(f"CRBM layer-wise pre-training (seed={seed})")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, ls=":")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "step03_pretrain_curves.png")
    plt.close(fig)
    log.info("已保存预训练曲线: %s", C.FIG_DIR / "step03_pretrain_curves.png")


if __name__ == "__main__":
    run()