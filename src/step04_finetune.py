# -*- coding: utf-8 -*-
"""
步骤 4 / Step 4 —— 展开并微调深度自编码器
====================================================================
论文 2.2 节：
  1. 把预训练好的各层 CRBM 权重展开(unroll)成对称深度自编码器；
  2. 用带动量的反向传播 (Eq.8-11) 微调全部参数，最小化输入-输出重构误差。

产出：
  models/deep_autoencoder_seed{seed}.pt    微调后的模型权重
  models/finetune_curves_seed{seed}.csv    微调学习曲线
  figures/step04_finetune_curve.png
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

import config as C
from src.autoencoder import DeepAutoencoder
from src.common import limit_threads, log, resolve_device, resolve_hidden_sizes, save_json, set_seed
from src.crbm import CRBM, CRBMConfig
from src.step03_pretrain_crbm import load_input_matrix


def load_pretrained_crbms(path, n_layers: int):
    ds = np.load(path, allow_pickle=True)
    out = []
    for li in range(1, n_layers + 1):
        cfg = CRBMConfig(n_visible=ds[f"layer{li}_W"].shape[0],
                         n_hidden=ds[f"layer{li}_W"].shape[1])
        m = CRBM(cfg)
        m.W = ds[f"layer{li}_W"].astype(np.float64)
        m.bv = ds[f"layer{li}_bv"].astype(np.float64)
        m.bh = ds[f"layer{li}_bh"].astype(np.float64)
        m.a_v = ds[f"layer{li}_a_v"].astype(np.float64)
        m.a_h = ds[f"layer{li}_a_h"].astype(np.float64)
        out.append(m)
    return out


def run(seed: int | None = None) -> dict:
    import torch
    import torch.nn as nn

    t0 = time.time()
    seed = int(seed if seed is not None else C.RANDOM_SEEDS[0])
    set_seed(seed)
    limit_threads(C.NUM_THREADS)
    device = resolve_device(C.DEVICE)

    log.info("=" * 68)
    log.info("STEP 4 | 深度自编码器微调 (seed=%d, device=%s)", seed, device)
    log.info("=" * 68)

    X, names, _ = load_input_matrix()
    X = X.astype(np.float32)
    input_dim = X.shape[1]
    hidden_sizes = resolve_hidden_sizes(input_dim, C.HIDDEN_SIZES)

    weights_path = C.MODEL_DIR / f"pretrained_crbms_seed{seed}.npz"
    crbms = load_pretrained_crbms(weights_path, len(hidden_sizes))

    dims = [input_dim] + list(hidden_sizes)
    model = DeepAutoencoder(dims, C.ACTIVATION, C.OUTPUT_ACTIVATION)
    model.load_pretrained_crbms(crbms)
    model.to(device)
    log.info("网络结构: %s", " - ".join(map(str, dims + dims[-2::-1])))

    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xt),
        batch_size=C.FINETUNE_BATCH_SIZE, shuffle=True, drop_last=False)

    if C.FINETUNE_OPTIMIZER == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=C.FINETUNE_LR,
                               weight_decay=C.FINETUNE_WEIGHT_DECAY)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=C.FINETUNE_LR,
                              momentum=C.FINETUNE_MOMENTUM,
                              weight_decay=C.FINETUNE_WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    hist = []
    log.info("开始微调: epochs=%d lr=%.2f momentum=%.2f weight_decay=%g batch=%d",
             C.FINETUNE_EPOCHS, C.FINETUNE_LR, C.FINETUNE_MOMENTUM,
             C.FINETUNE_WEIGHT_DECAY, C.FINETUNE_BATCH_SIZE)
    for ep in range(1, C.FINETUNE_EPOCHS + 1):
        model.train()
        tot, nb = 0.0, 0
        for (batch,) in loader:
            opt.zero_grad()
            out = model(batch)
            loss = loss_fn(out, batch)
            loss.backward()
            opt.step()
            tot += loss.item() * batch.shape[0]
            nb += batch.shape[0]
        ep_loss = tot / nb
        hist.append(ep_loss)
        if ep % 10 == 0 or ep == 1:
            log.info("    epoch %4d/%d  train-MSE=%.6f", ep, C.FINETUNE_EPOCHS, ep_loss)

    torch.save({"state_dict": model.state_dict(), "dims": dims,
                "activation": C.ACTIVATION, "out_activation": C.OUTPUT_ACTIVATION,
                "seed": seed, "input_features": names},
               C.MODEL_DIR / f"deep_autoencoder_seed{seed}.pt")
    pd.DataFrame({"epoch": range(1, len(hist) + 1), "train_mse": hist}).to_csv(
        C.MODEL_DIR / f"finetune_curves_seed{seed}.csv", index=False,
        encoding="utf-8-sig")

    _plot(hist, seed)

    report = {
        "seed": seed, "device": str(device), "dims": dims,
        "optimizer": C.FINETUNE_OPTIMIZER, "lr": C.FINETUNE_LR,
        "momentum": C.FINETUNE_MOMENTUM, "weight_decay": C.FINETUNE_WEIGHT_DECAY,
        "epochs": C.FINETUNE_EPOCHS, "final_train_mse": float(hist[-1]),
        "elapsed_sec": round(time.time() - t0, 2),
        "outputs": {
            "model": str(C.MODEL_DIR / f"deep_autoencoder_seed{seed}.pt"),
            "curves": str(C.MODEL_DIR / f"finetune_curves_seed{seed}.csv"),
            "figure": str(C.FIG_DIR / "step04_finetune_curve.png"),
        },
    }
    save_json(C.TABLE_DIR / f"step04_report_seed{seed}.json", report)
    log.info("STEP 4 完成，用时 %.1f s", time.time() - t0)
    return report


def _plot(hist, seed: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=C.FIGURE_DPI)
    ax.plot(range(1, len(hist) + 1), hist, color="#c0392b", lw=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reconstruction MSE")
    ax.set_title(f"Deep autoencoder fine-tuning (seed={seed})")
    ax.grid(alpha=0.3, ls=":")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "step04_finetune_curve.png")
    plt.close(fig)


if __name__ == "__main__":
    run()