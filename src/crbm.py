# -*- coding: utf-8 -*-
"""
连续受限玻尔兹曼机 (CRBM) —— 论文 2.1 节
====================================================================
Chen & Murray (2003) 连续型 RBM，作为深度自编码器逐层无监督预训练的基本单元。

可见层 v (连续, 归一化到 [0,1]) 与隐层 h 之间的能量模型:

    h_j = phi_j( sum_i w_ij v_i + b_j + sigma * N(0,1) )      (Eq.2)
    v_i = phi_i( sum_j w_ij h_j + b_i + sigma * N(0,1) )      (Eq.3)
    phi(x; a) = theta_L + (theta_H - theta_L) / (1 + exp(-a x))   (Eq.4)

学习采用最小对比散度 (CD-1) + 动量:
    dW  = <v h>_0 - <v h>_1          (Eq.5)
    dbv = <v>_0   - <v>_1
    dbh = <h>_0   - <h>_1
    噪声控制参数 a 按 Eq.(6) 用输入方差之差更新（越大 -> sigmoid 越陡）。

本实现做到: 批量 CD-1、动量、权重衰减、a 学习与范围保护、可行走复现。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class CRBMConfig:
    n_visible: int
    n_hidden: int
    sigma: float = 0.2            # 连续单元高斯噪声标准差
    lr: float = 0.3               # 权重/偏置学习率 (Eq.5,7)
    lr_noise: float = 0.01        # 噪声控制参数 a 的学习率
    momentum: float = 0.9         # 学习动量
    weight_decay: float = 1e-5    # 权重惩罚 ("learning cost")
    theta_l: float = 0.0
    theta_h: float = 1.0
    weight_init_std: float = 0.01
    a_init: float = 1.0
    a_min: float = 0.05
    a_max: float = 20.0
    learn_a: bool = False
    seed: int = 0


class CRBM:
    """连续 RBM，可见/隐层均使用带噪声控制的 sigmoid 单元。"""

    def __init__(self, cfg: CRBMConfig):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        s = cfg.weight_init_std
        self.W = rng.normal(0.0, s, size=(cfg.n_visible, cfg.n_hidden)).astype(np.float64)
        self.bv = rng.normal(0.0, s, size=(cfg.n_visible,)).astype(np.float64)
        self.bh = rng.normal(0.0, s, size=(cfg.n_hidden,)).astype(np.float64)
        self.a_v = np.full(cfg.n_visible, cfg.a_init, dtype=np.float64)
        self.a_h = np.full(cfg.n_hidden, cfg.a_init, dtype=np.float64)
        self._rng = rng
        # 动量缓存
        self._dW = np.zeros_like(self.W)
        self._dbv = np.zeros_like(self.bv)
        self._dbh = np.zeros_like(self.bh)
        self._dav = np.zeros_like(self.a_v)
        self._dah = np.zeros_like(self.a_h)
        self.history: List[float] = []

    # --------------------------------------------------------------
    # 激活函数 (Eq.4)
    # --------------------------------------------------------------
    def _phi(self, x: np.ndarray, a: np.ndarray) -> np.ndarray:
        z = np.clip(-a * x, -60.0, 60.0)
        g = 1.0 / (1.0 + np.exp(z))
        return self.cfg.theta_l + (self.cfg.theta_h - self.cfg.theta_l) * g

    def _phi_visible(self, x: np.ndarray) -> np.ndarray:
        return self._phi(x, self.a_v if self.cfg.learn_a else np.ones_like(self.a_v))

    def _phi_hidden(self, x: np.ndarray) -> np.ndarray:
        return self._phi(x, self.a_h if self.cfg.learn_a else np.ones_like(self.a_h))

    # --------------------------------------------------------------
    # 前向 / 重构 (Eq.2, Eq.3)
    # --------------------------------------------------------------
    def hidden_mean(self, v: np.ndarray, add_noise: bool = True) -> np.ndarray:
        pre = v @ self.W + self.bh
        if add_noise and self.cfg.sigma > 0:
            pre = pre + self.cfg.sigma * self._rng.normal(size=pre.shape)
        return self._phi_hidden(pre)

    def visible_mean(self, h: np.ndarray, add_noise: bool = True) -> np.ndarray:
        pre = h @ self.W.T + self.bv
        if add_noise and self.cfg.sigma > 0:
            pre = pre + self.cfg.sigma * self._rng.normal(size=pre.shape)
        return self._phi_visible(pre)

    def reconstruct(self, v: np.ndarray, n_gibbs: int = 1) -> np.ndarray:
        """确定性往返重构 (无噪声)，用于评估与特征提取。"""
        h = self.hidden_mean(v, add_noise=False)
        for _ in range(max(1, n_gibbs)):
            v = self.visible_mean(h, add_noise=False)
            h = self.hidden_mean(v, add_noise=False)
        return v

    def transform(self, v: np.ndarray) -> np.ndarray:
        """编码：返回隐层均值 (供逐层堆叠使用)。"""
        return self.hidden_mean(v, add_noise=False)

    # --------------------------------------------------------------
    # CD-1 训练 (Eq.5-7)
    # --------------------------------------------------------------
    def _cd_batch(self, v0: np.ndarray):
        h0 = self.hidden_mean(v0, add_noise=True)
        v1 = self.visible_mean(h0, add_noise=True)
        h1 = self.hidden_mean(v1, add_noise=True)
        return h0, v1, h1

    def _update_a(self, pre0: np.ndarray, pre1: np.ndarray, a: np.ndarray) -> np.ndarray:
        """Eq.(6) 的方差型更新：输入方差大 -> a 增大（sigmoid 更陡）。"""
        var0 = pre0.var(axis=0)
        var1 = pre1.var(axis=0)
        grad = (var0 - var1) / np.maximum(a, 1e-6) ** 2
        return grad

    def fit(self, X: np.ndarray, epochs: int = 200, batch_size: int = 256,
            verbose: bool = False, patience: int = 0) -> List[float]:
        n = X.shape[0]
        cfg = self.cfg
        best, best_err, bad = None, np.inf, 0
        for ep in range(1, epochs + 1):
            idx = self._rng.permutation(n)
            ep_err = 0.0
            nb = 0
            for start in range(0, n, batch_size):
                b = idx[start:start + batch_size]
                v0 = X[b]
                h0, v1, h1 = self._cd_batch(v0)
                # CD-1 梯度 (Eq.5-7)；动量把 lr 放在累积项内部，避免有效步长放大
                gW = (v0.T @ h0 - v1.T @ h1) / len(b)
                gbv = (v0 - v1).mean(axis=0)
                gbh = (h0 - h1).mean(axis=0)
                self._dW = cfg.momentum * self._dW + cfg.lr * (gW - cfg.weight_decay * self.W)
                self._dbv = cfg.momentum * self._dbv + cfg.lr * gbv
                self._dbh = cfg.momentum * self._dbh + cfg.lr * gbh
                self.W += self._dW
                self.bv += self._dbv
                self.bh += self._dbh
                if cfg.learn_a:
                    # 可见单元总输入 = W^T h + b_v；隐单元总输入 = W v + b_h
                    gv = self._update_a(h0 @ self.W.T + self.bv,
                                        h1 @ self.W.T + self.bv, self.a_v)
                    gh = self._update_a(v0 @ self.W + self.bh,
                                        v1 @ self.W + self.bh, self.a_h)
                    self._dav = cfg.momentum * self._dav + cfg.lr_noise * gv
                    self._dah = cfg.momentum * self._dah + cfg.lr_noise * gh
                    self.a_v = np.clip(self.a_v + self._dav, cfg.a_min, cfg.a_max)
                    self.a_h = np.clip(self.a_h + self._dah, cfg.a_min, cfg.a_max)
                ep_err += float(np.mean((v0 - v1) ** 2)) * len(b)
                nb += len(b)
            err = ep_err / max(1, nb)
            self.history.append(err)
            if verbose and (ep % 20 == 0 or ep == 1):
                print(f"    epoch {ep:4d}/{epochs}  recon-MSE={err:.6f}")
            if patience and err < best_err - 1e-7:
                best_err, best, bad = err, self._snapshot(), 0
            elif patience:
                bad += 1
                if bad >= patience:
                    self._restore(best)
                    break
        return self.history

    # --------------------------------------------------------------
    def _snapshot(self):
        return (self.W.copy(), self.bv.copy(), self.bh.copy(),
                self.a_v.copy(), self.a_h.copy())

    def _restore(self, snap):
        if snap is None:
            return
        self.W, self.bv, self.bh, self.a_v, self.a_h = [x.copy() for x in snap]

    # --------------------------------------------------------------
    def get_params(self) -> dict:
        return {
            "W": self.W.tolist(), "bv": self.bv.tolist(), "bh": self.bh.tolist(),
            "a_v": self.a_v.tolist(), "a_h": self.a_h.tolist(),
            "n_visible": self.cfg.n_visible, "n_hidden": self.cfg.n_hidden,
            "sigma": self.cfg.sigma, "theta_l": self.cfg.theta_l,
            "theta_h": self.cfg.theta_h,
        }

    @classmethod
    def from_params(cls, p: dict) -> "CRBM":
        cfg = CRBMConfig(n_visible=int(p["n_visible"]), n_hidden=int(p["n_hidden"]),
                         sigma=float(p["sigma"]), theta_l=float(p["theta_l"]),
                         theta_h=float(p["theta_h"]))
        obj = cls(cfg)
        obj.W = np.asarray(p["W"], dtype=np.float64)
        obj.bv = np.asarray(p["bv"], dtype=np.float64)
        obj.bh = np.asarray(p["bh"], dtype=np.float64)
        obj.a_v = np.asarray(p["a_v"], dtype=np.float64)
        obj.a_h = np.asarray(p["a_h"], dtype=np.float64)
        return obj