# -*- coding: utf-8 -*-
"""
深度自编码器 / Deep autoencoder
====================================================================
论文 2.2 节：把逐层预训练好的 CRBM 权重展开(unroll)成对称的深度自编码器，
再用带动量的反向传播微调全部参数 (Eq.8-11)。

网络结构示例（39 个变量）：
    39 -> 20 -> 10 -> 5 -> 10 -> 20 -> 39
    编码器 = 前三层；瓶颈 = 5 维编码；解码器 = 后三层（与前向对称）。
"""
from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn


def _act(name: str):
    name = (name or "sigmoid").lower()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name in ("relu",):
        return nn.ReLU()
    if name in ("tanh",):
        return nn.Tanh()
    if name in ("identity", "none", "linear"):
        return nn.Identity()
    raise ValueError(f"未知激活函数: {name}")


class DeepAutoencoder(nn.Module):
    """对称深度自编码器。dims = [输入, 隐1, 隐2, ..., 瓶颈]。"""

    def __init__(self, dims: Sequence[int], activation: str = "sigmoid",
                 out_activation: str = "sigmoid"):
        super().__init__()
        self.dims: List[int] = list(dims)
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.encoder.append(nn.Linear(dims[i], dims[i + 1]))
        for i in range(len(dims) - 1, 0, -1):
            self.decoder.append(nn.Linear(dims[i], dims[i - 1]))
        self.act = _act(activation)
        self.out_act = _act(out_activation)

    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for i, lin in enumerate(self.encoder):
            h = lin(h)
            if i < len(self.encoder) - 1:
                h = self.act(h)
        return h

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = z
        for i, lin in enumerate(self.decoder):
            h = lin(h)
            if i < len(self.decoder) - 1:
                h = self.act(h)
        return self.out_act(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    # ------------------------------------------------------------------
    def load_pretrained_crbms(self, crbms, scale: float = 1.0) -> None:
        """
        用逐层 CRBM 的权重初始化编码器；解码器用转置权重（tied 思想）。

        CRBM 定义 h = phi(W v + b_h)；线性层 Linear(in,out) 的权重形状为
        (out, in)，因此在初始化时需要把 CRBM 的 W (in, out) 转置写入。
        """
        n_enc = len(self.encoder)
        for i, crbm in enumerate(crbms):
            Wi = torch.as_tensor(crbm.W.T, dtype=torch.float32) * scale   # (out,in)
            bi = torch.as_tensor(crbm.bh, dtype=torch.float32)
            with torch.no_grad():
                self.encoder[i].weight.copy_(Wi)
                self.encoder[i].bias.copy_(bi)
        # 解码器：与对应编码层转置对称
        for j, lin in enumerate(self.decoder):
            enc = self.encoder[n_enc - 1 - j]
            with torch.no_grad():
                lin.weight.copy_(enc.weight.T)
                lin.bias.zero_()

    # ------------------------------------------------------------------
    def encoder_weight_norms(self) -> List[float]:
        return [float(lin.weight.detach().norm().cpu()) for lin in self.encoder]

    @staticmethod
    def build(input_dim: int, hidden_sizes: Sequence[int]) -> "DeepAutoencoder":
        dims = [input_dim] + list(hidden_sizes)
        return DeepAutoencoder(dims)