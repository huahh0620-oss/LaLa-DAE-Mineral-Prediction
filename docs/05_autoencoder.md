# `src/autoencoder.py` 分析文档

## 1. 文件职责

用 PyTorch 实现**对称深度自编码器**，并把 step03 训练好的 CRBM 权重
"展开（unroll）"进来作为初始值。对应论文 **2.2 节，Eq.(8)–(11)**。

## 2. 在流水线中的位置

```
step03 预训练产物 (pretrained_crbms_*.npz)
        │  load_pretrained_crbms()
        ▼
DeepAutoencoder  ──step04 微调──▶  deep_autoencoder_*.pt
                                        │
                                        └─step05 用它算重构误差
```

## 3. 网络结构

`dims = [输入维度] + HIDDEN_SIZES`，例如 12 个变量时：

```
dims = [12, 8, 4]
结构 = 12 - 8 - 4 - 8 - 12
       └─编码器─┘  └─解码器─┘
             瓶颈 4 维
```

| 组件 | 说明 |
| --- | --- |
| `self.encoder` | `ModuleList`，`Linear(dims[i] → dims[i+1])`，共 `len(dims)-1` 层 |
| `self.decoder` | 镜像：`Linear(dims[i+1] → dims[i])` |
| `self.act` | 隐藏层激活，默认 `Sigmoid`（与 CRBM 的连续单元一致） |
| `self.out_act` | 输出层激活，默认 `Sigmoid`（输出也在 [0,1]） |

## 4. 核心方法

| 方法 | 作用 |
| --- | --- |
| `encode(x)` | 逐层前向到瓶颈（最后一层不加激活） |
| `decode(z)` | 从瓶颈逐层还原到输入维度（最后一层加 `out_act`） |
| `forward(x)` | `decode(encode(x))`，即重构值 R |
| `load_pretrained_crbms(crbms)` | **权重展开**：把 CRBM 权重写进编码器；解码器用转置权重 |
| `encoder_weight_norms()` | 诊断用（观察各层权重尺度） |
| `DeepAutoencoder.build(input_dim, hidden_sizes)` | 便捷构造 |

### 4.1 权重展开的细节（容易出错的地方）

CRBM 的隐层状态是 `h = φ(W v + b_h)`，其中 `W` 形状为 `(n_in, n_out)`；
而 `nn.Linear(in, out)` 的权重形状是 `(out, in)`。因此写入时必须**转置**：

```python
Wi = torch.as_tensor(crbm.W.T)   # (out, in)
self.encoder[i].weight.copy_(Wi)
self.encoder[i].bias.copy_(crbm.bh)
```

解码器采用 **tied weights（权重共享思想）**：第 j 个解码层直接用对应编码层
权重的转置，偏置置零。这样解码器一开始就是编码器的"逆过程"，
与论文"把 CRBM 反向展开成解码器"的做法一致。

## 5. 微调（论文 Eq.8–11）

在 `step04_finetune.py` 中用带动量的 SGD 最小化 MSE：

```python
opt   = torch.optim.SGD(model.parameters(), lr=0.8,
                        momentum=0.9, weight_decay=1e-5)
lossf = nn.MSELoss()      # 对 D 个变量取平均的平方误差
```

论文 Eq.(8)(9) 的反向传播推导（含 noise-control 参数的导数）在标准
自动微分框架下由 PyTorch 自动完成，因此本文件不需要手写反向传播。

> 论文额外说明：微调"只是微调预训练阶段提取的特征，而非重新提取特征"。
> 代码中体现为 —— 微调时权重从预训练值出发、学习率 0.8 但只跑 200 轮，
> 实测训练 MSE 从 0.0146 快速降到 0.0033 后**基本不再下降**，
> 印证了论文"预训练已经学得很好"的结论。

## 6. 注意事项

1. `load_pretrained_crbms` 要求 `len(crbms)` 与 `len(encoder)` 一致，
   即预训练的层数与 `HIDDEN_SIZES` 长度必须匹配 —— 换网络结构后必须重跑 step03。
2. 若把 `ACTIVATION` 改成 `relu`，需同时保证数据非负（当前 [0,1] 满足），
   但 `OUTPUT_ACTIVATION` 建议保留 sigmoid 以匹配 [0,1] 值域。
3. 保存 checkpoint 时一并存了 `dims / activation / input_features`，
   便于 step05 直接重建模型而无需读配置。