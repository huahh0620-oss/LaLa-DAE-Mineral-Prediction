# `src/step05_anomaly_map.py` 分析文档

## 1. 文件职责

**步骤 5：重构误差 → 多元地化异常图**。对应论文 **Eq.(1)**。

把每个像元的多变量向量送进训练好的自编码器，用重构误差作为异常得分，
归一化到 [0,1] 后写回栅格。含本项目**改进点 ②（加权重构误差）**。

## 2. 输入 / 输出

| 输入 | 模型 `deep_autoencoder_seed{seed}.pt`、`dataset.npz`、矿点栅格 |
| --- | --- |

| 输出 | 内容 |
| --- | --- |
| `anomaly_map.tif` | 选定模式的异常图（**主结果**） |
| `anomaly_map_normalized.tif` | 同上（归一化，便于与论文阈值比较） |
| `anomaly_map_mse.tif` | 论文原式（不加权）异常图 |
| `anomaly_map_weighted.tif` | 加权重构误差异常图 |
| `cache/anomaly_scores.npz` | 两种模式的得分向量、掩膜、权重、变量名 |
| `figures/step05_anomaly_map.png` | 异常图 + 已知矿点 + 得分分布直方图 |

## 3. 两种异常得分

### 3.1 论文原式（`SCORE_MODE="mse"`）

$$\mathrm{RE}(i)=\frac{1}{D}\sum_{j=1}^{D}\bigl(I_{ij}-R_{ij}\bigr)^2$$

```python
def reconstruction_error(model, X, device):
    R = reconstruct(model, X, device)
    return ((X - R) ** 2).mean(axis=1)
```

### 3.2 加权重构误差（`SCORE_MODE="weighted"`，默认，改进点 ②）

$$\mathrm{RE}_w(i)=\sum_{j=1}^{D} w_j\bigl(I_{ij}-R_{ij}\bigr)^2,
\qquad w_j\propto\max\bigl(0,\ \mathrm{AUC}_j-0.5\bigr)$$

```python
def pathfinder_weights(X, deposit):
    w = [max(0.0, roc_auc_score(deposit, X[:, j]) - 0.5) for j in range(D)]
    return w / w.sum()

def weighted_error(model, X, device, weights):
    res2 = (X - R) ** 2
    return (res2 * weights).sum(axis=1)
```

**为什么有效**：不加权时，`Fe2O3` 这类方差大但判别力弱的变量会占据误差的
主要份额；加权后误差集中在 Cu-Au-Ag-Mo 等真正的指示元素上。
实测 ROC-AUC 从 0.734 提升到 0.756，t 极大值从 6.58 提升到 8.97。

## 4. 多种子集成

```python
if len(seeds) == 1:
    score_norm = per_norm[seeds[0]]
else:
    score_norm = np.mean([per_norm[s] for s in seeds], axis=0)
```

对每个种子分别把得分**归一化后再平均**（避免量纲差异），得到集成异常图。

## 5. 关键函数

| 函数 | 作用 |
| --- | --- |
| `_load_model(seed, device)` | 从 checkpoint 重建模型 |
| `reconstruct(model, X, device)` | 返回重构矩阵 R（分块推理，避免爆内存） |
| `reconstruction_error(...)` | 论文原式得分 |
| `pathfinder_weights(X, deposit)` | 判别力权重 |
| `weighted_error(...)` | 加权得分 |
| `score_mode()` | 读取并校验 `SCORE_MODE` |
| `_plot(...)` | 左：异常图+五角星；右：得分分布直方图 |
| `_find_reference_raster()` | 取一张地化栅格作为输出空间模板 |

## 6. 制图说明（针对"五角星看不清"的改进）

```python
axes[0].scatter(xx, yy, marker="*", s=C.STAR_SIZE*1.9,
                c="none", edgecolors="white", linewidths=1.9)   # 白色描边（底衬）
axes[0].scatter(xx, yy, marker="*", s=C.STAR_SIZE, c="k")       # 黑色星
```

**双层绘制**：先画一个更大的白色星作描边，再画黑色星。
这样无论背景是深蓝还是亮红，五角星都清晰可见（单层绘制在红色区几乎看不见）。

右图叠加"矿点得分中位数"与"全体像元中位数"两条参考线，
可以一眼看出矿点是否整体偏向高异常。

## 7. 注意事项

1. 矿点用于**加权**（`pathfinder_weights`）和**制图**，不参与模型训练 ——
   自编码器仍是无监督的。
2. `ANOMALY_MAP` 的 nodata 统一写为 `-9999`。
3. 分块大小 `batch=4096` 是为内存安全设置；数据很大时可调小。
4. `EVAL_RASTER` 可指定与 `POSITIVE_RASTER` 不同的矿点栅格用于出图。