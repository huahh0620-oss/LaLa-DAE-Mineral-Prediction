# `src/step01_prepare_data.py` 分析文档

## 1. 文件职责

**步骤 1：数据准备**。把 `Data_Path` 下的 39 个地化栅格读成一张
"样本 × 变量"矩阵，并完成对齐校验、掩膜、归一化。对应论文 **3.2 节 + Eq.(13)**。

## 2. 输入 / 输出

| 输入 | 说明 |
| --- | --- |
| `<Data_Path>/地化/*.tif` | 39 个地化栅格，网格必须对齐 |
| `<Data_Path>/<MASK_RASTER>` | 可选研究区掩膜 |

| 输出 | 内容 |
| --- | --- |
| `cache/dataset.npz` | 归一化样本矩阵 `X_norm`、原始值 `X_raw`、有效像元掩膜、变量名、栅格元数据、归一化参数 |
| `tables/feature_stats.csv` | 每个变量的 min/max/mean/std/分位数 |
| `tables/step01_report.json` | 数据准备报告 |

## 3. 处理流程

```
列出地化栅格 → 读取并对齐校验 → 展平成 (H*W, F)
    → 有效像元掩膜（全变量都有值）→ 可选研究区掩膜
    → 可选 ilr（原始浓度时）→ 统计 → 归一化到 [0,1] → 存 npz
```

### 3.1 展平与掩膜

```python
X_all = cube.reshape(n_band, -1).T          # (H*W, F)
valid_mask_flat = np.isfinite(X_all).all(axis=1)
X = X_all[valid_mask_flat]
```

**为什么要用"全变量都有值"作为有效判据**：自编码器要求每个样本是完整向量，
任一变量缺值都会让重构误差失去意义。本区 21218 个格网中 20289 个（95.6%）有效。

### 3.2 ilr 变换（可选）

`_ilr()` 实现了等距对数比变换（Egozcue et al., 2003），用于解决成分数据的
闭合效应。本区数据文件名含 `_CLR`，已是中心对数比结果，因此 **默认跳过**。
若换成原始浓度数据，把 `config.RAW_COMPOSITION = True` 即可自动启用。

### 3.3 归一化（论文 Eq.13）

$$y'_i=\frac{y_i-y_{\min}}{y_{\max}-y_{\min}}\,[1,6682]\ \rightarrow\ [0,1]$$

代码用 `fit_minmax` / `apply_minmax` 两步完成，并把参数一并存进 npz，
保证后续步骤与结果栅格可以精确反推。

## 4. 关键函数

| 函数 | 作用 |
| --- | --- |
| `_ilr(X)` | Helmert 正交基下的等距对数比变换 |
| `run()` | 步骤主入口，返回 report dict |

## 5. 输出 npz 的字段

| 键 | 形状 | 说明 |
| --- | --- | --- |
| `X_raw` | (N, F) | 归一化前的变量值 |
| `X_norm` | (N, F) | 归一化到 [0,1] 的变量值（**后续步骤使用**） |
| `valid_mask` | (H, W) | 有效像元布尔掩膜 |
| `feature_names` | (F,) | 变量名（来自文件名主干） |
| `transform` / `crs` | — | 栅格空间参考（供写栅格时复用） |
| `norm_min` / `norm_xmax` / `norm_lo` / `norm_hi` | (F,) | 归一化参数 |

## 6. 注意事项

1. **网格对齐是硬性要求**。若某个栅格与原网格不一致，`read_aligned_stack`
   会立即抛出 `ValueError` 并指出是哪个文件 —— 请先在地理信息软件里重采样对齐。
2. 变量名由文件名自动解析（`Cu_CLR_IDW.tif` → `Cu`）。
   如果文件名不规范，`feature_name_from_path` 会退回使用整个主干名，
   后续"按名字选元素"（如 `single_element_cu`）可能失效。
3. 本步骤不做任何行/列裁剪，裁剪通过 `MASK_RASTER` 实现。
4. 若数据是原始浓度（非 CLR/ilr），务必打开 `RAW_COMPOSITION`，
   否则闭合效应会让重构误差系统性失真。