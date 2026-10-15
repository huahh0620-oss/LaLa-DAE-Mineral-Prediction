# `src/step02_select_features.py` 分析文档

## 1. 文件职责

**步骤 2：矿化指示元素选择**（本项目相对论文的**改进点 ①**）。

用"单元素判别力"（以已知矿点为阳性样本的 ROC-AUC）给 39 个变量排序，
挑出对铜金矿化最敏感的前若干个元素，再做相关性去冗余。

## 2. 为什么需要这一步

论文直接把全部变量送进自编码器。但本区 39 个变量中有相当一部分
**与矿化负相关**（Sr 0.224、Zr 0.307、Be 0.358、Th 0.383、Cd 0.395 …），
把它们混进去会让重构误差更多反映岩性差异而非矿化，从而稀释异常信号。

实验证据（详见 `docs/优化说明.md`）：

| 变量集 | 变量数 | ROC-AUC | t 极大值 |
| --- | --- | --- | --- |
| 全部 39 个 | 39 | 0.680 | 4.77 |
| **前 12 个（本步骤选出）** | **12** | **0.734** | **6.58** |

## 3. 输入 / 输出

| 输入 | 说明 |
| --- | --- |
| `cache/dataset.npz` | step01 的产物 |
| 矿点栅格 | `EVAL_RASTER` 或 `POSITIVE_RASTER` |

| 输出 | 内容 |
| --- | --- |
| `cache/selected_features.npz` | `selected`（名字）与 `indices`（列下标） |
| `tables/feature_selection_auc.csv` | 全部变量的 AUC、是否入选、排名 |
| `tables/step02_report.json` | 选择参数与结果 |
| `figures/step02_feature_auc.png` | 判别力条形图（入选的标红） |

## 4. 三种选择模式

| `FEATURE_SELECTION_MODE` | 行为 |
| --- | --- |
| `"auc"`（默认） | 按 AUC 从高到低挑选，直到达到 `N_SELECTED_FEATURES` 或 AUC 低于 `SELECTION_MIN_AUC` |
| `"manual"` | 使用 `config.MANUAL_FEATURES`（知识驱动的 IOCG 指示元素组合） |
| `"none"` | 使用全部变量（**等价论文做法**） |

## 5. 核心算法

### 5.1 判别力与方向

```python
auc[j] = roc_auc_score(deposit, X[:, j])
```

* `AUC > 0.5` → **正指示**（值越高越靠近矿化）
* `AUC ≈ 0.5` → 无判别力
* `AUC < 0.5` → **反指示**（与矿化负相关）

把 `AUC < 0.5` 的元素排除，是本步骤最关键的一步。

### 5.2 相关性去冗余

```python
if C.SELECTION_MAX_CORR > 0 and selected:
    r = |corrcoef(X[:, j], X[:, selected]).T|
    redundant = any(r > C.SELECTION_MAX_CORR)
```

如果候选变量与**已入选**变量的 |相关系数| 超过 0.90，就跳过它
（因为信息高度重复）。设 `SELECTION_MAX_CORR = 0` 可关闭去冗余。

### 5.3 自动选出的结果

```
Cu(0.738), F(0.701), Mo(0.692), Au(0.661), Ag(0.649), Mn(0.615),
As(0.608), B(0.601), Na2O(0.561), Fe2O3(0.531), Ba(0.524), Ni(0.520)
```

这与拉拉式铜金矿（铁氧化物-铜-金型）的地质认识高度一致：

| 元素 | 地质含义 |
| --- | --- |
| Cu, Au, Ag, Mo | 主成矿元素组合 |
| F, Ba | 挥发分 / 热液流体标志 |
| As, B | 前缘晕元素 |
| Mn, Fe2O3, Ni, Na2O | 蚀变与围岩标志（钠化、铁锰碳酸盐化） |

## 6. 关键函数

| 函数 | 作用 |
| --- | --- |
| `run()` | 步骤主入口 |
| `load_selected_indices()` | **供 step03 调用**，返回入选变量的列下标 |
| `_plot_auc(tbl, selected)` | 绘制判别力条形图 |

## 7. ⚠️ 使用时的诚实性提醒

本步骤用**已知矿点**来挑选指示元素，属于勘查地球化学中的标准做法
（pathfinder element selection），但它意味着**变量选择与评价使用了同一套标签**，
会让 ROC-AUC 偏乐观。因此在论文中必须做到：

1. 明确说明"指示元素由已知矿点标定"；
2. **同时报告 `FEATURE_SELECTION_MODE="none"` 的结果**作为对照（论文口径）；
3. 若有独立矿点集，用它做外部验证（本项目可用 `positive.tif` 的 58 个像元
   作为半独立验证集，见 `docs/优化说明.md` 中的 DEP58 列）。

## 8. 注意事项

1. 必须在 step01 之后运行（依赖 `dataset.npz`）。
2. 矿点像元数为 0 时直接报错，避免选出无意义的变量集。
3. 修改 `N_SELECTED_FEATURES` 会同时改变下游网络结构（`HIDDEN_SIZES="auto"`），
   属于预期行为。