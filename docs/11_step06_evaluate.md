# `src/step06_evaluate.py` 分析文档

## 1. 文件职责

**步骤 6：空间相关性评价与异常分级**。对应论文 **4.2 节**。

用 Student's t 值度量异常图与已知矿点的空间相关性，把异常分成
高 / 中 / 低三级，并给出 ROC-AUC、捕获率、成功度曲线等定量指标。

## 2. 输入 / 输出

| 输入 | `cache/anomaly_scores.npz`、矿点栅格 |
| --- | --- |

| 输出 | 内容 |
| --- | --- |
| `anomaly_classes.tif` | 高(3)/中(2)/低(1) 分级栅格 |
| `tables/step06_threshold_scan.csv` | 阈值扫描：面积占比、捕获率、富集比、t 值 |
| `tables/step06_anomaly_classes.csv` | 三级异常的面积与矿点占比 |
| `tables/step06_deposit_pixels.csv` | **每个矿点像元**的得分、分位、所属级别 |
| `tables/step06_metrics.csv` | 综合指标 |
| `figures/step06_tvalue.png` | t 值—阈值曲线 |
| `figures/step06_anomaly_classes.png` | 分级图 + 五角星 + 面积/矿点占比柱状图 |
| `figures/step06_roc_capture.png` | ROC 曲线 + 预测面积成功率曲线 |

## 3. Student's t 值（Bonham-Carter, 1994）

对给定阈值 τ，把异常图二值化为 B = {score ≥ τ}，则

$$t=\frac{D_1-D\cdot A_1/A}{\sqrt{D\cdot \frac{A_1}{A}\left(1-\frac{A_1}{A}\right)}}$$

| 符号 | 含义 |
| --- | --- |
| `A` | 研究区总像元数 |
| `A1` | 异常区像元数 |
| `D` | 已知矿点总数 |
| `D1` | 落在异常区内的矿点数 |

**直观理解**：分子是"实际落在异常区的矿点数"与"随机情况下的期望矿点数"之差；
分母是随机波动尺度。因此 t 越大表示异常区对矿点的**富集越显著**，
t > 1.96 表示 95% 置信水平下显著。

```python
def student_t_value(score, deposit, threshold):
    anom = score >= threshold
    p = anom.sum() / score.size
    return (n_da - n_d * p) / np.sqrt(n_d * p * (1 - p))
```

## 4. 三级分级的阈值规则（改进点 ④）

| 级别 | 规则 |
| --- | --- |
| **高异常** | `score ≥ thr_max`，`thr_max = argmax(t)`（t 值取极大处的阈值） |
| **中异常** | `thr_mod ≤ score < thr_max` |
| **低异常** | `score < thr_mod` |

`thr_mod` 取**"富集比 ≥ `MODERATE_ENRICHMENT_MIN`（默认 2）的最低阈值"**：

```python
enr = scan[(scan.enrichment >= C.MODERATE_ENRICHMENT_MIN) & (scan.threshold <= thr_max)]
thr_mod = enr.threshold.min()
```

**为什么不用论文的 `t ≥ 1.96` 作为中异常下界**：
本区只有 23 个矿点，`t ≥ 1.96` 对应的阈值极低（0.032），会把 **71% 的面积**
划成中异常，完全失去分级意义。论文中该比例是 34.1%（其矿点数远多于本区）。
改用富集比规则后，本项目中异常占 35.6% 面积，与论文可直接对比。

> 代码仍然计算并报告了 `t ≥ 1.96` 的阈值（`threshold_t_critical`），
> 以便在论文中同时给出"论文口径"与"本项目口径"两组数字。

## 5. 关键函数

| 函数 | 作用 |
| --- | --- |
| `student_t_value(score, deposit, threshold)` | 单个阈值的 t 值 |
| `scan_thresholds(score, deposit, n)` | 在分位阈值上扫描，返回 DataFrame |
| `_roc_capture(score, deposit)` | ROC/成功率曲线数据与 AUC、AP |
| `run(seed)` | 步骤主入口 |
| `_plot_tvalue` / `_plot_classes` / `_plot_roc_capture` | 三张图 |

## 6. 制图说明（针对"五角星"的改进）

`step06_anomaly_classes.png` 采用左右双面板：

* **左图**：分级地图 + 双层绘制的五角星（白色描边 + 黑色实心），
  图例中注明矿点数量；
* **右图**：三级异常的"面积占比 vs 矿点占比"对比柱状图 ——
  可以一眼看出高异常区"面积很小但矿点占比很高"。

另外 `step06_deposit_pixels.csv` 逐个列出**每个矿点像元**的
得分、分位与所属级别，便于逐个核查"哪个五角星没对上"。

## 7. 本区实测结果

| 指标 | 数值 |
| --- | --- |
| Student's t 极大值 | **14.18**（阈值 0.341） |
| 高异常面积占比 | 1.00% |
| 高异常含已知矿点 | 30.4% |
| 中异常面积占比 / 含矿点 | 35.6% / 43.5% |
| 中+高异常含矿点 | **73.9%** |
| ROC-AUC | 0.766 |
| 矿点得分中位分位 | 0.89 |
| Top 10% 面积捕获矿点 | 43.5% |

## 8. 注意事项

1. t 值曲线在**高阈值端可能单调上升**（本区就是这样），此时
   `argmax(t)` 落在扫描区间的下端（top 1%）—— 说明"越高的异常越富集"，
   是好事，但要注意 top 1% 只有 203 个像元，制图上要保证可见。
2. 只有 23 个矿点，**t 值与 AUC 的置信区间都很宽**，结论表述应留有余地。
3. 用 `read_deposit_aligned` 算指标、用 `read_deposit_grid` 画图，
   两者不可混用（形状不同）。