# `config.py` 分析文档

## 1. 文件职责

工程**唯一需要用户修改**的文件。集中存放：

* 数据路径（与代码解耦的关键）
* 数据目录布局约定
* 预处理、网络、训练、评价、制图的全部超参数

所有 `src/step*.py` 都只从本文件读取配置，脚本内部**不出现任何绝对路径**。

## 2. 在流水线中的位置

```
config.py  ──被所有步骤导入──▶  step01 ... step07
```

属"全局参数层"，不参与计算，但决定了每次运行的行为。

## 3. 关键配置逐项说明

### 3.1 数据路径（第 1 节）

| 变量 | 说明 |
| --- | --- |
| `Data_Path` | 数据根目录。优先级：环境变量 `LALA_DATA_PATH` > 默认值 `D:\LaLa_Data` |
| `GEOCHEM_SUBDIR` | 地化栅格子目录名，默认 `地化` |
| `POSITIVE_RASTER` | 已知矿点栅格文件名，默认 `positive_IOCG.tif` |
| `MASK_RASTER` | 可选研究区掩膜；留空表示不使用 |
| `AUTO_LOCATE_GEOCHEM` | 找不到 `地化/` 时是否递归搜索 |
| `POSITIVE_SEARCH_PARENTS` | 矿点栅格向上搜索的目录层数 |

**为什么要这样做**：数据可能放在 `D:\`、`F:\` 甚至移动硬盘上，而代码要能
原样复制到任何机器。把路径收敛到**一个变量**，就实现了"程序与数据解耦"。

### 3.2 输出路径（第 2 节）

`OUTPUT_DIR` 及其下的 `figures / tables / models / cache` 四个子目录，
在导入 `config` 时自动创建，因此步骤脚本无需再关心目录是否存在。

### 3.3 预处理（第 3 节）

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `RAW_COMPOSITION` | `False` | 数据是原始浓度时置 `True`，会自动做 ilr 变换 |
| `NORMALIZE_RANGE` | `(0,1)` | 论文 Eq.(13)：线性拉伸到 [0,1] |

本区数据文件名带 `_CLR_IDW`，说明已是**中心对数比（CLR）**结果，
闭合效应已解决，因此默认跳过 ilr。

### 3.4 指示元素选择（第 4 节，改进点 ①）

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `FEATURE_SELECTION_MODE` | `"auc"` | `auc` 自动挑选 / `manual` 手工列表 / `none` 用全部 |
| `N_SELECTED_FEATURES` | `12` | 挑选个数 |
| `SELECTION_MIN_AUC` | `0.5` | 低于该判别力的元素直接排除（去掉反指示元素） |
| `SELECTION_MAX_CORR` | `0.90` | 相关性去冗余阈值，0 表示不去冗余 |
| `MANUAL_FEATURES` | 18 个元素 | 知识驱动组合，供 `manual` 模式使用 |

### 3.5 网络结构（第 5 节，改进点 ③）

```python
HIDDEN_SIZES = "auto"     # → 两个递减隐藏层：0.70n 与 0.35n
                          # 12 个变量 → [8, 4]；39 个变量 → [27, 14]
```

改成 `[20, 10, 5]` 即可复现论文的 3 层折半结构。

### 3.6 CRBM 预训练（第 6 节）

论文 2.1 节参数：`CRBM_EPOCHS=200`、`CRBM_LR=0.3`、`CRBM_MOMENTUM=0.9`、
`CRBM_SIGMA=0.2`、`CRBM_WEIGHT_DECAY=1e-5`（论文的 learning cost）。

> ⚠️ **重要**：`CRBM_LEARN_A` 默认 `False`。论文 Eq.(6) 用输入方差之差更新
> 噪声控制参数 `a`，但实测在本数据上 `a` 会单调增大到上界、使 sigmoid 饱和、
> 重构误差发散。固定 `a=1` 时训练稳定。这是本项目对论文的一处**必要偏离**，
> 详见 `04_crbm.md`。

### 3.7 微调（第 7 节）

论文 2.2 节参数：`FINETUNE_EPOCHS=200`、`FINETUNE_LR=0.8`、
`FINETUNE_MOMENTUM=0.9`、`FINETUNE_WEIGHT_DECAY=1e-5`，
优化器为带动量的 SGD。

### 3.8 随机性（第 8 节）

`RANDOM_SEEDS` 支持多个种子；多于一个时，各模型训练完成后对
**归一化异常得分取平均**得到集成异常图（step05 内实现）。

### 3.9 评价与分级（第 9 节，改进点 ④）

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `T_CRITICAL` | 1.96 | t 值显著性门槛（论文口径，仅作参考对比） |
| `N_THRESHOLD_CANDIDATES` | 100 | 阈值扫描点数 |
| `MODERATE_ENRICHMENT_MIN` | 2.0 | 中异常下界的富集比门槛（**本项目实际使用**） |
| `CAPTURE_PERCENTS` | `[1,5,10,20]` | 报告 Top-X% 面积捕获率 |

### 3.10 异常得分模式（第 10 节，改进点 ②）

```python
SCORE_MODE = "weighted"   # 加权重构误差（默认）
                          # "mse" = 论文原式
```

### 3.11 制图（第 11 节）

`FIGURE_DPI`、`STAR_SIZE`（矿点五角星尺寸）—— 这两个参数是"让五角星看得清"
的直接开关。若图中星标仍偏小，把 `STAR_SIZE` 从 130 调到 200 即可。

### 3.12 基准测试（第 12 节）

`BENCHMARK_METHODS` 列出参与对比的 7 种方法，`BENCHMARK_SEEDS` 控制重复次数。

## 4. 环境变量覆盖速查

| 环境变量 | 等价配置 |
| --- | --- |
| `LALA_DATA_PATH` | `Data_Path` |
| `LALA_OUTPUT_DIR` | `OUTPUT_DIR` |
| `LALA_POSITIVE_RASTER` | `POSITIVE_RASTER` |
| `LALA_FEATURE_MODE` | `FEATURE_SELECTION_MODE` |
| `LALA_N_FEATURES` | `N_SELECTED_FEATURES` |
| `LALA_SCORE_MODE` | `SCORE_MODE` |
| `LALA_SEEDS` | `RANDOM_SEEDS`（逗号分隔） |
| `LALA_CRBM_LEARN_A` | `CRBM_LEARN_A` |
| `LALA_ENRICH_MIN` | `MODERATE_ENRICHMENT_MIN` |
| `LALA_DEVICE` | `DEVICE`（`auto`/`cpu`/`cuda`） |

## 5. 注意事项

1. **不要**在步骤脚本里写绝对路径，这是本工程"可迁移"的前提。
2. 修改 `N_SELECTED_FEATURES` 会同时改变网络结构（`auto` 按输入维度生成），
   无需手工改 `HIDDEN_SIZES`。
3. `T_CRITICAL` 只影响报告中的参考列；实际分级用 `MODERATE_ENRICHMENT_MIN`。
4. 配置在**导入时**读取环境变量，所以必须在运行前设好环境变量。