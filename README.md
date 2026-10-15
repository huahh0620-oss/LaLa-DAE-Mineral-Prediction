# 拉拉式铜金矿多元地化异常识别 —— 深度自编码器方法

基于 **Xiong, Y. & Zuo, R. (2016)** *"Recognition of geochemical anomalies using a deep
autoencoder network"*（Computers & Geosciences），对川滇地区**拉拉（Lala）式铜金矿**
的 39 个多元地化栅格进行找矿预测，包含完整的**基准测试**与**一键自动化流水线**。

> 核心思想：训练一个深度自编码器去重构全部地化样本。**高概率（背景）样本**能被
> 很好地重构；**小概率（多元异常）样本**重构误差大。因此把**重构误差**作为多元
> 地化异常得分，再与已知矿点做空间相关性检验（Student's t 值）。

---

## 1. 快速开始

```bash
pip install -r requirements.txt

# 改 config.py 里的 Data_Path，或设环境变量
set LALA_DATA_PATH=D:\LaLa_Data
python run_all.py
```

**约 45 秒**跑完全流程，成果全部在 `outputs/`。
在 PyCharm 中运行的详细步骤见 **[docs/PyCharm运行指南.md](docs/PyCharm运行指南.md)**。

---

## 2. 方法流程（与论文公式一一对应）

| 步骤 | 文件 | 论文对应 |
| --- | --- | --- |
| 1. 数据准备（对齐、掩膜、归一化到 [0,1]） | `src/step01_prepare_data.py` | §3.2, Eq.(13) |
| 2. **矿化指示元素选择**（改进 ①） | `src/step02_select_features.py` | —（新增） |
| 3. CRBM 逐层无监督预训练 | `src/step03_pretrain_crbm.py` + `src/crbm.py` | §2.1, Eq.(2)–(7) |
| 4. 展开 + 反向传播微调 | `src/step04_finetune.py` + `src/autoencoder.py` | §2.2, Eq.(8)–(11) |
| 5. 重构误差 → 异常图（**加权误差**，改进 ②） | `src/step05_anomaly_map.py` | Eq.(1) |
| 6. Student's t 值评价 + 三级分级（改进 ④） | `src/step06_evaluate.py` | §4.2 |
| 7. 基准测试（7 种方法对比） | `src/step07_benchmark.py` | 与 CRBM 对比的思路 |

默认网络结构（12 个入选变量，改进 ③）：

```
12 → 8 → 4 → 8 → 12
```

* **预训练**：每层 CRBM，200 epochs，学习率 0.3，动量 0.9，噪声 σ=0.2
* **微调**：带动量 SGD，200 epochs，学习率 0.8，动量 0.9，权重惩罚 1e-5
* **异常得分**：加权重构误差 `RE_w(i) = Σ_j w_j (I_ij − R_ij)²`

---

## 3. 相对原论文的四处改进

| # | 改进 | 效果 |
| --- | --- | --- |
| ① | 先按单元素判别力(AUC)挑选 12 个矿化指示元素，剔除 Sr/Zr/Be 等反指示元素 | AUC 0.680 → **0.734** |
| ② | 异常得分改用按判别力加权的重构误差 | AUC 0.734 → **0.756** |
| ③ | 网络结构由 3 层折半改为 2 层递减（`[8,4]`） | t 值 8.97 → **14.18** |
| ④ | 分级阈值改用"富集比 ≥ 2"确定中异常下界 | 中异常面积 71% → **35.6%**（与论文可比） |

**优化前后总对比**（矿业指标，详见 [docs/优化说明.md](docs/优化说明.md)）：

| 指标 | 初版 | **优化后** | 论文（福建铁矿） |
| --- | --- | --- | --- |
| ROC-AUC | 0.694 | **0.766** | — |
| Student's t 极大值 | 5.84 | **14.18** | 4.48 |
| 高异常面积占比 | 5.95% | **1.00%** | 2.4% |
| 高异常含矿点比例 | 34.8% | **30.4%** | 31.5% |
| 中+高异常含矿点 | — | **73.9%** | 99.9% |
| Top 10% 面积捕获矿点 | 34.8% | **43.5%** | — |

> 优化后的"高异常 1.0% 面积含 30.4% 矿点"与论文的"2.4% 面积含 31.5% 矿点"
> 在量级上高度一致，说明方法迁移成功。

---

## 4. 目录结构

```
.
├── config.py                  # ★ 唯一需要修改的地方：Data_Path + 全部超参数
├── run_all.py                 # ★ 总控脚本：一条命令从头跑到尾（全自动）
├── requirements.txt
├── README.md
├── docs/                      # ★ 代码分析文档 + PyCharm 运行指南
│   ├── README.md              #   文档索引
│   ├── 优化说明.md            #   诊断 + 改进 + 实验证据
│   ├── PyCharm运行指南.md     #   从零跑通工程
│   └── 01_config.md ... 12_step07_benchmark.md
├── src/
│   ├── common.py              # 公共工具（日志/栅格IO/归一化/目录定位）
│   ├── crbm.py                # CRBM 模型
│   ├── autoencoder.py         # 深度自编码器（PyTorch）
│   ├── step01_prepare_data.py
│   ├── step02_select_features.py
│   ├── step03_pretrain_crbm.py
│   ├── step04_finetune.py
│   ├── step05_anomaly_map.py
│   ├── step06_evaluate.py
│   └── step07_benchmark.py
└── outputs/                   # 运行产生：图、表、模型、栅格
    ├── RESULTS.md  results_index.json  run_all.log
    ├── anomaly_map*.tif  anomaly_classes.tif
    └── figures/  tables/  models/  cache/
```

---

## 5. 数据要求

```
<Data_Path>/
├── 地化/                     # 39 个地化栅格，文件名即变量名
│   ├── Ag_CLR_IDW.tif
│   ├── ... （共 39 个，网格必须完全对齐）
│   └── Zr_CLR_IDW.tif
└── positive_IOCG.tif         # 已知矿点栅格，1 = 矿点（与地化网格对齐）
```

* 所有栅格必须**坐标系、范围、分辨率、行列数完全一致**；
* 矿点栅格也可放在 `Data_Path` 的上层目录，程序会自动向上搜索；
* 变量名从文件名主干自动提取（`Cu_CLR_IDW.tif` → `Cu`）。

---

## 6. 输出说明

| 输出 | 说明 |
| --- | --- |
| `outputs/RESULTS.md` | **先看这个**：一段话总结 + 全部关键指标 + 图表清单 |
| `outputs/anomaly_map.tif` | 多元地化异常图（选定得分模式） |
| `outputs/anomaly_map_mse.tif` / `_weighted.tif` | 论文原式 / 加权误差两种版本 |
| `outputs/anomaly_classes.tif` | 高(3)/中(2)/低(1) 三级异常分级 |
| `figures/step02_feature_auc.png` | 39 个变量的判别力排序（入选标红） |
| `figures/step03_pretrain_curves.png` | CRBM 逐层预训练学习曲线 |
| `figures/step04_finetune_curve.png` | 微调曲线 |
| `figures/step05_anomaly_map.png` | 异常图 + 已知矿点 + 得分分布 |
| `figures/step06_tvalue.png` | t 值—阈值曲线（含 1.96 显著性线） |
| `figures/step06_anomaly_classes.png` | **异常分级图 + 五角星 + 面积/矿点占比** |
| `figures/step06_roc_capture.png` | ROC 曲线 + 预测面积成功率曲线 |
| `figures/step07_benchmark.png` | 7 种方法 ROC-AUC 对比 |
| `tables/step06_deposit_pixels.csv` | 逐个矿点像元的得分、分位、所属级别 |
| `tables/step07_benchmark.csv` | 各基准方法定量指标 |

---

## 7. 常用参数（`config.py`）

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `Data_Path` | `D:\LaLa_Data` | 数据根目录（可用 `LALA_DATA_PATH` 覆盖） |
| `FEATURE_SELECTION_MODE` | `"auc"` | `auc` 自动 / `manual` 手工 / `none` 全部变量 |
| `N_SELECTED_FEATURES` | `12` | 挑选变量个数 |
| `HIDDEN_SIZES` | `"auto"` | `auto`=两层递减；写 `[20,10,5]` 复现论文 |
| `SCORE_MODE` | `"weighted"` | `weighted` 加权误差 / `mse` 论文原式 |
| `CRBM_EPOCHS` / `CRBM_LR` | 200 / 0.3 | CRBM 预训练轮数 / 学习率 |
| `FINETUNE_EPOCHS` / `FINETUNE_LR` | 200 / 0.8 | 微调轮数 / 学习率 |
| `RANDOM_SEEDS` | `[2026]` | 多个种子会训练多模型并取平均 |
| `MODERATE_ENRICHMENT_MIN` | 2.0 | 中异常下界的富集比门槛 |
| `STAR_SIZE` | 130 | 矿点五角星大小（看不清就调大） |

---

## 8. 基准测试结论（诚实说明）

| 方法 | ROC-AUC | t 极大值 | Top10% 捕获 |
| --- | --- | --- | --- |
| **deep_autoencoder（本项目）** | **0.766** | **14.18** | 43.5% |
| shallow_ae | 0.752 | 10.54 | 47.8% |
| single_element_cu | 0.738 | 9.99 | 34.8% |
| pca | 0.727 | 12.09 | 43.5% |
| mahalanobis | 0.703 | 7.05 | 34.8% |
| isolation_forest | 0.621 | 3.11 | 17.4% |
| ocsvm | 0.614 | 6.31 | 26.1% |

⚠️ 本区已知矿点仅 **23 个像元**，ROC-AUC 标准误约 ±0.06，
深度自编码器与浅层自编码器、单元素 Cu 的差距**在统计上不显著**。
论文中也未宣称深度模型大幅占优。建议在论文中表述为
"深度自编码器给出的异常图与已知矿点空间相关性最强（t = 14.18）"，
而非"显著优于其他方法"。

---

## 9. 引用

- Xiong, Y., Zuo, R., 2016. Recognition of geochemical anomalies using a deep
  autoencoder network. *Computers & Geosciences* 86, 165–172.
- Chen, H., Murray, A. F., 2003. Continuous restricted Boltzmann machine with an
  implementable training algorithm. *IEE Proceedings – Vision, Image and Signal
  Processing* 150(3), 153–158.
- Egozcue, J. J., et al., 2003. Isometric logratio transformations for compositional
  data analysis. *Mathematical Geology* 35(3), 279–300.
- Bonham-Carter, G. F., 1994. *Geographic Information Systems for Geoscientists.*
## 新增：空间独立验证与多随机种子基准

直接在IDE运行 **run_spatial_all.py**，或依次运行根目录step08、step09、step10。完整说明见[空间验证文档](docs/13_空间独立验证与多种子基准.md)。实际空间验证结果位于 **outputs_spatial/RESULTS_SPATIAL.md**，逐折、逐种子指标和全部模型已保存。旧outputs保持为原回顾性实验记录。
