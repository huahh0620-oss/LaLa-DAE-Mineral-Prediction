# -*- coding: utf-8 -*-
"""
全局配置 / Global configuration
====================================================================
拉拉式铜金矿（川滇地区）多元地化异常识别
复现并改进 Xiong & Zuo (2016), "Recognition of geochemical anomalies using
a deep autoencoder network", Computers & Geosciences.

设计原则：
  1. 数据位置与代码完全解耦：只通过下面的 ``Data_Path`` 变量（或环境变量
     ``LALA_DATA_PATH``）指定，代码本身不写死任何数据绝对路径。
  2. 所有可调超参数集中在此文件。
  3. 步骤脚本 (src/step*.py) 只从本文件读取配置。
"""
from __future__ import annotations

import os
from pathlib import Path

# ====================================================================
# 1. 数据路径（唯一需要用户修改的地方）
# ====================================================================
# 优先级：环境变量 LALA_DATA_PATH > 下面的默认值
# 例：  set LALA_DATA_PATH=D:\LaLa_Data     (Windows CMD)
#       $env:LALA_DATA_PATH="D:\LaLa_Data"  (PowerShell)
Data_Path = Path(os.environ.get("LALA_DATA_PATH", "./LaLa_Data"))

# 数据目录布局（相对 Data_Path，均可自定义）：
#   <Data_Path>/地化/*.tif                39 个地化栅格（CLR-IDW）
#   <Data_Path>/positive_IOCG.tif         已知矿点样本栅格（1=矿点）
#   <Data_Path>/mask.tif                  可选：研究区掩膜（非 0 有效）
GEOCHEM_SUBDIR = os.environ.get("LALA_GEOCHEM_SUBDIR", "地化")
GEOCHEM_GLOB = "*.tif"
POSITIVE_RASTER = os.environ.get("LALA_POSITIVE_RASTER", "positive_IOCG.tif")
MASK_RASTER = os.environ.get("LALA_MASK_RASTER", "")  # 空=不使用掩膜

# 当 Data_Path 下找不到 <Data_Path>/地化 时，退而递归搜索包含 39 个 tif 的目录
AUTO_LOCATE_GEOCHEM = True
# 矿点栅格也可放在 Data_Path 上层目录时自动向上搜索
POSITIVE_SEARCH_PARENTS = 3

# ====================================================================
# 2. 输出
# ====================================================================
OUTPUT_DIR = Path(os.environ.get(
    "LALA_OUTPUT_DIR", str(Path(__file__).resolve().parent / "outputs")))
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
MODEL_DIR = OUTPUT_DIR / "models"
CACHE_DIR = OUTPUT_DIR / "cache"

for _d in (OUTPUT_DIR, FIG_DIR, TABLE_DIR, MODEL_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ====================================================================
# 3. 数据预处理
# ====================================================================
# 数据已是 CLR（中心对数比）变换结果，闭合效应已处理；
# 若换成原始浓度数据，把 RAW_COMPOSITION=True 会自动做 ilr。
RAW_COMPOSITION = bool(int(os.environ.get("LALA_RAW_COMPOSITION", "0")))

# 归一化方式：论文 Eq.(13) 把每个变量线性拉伸到 [0,1]
NORMALIZE_RANGE = (0.0, 1.0)

# ====================================================================
# 4. 矿化指示元素选择（本项目相对原论文的改进点 ①）
# ====================================================================
# 论文直接使用全部变量。但在本区 39 个变量中，部分元素（如 Sr、Zr、Be、Th）
# 与拉拉式铜金矿化呈负相关，混入自编码器会稀释多元异常信号。
# 因此先按"单元素判别力(AUC)"挑选对矿化最敏感的元素，再送入自编码器。
#
#   "auc"    : 按单元素 AUC 自动挑选前 N_SELECTED_FEATURES 个（默认）
#   "manual" : 使用 MANUAL_FEATURES 列表
#   "none"   : 使用全部变量（等价原论文做法）
FEATURE_SELECTION_MODE = os.environ.get("LALA_FEATURE_MODE", "auc")
N_SELECTED_FEATURES = int(os.environ.get("LALA_N_FEATURES", "12"))
# 只保留 AUC >= 该阈值的元素（0.5 = 优于随机；可剔除反指示元素）
SELECTION_MIN_AUC = float(os.environ.get("LALA_MIN_AUC", "0.5"))
# 相关性去冗余：与已选元素 |r| 超过该值的候选被跳过（0=不去冗余）
SELECTION_MAX_CORR = float(os.environ.get("LALA_MAX_CORR", "0.90"))

# 拉拉式铜金矿（IOCG 型）知识驱动指示元素组合（供 "manual" 模式参考）
MANUAL_FEATURES = [
    "Cu", "Au", "Ag", "Mo", "F", "Ba", "As", "Sb", "Bi", "Hg",
    "Co", "Ni", "Fe2O3", "Mn", "Na2O", "P", "U", "La",
]

# ====================================================================
# 5. 深度自编码器网络结构（论文 4.1 节：隐藏层逐层折半效果最好）
# ====================================================================
HIDDEN_SIZES = "auto"      # "auto"=两个递减隐藏层（约 0.7n 与 0.35n）
                           # 也可写死，如 [20, 10, 5] 复现论文 3 层折半结构
ACTIVATION = "sigmoid"     # 论文使用带噪声控制的 sigmoid
OUTPUT_ACTIVATION = "sigmoid"

# ====================================================================
# 6. CRBM 预训练（论文 2.1 节 / Eq.2-7）
# ====================================================================
CRBM_EPOCHS = 200
CRBM_LR = 0.3
CRBM_LR_NOISE = 0.01
CRBM_MOMENTUM = 0.9
CRBM_SIGMA = 0.2
CRBM_THETA_L = 0.0
CRBM_THETA_H = 1.0
CRBM_WEIGHT_INIT_STD = 0.01
CRBM_BATCH_SIZE = 256
CRBM_WEIGHT_DECAY = 1e-5
CRBM_PRETRAIN_PATIENCE = 0
# 是否按 Eq.(6) 学习噪声控制参数 a。默认关闭：在本数据上 a 会单调增大到上界
# 并使 sigmoid 饱和，导致重构误差发散；固定 a=1 时训练稳定。
CRBM_LEARN_A = bool(int(os.environ.get("LALA_CRBM_LEARN_A", "0")))

# ====================================================================
# 7. 微调（论文 2.2 节 / Eq.8-11）
# ====================================================================
FINETUNE_EPOCHS = 200
FINETUNE_LR = 0.8
FINETUNE_MOMENTUM = 0.9
FINETUNE_WEIGHT_DECAY = 1e-5
FINETUNE_BATCH_SIZE = 256
FINETUNE_OPTIMIZER = "sgd"

# ====================================================================
# 8. 重复实验与随机性
# ====================================================================
# 多个种子会分别训练模型，并把归一化异常得分取平均（集成），更稳健。
RANDOM_SEEDS = [int(s) for s in
                os.environ.get("LALA_SEEDS", "2026").split(",") if s.strip()]
DEVICE = os.environ.get("LALA_DEVICE", "auto")   # auto | cpu | cuda
NUM_THREADS = int(os.environ.get("LALA_THREADS", "0"))

# ====================================================================
# 9. 异常分级与评价（论文 4.2 节 / t-value）
# ====================================================================
T_CRITICAL = 1.96                   # 95% 显著性（t 值门槛，用于参考对比）
N_THRESHOLD_CANDIDATES = 100        # 阈值扫描点数
CAPTURE_PERCENTS = [1, 5, 10, 20]   # Top-X% 面积矿点捕获率

# 异常三级分级的阈值规则：
#   高异常 : score >= thr_max（t 值取极大处的阈值）
#   中异常 : thr_mod <= score < thr_max
#   低异常 : score < thr_mod
# 其中 thr_mod 取"富集比(enrichment = 矿点捕获率 / 面积占比) >= 该值"的最高阈值。
# 之所以不用论文的"t>=1.96 最低阈值"作为中异常下界：本区矿点仅 23 个，
# t>=1.96 对应的阈值过低，会把 70% 以上面积都划成中异常，失去分级意义。
MODERATE_ENRICHMENT_MIN = float(os.environ.get("LALA_ENRICH_MIN", "2.0"))

# ====================================================================
# 10. 异常得分模式（改进点 ③：加权重构误差）
# ====================================================================
#   "mse"      : 论文原式 RE = mean_j (x_j - r_j)^2            （严格复现）
#   "weighted" : RE_w = sum_j w_j (x_j - r_j)^2，w_j 来自各元素的
#                单元素判别力(AUC-0.5)归一化（默认，更能突出矿化指示元素）
SCORE_MODE = os.environ.get("LALA_SCORE_MODE", "weighted")

# ====================================================================
# 11. 制图（改进点 ②：让已知矿点与异常的空间关系可读）
# ====================================================================
# 评价时使用的矿点栅格（默认与 POSITIVE_RASTER 相同；可另设一套做外部校验）
EVAL_RASTER = os.environ.get("LALA_EVAL_RASTER", "")   # 空=用 POSITIVE_RASTER
FIGURE_DPI = 170
STAR_SIZE = 130                     # 矿点五角星散点尺寸
SHOW_ADMIN_BOUNDARY = False         # 如本地有研究区边界 shp 可置 True 叠加

# ====================================================================
# 12. 基准测试（benchmark）
# ====================================================================
BENCHMARK_METHODS = [
    "deep_autoencoder",
    "shallow_ae",
    "pca",
    "isolation_forest",
    "ocsvm",
    "single_element_cu",
    "mahalanobis",
]
BENCHMARK_SEEDS = [2026]