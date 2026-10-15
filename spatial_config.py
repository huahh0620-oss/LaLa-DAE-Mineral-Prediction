"""空间嵌套验证配置。直接在IDE运行 run_spatial_all.py，无需命令行参数。"""
from pathlib import Path
import config as C
ROOT = Path(__file__).resolve().parent
DATA_PATH = C.Data_Path if C.Data_Path.is_absolute() else ROOT / C.Data_Path
OUTPUT = ROOT / 'outputs_spatial'
OUTER_FOLDS = 5
CLUSTER_RADIUS_PIXELS = 6.0  # DBSCAN相邻连接半径12km，簇为距离代理而非矿床认定
BUFFER_PIXELS = 3.0         # 训练像元到整个留出区最近像元距离必须>6km
SPLIT_SEED = 1701           # 固定空间区，所有方法/训练种子共用
SEEDS = [2026, 2027, 2028]
INNER_FOLDS = 2             # 每次外层训练区内再作2折空间验证
TUNING_SEED = 2026          # 内层调参固定种子，外层重复检验训练随机性
CRBM_EPOCHS = 40            # 本次预算，区别原回顾实验的200轮
FINETUNE_EPOCHS = 80
BATCH_SIZE = 512
MAX_FIT_PIXELS = 8000       # 每次只在可用训练区内固定抽样，所有方法共享
OCSVM_MAX_FIT = 3000
VARIANTS = [('deep_autoencoder',12),('shallow_ae',12),('pca',12),
            ('isolation_forest',12),('ocsvm',12),('mahalanobis',12),
            ('single_element_cu',1),('deep_autoencoder',26),('deep_autoencoder',39)]
# 仅使用内层验证AP选择，平局按此预先固定顺序。禁用外层测试选参。
GRIDS = {'deep_autoencoder':[{'lr':0.2},{'lr':0.8}],
         'shallow_ae':[{'lr':0.2},{'lr':0.8}],
         'pca':[{'components':3},{'components':5}],
         'isolation_forest':[{'max_samples':256},{'max_samples':1024}],
         'ocsvm':[{'nu':0.02},{'nu':0.10}],
         'mahalanobis':[{'shrinkage':0.05},{'shrinkage':0.2}],
         'single_element_cu':[{}]}
