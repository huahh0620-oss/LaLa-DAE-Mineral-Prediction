# `src/step07_benchmark.py` 分析文档

## 1. 文件职责

**步骤 7：基准测试**。把本方法（CRBM 预训练 + 微调的深度自编码器）与
多种常用方法放在**同一套变量、同一评价协议**下对比，回答
"深度自编码器是否真的更好"。

## 2. 对比的 7 种方法

| 方法 | 归属 | 异常得分定义 |
| --- | --- | --- |
| `deep_autoencoder` | 本项目 | 加权重构误差（或 `SCORE_MODE` 指定） |
| `shallow_ae` | 自编码器族 | 单隐层（瓶颈 5 维）自编码器的重构误差 |
| `pca` | 线性降维 | PCA 前 k 个主成分的重构误差 |
| `mahalanobis` | 统计 | 到数据中心点的马氏距离 |
| `isolation_forest` | 集成学习 | `−score_samples` |
| `ocsvm` | 单类分类 | `−decision_function` |
| `single_element_cu` | 传统单变量 | Cu 元素归一化值 |

## 3. 评价协议（与 step06 完全一致）

每个方法产生一个得分向量后，统一计算：

* ROC-AUC、Average Precision
* Student's t 极大值及对应阈值
* Top 1% / 5% / 10% / 20% 面积矿点捕获率
* 矿点得分中位分位、矿点/背景均值比

`evaluate_score()` 与 `scan_thresholds()` 直接从 `step06_evaluate` 导入，
**保证主方法与基准方法用的是同一套指标定义**。

## 4. 关键实现细节

### 4.1 复用已训练的深度模型

```python
def score_deep_autoencoder(X, seed, deposit):
    if seed == C.RANDOM_SEEDS[0] and (CACHE_DIR/"anomaly_scores.npz").exists():
        return np.load(...)["score_raw"]        # 直接复用 step05 的结果
    ...                                          # 否则现场训练
```

避免重复训练，也保证基准表里的 `deep_autoencoder` 与主结果**完全一致**。

### 4.2 各基准方法的实现要点

| 方法 | 实现 | 注意 |
| --- | --- | --- |
| `score_pca` | `PCA(n_components=k)` 后 `inverse_transform` | k 取 `min(5, D-1)` |
| `score_mahalanobis` | 协方差逆矩阵 + `einsum` | 需加 `1e-6` 对角正则，否则奇异 |
| `score_isolation_forest` | `IsolationForest(n_estimators=300)` | 取负号使"越大越异常" |
| `score_ocsvm` | `OneClassSVM(kernel="rbf", nu=0.05)` | **O(n²) 很慢**，只随机抽 5000 个样本拟合 |
| `score_shallow_ae` | `Linear(D→5)→Sigmoid→Linear(5→D)→Sigmoid` | 与深度模型同优化器/学习率，唯一区别是"只有一层隐层" |
| `score_single_element` | 直接取 Cu 那一列 | 找不到 Cu 时退回第 0 列 |

## 5. 本区实测结果

| 方法 | ROC-AUC | t 极大值 | Top10% 捕获 |
| --- | --- | --- | --- |
| **deep_autoencoder** | **0.766** | **14.18** | 43.5% |
| shallow_ae | 0.752 | 10.54 | 47.8% |
| single_element_cu | 0.738 | 9.99 | 34.8% |
| pca | 0.727 | 12.09 | 43.5% |
| mahalanobis | 0.703 | 7.05 | 34.8% |
| isolation_forest | 0.621 | 3.11 | 17.4% |
| ocsvm | 0.614 | 6.31 | 26.1% |

**结论与诚实说明**：

1. 深度自编码器在 AUC 与 t 值上**排名第一**，且 **t 值（14.18）明显领先**
   —— 说明它识别出的高异常区对矿点的富集能力最强。
2. 但浅层自编码器（0.752）与单元素 Cu（0.738）紧随其后，
   **23 个矿点带来的统计不确定性（AUC 标准误约 ±0.06）使这些差距不显著**。
3. 论文本身也报告深度自编码器与 CRBM 的结果"相似"，并未宣称深度模型大幅占优。
4. 因此论文中应表述为"深度自编码器给出的异常图与已知矿点具有最强的空间
   相关性（t = 14.18）"，而**不要**写成"显著优于其他方法"。

## 6. 关键函数

| 函数 | 作用 |
| --- | --- |
| `score_*` 系列 | 各方法的异常得分计算 |
| `METHOD_FUNCS` | 方法名 → 计算函数的映射表 |
| `evaluate_score(score, deposit)` | 统一评价 |
| `run()` | 步骤主入口 |
| `_plot(df, roc_store)` | 左：ROC 对比曲线；右：AUC 柱状图（本方法标红） |

## 7. 注意事项

1. OCSVM 会对 2 万个样本做 `decision_function` 预测，虽然不是训练但也不快；
   若追求速度可只对有效像元抽样评价。
2. 所有基准方法都使用**同一套变量子集**（`load_input_matrix()`），
   否则对比不公平。
3. 若要加入新方法，只需写一个 `score_xxx(X, names, seed, deposit)` 函数
   并注册到 `METHOD_FUNCS` 与 `config.BENCHMARK_METHODS`。