# `src/step03_pretrain_crbm.py` 分析文档

## 1. 文件职责

**步骤 3：CRBM 逐层无监督预训练（贪心逐层）**。对应论文 **2.1 节**。

把若干个 CRBM 依次堆叠：第 1 层在归一化后的输入上学权重，其隐层输出
作为第 2 层的"可见层"输入，依此类推，为深度自编码器提供初始权重。

## 2. 输入 / 输出

| 输入 | 说明 |
| --- | --- |
| `cache/dataset.npz` | step01 的样本矩阵 |
| `cache/selected_features.npz` | step02 的变量选择结果 |

| 输出 | 内容 |
| --- | --- |
| `models/pretrained_crbms_seed{seed}.npz` | 各层 CRBM 的 W / bv / bh / a_v / a_h |
| `models/pretrain_curves_seed{seed}.csv` | 各层逐 epoch 的重构 MSE |
| `figures/step03_pretrain_curves.png` | 预训练学习曲线（对数纵轴） |
| `tables/step03_report_seed{seed}.json` | 报告 |

## 3. 核心流程

```python
X, names, idx = load_input_matrix()          # 取变量子集
hidden_sizes  = resolve_hidden_sizes(...)    # "auto" → [8, 4]
reps = X.copy()
for li, (n_in, n_out) in enumerate(zip(dims[:-1], dims[1:]), start=1):
    m = CRBM(CRBMConfig(n_visible=n_in, n_hidden=n_out, seed=seed+li, ...))
    m.fit(reps, epochs=200, batch_size=256)
    reps = m.transform(reps)                 # 无噪隐层输出 → 下一层的输入
```

### 3.1 逐层堆叠的要点

* 每层用**独立的随机种子**（`seed + li`），保证层间初始化不重复；
* `m.transform()` 必须是无噪声版本，否则噪声会逐层累积放大；
* 每层重新把 `reps` 作为可见层数据，因此第 2、3 层的输入取值范围仍是 [0,1]。

### 3.2 关键公共函数 `load_input_matrix()`

```python
def load_input_matrix():
    ds = np.load(CACHE_DIR / "dataset.npz")
    X, names = ds["X_norm"], ds["feature_names"]
    idx = list(range(len(names))) if C.FEATURE_SELECTION_MODE == "none" \
          else load_selected_indices()
    return X[:, idx], [names[i] for i in idx], idx
```

**step03 / step04 / step05 / step07 都调用它**，从而保证四个步骤
使用的是**同一套变量子集** —— 这是避免"训练用 12 个变量、评价用 39 个变量"
这类低级错误的关键设计。

## 4. 关键函数

| 函数 | 作用 |
| --- | --- |
| `load_input_matrix()` | 读取并按 step02 结果裁剪变量子集（**被多个步骤复用**） |
| `run(seed)` | 步骤主入口：逐层训练 + 保存 + 出图 |
| `_plot_curves(curves, seed)` | 绘制各层学习曲线 |

## 5. 保存的 npz 字段

| 键 | 说明 |
| --- | --- |
| `layer{i}_W` | 第 i 层权重，形状 `(n_in, n_out)` |
| `layer{i}_bv` / `layer{i}_bh` | 偏置 |
| `layer{i}_a_v` / `layer{i}_a_h` | 噪声控制参数 |
| `n_layers` | 层数（step04 据此加载） |
| `input_dim` / `hidden_sizes` | 结构记录 |

## 6. 训练曲线怎么读

正常情况：每层 MSE 在前 20–40 个 epoch 快速下降，之后趋于平稳并缓慢下降，
曲线在对数坐标下呈"L"形。本项目实测：

| 层 | 输入→输出 | 200 epoch 后 MSE |
| --- | --- | --- |
| 1 | 12 → 8 | ≈ 0.0040 |
| 2 | 8 → 4 | ≈ 0.0030 |

若曲线出现**先降后升**，说明训练发散 —— 检查是否误开了
`CRBM_LEARN_A` 或把 `CRBM_LR`/`CRBM_MOMENTUM` 调得过大。

## 7. 注意事项

1. 层数与 `HIDDEN_SIZES` 必须一致；改网络结构后**必须重跑本步骤**，
   否则 step04 会因 npz 层数不匹配而报错。
2. 每层的 `n_visible` 由上一层决定，改动 `N_SELECTED_FEATURES` 会连锁改变全部层。
3. 训练耗时与样本量、层数、epoch 数成正比；本区 20289 个像元、2 层结构
   在纯 CPU 上约 7 秒。