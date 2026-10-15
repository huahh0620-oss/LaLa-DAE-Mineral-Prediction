# PyCharm 运行指南 / Run this project in PyCharm

本指南带你从零把工程在 PyCharm 里跑通一遍。全程约 15 分钟（含环境安装），
之后每次完整运行只需约 45 秒。

---

## 0. 前置条件

| 项 | 要求 |
| --- | --- |
| 操作系统 | Windows 10/11（本工程在 Windows PowerShell 下开发验证） |
| Python | **3.10 ~ 3.13**（已在 Python 3.13.7 上验证通过） |
| PyCharm | Community 版即可（不需要 Professional） |
| 内存 | ≥ 8 GB（本区数据 206×103×39，非常小） |
| 磁盘 | ≥ 2 GB（主要给 PyTorch） |
| 数据 | 一个数据根目录，内含 `地化/`（39 个 tif）与矿点栅格 |

> GPU 不是必需的。本区数据只有约 2 万个像元、网络只有几千个参数，
> **CPU 比 GPU 更快**（GPU 启动开销占主导）。

---

## 1. 准备数据目录

把数据整理成下面这种结构（目录名可以自定义，但要记住绝对路径）：

```
D:\LaLa_Data\
├── 地化\
│   ├── Ag_CLR_IDW.tif
│   ├── Al2O3_CLR_IDW.tif
│   ├── ...                    ← 共 39 个，网格必须完全一致
│   └── Zr_CLR_IDW.tif
└── positive_IOCG.tif          ← 已知矿点栅格，1 = 矿点，与地化网格对齐
```

**检查清单**：

- [ ] 39 个 tif 的坐标系、范围、行列数完全相同（可用 QGIS 检查）
- [ ] 矿点栅格的网格与地化栅格一致，值为 1 的像元即已知矿点
- [ ] 目录路径中可以有中文，但**不要有空格**（本指南以 `D:\LaLa_Data` 为例）

---

## 2. 在 PyCharm 中打开工程

1. `File` → `Open...`
2. 选择工程根目录（含 `config.py`、`run_all.py`、`src/` 的那个文件夹）
3. 若提示 "Trust Project?"，选择 **Trust Project**

打开后左侧项目树应是：

```
矿产预测
├── config.py
├── run_all.py
├── requirements.txt
├── README.md
├── docs/          ← 你正在读的文档
├── src/
└── outputs/       ← 运行后产生
```

---

## 3. 配置 Python 解释器

### 3.1 推荐：用虚拟环境（干净、可复现）

1. `File` → `Settings`（或 `Ctrl+Alt+S`）
2. 左侧 `Project: <你的工程名>` → `Python Interpreter`
3. 点齿轮 → `Add...` → 选 `Virtualenv Environment`
4. `Location` 用工程下的 `.venv`，`Base interpreter` 选已安装的 Python 3.10–3.13
5. `OK` 创建

### 3.2 也可以直接用已有解释器

如果你机器上已装好 numpy/rasterio/torch，直接在解释器下拉框里选中它即可。

---

## 4. 安装依赖

### 方式 A：用 PyCharm 终端（推荐）

打开 PyCharm 底部的 `Terminal`（`Alt+F12`），确认提示符前有 `(.venv)`，然后：

```bash
pip install -r requirements.txt
```

国内网络建议加清华镜像，速度快很多：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 方式 B：用 conda

```bash
conda create -n lala python=3.11 -y
conda activate lala
conda install -c conda-forge numpy pandas scipy scikit-learn rasterio matplotlib -y
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 4.1 依赖清单（`requirements.txt`）

| 包 | 用途 | 在哪一步用 |
| --- | --- | --- |
| `numpy` `pandas` `scipy` | 数值计算、表格 | 全部 |
| `scikit-learn` | ROC-AUC、PCA、隔离森林、OCSVM | step02/06/07 |
| `rasterio` | 读写 GeoTIFF | step01/05/06 |
| `matplotlib` | 出图 | step02~07 |
| `torch` | 深度自编码器 | step04/05/07 |

### 4.2 安装自检

在终端运行：

```bash
python -c "import numpy, pandas, scipy, sklearn, rasterio, matplotlib, torch; print('all ok', torch.__version__)"
```

打印 `all ok 2.x.x` 即安装成功。**若某个包导入失败，先解决它再继续。**

> `rasterio` 安装失败最常见，多半是 pip 版本过旧：
> 先 `python -m pip install --upgrade pip`，再重装。

---

## 5. 配置运行参数（两种方式，任选一种）

### 方式 A：直接改 `config.py`（最直观，推荐先用这个）

打开 `config.py`，把第 1 节的路径改成你的数据目录：

```python
Data_Path = Path(os.environ.get("LALA_DATA_PATH", r"D:\LaLa_Data"))
                                                    ^^^^^^^^^^^^^^^
                                                    改成你的实际路径
```

如果你把矿点文件放在别处，再改：

```python
POSITIVE_RASTER = os.environ.get("LALA_POSITIVE_RASTER", "positive_IOCG.tif")
```

### 方式 B：在 PyCharm 运行配置里设环境变量（不改代码）

1. `Run` → `Edit Configurations...`
2. 左上 `+` → `Python`
3. 填写：

| 字段 | 值 |
| --- | --- |
| `Name` | `run_all` |
| `Script path` | `<工程根目录>\run_all.py` |
| `Parameters` | （留空；见第 6 节的可选参数） |
| `Working directory` | **`<工程根目录>`** ← 必须，否则找不到 `src` |
| `Python interpreter` | 第 3 步选好的解释器 |

4. 展开 `Environment variables`，点右侧图标，添加：

```
LALA_DATA_PATH=D:\LaLa_Data
LALA_POSITIVE_RASTER=positive_IOCG.tif
```

> ⚠️ **`Working directory` 必须设成工程根目录**。本工程用
> `import config` / `from src import ...` 的方式导入模块，
> 如果工作目录不对会报 `ModuleNotFoundError: No module named 'config'`。

---

## 6. 运行整个工程（一键跑完）

### 6.1 用 PyCharm 运行

1. 在项目树里右键 `run_all.py` → `Run 'run_all'`
   （或打开 `run_all.py` 后按 `Ctrl+Shift+F10`）
2. 下方 `Run` 窗口会滚动输出日志

**成功的标志**（最后几行）：

```
[INFO] 流程全部完成 ✓  总用时 43.2 s
[INFO] 输出目录：<工程根目录>\outputs
```

### 6.2 用终端运行（等价）

```bash
cd <工程根目录>
python run_all.py
```

### 6.3 可选参数

| 需求 | 命令 |
| --- | --- |
| 指定数据目录 | `python run_all.py --data-path "D:\LaLa_Data"` |
| 多种子集成（更稳健，耗时 ×N） | `python run_all.py --seeds 2026,2027,2028` |
| 严格复现论文（用全部 39 个变量） | `python run_all.py --feature-mode none` |
| 改选前 16 个变量 | `python run_all.py --n-features 16` |
| 换矿点样本 | `python run_all.py --positive-raster positive.tif` |
| 跳过基准测试（省约 15 秒） | `python run_all.py --skip-benchmark` |

在 PyCharm 里把这些参数填到运行配置的 `Parameters` 字段即可。

### 6.4 预期耗时

| 步骤 | 耗时（CPU，本区数据） |
| --- | --- |
| step01 数据准备 | < 1 s |
| step02 指示元素选择 | ≈ 2 s |
| step03 CRBM 预训练 | ≈ 7 s |
| step04 微调 | ≈ 19 s |
| step05 异常图 | < 1 s |
| step06 评价与分级 | ≈ 1 s |
| step07 基准测试 | ≈ 15 s |
| **合计** | **≈ 45 s** |

---

## 7. 运行后检查成果

在项目树里展开 `outputs/`：

```
outputs/
├── RESULTS.md                 ← 先看这个：一段话总结 + 指标表
├── run_all.log                ← 完整日志（排错用）
├── results_index.json
├── anomaly_map.tif            ← 多元地化异常图（主结果）
├── anomaly_map_mse.tif        ← 论文原式误差版本
├── anomaly_map_weighted.tif   ← 加权误差版本
├── anomaly_classes.tif        ← 高/中/低分级
├── cache/                     ← 中间数据（dataset.npz 等）
├── figures/                   ← 8 张图
│   ├── step02_feature_auc.png
│   ├── step03_pretrain_curves.png
│   ├── step04_finetune_curve.png
│   ├── step05_anomaly_map.png
│   ├── step06_anomaly_classes.png     ← 含五角星的异常分级图
│   ├── step06_roc_capture.png
│   ├── step06_tvalue.png
│   └── step07_benchmark.png
├── models/                    ← 训练好的模型
└── tables/                    ← 11 个 CSV/JSON 结果表
```

**在 PyCharm 里看图**：直接双击 `figures/*.png`，PyCharm 会内嵌预览。

**重点看三张图**：

| 图 | 该看什么 |
| --- | --- |
| `step06_anomaly_classes.png` | 五角星（已知矿点）是否落在红色/橙色区；右侧柱状图看"高异常面积小但矿点占比高" |
| `step05_anomaly_map.png` | 异常得分的连续分布；右侧直方图看矿点中位数是否明显偏右 |
| `step07_benchmark.png` | 本方法（红条）与其他 6 种方法的 ROC-AUC 对比 |

**核对五角星对不对得上**：打开 `tables/step06_deposit_pixels.csv`，
逐行看每个矿点像元的 `percentile` 与 `class`。
本项目优化后：23 个矿点里 7 个落在 high、10 个落在 moderate，
中位分位 0.89。

---

## 8. 只运行某一步（调试用）

每个 `stepXX_*.py` 都有 `if __name__ == "__main__": run()`，
可以单独运行。**在 PyCharm 里右键文件 → `Run`**，或：

```bash
python -m src.step01_prepare_data     # 必须从工程根目录运行
python -m src.step02_select_features
python -m src.step03_pretrain_crbm
python -m src.step04_finetune
python -m src.step05_anomaly_map
python -m src.step06_evaluate
python -m src.step07_benchmark
```

⚠️ 注意步骤间的依赖：

```
step01 ─▶ step02 ─▶ step03 ─▶ step04 ─▶ step05 ─▶ step06 ─▶ step07
  数据      变量      预训练      微调      异常图     评价      基准
```

单独运行某一步时，**它依赖的前置步骤必须已经跑过**。
例如改了网络结构后要重跑 step03 和 step04，step05 才能用新模型。

---

## 9. 常见报错与解决

| 报错 | 原因 | 解决 |
| --- | --- | --- |
| `ModuleNotFoundError: No module named 'config'` | 工作目录不对 | 把运行配置的 `Working directory` 设为工程根目录 |
| `FileNotFoundError: 数据目录不存在：D:\LaLa_Data` | `Data_Path` 不对 | 改 `config.py` 或设 `LALA_DATA_PATH` 环境变量 |
| `FileNotFoundError: 未找到地化目录` | 数据目录结构不符 | 确认 `地化/` 下有 39 个 tif；或设 `AUTO_LOCATE_GEOCHEM=True` 自动搜索 |
| `ValueError: 栅格 xxx.tif 与参考网格不对齐` | 某个 tif 坐标系/范围不一致 | 用 QGIS 重采样对齐后再跑 |
| `ValueError: 矿点栅格网格 (a,b) 与地化网格 (c,d) 不一致` | 矿点栅格没对齐 | 把矿点栅格重采样到与地化栅格一致 |
| `FileNotFoundError: 未找到栅格 positive_IOCG.tif` | 矿点文件不在搜索范围内 | 把它放到 `Data_Path` 下，或用 `LALA_POSITIVE_RASTER` 指定 |
| `cannot reshape array of size 20289 into shape (206,103)` | 把"有效像元向量"当成"完整网格"用了 | 检查是否误用 `read_deposit_aligned` 的返回值去画图（画图要用 `read_deposit_grid`） |
| 控制台中文显示成乱码 | Windows 控制台默认 GBK | 本工程已在 `src/common.py` 里把 stdout 重设为 UTF-8；若仍乱码，在运行配置里加环境变量 `PYTHONIOENCODING=utf-8` |
| `ModuleNotFoundError: No module named 'rasterio'` | 依赖没装全 | `pip install -r requirements.txt` |
| `pip` 安装 rasterio 失败 | pip 过旧或网络问题 | `python -m pip install --upgrade pip`，再用国内镜像重装 |
| 训练出现 `train-MSE` 先降后升 | CRBM 训练发散 | 确认 `CRBM_LEARN_A=False`；不要随意调大 `CRBM_LR`/`CRBM_MOMENTUM` |
| 图上五角星太小 / 看不清 | 显示尺寸参数偏小 | 调大 `config.STAR_SIZE`（默认 130，可试 200） |
| 出图中文变成方框 | matplotlib 中文字体缺失 | 本工程图件**全部使用英文标注**，正常情况不会出现；若你自己加了中文标注，需设置 `plt.rcParams['font.sans-serif']` |

---

## 10. 改动前后对比的操作方式

| 目标 | 操作 |
| --- | --- |
| 本项目默认（推荐，已优化） | 什么都不改，直接跑 |
| 严格复现论文（39 变量 + 原式误差 + 3 层折半） | `--feature-mode none`；并把 `config.SCORE_MODE` 改成 `"mse"`、`config.HIDDEN_SIZES` 改成 `[20, 10, 5]` |
| 只换矿点样本 | `--positive-raster positive.tif` |
| 用知识驱动的 IOCG 元素组合 | `config.FEATURE_SELECTION_MODE = "manual"` |
| 更稳健（多种子集成） | `--seeds 2026,2027,2028`（耗时约 2.5 分钟） |
| 只跑主流程、不要基准 | `--skip-benchmark` |

---

## 11. 一页速查卡

```bash
# 1) 建环境
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 2) 改数据路径（二选一）
#    a. 编辑 config.py 的 Data_Path
#    b. set LALA_DATA_PATH=D:\LaLa_Data

# 3) 一键运行
python run_all.py

# 4) 看结果
#    outputs/RESULTS.md
#    outputs/figures/step06_anomaly_classes.png
#    outputs/tables/step06_deposit_pixels.csv
```

PyCharm 里对应：`File → Open` 工程 → `Settings → Python Interpreter` 选 `.venv`
→ `Run → Edit Configurations` 新建 Python 配置（Script=`run_all.py`,
Working directory=工程根目录, Environment variables=`LALA_DATA_PATH=...`）
→ 点绿色三角运行。