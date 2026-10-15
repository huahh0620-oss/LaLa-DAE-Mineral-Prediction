# `src/common.py` 分析文档

## 1. 文件职责

**跨步骤复用的工具库**（不是"步骤"）。提供日志、随机性、栅格 IO、
归一化、目录定位、网络结构解析、矿点读取等公共能力。

## 2. 在流水线中的位置

```
common.py  ──被 step01~step07 全部导入──▶  提供基础能力
```

## 3. 函数清单

### 3.1 日志

| 函数 | 说明 |
| --- | --- |
| `get_logger(name)` | 返回带统一格式的单例 logger `log` |
| `add_file_logger(logger, path)` | 追加一个 UTF-8 文件处理器 |

文件开头还把 `sys.stdout / sys.stderr` 重新配置为 UTF-8，
**解决 Windows 控制台打印中文乱码**的问题。

### 3.2 随机性

| 函数 | 说明 |
| --- | --- |
| `set_seed(seed)` | 同时固定 `random`、`numpy`、`PYTHONHASHSEED`、`torch` |
| `resolve_device(spec)` | `auto` 时自动选 cuda/cpu |
| `limit_threads(n)` | 限制 OpenMP/MKL/torch 线程数，避免小数据被多线程拖慢 |

### 3.3 栅格 IO（核心）

| 函数 | 说明 |
| --- | --- |
| `read_aligned_stack(paths)` | 读取一组栅格 → `(bands, rows, cols)` cube + 元数据；**逐张校验网格是否对齐**，并把 nodata 转成 `NaN` |
| `write_raster` / `write_raster_like` | 写单波段 GeoTIFF（自动继承原 grid 的 transform/CRS，LZW 压缩） |
| `list_geochem_rasters(dir)` | 按文件名排序列出栅格 |
| `feature_name_from_path(path)` | `Cu_CLR_IDW.tif` → `Cu`（去掉 `_CLR_IDW`/`_CLR`/`_IDW` 后缀） |

`read_aligned_stack` 里对齐校验的判断条件：

```python
if (width != ref.width or height != ref.height
        or str(crs) != str(ref.crs) or transform != ref.transform):
    raise ValueError(...)
```

这一步很关键：**地化插值栅格最常见的问题就是网格没对齐**，
若不对齐会让"每个像元的 39 维向量"根本不存在。宁可尽早报错。

### 3.4 归一化（论文 Eq.13）

| 函数 | 说明 |
| --- | --- |
| `fit_minmax(X, lo, hi)` | 按列求 `min/max`，返回参数元组 |
| `apply_minmax(X, params)` | 线性拉伸到 `[lo, hi]` |
| `inverse_minmax` | 反归一化 |
| `zscore(X)` | 标准化（备用） |

拆成 fit/apply 两步，是为了保证**归一化参数可复用**（例如把新数据
用同一套参数变换），也便于把参数随数据集一起存进 `dataset.npz`。

### 3.5 目录定位（数据解耦的关键）

| 函数 | 说明 |
| --- | --- |
| `locate_geochem_dir(...)` | 依次尝试 `Data_Path/地化` → 递归搜索含 39 个 tif 的目录 |
| `locate_raster(...)` | 在 `Data_Path` 及其上若干级目录中查找指定文件名 |
| `locate_eval_raster(...)` | 评价用矿点栅格；`EVAL_RASTER` 为空时退回 `POSITIVE_RASTER` |

设计目的：用户把数据放在哪里都能跑，不必改代码。

### 3.6 网络结构与矿点读取

| 函数 | 说明 |
| --- | --- |
| `resolve_hidden_sizes(input_dim, spec)` | `"auto"` → 两个递减隐藏层（0.70n / 0.35n）；也可传显式列表 |
| `read_deposit_aligned(path, valid_mask, shape)` | 读矿点栅格 → 返回与有效像元一一对应的**布尔向量** |
| `read_deposit_grid(path, shape)` | 同上，但返回完整的 `(H, W)` 0/1 网格（**供绘图叠加用**） |

> ⚠️ 这两个矿点读取函数很容易混淆：
> `read_deposit_aligned` 返回长度 = 有效像元数（如 20289），
> `read_deposit_grid` 返回 `(206, 103)` 的完整网格。
> 计算指标用前者，画图用后者。混用会报
> `cannot reshape array of size 20289 into shape (206,103)`。

### 3.7 小工具

`ensure_dir`、`save_json`、`load_json`、`pretty_print_dict`。

## 4. 注意事项

1. `read_aligned_stack` 会把 `<= -3e38` 的值也当作 nodata —— 因为部分
   GIS 软件在写 float32 栅格时用 `-3.4e38` 表示无效值，而 `nodata` 标签可能丢失。
2. 所有归一化、插值产生的 `NaN` 会在写栅格时统一替换为 `-9999`。
3. `set_seed` 里对 torch 的调用做了 `try/except`，因此在没装 torch 的环境里
   也能导入本模块（例如只跑 step01）。