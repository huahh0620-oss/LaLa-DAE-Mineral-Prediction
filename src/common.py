# -*- coding: utf-8 -*-
"""
公共工具 / Shared utilities
====================================================================
日志、随机种子、栅格读写、目录定位、归一化等跨步骤复用的函数。
本文件不是"步骤"，而是被 step01~step06 共同调用的工具库。
"""
from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------
# 让 Windows 控制台也能打印中文
# ---------------------------------------------------------------------
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass


# ---------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------
_LOGGER_NAME = "lala"


def get_logger(name: str = "lala", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)-7s %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


log = get_logger()


def add_file_logger(logger: logging.Logger, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)


# ---------------------------------------------------------------------
# 随机性
# ---------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def resolve_device(spec: str = "auto"):
    import torch
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def limit_threads(num_threads: int) -> None:
    if num_threads and num_threads > 0:
        os.environ.setdefault("OMP_NUM_THREADS", str(num_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(num_threads))
        try:
            import torch
            torch.set_num_threads(num_threads)
        except Exception:
            pass


# ---------------------------------------------------------------------
# 栅格 IO
# ---------------------------------------------------------------------
def read_aligned_stack(paths: Sequence[Path]) -> Tuple[np.ndarray, dict]:
    """
    读取一组栅格，要求网格完全对齐，返回 (cube, meta)。

    cube: (bands, rows, cols) float32，nodata 已替换为 np.nan
    meta: 包含 profile / transform / crs / nodata / shape 等
    """
    import rasterio

    profiles = []
    arrays: List[np.ndarray] = []
    for p in paths:
        with rasterio.open(p) as ds:
            profiles.append(ds.profile.copy())
            arr = ds.read(1).astype("float32")
            nd = ds.nodata
            if nd is not None:
                arr = np.where(arr == nd, np.nan, arr)
            # 数据中可能用极大负值当 nodata
            arr[arr <= -3.0e38] = np.nan
            arrays.append(arr)

    ref = profiles[0]
    for p, prof in zip(paths, profiles[1:]):
        if (prof["width"] != ref["width"] or prof["height"] != ref["height"]
                or str(prof.get("crs")) != str(ref.get("crs"))
                or prof["transform"] != ref["transform"]):
            raise ValueError(f"栅格 {p.name} 与参考网格不对齐，请先做重采样/对齐。")

    cube = np.stack(arrays, axis=0)
    meta = {
        "profile": ref,
        "transform": ref["transform"],
        "crs": str(ref.get("crs")),
        "nodata": ref.get("nodata"),
        "shape": (ref["height"], ref["width"]),
        "bounds": tuple(rasterio.transform.array_bounds(ref["height"], ref["width"], ref["transform"])),
    }
    return cube, meta


def write_raster(path: Path, array: np.ndarray, profile: dict, nodata: float = -9999.0) -> None:
    import rasterio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prof = dict(profile)
    prof.update(driver="GTiff", dtype="float32", count=1, nodata=nodata,
                compress="lzw")
    out = array.astype("float32")
    out = np.where(np.isfinite(out), out, nodata)
    with rasterio.open(path, "w", **prof) as ds:
        ds.write(out, 1)


def write_raster_like(path: Path, array: np.ndarray, meta: dict,
                      nodata: float = -9999.0) -> None:
    write_raster(path, array, meta["profile"], nodata=nodata)


# ---------------------------------------------------------------------
# 归一化（论文 Eq.13：线性拉伸到 [0,1]）
# ---------------------------------------------------------------------
def fit_minmax(X: np.ndarray, lo: float = 0.0, hi: float = 1.0):
    """按列拟合 min-max 参数。返回 (min, max, lo, hi)。"""
    xmin = np.nanmin(X, axis=0)
    xmax = np.nanmax(X, axis=0)
    span = xmax - xmin
    span[span == 0] = 1.0
    return xmin, xmax, lo, hi


def apply_minmax(X: np.ndarray, params) -> np.ndarray:
    xmin, xmax, lo, hi = params
    span = xmax - xmin
    span = np.where(span == 0, 1.0, span)
    return lo + (X - xmin) / span * (hi - lo)


def inverse_minmax(Xn: np.ndarray, params) -> np.ndarray:
    xmin, xmax, lo, hi = params
    return xmin + (Xn - lo) / (hi - lo) * (xmax - xmin)


def zscore(X: np.ndarray, axis: int = 0):
    mu = np.nanmean(X, axis=axis, keepdims=True)
    sd = np.nanstd(X, axis=axis, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    return (X - mu) / sd, (mu, sd)


# ---------------------------------------------------------------------
# 目录定位（把代码与数据解耦的关键）
# ---------------------------------------------------------------------
def locate_geochem_dir(data_path: Path, subdir: str, glob_pat: str = "*.tif",
                       auto: bool = True, expect: int | None = None) -> Path:
    direct = Path(data_path) / subdir
    if direct.is_dir() and any(direct.glob(glob_pat)):
        return direct
    if not auto:
        raise FileNotFoundError(f"未找到地化目录：{direct}")
    # 递归搜索：选取含 tif 数量最多的目录
    best, best_n = None, 0
    for p in Path(data_path).rglob("*"):
        if p.is_dir():
            n = len(list(p.glob(glob_pat)))
            if expect is not None and n == expect:
                return p
            if n > best_n:
                best, best_n = p, n
    if best is None:
        raise FileNotFoundError(f"在 {data_path} 下未找到任何地化栅格。")
    return best


def locate_raster(data_path: Path, name: str, search_parents: int = 3) -> Path:
    data_path = Path(data_path)
    direct = data_path / name
    if direct.exists():
        return direct
    for p in data_path.rglob(name):
        return p
    for parent in list(data_path.parents)[:max(0, search_parents)]:
        cand = parent / name
        if cand.exists():
            return cand
        for p in parent.rglob(name):
            return p
    raise FileNotFoundError(
        f"未找到栅格 {name}（已搜索 {data_path} 及其上 {search_parents} 级目录）。")


def list_geochem_rasters(geochem_dir: Path, glob_pat: str = "*.tif") -> List[Path]:
    files = sorted(p for p in Path(geochem_dir).glob(glob_pat)
                   if not p.name.startswith("."))
    return files


def feature_name_from_path(p: Path) -> str:
    """Ag_CLR_IDW.tif -> Ag；保留文件名主干作为后备。"""
    stem = Path(p).stem
    for suffix in ("_CLR_IDW", "_CLR", "_IDW"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


# ---------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------
def ensure_dir(p: Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pretty_print_dict(d: dict, title: str = "") -> None:
    if title:
        log.info(title)
    for k, v in d.items():
        log.info("    %-28s %s", k, v)

# ---------------------------------------------------------------------
# 网络结构与矿点读取辅助
# ---------------------------------------------------------------------
def resolve_hidden_sizes(input_dim: int, spec) -> list:
    """把 config.HIDDEN_SIZES 解析成具体的隐藏层宽度列表。

    spec = "auto" 时采用两个递减隐藏层（约输入的 70% 和 35%）：
        12 个变量 -> [8, 4]     39 个变量 -> [27, 14]

    这是在本区数据上对 3 种深度（1/2/3 个隐藏层）× 多种宽度做对比实验后
    选出的稳健结构（见 docs/优化说明.md）：2 层递减结构的 ROC-AUC 与
    t 值在 3 个随机种子和两套矿点样本上都优于 3 层折半结构。

    若要严格复现论文的 3 层折半结构，把 config.HIDDEN_SIZES 显式写成
    如 [20, 10, 5] 即可。
    """
    if isinstance(spec, str) and spec.strip().lower() == "auto":
        h1 = max(3, int(round(input_dim * 0.70)))
        h2 = max(2, int(round(input_dim * 0.35)))
        h2 = min(h2, h1 - 1) if h1 > 2 else 2
        return [h1, h2]
    return [int(v) for v in spec]


def locate_eval_raster(data_path: Path, eval_name: str, positive_name: str,
                       search_parents: int = 3) -> Path:
    """评价用矿点栅格：EVAL_RASTER 为空则退回 POSITIVE_RASTER。"""
    name = eval_name or positive_name
    return locate_raster(Path(data_path), name, search_parents)


def read_deposit_aligned(path: Path, valid_mask: np.ndarray,
                         expect_shape: tuple | None = None) -> np.ndarray:
    """读取矿点栅格，返回与研究区有效像元一一对应的布尔向量。

    path         矿点栅格路径
    valid_mask   (H, W) 或 (H*W,) 的布尔掩膜
    expect_shape 可选 (H, W)，用于校验网格是否一致
    """
    import rasterio

    valid_mask = np.asarray(valid_mask).astype(bool)
    with rasterio.open(path) as ds:
        grid = ds.read(1)
        nod = ds.nodata
        if expect_shape is not None and grid.shape != tuple(expect_shape):
            raise ValueError(
                f"矿点栅格 {Path(path).name} 网格 {grid.shape} 与地化网格 {expect_shape} 不一致。")
    if nod is not None:
        grid = np.where(grid == nod, 0, grid)
    grid = np.where(np.isfinite(grid), grid, 0)
    flat = grid.reshape(-1)
    return (flat[valid_mask.reshape(-1)] == 1)
def read_deposit_grid(path: Path, expect_shape: tuple | None = None) -> np.ndarray:
    """读取矿点栅格，返回完整 (H, W) 的 0/1 网格（供绘图用）。"""
    import rasterio

    with rasterio.open(path) as ds:
        grid = ds.read(1)
        nod = ds.nodata
        if expect_shape is not None and grid.shape != tuple(expect_shape):
            raise ValueError(
                f"矿点栅格 {Path(path).name} 网格 {grid.shape} 与地化网格 {expect_shape} 不一致。")
    if nod is not None:
        grid = np.where(grid == nod, 0, grid)
    grid = np.where(np.isfinite(grid), grid, 0)
    return (grid == 1)