# -*- coding: utf-8 -*-
"""
总控脚本 / Master pipeline
====================================================================
一次性、全自动地从头跑到尾：

    step01 数据准备  -> step02 指示元素选择 -> step03 CRBM 预训练
    -> step04 微调   -> step05 异常图        -> step06 评价与分级
    -> step07 基准测试

用法：
    python run_all.py
    python run_all.py --data-path "D:\\LaLa_Data"
    python run_all.py --seeds 2026,2027,2028
    python run_all.py --feature-mode none      # 等价原论文（用全部变量）
    python run_all.py --skip-benchmark

全过程无需人工干预；任何一步失败都会记录日志并给出明确报错。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="拉拉式铜金矿深度自编码器地化异常识别（全自动总控）")
    p.add_argument("--data-path", default=None, help="数据根目录；等价于环境变量 LALA_DATA_PATH")
    p.add_argument("--output-dir", default=None, help="输出目录")
    p.add_argument("--seeds", default=None, help="逗号分隔的随机种子，例如 2026,2027,2028")
    p.add_argument("--positive-raster", default=None, help="矿点栅格文件名，例如 positive_IOCG.tif")
    p.add_argument("--feature-mode", default=None, choices=["auc", "manual", "none"],
                   help="变量选择方式：auc(默认) / manual / none(全部变量)")
    p.add_argument("--n-features", default=None, help="选择前 N 个变量")
    p.add_argument("--skip-benchmark", action="store_true", help="跳过 step07 基准测试")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.seeds:
        os.environ["LALA_SEEDS"] = args.seeds
    if args.data_path:
        os.environ["LALA_DATA_PATH"] = args.data_path
    if args.output_dir:
        os.environ["LALA_OUTPUT_DIR"] = args.output_dir
    if args.positive_raster:
        os.environ["LALA_POSITIVE_RASTER"] = args.positive_raster
    if args.feature_mode:
        os.environ["LALA_FEATURE_MODE"] = args.feature_mode
    if args.n_features:
        os.environ["LALA_N_FEATURES"] = args.n_features

    import config as C
    from src.common import add_file_logger, log

    log_path = Path(C.OUTPUT_DIR) / "run_all.log"
    add_file_logger(log, log_path)

    seeds = list(C.RANDOM_SEEDS)
    log.info("#" * 72)
    log.info("# 拉拉式铜金矿多元地化异常识别 —— 深度自编码器方法（Xiong & Zuo, 2016 改进版）")
    log.info("#" * 72)
    log.info("Data_Path      = %s", C.Data_Path)
    log.info("Output_DIR     = %s", C.OUTPUT_DIR)
    log.info("矿点栅格       = %s", C.POSITIVE_RASTER)
    log.info("变量选择       = %s (n=%s)", C.FEATURE_SELECTION_MODE, C.N_SELECTED_FEATURES)
    log.info("随机种子       = %s", seeds)
    log.info("设备           = %s", C.DEVICE)
    log.info("日志           = %s", log_path)

    t_all = time.time()
    summary, failed = {}, []

    from src import (step01_prepare_data, step02_select_features, step03_pretrain_crbm,
                     step04_finetune, step05_anomaly_map, step06_evaluate)

    steps = [("step01 数据准备", lambda: step01_prepare_data.run()),
             ("step02 指示元素选择", lambda: step02_select_features.run()),
             ("step03/04 预训练+微调", lambda: _train_all(step03_pretrain_crbm,
                                                          step04_finetune, seeds)),
             ("step05 异常图", lambda: step05_anomaly_map.run(seeds)),
             ("step06 评价与分级", lambda: step06_evaluate.run(seeds[0]))]

    if not args.skip_benchmark:
        from src import step07_benchmark
        steps.append(("step07 基准测试", lambda: step07_benchmark.run()))

    for name, fn in steps:
        log.info("")
        log.info(">>> %s", name)
        try:
            summary[name] = fn()
        except Exception as e:  # noqa: BLE001
            failed.append((name, e))
            log.error("步骤 [%s] 失败: %s", name, e)
            log.error(traceback.format_exc())
            if name.startswith("step01") or name.startswith("step02"):
                break

    try:
        _write_summary(summary, seeds, time.time() - t_all)
    except Exception as e:  # noqa: BLE001
        log.warning("汇总报告生成失败: %s", e)

    log.info("")
    log.info("#" * 72)
    if failed:
        log.warning("流程结束，但有 %d 个步骤失败：%s", len(failed), ", ".join(n for n, _ in failed))
    else:
        log.info("流程全部完成 ✓  总用时 %.1f s", time.time() - t_all)
    log.info("输出目录：%s", C.OUTPUT_DIR)
    log.info("#" * 72)
    return 1 if failed else 0


def _train_all(s3, s4, seeds):
    out = {}
    for s in seeds:
        out[f"step03_seed{s}"] = s3.run(s)
        out[f"step04_seed{s}"] = s4.run(s)
    return out


def _write_summary(summary: dict, seeds, elapsed: float) -> None:
    import json
    import config as C
    import pandas as pd
    from src.common import log

    def g(key):
        for k, v in summary.items():
            if key in k and isinstance(v, dict):
                return v
        return None

    idx = {
        "method": "Deep autoencoder (CRBM pre-training + fine-tuning), "
                  "with pathfinder-element selection",
        "reference": "Xiong, Y., Zuo, R., 2016. Recognition of geochemical anomalies "
                     "using a deep autoencoder network. Computers & Geosciences.",
        "data_path": str(C.Data_Path), "seeds": seeds,
        "elapsed_sec": round(elapsed, 2),
        "step_reports": {k: (v if isinstance(v, dict) else str(v)) for k, v in summary.items()},
        "outputs": {
            "anomaly_map": str(C.OUTPUT_DIR / "anomaly_map.tif"),
            "anomaly_map_normalized": str(C.OUTPUT_DIR / "anomaly_map_normalized.tif"),
            "anomaly_classes": str(C.OUTPUT_DIR / "anomaly_classes.tif"),
            "figures": sorted(str(p) for p in C.FIG_DIR.glob("*.png")),
            "tables": sorted(str(p) for p in C.TABLE_DIR.glob("*.csv")),
            "models": sorted(str(p) for p in C.MODEL_DIR.glob("*")),
        },
    }
    out = C.OUTPUT_DIR / "results_index.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(idx, f, ensure_ascii=False, indent=2, default=str)

    md = _markdown_summary(g("step01"), g("step02"), g("step06"), g("step07"), seeds, elapsed, C)
    (C.OUTPUT_DIR / "RESULTS.md").write_text(md, encoding="utf-8")
    log.info("汇总索引: %s；成果摘要: %s", out, C.OUTPUT_DIR / "RESULTS.md")


def _markdown_summary(s1, s2, s6, s7, seeds, elapsed, C) -> str:
    import pandas as pd
    L = ["# 运行成果摘要 / Results summary", "",
         f"- 数据目录: `{C.Data_Path}`", f"- 随机种子: `{seeds}`",
         f"- 总用时: {elapsed:.1f} s"]
    if s1:
        L += [f"- 地化变量总数: {s1.get('n_features')}",
              f"- 有效像元数: {s1.get('n_pixels_valid')}"]
    L.append("")
    if s2:
        L += ["## 指示元素选择", "",
              f"- 方式: `{s2.get('mode')}`；入选 {s2.get('n_selected')} / {s2.get('n_total_features')} 个变量",
              f"- 入选变量: {', '.join(s2.get('selected_features', []))}", ""]
    if s6:
        m = s6.get("metrics", {})
        L += ["## 深度自编码器异常识别", "",
              f"- ROC-AUC: **{m.get('roc_auc', float('nan')):.3f}**",
              f"- Student's t 极大值: **{s6.get('t_max', float('nan')):.2f}** (阈值 {s6.get('threshold_t_max', float('nan')):.3f})",
              f"- t=1.96 门槛: {s6.get('threshold_t_critical', float('nan')):.3f}",
              f"- 高异常区面积占比: {100 * m.get('high_class_area_fraction', 0):.2f}%",
              f"- 高异常区含已知矿点: {100 * m.get('high_class_deposit_fraction', 0):.1f}%",
              f"- 中+高异常含已知矿点: {100 * m.get('moderate_or_high_deposit_fraction', 0):.1f}%",
              f"- 矿点得分中位分位: {m.get('median_deposit_percentile', float('nan')):.2f}"]
        for pct in (1, 5, 10, 20):
            k = f"capture_top{pct}pct"
            if k in m:
                L.append(f"- Top {pct}% 面积矿点捕获率: {100 * m[k]:.1f}%")
        L.append("")
    if s7 and s7.get("results"):
        df = pd.DataFrame(s7["results"]).sort_values("roc_auc", ascending=False)
        L += ["## 基准对比 (ROC-AUC)", "",
              "| 方法 | ROC-AUC | t_max | Top10% 捕获 |", "| --- | --- | --- | --- |"]
        for _, r in df.iterrows():
            L.append(f"| {r['method']} | {r['roc_auc']:.3f} | {r['t_max']:.2f} | "
                     f"{100 * r['capture_top10pct']:.1f}% |")
        L.append("")
    L += ["## 图表清单", ""]
    for p in sorted(C.FIG_DIR.glob("*.png")):
        L.append(f"- `figures/{p.name}`")
    for p in sorted(C.TABLE_DIR.glob("*.csv")):
        L.append(f"- `tables/{p.name}`")
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    sys.exit(main())