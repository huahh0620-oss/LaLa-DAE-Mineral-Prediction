# `run_all.py` 分析文档

## 1. 文件职责

**总控脚本**。一条命令按顺序驱动 step01 → step07，全过程无人工介入；
任何一步失败都会被捕获、写日志，并在最后统一汇报。

## 2. 在流水线中的位置

```
run_all.py
   ├─ step01 数据准备
   ├─ step02 指示元素选择
   ├─ step03 CRBM 预训练      ┐ 对 RANDOM_SEEDS 中每个种子各跑一次
   ├─ step04 微调             ┘
   ├─ step05 异常图（多种子时做集成）
   ├─ step06 评价与分级
   └─ step07 基准测试（可用 --skip-benchmark 跳过）
```

## 3. 命令行参数

| 参数 | 等价环境变量 | 说明 |
| --- | --- | --- |
| `--data-path` | `LALA_DATA_PATH` | 数据根目录 |
| `--output-dir` | `LALA_OUTPUT_DIR` | 输出目录 |
| `--seeds` | `LALA_SEEDS` | 逗号分隔，如 `2026,2027,2028` |
| `--positive-raster` | `LALA_POSITIVE_RASTER` | 矿点栅格文件名 |
| `--feature-mode` | `LALA_FEATURE_MODE` | `auc` / `manual` / `none` |
| `--n-features` | `LALA_N_FEATURES` | 选择前 N 个变量 |
| `--skip-benchmark` | — | 跳过 step07，节省约 15 秒 |

## 4. 核心逻辑

### 4.1 参数覆盖顺序（`main()` 开头）

```
命令行参数  →  环境变量  →  config.py 默认值
```

命令行参数通过写回 `os.environ` 生效，因此**必须在 `import config` 之前**完成，
代码正是这样安排的（先 `os.environ[...] = ...`，再 `import config as C`）。

### 4.2 步骤注册表

```python
steps = [("step01 数据准备", lambda: step01_prepare_data.run()), ...]
```

每一步统一封装成 `(名称, 无参函数)`，便于循环执行、统一记日志、统一捕获异常。

### 4.3 容错策略

* `step01` / `step02` 失败 → **立即中断**（后续步骤依赖它们的数据与变量选择结果）；
* 其余步骤失败 → 记录错误后继续（例如基准测试失败不影响主结果）。

### 4.4 输出汇总

结束后生成两份文件：

| 文件 | 内容 |
| --- | --- |
| `outputs/results_index.json` | 机器可读的完整索引（含每一步的 report） |
| `outputs/RESULTS.md` | **人类可读**的成果摘要：核心指标表 + 基准对比表 + 图表清单 |

### 4.5 辅助函数

| 函数 | 作用 |
| --- | --- |
| `_train_all(s3, s4, seeds)` | 对每个种子依次跑预训练与微调 |
| `_write_summary(...)` | 写 `results_index.json` 与 `RESULTS.md` |
| `_markdown_summary(...)` | 组装 Markdown 摘要文本 |

## 5. 日志

* 屏幕输出：带时间戳的 INFO 日志（`src/common.py` 中的 `log`）；
* 文件输出：`outputs/run_all.log`（通过 `add_file_logger` 追加）。

排查问题时**优先看日志文件**，它包含每一步的完整输出。

## 6. 退出码

| 退出码 | 含义 |
| --- | --- |
| 0 | 全部步骤成功 |
| 1 | 有步骤失败（失败清单在日志末尾） |

在 PyCharm 中运行后，可直接看运行窗口最后的 `流程全部完成 ✓` 或 `有 N 个步骤失败`。

## 7. 注意事项

1. 必须**在工程根目录**运行（`python run_all.py`），因为 `src` 是包、`config` 是顶层模块。
2. `--seeds` 给多个种子会让总耗时近似线性增长（每种子约 25 秒 CPU 训练）。
3. `run_all.py` 不做数据校验以外的任何人工确认，适合放进计划任务自动跑。