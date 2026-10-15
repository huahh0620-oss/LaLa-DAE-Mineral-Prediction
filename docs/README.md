# 文档索引 / Documentation index

本目录是「拉拉式铜金矿多元地化异常识别（深度自编码器方法）」工程的配套分析文档。
每个主要代码文件对应一份分析文档，另有两份专题文档。

## 阅读顺序建议

| 序号 | 文档 | 对应代码 | 说明 |
| --- | --- | --- | --- |
| 00 | [优化说明.md](优化说明.md) | — | 相对原论文做了哪些改进、为什么、效果如何 |
| 01 | [01_config.md](01_config.md) | `config.py` | 全局配置与参数含义 |
| 02 | [02_run_all.md](02_run_all.md) | `run_all.py` | 总控脚本：一条命令跑完全流程 |
| 03 | [03_common.md](03_common.md) | `src/common.py` | 公共工具函数库 |
| 04 | [04_crbm.md](04_crbm.md) | `src/crbm.py` | CRBM 连续受限玻尔兹曼机 |
| 05 | [05_autoencoder.md](05_autoencoder.md) | `src/autoencoder.py` | 深度自编码器（PyTorch） |
| 06 | [06_step01_prepare_data.md](06_step01_prepare_data.md) | `src/step01_prepare_data.py` | 步骤 1：数据准备 |
| 07 | [07_step02_select_features.md](07_step02_select_features.md) | `src/step02_select_features.py` | 步骤 2：指示元素选择 |
| 08 | [08_step03_pretrain_crbm.md](08_step03_pretrain_crbm.md) | `src/step03_pretrain_crbm.py` | 步骤 3：CRBM 逐层预训练 |
| 09 | [09_step04_finetune.md](09_step04_finetune.md) | `src/step04_finetune.py` | 步骤 4：展开与微调 |
| 10 | [10_step05_anomaly_map.md](10_step05_anomaly_map.md) | `src/step05_anomaly_map.py` | 步骤 5：异常图生成 |
| 11 | [11_step06_evaluate.md](11_step06_evaluate.md) | `src/step06_evaluate.py` | 步骤 6：评价与分级 |
| 12 | [12_step07_benchmark.md](12_step07_benchmark.md) | `src/step07_benchmark.py` | 步骤 7：基准测试 |
| 13 | [PyCharm运行指南.md](PyCharm运行指南.md) | — | 在 PyCharm 中从零跑通工程的完整流程 |

## 方法一句话概括

把 39 个地化栅格的每个像元看作一个 39 维样本；先挑出对铜金矿化敏感的元素，
再用 **CRBM 逐层预训练 + 反向传播微调的深度自编码器**学习背景样本的多元结构；
重构误差大的像元即为多元地化异常；最后用 **Student's t 值**与已知矿点做空间
相关性检验并分级（Xiong & Zuo, 2016）。