# PhoCoNet 实验记录

| 实验 | 代码入口 | 设置 |
|---|---|---|
| 主实验 | `experiments/run_main.py` | 三个数据集，窗口长度 4、8、12、16、24，训练种子 2021–2025 |
| 信息集消融 | `experiments/run_information_sets.py` | T、TH、Q、TQ、THQ，三个数据集，训练种子 2021–2025 |
| 辅助变量噪声 | `experiments/run_sensor_noise.py` | SNR 为 100、50、30、20、10、5、0 dB，训练种子 2021，噪声种子 20210726 |
| 历史 TP 缺失 | `experiments/run_history_tp_missingness.py` | 缺失率 0、0.1、0.3、0.5、0.7，非零缺失率各 10 组掩码，训练种子 2021 |
| GradientSHAP | `experiments/run_shap.py` | 每站 435 个测试窗口、8 个训练背景、16 个积分点，训练种子 2021 |
| 仅 Q 的 MLP / LSTM | `experiments/run_q_only_baselines.py` | 三个数据集，训练种子 2021，各模型 3 次训练 |

## 数据与训练

模型入口为 `phoconet.PhoCoNet`。三个数据集为桔子洲、三角洲、捞刀河，包含 TP 和 WT、pH、DO、Cond、Turb、PI、AN、TN 八个辅助变量。T 表示历史 TP，H 表示历史辅助观测，Q 表示估计窗口内在线辅助观测。

历史窗口长度 96，分块长度 6。训练使用 Adam、MSE、batch=64、初始学习率 0.001、type1 学习率调度，最多 100 epochs，patience=10，按验证集选择检查点。

主实验先取前 `int(n*0.8)` 个时间点为训练验证集，再取其中前 `int(tv_end*0.875)` 个时间点训练。n=2286 时为训练 1599、验证 229、测试 458 点。信息集消融使用 `int(n*0.7)` 和 `int(n*0.8)` 两个边界。标准化统计量均只由对应训练集估计。

主实验、噪声和 MLP/LSTM 对照的 RMSE 为逐窗口 RMSE 均值。信息集实验的 `rmse_norm` 为全局 RMSE，缺失实验同时保存全局 RMSE 和逐窗口 RMSE 均值。

## 运行

```powershell
python -B scripts/prepare_data.py --source PATH_TO_STATION_CSV_DIRECTORY
python -B -m experiments.run_suite --run-id phoconet_20261027 --prepare-only
python -B -m experiments.run_suite --run-id phoconet_20261027 --execute
```

统一入口依次执行主实验、噪声、缺失、SHAP 和信息集消融。噪声、缺失和 SHAP 使用主实验 h24/seed2021 的检查点。

```powershell
python -B -m experiments.run_q_only_baselines --run-id q_only_20261027 --reference-run-id phoconet_20261027 --seed 2021
python -B -m experiments.analyze_q_only_baselines --run-id q_only_20261027
```

MLP 为逐时刻 8→64→64→1，GELU、dropout=0.1，共 4801 个参数。LSTM 为单层单向 8→64，使用 dropout=0.1 和 64→1 读出，共 19009 个参数，每个窗口重置隐状态。两种基线只读取 Q。

## 文件

结果保存在 `result/RUN_ID/`，包括配置、训练日志、检查点、逐窗口预测和指标。批次同时保存源代码快照、数据哈希、环境及执行命令。

[2026-10-27 实验结果](experiment_records/20261027/README.md)

```powershell
python -B -m pytest tests/test_phoconet.py tests/test_model_names.py tests/test_pipeline.py tests/test_q_only_baselines.py -q
python -B scripts/validate_results.py --run-id phoconet_20261027 --profile formal
python -B scripts/package_source.py --output output/PhoCoNet_source_20261027.zip
```
