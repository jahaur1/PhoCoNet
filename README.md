# PhoCoNet

PhoCoNet estimates total phosphorus using historical TP, historical auxiliary measurements, and online auxiliary observations.

[实验代码与运行记录](EXPERIMENTS.md)

```powershell
python -m pip install -r requirements.txt
python -B scripts/prepare_data.py --source PATH_TO_STATION_CSV_DIRECTORY
python -B -m experiments.run_suite --run-id phoconet_20261027 --execute
```
