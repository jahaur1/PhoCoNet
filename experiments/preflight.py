"""Check the formal pipeline environment and data before starting training."""
from __future__ import annotations
import importlib
import json
import platform
import sys
from pathlib import Path

def check() -> dict:
    names = ("torch", "numpy", "pandas", "sklearn", "scipy", "einops", "matplotlib", "shap")
    versions = {}
    for name in names:
        module = importlib.import_module(name)
        versions[name] = str(getattr(module, "__version__", "unknown"))
    from experiments.config import DATASETS, TARGET_CHANNEL
    from experiments.information_value_core import AUXILIARIES, load_frame
    from experiments.paths import ROOT
    from ts_benchmark.data.utils import read_data
    from experiments import run_main, run_sensor_noise, run_history_tp_missingness, run_shap, run_information_sets
    import torch
    from ts_benchmark.data.data_source import LocalForecastingDataSource
    source = LocalForecastingDataSource()
    datasets = {}
    for station in DATASETS:
        file = ROOT / "dataset" / "forecasting" / (station + ".csv")
        frame = load_frame(file, "full8")
        benchmark = read_data(str(file))
        if benchmark.columns[TARGET_CHANNEL] != "TP":
            raise ValueError(f"{station}: benchmark target channel is not TP")
        if list(benchmark.drop(columns="TP").columns) != list(AUXILIARIES):
            raise ValueError(f"{station}: auxiliary channel order differs between runners")
        if not benchmark.loc[:, frame.columns].equals(frame):
            raise ValueError(f"{station}: benchmark and controlled data readers disagree")
        windows = len(frame) - int(len(frame) * 0.8) - 24 + 1
        if windows != 435:
            raise ValueError(f"{station}: historical-missingness protocol expects 435 windows, got {windows}")
        datasets[station] = {"rows": len(frame), "test_windows_h24": windows, "columns": list(frame.columns)}
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "versions": versions, "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "datasets": datasets}

if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
