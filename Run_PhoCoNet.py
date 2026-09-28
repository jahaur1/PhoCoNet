"""Run the default PhoCoNet experiment at a 24-step estimation horizon."""

from __future__ import annotations

import json
import subprocess
import sys

from experiments.config import (
    DATASETS,
    MODEL_NAME,
    TARGET_CHANNEL,
    params_for,
)


def main() -> int:
    for dataset in DATASETS:
        command = [
            sys.executable,
            "scripts/run_benchmark.py",
            "--config-path",
            "rolling_forecast_config.json",
            "--data-name-list",
            f"{dataset}.csv",
            "--strategy-args",
            json.dumps({"horizon": 24, "target_channel": [TARGET_CHANNEL]}),
            "--model-name",
            MODEL_NAME,
            "--model-hyper-params",
            json.dumps(params_for(dataset, 24)),
            "--gpus",
            "0",
            "--num-workers",
            "1",
            "--timeout",
            "60000",
            "--save-path",
            f"{dataset}/PhoCoNet_TP",
            "--deterministic",
            "full",
        ]
        completed = subprocess.run(command, check=False)
        if completed.returncode:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
