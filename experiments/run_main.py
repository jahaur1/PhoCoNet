"""Run PhoCoNet on three stations, five horizons, and five training seeds."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd

from experiments.config import (
    DATASETS,
    HORIZONS,
    MODEL_NAME,
    SEEDS,
    TARGET_CHANNEL,
    params_for,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "result" / "phoconet_main"
METRICS = ("mse_norm", "mae_norm", "rmse_norm")


def parse_report(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in list(csv.reader(handle))[1:]:
            if len(row) >= 3 and row[1] in METRICS:
                values[row[1]] = float(row[2])
    if set(values) != set(METRICS):
        raise ValueError(f"invalid report: {path}")
    return values


def valid_report(directory: Path) -> Path | None:
    candidates = sorted(
        directory.glob("test_report*.csv"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        try:
            parse_report(path)
            return path
        except (OSError, UnicodeError, csv.Error, ValueError):
            continue
    return None


def run_one(
    station: str,
    horizon: int,
    seed: int,
    epochs: int,
    gpu: int,
    force: bool,
) -> dict[str, object]:
    directory = OUTPUT / station / f"h{horizon}" / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    report = None if force else valid_report(directory)
    params = params_for(station, horizon, epochs)
    if report is None:
        command = [
            sys.executable,
            "-B",
            "-u",
            "scripts/run_benchmark.py",
            "--config-path",
            "rolling_forecast_config.json",
            "--data-name-list",
            f"{station}.csv",
            "--strategy-args",
            json.dumps(
                {"horizon": horizon, "target_channel": [TARGET_CHANNEL]}
            ),
            "--model-name",
            MODEL_NAME,
            "--model-hyper-params",
            json.dumps(params),
            "--seed",
            str(seed),
            "--deterministic",
            "full",
            "--save-true-pred",
            "true",
            "--gpus",
            str(gpu),
            "--num-workers",
            "1",
            "--timeout",
            "60000",
            "--save-path",
            directory.relative_to(ROOT / "result").as_posix(),
        ]
        (directory / "request.json").write_text(
            json.dumps(
                {
                    "model": "PhoCoNet",
                    "station": station,
                    "horizon": horizon,
                    "seed": seed,
                    "params": params,
                    "argv": command,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        environment = {
            **os.environ,
            "PYTHONHASHSEED": str(seed),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONIOENCODING": "utf-8",
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
        }
        print(f"[run] {station}/h{horizon}/seed_{seed}", flush=True)
        with (directory / "training.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode:
            raise RuntimeError(
                f"training failed: {station}/h{horizon}/seed_{seed}"
            )
        report = valid_report(directory)
        if report is None:
            raise RuntimeError(
                f"no report: {station}/h{horizon}/seed_{seed}"
            )
    else:
        print(f"[skip] {station}/h{horizon}/seed_{seed}", flush=True)
    values = parse_report(report)
    return {
        "model": "PhoCoNet",
        "station": station,
        "horizon": horizon,
        "seed": seed,
        **values,
        "report": str(report.relative_to(ROOT)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--horizons", nargs="+", type=int, choices=HORIZONS, default=HORIZONS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    matrix = [
        (seed, station, horizon)
        for seed in args.seeds
        for station in args.datasets
        for horizon in args.horizons
    ]
    for index, (seed, station, horizon) in enumerate(matrix, start=1):
        print(f"[progress] {index}/{len(matrix)}", flush=True)
        rows.append(
            run_one(
                station,
                horizon,
                seed,
                args.epochs,
                args.gpu,
                args.force,
            )
        )
        pd.DataFrame(rows).to_csv(
            OUTPUT / "metrics_partial.csv", index=False
        )

    frame = pd.DataFrame(rows).sort_values(["seed", "station", "horizon"])
    frame.to_csv(OUTPUT / "metrics_by_seed.csv", index=False)
    frame.groupby("seed", as_index=False)[list(METRICS)].mean().to_csv(
        OUTPUT / "macro_by_seed.csv", index=False
    )
    frame.groupby(["station", "horizon"], as_index=False).agg(
        mae_mean=("mae_norm", "mean"),
        mae_sd=("mae_norm", "std"),
        rmse_mean=("rmse_norm", "mean"),
        rmse_sd=("rmse_norm", "std"),
    ).to_csv(OUTPUT / "cell_mean_sd.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
