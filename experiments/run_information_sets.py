"""Run T, TH, Q, TQ, and THQ with the locked PhoCoNet backbone."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import information_value_core as base
from experiments.information_value_normalization import QNormControlledModel
from ts_benchmark.baselines.phoconet.models.phoconet import (
    Model as PhoCoNetModel,
)

OUTPUT = ROOT / "result" / "phoconet_information_sets"


def model_config(channels: int, history: int, horizon: int, width: int):
    config = base.model_config(channels, history, horizon, width, patch=6)
    config.attn_mode = "none"
    config.ia_layers = 0
    config.ca_layers = 0
    config.q_variable_attention = False
    config.q_aligned_variable_attention = False
    config.q_aligned_variable_residual = False
    config.q_aligned_variable_modulation = True
    config.q_aligned_modulation_dim = 32
    config.q_aligned_modulation_hidden = 32
    config.q_aligned_modulation_max = 0.25
    config.q_aligned_modulation_output_scale = 0.01
    config.late_aggregation = "attention"
    config.late_dim = 32
    config.late_heads = 4
    config.late_hidden = 32
    config.late_output_scale = 1e-2
    config.late_gate_init = 0.1
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", choices=base.STATIONS,
        default=list(base.STATIONS),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[2021, 2022, 2023, 2024, 2025])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument(
        "--data-dir", type=Path,
        default=ROOT / "dataset" / "forecasting",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.threads <= 0 or args.epochs <= 0 or args.batch_size <= 0:
        parser.error("threads, epochs, and batch size must be positive")
    if len(set(args.datasets)) != len(args.datasets):
        parser.error("duplicate datasets")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("duplicate seeds")
    return args


def summarize(rows: list[dict], output: Path) -> None:
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "metrics_by_run.csv", index=False)
    summary = (
        frame.groupby(["station", "condition"], as_index=False)
        .agg(
            n_seeds=("seed", "count"),
            mse_mean=("mse_norm", "mean"),
            mse_sd=("mse_norm", "std"),
            mae_mean=("mae_norm", "mean"),
            mae_sd=("mae_norm", "std"),
            rmse_mean=("rmse_norm", "mean"),
            rmse_sd=("rmse_norm", "std"),
        )
    )
    summary.to_csv(output / "summary.csv", index=False)

    indexed = {
        (row.station, int(row.seed), row.condition): row
        for row in frame.itertuples(index=False)
    }
    paired_rows = []
    for station in base.STATIONS:
        for seed in sorted(frame.seed.unique()):
            for contrast, (before, after) in base.CONTRASTS.items():
                left = indexed[station, int(seed), before]
                right = indexed[station, int(seed), after]
                for metric in ("mse_norm", "mae_norm", "rmse_norm"):
                    reduction = getattr(left, metric) - getattr(right, metric)
                    paired_rows.append(
                        {
                            "station": station,
                            "seed": int(seed),
                            "contrast": contrast,
                            "metric": metric,
                            "before": getattr(left, metric),
                            "after": getattr(right, metric),
                            "reduction": reduction,
                            "reduction_pct": 100.0 * reduction / getattr(left, metric),
                        }
                    )
    pd.DataFrame(paired_rows).to_csv(output / "paired_contrasts.csv", index=False)

    macro = (
        frame.groupby("condition", as_index=False)[
            ["mse_norm", "mae_norm", "rmse_norm"]
        ]
        .mean()
        .set_index("condition")
        .reindex(base.CONDITIONS)
    )
    lines = [
        "# PhoCoNet controlled information-value experiment",
        "",
        "T = historical TP; H = historical auxiliary variables; Q = current "
        "online auxiliary observations. Every condition is independently trained "
        "from an identical initialization within each station and seed.",
        "",
        "Protocol: Full-8, H=96, F=24, period=6, no backbone attention, "
        "QNorm safe normalization, MSE training.",
        "",
        "## Per-station results",
        "",
        "| Station | Information | MSE | MAE | RMSE |",
        "|---|---|---:|---:|---:|",
    ]
    for station in base.STATIONS:
        subset = frame.loc[frame.station.eq(station)]
        for condition in base.CONDITIONS:
            group = subset.loc[subset.condition.eq(condition)]
            lines.append(
                f"| {station} | {condition} | {group.mse_norm.mean():.6f} "
                f"| {group.mae_norm.mean():.6f} | {group.rmse_norm.mean():.6f} |"
            )
    lines.extend(
        [
            "",
            "## Macro averages",
            "",
            "| Information | MSE | MAE | RMSE |",
            "|---|---:|---:|---:|",
        ]
    )
    for condition, row in macro.iterrows():
        lines.append(
            f"| {condition} | {row.mse_norm:.6f} | {row.mae_norm:.6f} "
            f"| {row.rmse_norm:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Macro information contrasts",
            "",
            "Positive reduction means the added information lowers error.",
            "",
            "| Contrast | MSE reduction | MAE reduction | RMSE reduction |",
            "|---|---:|---:|---:|",
        ]
    )
    for contrast, (before, after) in base.CONTRASTS.items():
        reductions = []
        for metric in ("mse_norm", "mae_norm", "rmse_norm"):
            value = 100.0 * (
                macro.loc[before, metric] - macro.loc[after, metric]
            ) / macro.loc[before, metric]
            reductions.append(value)
        lines.append(
            f"| {contrast}: {before} to {after} | {reductions[0]:+.2f}% "
            f"| {reductions[1]:+.2f}% | {reductions[2]:+.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Evidence boundary",
            "",
            "This is a five-seed structural experiment. It quantifies training-seed "
            "variability, but the three stations are not a population sample. The metrics use "
            "global block RMSE from the controlled information-value framework and "
            "must not be pooled numerically with the benchmark's mean-window RMSE.",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def main() -> int:
    args = parse_args()
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else args.device
    )
    torch.set_num_threads(args.threads)
    history, horizon, width = 96, 24, 128
    run_args = argparse.Namespace(
        lr=args.lr,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
    )
    settings = {
        "datasets": list(args.datasets),
        "seeds": list(args.seeds),
        "protocol": "full8",
        "normalization": "safe-revin-qnorm",
        "backbone": "PhoCoNet",
        "history": history,
        "horizon": horizon,
        "period": 6,
        "width": width,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "lr": args.lr,
    }
    manifest = {
        "settings": settings,
        "resolved_device": str(device),
        "torch_version": torch.__version__,
        "conditions": {key: asdict(value) for key, value in base.CONDITIONS.items()},
        "datasets": {},
    }
    prepared = {}
    for station in args.datasets:
        path = args.data_dir / f"{station}.csv"
        data = base.load_frame(path, "full8")
        origins = base.split_origins(len(data), history, horizon)
        mean, scale = base.fit_scaler(data.to_numpy(), int(len(data) * 0.7))
        values = ((data.to_numpy() - mean) / scale).astype(np.float32)
        prepared[station] = (
            values,
            base.calendar_marks(data.index),
            origins,
            mean,
            scale,
            data.index,
        )
        manifest["datasets"][station] = {
            "sha256": base.file_hash(path),
            "rows": len(data),
            "columns": list(data.columns),
            "alpha_init": base.ALPHA_INIT[station],
        }

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError("output manifest differs; use a new directory")
    else:
        if args.summarize_only or any(args.output.iterdir()):
            raise ValueError("expected empty output or matching manifest")
        base.write_json(manifest_path, manifest)

    rows: list[dict] = []
    total = len(args.datasets) * len(args.seeds) * len(base.CONDITIONS)
    index = 0
    for station in args.datasets:
        values, marks, origins, mean, scale, dates = prepared[station]
        for seed in args.seeds:
            initial_hash = None
            for condition, information in base.CONDITIONS.items():
                index += 1
                directory = args.output / station / f"seed_{seed}" / condition
                completion = directory / "completed.json"
                if completion.exists():
                    completed = json.loads(completion.read_text(encoding="utf-8"))
                    fingerprint = completed["initial_state_sha256"]
                    print(f"[resume {index}/{total}] {station}/{seed}/{condition}", flush=True)
                else:
                    if args.summarize_only:
                        raise ValueError(f"incomplete matrix: {directory}")
                    print(f"[train {index}/{total}] {station}/{seed}/{condition}", flush=True)
                    directory.mkdir(parents=True, exist_ok=True)
                    base.seed_everything(seed)
                    config = model_config(values.shape[1], history, horizon, width)
                    model = QNormControlledModel(
                        config,
                        information,
                        "safe-revin",
                        base.ALPHA_INIT[station],
                        backbone_cls=PhoCoNetModel,
                    ).to(device)
                    fingerprint = base.state_hash(model)
                    training = base.Windows(values, marks, origins["train"], history, horizon)
                    validation = base.Windows(values, marks, origins["validation"], history, horizon)
                    training_result = base.fit(
                        model,
                        base.loader(training, args.batch_size, seed, True),
                        base.loader(validation, args.batch_size, seed),
                        run_args,
                        device,
                        directory,
                    )
                    test = base.Windows(values, marks, origins["test"], history, horizon)
                    actual, predicted, positions = base.predict(
                        model,
                        base.loader(test, args.batch_size, seed),
                        device,
                    )
                    np.savez_compressed(
                        directory / "predictions_clean.npz",
                        actual_norm=actual,
                        predicted_norm=predicted,
                        actual_raw=actual * scale[0] + mean[0],
                        predicted_raw=predicted * scale[0] + mean[0],
                        origins=positions,
                        target_timestamps=dates.asi8[
                            positions[:, None] + np.arange(horizon)
                        ],
                    )
                    metrics = {
                        "station": station,
                        "seed": seed,
                        "condition": condition,
                        "test_windows": len(actual),
                        "parameters": sum(p.numel() for p in model.parameters()),
                        **training_result,
                        **base.error_metrics(actual, predicted, float(scale[0])),
                    }
                    completed = {
                        "initial_state_sha256": fingerprint,
                        "artifacts": ["best.pt", "training.csv", "predictions_clean.npz"],
                        "metrics": metrics,
                    }
                    base.write_json(completion, completed)
                if initial_hash is not None and initial_hash != fingerprint:
                    raise RuntimeError("paired conditions have different initial weights")
                initial_hash = fingerprint
                rows.append(completed["metrics"])
                pd.DataFrame(rows).to_csv(args.output / "metrics_partial.csv", index=False)
    summarize(rows, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
