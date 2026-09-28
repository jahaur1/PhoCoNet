"""Evaluate PhoCoNet under SNR-controlled auxiliary-sensor noise.

Gaussian noise is added at test time only to online auxiliary variables in
the historical and estimation windows. Verified historical TP remains clean.

The noise standard deviation follows

    sigma_noise / sigma_signal = 10 ** (-SNR_dB / 20).

Default outputs
---------------
result/phoconet_sensor_noise/
    sensor_noise_metrics.csv       one row per model/dataset/SNR/noise seed
    sensor_noise_summary.csv       mean and standard deviation across seeds
    sensor_noise_table.md          compact exact-value table
    sensor_noise_table.tex         reference-style LaTeX table
    experiment_manifest.json       complete evaluation configuration

Examples
--------
Run from the repository root with
``python -m experiments.run_sensor_noise``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from experiments.config import (
    BASE_PARAMS as PHOCONET_BASE_PARAMS,
    DATASETS,
    DATASET_ALPHA_INIT,
    MODEL_LABEL,
    MODEL_NAME,
    TARGET_CHANNEL,
    validate_params,
)

ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "result"
SAVE_FAMILY = "phoconet_sensor_noise"
OUTPUT_DIR = RESULT_ROOT / SAVE_FAMILY
TP_CHANNEL = TARGET_CHANNEL

DATASET_LABELS = {
    "juzizhou": "Juzizhou",
    "sanjiaozhou": "Sanjiaozhou",
    "laodaohe": "Laodaohe",
}
DEFAULT_SNR_DB = (100.0, 50.0, 30.0, 20.0, 10.0, 5.0, 0.0)

BASE_PARAMS = {**PHOCONET_BASE_PARAMS, "input_missing_rate": 0.0}

MODEL_CONFIGS = {
    MODEL_LABEL: {
        "model_name": MODEL_NAME,
        "adapter": None,
        "runner": "scripts/run_benchmark.py",
        "uses_phoconet_params": True,
    },
}
DEFAULT_MODELS = (MODEL_LABEL,)

REQUIRED_METRICS = ("mse_norm", "mae_norm", "rmse_norm")


@dataclass(frozen=True)
class NoiseRecord:
    model: str
    dataset: str
    snr_db: float
    noise_std_ratio: float
    train_seed: int
    noise_seed: int
    mse_norm: float
    mae_norm: float
    rmse_norm: float
    report_path: str


@dataclass(frozen=True)
class NoiseSummary:
    model: str
    dataset: str
    snr_db: float
    noise_std_ratio: float
    n_noise_seeds: int
    mse_mean: float
    mse_std: float
    mae_mean: float
    mae_std: float
    rmse_mean: float
    rmse_std: float
    mse_change_pct: float
    mae_change_pct: float
    rmse_change_pct: float


def snr_to_noise_ratio(snr_db: float) -> float:
    """Convert amplitude SNR in dB to noise-standard-deviation ratio."""
    if not math.isfinite(snr_db):
        raise ValueError(f"SNR must be finite, got {snr_db!r}")
    return 10.0 ** (-snr_db / 20.0)


def snr_token(snr_db: float) -> str:
    """Return a path-safe and stable representation of an SNR value."""
    if float(snr_db).is_integer():
        token = str(int(snr_db))
    else:
        token = f"{snr_db:g}".replace(".", "p")
    return token.replace("-", "neg")


def condition_dir(
    model: str, dataset: str, snr_db: float, noise_seed: int
) -> Path:
    return (
        RESULT_ROOT
        / dataset
        / SAVE_FAMILY
        / model
        / f"snr_{snr_token(snr_db)}db_seed{noise_seed}_TP"
    )


def parse_report(report_path: Path) -> dict[str, float] | None:
    """Read the normalized metrics from one benchmark report."""
    try:
        with report_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
    except (OSError, UnicodeError, csv.Error):
        return None
    if len(rows) < 2:
        return None

    metrics: dict[str, float] = {}
    for row in rows[1:]:
        if len(row) < 3 or row[1] not in REQUIRED_METRICS:
            continue
        try:
            metrics[row[1]] = float(row[2])
        except ValueError:
            return None
    return metrics if all(metric in metrics for metric in REQUIRED_METRICS) else None


def latest_valid_report(
    model: str, dataset: str, snr_db: float, noise_seed: int
) -> Path | None:
    reports = sorted(
        condition_dir(model, dataset, snr_db, noise_seed).glob("test_report*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return next((path for path in reports if parse_report(path) is not None), None)


def build_command(
    model: str,
    dataset: str,
    snr_db: float,
    noise_seed: int,
    *,
    horizon: int,
    num_epochs: int | None,
    train_seed: int,
    gpus: Sequence[int],
    timeout: float,
) -> list[str]:
    noise_ratio = snr_to_noise_ratio(snr_db)
    model_config = MODEL_CONFIGS[model]
    if model_config.get("uses_phoconet_params", False):
        params = dict(BASE_PARAMS)
        params["alpha_init"] = DATASET_ALPHA_INIT[dataset]
    else:
        params = dict(model_config["params"][dataset])
        params["pred_len"] = horizon
        if model_config.get("adapter") == "transformer_adapter":
            params.setdefault("label_len", horizon)
    params.update(
        {
            "horizon": horizon,
            "input_noise_level": noise_ratio,
            "input_noise_scope": "auxiliary",
            "input_noise_seed": noise_seed,
            "sensor_noise_in_base": True,
        }
    )
    if model == MODEL_LABEL:
        validate_params(params)
    if num_epochs is not None:
        params["num_epochs"] = num_epochs

    # Only the test-time noise configuration changes across SNR conditions.
    # The digest prevents reuse after any model or training setting changes.
    noise_only_keys = {
        "input_noise_level",
        "input_noise_scope",
        "input_noise_seed",
        "sensor_noise_in_base",
        "robustness_checkpoint_cache",
    }
    training_signature = {
        key: value for key, value in params.items() if key not in noise_only_keys
    }
    signature_digest = hashlib.sha256(
        json.dumps(training_signature, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:12]
    checkpoint_cache = (
        OUTPUT_DIR
        / "model_cache"
        / model
        / dataset
        / f"h{horizon}_seed{train_seed}_{signature_digest}.pt"
    )
    params["robustness_checkpoint_cache"] = str(checkpoint_cache)

    strategy_args = {"horizon": horizon, "target_channel": [TP_CHANNEL]}
    save_path = condition_dir(model, dataset, snr_db, noise_seed).relative_to(
        RESULT_ROOT
    )
    runner = model_config.get("runner", "scripts/run_benchmark.py")

    command = [
        sys.executable,
        f"./{runner}",
        "--config-path",
        "rolling_forecast_config.json",
        "--data-name-list",
        f"{dataset}.csv",
        "--strategy-args",
        json.dumps(strategy_args),
        "--model-name",
        model_config["model_name"],
        "--model-hyper-params",
        json.dumps(params),
    ]
    adapter = model_config.get("adapter")
    if adapter:
        command.extend(["--adapter", adapter])
    command.extend([
        "--gpus",
        *[str(gpu) for gpu in gpus],
        "--num-workers",
        "1",
        "--timeout",
        str(timeout),
        "--save-path",
        save_path.as_posix(),
        "--seed",
        str(train_seed),
        "--save-true-pred",
        "false",
        "--deterministic",
        "full",
    ])
    return command


def run_condition(
    model: str,
    dataset: str,
    snr_db: float,
    noise_seed: int,
    *,
    horizon: int,
    num_epochs: int | None,
    train_seed: int,
    gpus: Sequence[int],
    timeout: float,
    force: bool,
    dry_run: bool,
) -> bool:
    existing = latest_valid_report(model, dataset, snr_db, noise_seed)
    label = f"{model}/{dataset}/SNR={snr_db:g}dB/noise_seed={noise_seed}"
    if existing is not None and not force:
        print(f"[skip] {label}: {existing.name}", flush=True)
        return True

    command = build_command(
        model,
        dataset,
        snr_db,
        noise_seed,
        horizon=horizon,
        num_epochs=num_epochs,
        train_seed=train_seed,
        gpus=gpus,
        timeout=timeout,
    )
    ratio = snr_to_noise_ratio(snr_db)
    print(f"[run] {label}  sigma_noise/sigma_signal={ratio:.6g}", flush=True)
    if dry_run:
        print("      " + subprocess.list2cmdline(command), flush=True)
        return True

    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode != 0:
        print(f"[failed] {label}: exit code {completed.returncode}", flush=True)
        return False
    report = latest_valid_report(model, dataset, snr_db, noise_seed)
    if report is None:
        print(f"[failed] {label}: no valid test report", flush=True)
        return False
    print(f"[done] {label}: {report.name}", flush=True)
    return True


def collect_records(
    models: Sequence[str],
    datasets: Sequence[str],
    snr_values: Sequence[float],
    noise_seeds: Sequence[int],
    train_seed: int,
) -> tuple[list[NoiseRecord], list[str]]:
    records: list[NoiseRecord] = []
    missing: list[str] = []
    for model in models:
        for dataset in datasets:
            for snr_db in snr_values:
                for noise_seed in noise_seeds:
                    report = latest_valid_report(
                        model, dataset, snr_db, noise_seed
                    )
                    if report is None:
                        missing.append(
                            f"{model}/{dataset}/SNR={snr_db:g}/seed={noise_seed}"
                        )
                        continue
                    metrics = parse_report(report)
                    if metrics is None:
                        missing.append(str(report))
                        continue
                    records.append(
                        NoiseRecord(
                            model=model,
                            dataset=dataset,
                            snr_db=snr_db,
                            noise_std_ratio=snr_to_noise_ratio(snr_db),
                            train_seed=train_seed,
                            noise_seed=noise_seed,
                            mse_norm=metrics["mse_norm"],
                            mae_norm=metrics["mae_norm"],
                            rmse_norm=metrics["rmse_norm"],
                            report_path=str(report.relative_to(ROOT)),
                        )
                    )
    return records, missing


def mean_and_std(values: Iterable[float]) -> tuple[float, float]:
    values_list = list(values)
    if not values_list:
        raise ValueError("Cannot summarize an empty metric list")
    mean_value = statistics.fmean(values_list)
    std_value = statistics.stdev(values_list) if len(values_list) > 1 else 0.0
    return mean_value, std_value


def percentage_change(value: float, reference: float) -> float:
    if reference == 0:
        return math.nan
    return 100.0 * (value - reference) / reference


def aggregate_records(
    records: Sequence[NoiseRecord],
    models: Sequence[str],
    datasets: Sequence[str],
    snr_values: Sequence[float],
) -> list[NoiseSummary]:
    grouped: dict[tuple[str, str, float], list[NoiseRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.model, record.dataset, record.snr_db)].append(record)

    summaries: list[NoiseSummary] = []
    for model in models:
        for dataset in datasets:
            dataset_rows: list[dict[str, float | int]] = []
            for snr_db in snr_values:
                group = grouped.get((model, dataset, snr_db), [])
                if not group:
                    continue
                mse_mean, mse_std = mean_and_std(item.mse_norm for item in group)
                mae_mean, mae_std = mean_and_std(item.mae_norm for item in group)
                rmse_mean, rmse_std = mean_and_std(item.rmse_norm for item in group)
                dataset_rows.append(
                    {
                        "snr_db": snr_db,
                        "noise_std_ratio": snr_to_noise_ratio(snr_db),
                        "n": len(group),
                        "mse_mean": mse_mean,
                        "mse_std": mse_std,
                        "mae_mean": mae_mean,
                        "mae_std": mae_std,
                        "rmse_mean": rmse_mean,
                        "rmse_std": rmse_std,
                    }
                )
            if not dataset_rows:
                continue

            # The largest SNR is the near-clean reference condition.
            reference = max(dataset_rows, key=lambda row: float(row["snr_db"]))
            for row in dataset_rows:
                summaries.append(
                    NoiseSummary(
                        model=model,
                        dataset=dataset,
                        snr_db=float(row["snr_db"]),
                        noise_std_ratio=float(row["noise_std_ratio"]),
                        n_noise_seeds=int(row["n"]),
                        mse_mean=float(row["mse_mean"]),
                        mse_std=float(row["mse_std"]),
                        mae_mean=float(row["mae_mean"]),
                        mae_std=float(row["mae_std"]),
                        rmse_mean=float(row["rmse_mean"]),
                        rmse_std=float(row["rmse_std"]),
                        mse_change_pct=percentage_change(
                            float(row["mse_mean"]), float(reference["mse_mean"])
                        ),
                        mae_change_pct=percentage_change(
                            float(row["mae_mean"]), float(reference["mae_mean"])
                        ),
                        rmse_change_pct=percentage_change(
                            float(row["rmse_mean"]), float(reference["rmse_mean"])
                        ),
                    )
                )
    return summaries


def write_dataclass_csv(path: Path, rows: Sequence[object]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    dictionaries = [asdict(row) for row in rows]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dictionaries[0]))
        writer.writeheader()
        writer.writerows(dictionaries)


def display_metric(mean_value: float, std_value: float, n: int) -> str:
    return (
        f"{mean_value:.4f} +/- {std_value:.4f}"
        if n > 1
        else f"{mean_value:.4f}"
    )


def latex_metric(mean_value: float, std_value: float, n: int) -> str:
    return (
        f"{mean_value:.4f} $\\pm$ {std_value:.4f}"
        if n > 1
        else f"{mean_value:.4f}"
    )


def write_markdown_table(path: Path, summaries: Sequence[NoiseSummary]) -> None:
    lines = [
        "# Sensor-noise robustness",
        "",
        "Gaussian noise is applied at test time only to online auxiliary variables; ",
        "verified historical TP remains unchanged. Metrics are normalized and lower is better.",
        "",
        "| Model | Dataset | SNR (dB) | MSE | MAE | RMSE | Delta MSE vs. 100 dB |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.model,
                    DATASET_LABELS[row.dataset],
                    f"{row.snr_db:g}",
                    display_metric(row.mse_mean, row.mse_std, row.n_noise_seeds),
                    display_metric(row.mae_mean, row.mae_std, row.n_noise_seeds),
                    display_metric(row.rmse_mean, row.rmse_std, row.n_noise_seeds),
                    f"{row.mse_change_pct:+.2f}%",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex_table(
    path: Path,
    summaries: Sequence[NoiseSummary],
    models: Sequence[str],
    datasets: Sequence[str],
) -> None:
    by_key = {
        (row.model, row.dataset, row.snr_db): row for row in summaries
    }
    table_spec = "ll" + "cc" * len(models)
    model_header = " & ".join(
        rf"\multicolumn{{2}}{{c}}{{{model}}}" for model in models
    )
    metric_header = " & ".join(["MSE & MAE"] * len(models))
    cmidrules = " ".join(
        rf"\cmidrule(lr){{{3 + 2 * index}-{4 + 2 * index}}}"
        for index in range(len(models))
    )

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Robustness to additive Gaussian sensor noise. Noise is applied at test time to historical and estimation-window auxiliary measurements, while verified historical TP remains unchanged. The noise ratio is $\sigma_n/\sigma_x=10^{-\mathrm{SNR}/20}$. Values are normalized and lower is better; when multiple noise seeds are used, results are reported as mean $\pm$ standard deviation.}",
        r"\label{tab:sensor_noise_robustness}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        rf"\begin{{tabular}}{{{table_spec}}}",
        r"\toprule",
        f"Dataset & SNR (dB) & {model_header}" + r"\\",
        cmidrules,
        f" & & {metric_header}" + r"\\",
        r"\midrule",
    ]
    included = 0
    for dataset in datasets:
        snr_rows = sorted(
            {
                row.snr_db
                for row in summaries
                if row.dataset == dataset and row.model in models
            },
            reverse=True,
        )
        if not snr_rows:
            continue
        if included:
            lines.append(r"\midrule")
        included += 1
        for index, snr_db in enumerate(snr_rows):
            dataset_cell = (
                rf"\multirow{{{len(snr_rows)}}}{{*}}{{{DATASET_LABELS[dataset]}}}"
                if index == 0
                else ""
            )
            available = [
                by_key[(model, dataset, snr_db)]
                for model in models
                if (model, dataset, snr_db) in by_key
            ]
            best_mse = min((row.mse_mean for row in available), default=math.nan)
            best_mae = min((row.mae_mean for row in available), default=math.nan)
            cells: list[str] = []
            for model in models:
                row = by_key.get((model, dataset, snr_db))
                if row is None:
                    cells.extend(["--", "--"])
                    continue
                mse_text = latex_metric(
                    row.mse_mean, row.mse_std, row.n_noise_seeds
                )
                mae_text = latex_metric(
                    row.mae_mean, row.mae_std, row.n_noise_seeds
                )
                if len(models) > 1 and math.isclose(row.mse_mean, best_mse):
                    mse_text = rf"\textbf{{{mse_text}}}"
                if len(models) > 1 and math.isclose(row.mae_mean, best_mae):
                    mae_text = rf"\textbf{{{mae_text}}}"
                cells.extend([mse_text, mae_text])
            lines.append(
                f"{dataset_cell} & {snr_db:g} & "
                + " & ".join(cells)
                + r"\\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(args: argparse.Namespace) -> None:
    manifest = {
        "models": args.models,
        "datasets": args.datasets,
        "snr_db": args.snr_db,
        "noise_std_ratio": {
            f"{value:g}": snr_to_noise_ratio(value) for value in args.snr_db
        },
        "noise_scope": "historical and estimation-window auxiliary variables only",
        "verified_historical_tp_perturbed": False,
        "noise_distribution": "zero-mean Gaussian",
        "noise_seeds": args.noise_seeds,
        "train_seed": args.train_seed,
        "horizon": args.horizon,
        "num_epochs": args.num_epochs,
        "checkpoint_reuse": "one clean-data checkpoint per model/dataset",
        "phoconet_hyperparameters": BASE_PARAMS,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def summarize(args: argparse.Namespace) -> tuple[int, int]:
    records, missing = collect_records(
        args.models, args.datasets, args.snr_db, args.noise_seeds, args.train_seed
    )
    summaries = aggregate_records(
        records, args.models, args.datasets, args.snr_db
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_dataclass_csv(OUTPUT_DIR / "sensor_noise_metrics.csv", records)
    write_dataclass_csv(OUTPUT_DIR / "sensor_noise_summary.csv", summaries)
    write_markdown_table(OUTPUT_DIR / "sensor_noise_table.md", summaries)
    write_latex_table(
        OUTPUT_DIR / "sensor_noise_table.tex",
        summaries,
        args.models,
        args.datasets,
    )
    write_manifest(args)

    print(f"[summary] completed reports: {len(records)}", flush=True)
    print(f"[summary] missing reports: {len(missing)}", flush=True)
    if missing:
        print("[summary] missing conditions:", flush=True)
        for item in missing:
            print(f"  - {item}", flush=True)
    print(f"[summary] outputs: {OUTPUT_DIR}", flush=True)
    return len(records), len(missing)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate model robustness under SNR-controlled sensor noise."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_CONFIGS),
        default=list(DEFAULT_MODELS),
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASETS, default=list(DATASETS)
    )
    parser.add_argument(
        "--snr-db", nargs="+", type=float, default=list(DEFAULT_SNR_DB)
    )
    parser.add_argument("--noise-seeds", nargs="+", type=int, default=[20210726])
    parser.add_argument("--train-seed", type=int, default=2021)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Override every model's configured epoch budget.",
    )
    parser.add_argument("--gpus", nargs="+", type=int, default=[0])
    parser.add_argument("--timeout", type=float, default=60000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    if args.horizon <= 0:
        parser.error("--horizon must be positive")
    if args.num_epochs is not None and args.num_epochs <= 0:
        parser.error("--num-epochs must be positive")
    if len(set(args.snr_db)) != len(args.snr_db):
        parser.error("--snr-db contains duplicate values")
    if len(set(args.noise_seeds)) != len(args.noise_seeds):
        parser.error("--noise-seeds contains duplicate values")
    args.snr_db = sorted(args.snr_db, reverse=True)
    return args


def main() -> int:
    args = parse_args()
    total = (
        len(args.models)
        * len(args.datasets)
        * len(args.snr_db)
        * len(args.noise_seeds)
    )
    print("=" * 78)
    print("SNR-controlled sensor-noise robustness")
    print("=" * 78)
    print(f"Models: {', '.join(args.models)}")
    print(f"Datasets: {', '.join(args.datasets)}")
    print("SNR (dB): " + ", ".join(f"{value:g}" for value in args.snr_db))
    print("Noise seeds: " + ", ".join(map(str, args.noise_seeds)))
    print(f"Planned conditions: {total}")
    print("Noise scope: auxiliary sensor variables only; historical TP stays clean")

    failures: list[str] = []
    if not args.summarize_only:
        for model in args.models:
            for dataset in args.datasets:
                for snr_db in args.snr_db:
                    for noise_seed in args.noise_seeds:
                        success = run_condition(
                            model,
                            dataset,
                            snr_db,
                            noise_seed,
                            horizon=args.horizon,
                            num_epochs=args.num_epochs,
                            train_seed=args.train_seed,
                            gpus=args.gpus,
                            timeout=args.timeout,
                            force=args.force,
                            dry_run=args.dry_run,
                        )
                        if not success:
                            failures.append(
                                f"{model}/{dataset}/SNR={snr_db:g}/"
                                f"seed={noise_seed}"
                            )

    if args.dry_run:
        print("Dry run completed; no experiment or summary files were written.")
        return 0

    _, missing_count = summarize(args)
    if failures:
        print("Failed runs: " + ", ".join(failures), flush=True)
    return 1 if failures or missing_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
