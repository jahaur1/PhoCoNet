"""Evaluate PhoCoNet under timestamp-consistent historical TP missingness.

Each nonzero missingness rate uses ten independently drawn global timestamp
masks. The realized uniform arrays are archived with the outputs. An existing
mask directory can optionally be supplied to reproduce the same realizations.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.config import DATASETS
from experiments.run_main import params_for
from experiments.information_value_core import AUXILIARIES, load_frame
from ts_benchmark.baselines.phoconet.phoconet import (
    PhoCoNet,
)
from ts_benchmark.evaluation.strategy.rolling_forecast import (
    RollingForecastEvalBatchMaker,
    RollingForecastPredictBatchMaker,
)


ROOT = Path(__file__).resolve().parents[1]
RATES = (0.0, 0.1, 0.3, 0.5, 0.7)
HISTORY = 96
HORIZON = 24
TRAIN_SEED = 2021
MASK_REPETITIONS = 10
METRICS = ("mse_norm", "mae_norm", "rmse_norm", "mean_window_rmse_norm")
MODEL_LABEL = "PhoCoNet"
MAIN_MODEL_LABEL = "PhoCoNet"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Refusing to write an empty table")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def eligible_history(n_rows: int, origins: np.ndarray) -> np.ndarray:
    eligible = np.zeros(n_rows, dtype=bool)
    for origin in origins:
        eligible[origin - HISTORY : origin] = True
    return eligible


def checkpoint_for(station: str, cache_root: Path) -> Path:
    candidates = sorted(
        (cache_root / MODEL_LABEL / station).glob(
            f"h{HORIZON}_seed{TRAIN_SEED}_*.pt"
        )
    )
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one PhoCoNet checkpoint for {station}, "
            f"found {len(candidates)}"
        )
    return candidates[0]


def load_final_model(
    station: str,
    frame: pd.DataFrame,
    checkpoint: Path,
) -> PhoCoNet:
    params = {
        **params_for(station, HORIZON, 100),
        "norm": True,
        "robustness_checkpoint_cache": str(checkpoint.resolve()),
    }
    model = PhoCoNet(**params)
    validation_end = int(len(frame) * 0.8)
    model.forecast_fit(
        frame.loc[:, ["TP"]].iloc[:validation_end],
        covariates={"exog": frame.loc[:, list(AUXILIARIES)].iloc[:validation_end]},
        train_ratio_in_tv=0.875,
    )
    if model.model is None or model.check_point is None:
        raise RuntimeError("PhoCoNet checkpoint was not loaded")
    if (
        model.config.attn_mode != "none"
        or int(model.config.ia_layers) != 0
        or int(model.config.ca_layers) != 0
        or not bool(model.config.q_aligned_variable_modulation)
        or bool(model.config.q_aligned_variable_attention)
        or bool(model.config.q_aligned_variable_residual)
    ):
        raise RuntimeError("Loaded model is not the frozen final PhoCoNet architecture")
    return model


def predict(
    model: PhoCoNet,
    frame: pd.DataFrame,
    origins: np.ndarray,
    missing: np.ndarray,
    training_tp_mean: float,
) -> tuple[np.ndarray, np.ndarray]:
    target_input = frame.loc[:, ["TP"]].copy()
    target_input.loc[missing, "TP"] = training_tp_mean
    exog = frame.loc[:, list(AUXILIARIES)]
    maker = RollingForecastEvalBatchMaker(
        target_input, origins.tolist(), {"exog": exog}
    )
    prediction_maker = RollingForecastPredictBatchMaker(maker)
    exog_future = maker.make_batch_eval(HORIZON)["covariates"]["exog"]
    chunks: list[np.ndarray] = []
    batch_index = 0
    while prediction_maker.has_more_batches():
        chunks.append(
            model.batch_forecast(
                HORIZON, prediction_maker, exog_future, batch_index
            )
        )
        batch_index += 1
    predicted = np.concatenate(chunks, axis=0)
    target = frame.TP.to_numpy()
    actual = target[origins[:, None] + np.arange(HORIZON)][..., None]
    if predicted.shape != actual.shape or not np.isfinite(predicted).all():
        raise RuntimeError(
            f"Invalid prediction array: predicted={predicted.shape}, actual={actual.shape}"
        )
    return actual, predicted


def error_metrics(actual: np.ndarray, predicted: np.ndarray, scale: float) -> dict:
    error = (predicted.astype(np.float64) - actual.astype(np.float64)) / scale
    mse = float(np.mean(error**2))
    return {
        "mse_norm": mse,
        "mae_norm": float(np.mean(np.abs(error))),
        "rmse_norm": float(np.sqrt(mse)),
        "mean_window_rmse_norm": float(
            np.sqrt(np.mean(error**2, axis=(1, 2))).mean()
        ),
    }


def summarize(rows: list[dict], output: Path) -> list[dict]:
    summaries: list[dict] = []
    for station in DATASETS:
        clean = next(
            row
            for row in rows
            if row["station"] == station and row["missing_rate"] == 0
        )
        for rate in RATES:
            group = [
                row
                for row in rows
                if row["station"] == station and row["missing_rate"] == rate
            ]
            expected = 1 if rate == 0 else MASK_REPETITIONS
            if len(group) != expected:
                raise RuntimeError(
                    f"Incomplete matrix for {station}/{rate}: {len(group)}"
                )
            item: dict = {
                "station": station,
                "missing_rate": rate,
                "n_masks": len(group),
            }
            for metric in METRICS:
                values = np.asarray([row[metric] for row in group], dtype=float)
                item[f"{metric}_mean"] = float(values.mean())
                item[f"{metric}_sd"] = (
                    float(values.std(ddof=1)) if len(values) > 1 else None
                )
            changes = np.asarray(
                [100 * (row["mse_norm"] - clean["mse_norm"]) / clean["mse_norm"]
                 for row in group],
                dtype=float,
            )
            item["mse_increase_pct_mean"] = float(changes.mean())
            item["mse_increase_pct_sd"] = (
                float(changes.std(ddof=1)) if len(changes) > 1 else None
            )
            for key in ("actual_unique_missing_rate", "actual_window_missing_rate"):
                values = np.asarray([row[key] for row in group], dtype=float)
                item[f"{key}_mean"] = float(values.mean())
                item[f"{key}_sd"] = (
                    float(values.std(ddof=1)) if len(values) > 1 else None
                )
            summaries.append(item)
    write_csv(output / "summary.csv", summaries)
    return summaries


def make_analysis(summaries: list[dict], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = output / "figures"
    figure_dir.mkdir(exist_ok=False)
    for metric, ylabel, stem in (
        ("mse_norm", "Normalized block MSE", "missing_rate_mse"),
        ("mse_increase_pct", "MSE increase from clean (%)", "relative_mse_increase"),
    ):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
        for axis, station in zip(axes, DATASETS):
            group = [row for row in summaries if row["station"] == station]
            axis.errorbar(
                [100 * row["missing_rate"] for row in group],
                [row[f"{metric}_mean"] for row in group],
                yerr=[row[f"{metric}_sd"] or 0 for row in group],
                marker="o",
                capsize=4,
                linewidth=1.8,
                color="#256b91",
            )
            axis.set(
                title=station.capitalize(),
                xlabel="Historical TP missingness (%)",
                ylabel=ylabel,
                xticks=[0, 10, 30, 50, 70],
            )
            axis.grid(alpha=0.22)
            if metric == "mse_increase_pct":
                axis.axhline(0, color="grey", linewidth=0.7)
        fig.suptitle(
            "PhoCoNet; frozen seed-2021 checkpoint; SD across 10 masks",
            fontsize=11,
        )
        fig.savefig(figure_dir / f"{stem}.png", dpi=220)
        fig.savefig(figure_dir / f"{stem}.svg")
        plt.close(fig)

    lines = [
        "# PhoCoNet 历史 TP 随机缺失敏感性分析",
        "",
        "## Protocol",
        "",
        "PhoCoNet (variable-isolated history encoding + target-conditioned bounded modulation), "
        "Full-8, H=96, F=24. The seed-2021 checkpoint is frozen; there is no retraining or tuning.",
        "Historical TP is independently masked by global timestamp at 0%, 10%, 30%, 50%, and 70%. "
        "The realized uniform draws are archived so the same missing locations can be reused in later comparisons.",
        "Each nonzero rate has 10 independent mask realizations. Missing TP is replaced by the training TP mean before model scaling and window RevIN.",
        "All auxiliary histories, estimation-window auxiliaries, labels, splits, and test origins remain unchanged.",
        "",
        "## Results",
        "",
        "| Station | Missing | Masks | MSE mean ± SD | MAE mean ± SD | MSE change |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        mse_sd = "NA" if row["mse_norm_sd"] is None else f"{row['mse_norm_sd']:.6f}"
        mae_sd = "NA" if row["mae_norm_sd"] is None else f"{row['mae_norm_sd']:.6f}"
        lines.append(
            f"| {row['station']} | {row['missing_rate']:.0%} | {row['n_masks']} | "
            f"{row['mse_norm_mean']:.6f} ± {mse_sd} | "
            f"{row['mae_norm_mean']:.6f} ± {mae_sd} | "
            f"{row['mse_increase_pct_mean']:+.2f}% |"
        )
    lines.extend(["", "## Interpretation limits", ""])
    lines.extend(
        [
            "The standard deviation describes sensitivity to missing locations for one frozen training run; it is not training-seed uncertainty or a confidence interval.",
            "Overlapping test windows are not treated as independent replicates. No p-values are reported.",
            "The experiment represents sporadic MCAR-like loss with mean imputation, not contiguous outages, concentration-dependent missingness, or laboratory reporting delay.",
        ]
    )
    (output / "analysis-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "stats-appendix.md").write_text(
        "# Statistical scope\n\n"
        "Primary unit: one global timestamp mask applied to one fixed station/test period/checkpoint.\n"
        "Ten independent masks per nonzero rate; rates are nested within each realization. The clean reference is deterministic.\n"
        "Reported uncertainty is sample SD (ddof=1) across mask locations. It excludes training-seed, station-population, and time-period uncertainty.\n"
        "MSE and MAE aggregate all 435 x 24 forecast errors. Global RMSE is sqrt(global MSE); mean-window RMSE is archived separately.\n"
        "No inferential test is used because masks are perturbation replicates around one fixed test set and checkpoint.\n",
        encoding="utf-8",
    )
    (output / "figure-catalog.md").write_text(
        "# Figure catalog\n\n"
        "- `figures/missing_rate_mse.{png,svg}`: absolute normalized block MSE versus nominal missingness. Error bars are sample SD across 10 masks.\n"
        "- `figures/relative_mse_increase.{png,svg}`: percentage MSE change from each station's clean PhoCoNet reference.\n\n"
        "Both figures use `summary.csv`. They show the direction, magnitude, and location sensitivity of degradation; they do not show training uncertainty.\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "result/phoconet_history_tp_missingness",
    )
    parser.add_argument(
        "--mask-source",
        type=Path,
        default=None,
        help="optional directory containing <station>/mask_uniforms.npz",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=ROOT / "result/phoconet_sensor_noise/model_cache",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    mask_source = args.mask_source.resolve() if args.mask_source else None
    checkpoint_root = args.checkpoint_root.resolve()
    for path in (output, checkpoint_root):
        path.relative_to(ROOT / "result")
    if mask_source is not None:
        mask_source.relative_to(ROOT / "result")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be empty; formal results are never overwritten")
    output.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    manifest: dict = {
        "title": "PhoCoNet robustness to random missingness in historical TP",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_LABEL,
        "architecture": {
            "fusion": "variable-isolated history encoding with late target-conditioned aggregation",
            "temporal_attention": False,
            "variable_attention": "late target-conditioned bounded modulation",
            "ia_layers": 0,
            "ca_layers": 0,
            "attn_mode": "none",
            "query_correction": "bounded Q-conditioned modulation",
        },
        "train_seed": TRAIN_SEED,
        "training_runs": 0,
        "history": HISTORY,
        "horizon": HORIZON,
        "rates": list(RATES),
        "nonzero_mask_repetitions": MASK_REPETITIONS,
        "mask_rng": (
            "archived uniform draws reused"
            if mask_source is not None
            else "NumPy default_rng OS entropy; realized arrays archived"
        ),
        "mask_source": (
            str(mask_source.relative_to(ROOT))
            if mask_source is not None
            else None
        ),
        "imputation": "raw training TP mean before training scaling and window RevIN",
        "device": device,
        "torch_version": str(torch.__version__),
        "runner_sha256": sha256(Path(__file__)),
        "stations": {},
    }
    write_json(output / "manifest.json", manifest)

    main_source = ROOT / "result/phoconet_main"
    main_metrics = pd.read_csv(main_source / "metrics_by_seed.csv")
    rows: list[dict] = []
    print("PhoCoNet; 123 frozen-checkpoint evaluations; no training", flush=True)
    for station in DATASETS:
        data_path = ROOT / "dataset/forecasting" / f"{station}.csv"
        frame = load_frame(data_path, "full8")
        validation_end = int(len(frame) * 0.8)
        train_end = int(validation_end * 0.875)
        origins = np.arange(validation_end, len(frame) - HORIZON + 1, dtype=np.int64)
        if len(origins) != 435:
            raise ValueError(f"Unexpected window count for {station}: {len(origins)}")
        eligible = eligible_history(len(frame), origins)
        training_mean = float(frame.TP.iloc[:train_end].mean())
        training_scale = float(frame.TP.iloc[:train_end].std(ddof=0))

        uniforms_file = (
            mask_source / station / "mask_uniforms.npz"
            if mask_source is not None
            else None
        )
        if uniforms_file is None:
            uniforms = np.random.default_rng().random(
                (MASK_REPETITIONS, len(frame))
            )
        else:
            with np.load(uniforms_file) as archive:
                uniforms = archive["uniforms"]
        if uniforms.shape != (MASK_REPETITIONS, len(frame)):
            raise ValueError(f"Unexpected archived mask shape for {station}")
        checkpoint = checkpoint_for(station, checkpoint_root)
        model = load_final_model(station, frame, checkpoint)
        station_dir = output / station
        station_dir.mkdir()
        np.savez_compressed(station_dir / "mask_uniforms.npz", uniforms=uniforms)

        clean_reference = main_metrics[
            (main_metrics.station == station)
            & (main_metrics.model == MAIN_MODEL_LABEL)
            & (main_metrics.horizon == HORIZON)
            & (main_metrics.seed == TRAIN_SEED)
        ]
        if len(clean_reference) != 1:
            raise ValueError(f"Missing main-experiment clean reference for {station}")
        for rate in RATES:
            repetitions = 1 if rate == 0 else MASK_REPETITIONS
            for realization in range(repetitions):
                missing = (uniforms[realization] < rate) & eligible
                actual, predicted = predict(
                    model, frame, origins, missing, training_mean
                )
                metrics = error_metrics(actual, predicted, training_scale)
                if rate == 0:
                    np.testing.assert_allclose(
                        metrics["mse_norm"],
                        float(clean_reference.iloc[0].mse_norm),
                        rtol=1e-5,
                        atol=1e-7,
                    )
                name = f"rate_{int(rate * 100):02d}_mask_{realization + 1:02d}.npz"
                np.savez_compressed(
                    station_dir / name,
                    predicted=predicted,
                    actual=actual,
                    origins=origins,
                    missing=missing,
                )
                window_missing = np.asarray(
                    [missing[origin - HISTORY : origin].mean() for origin in origins]
                )
                rows.append(
                    {
                        "station": station,
                        "missing_rate": rate,
                        "mask_id": realization + 1,
                        "test_windows": len(origins),
                        "eligible_unique_timestamps": int(eligible.sum()),
                        "missing_unique_timestamps": int(missing.sum()),
                        "actual_unique_missing_rate": float(missing[eligible].mean()),
                        "actual_window_missing_rate": float(window_missing.mean()),
                        **metrics,
                        "archive": str((station_dir / name).relative_to(ROOT)),
                    }
                )
                print(
                    f"[{len(rows):03d}/123] {station} rate={rate:.0%} "
                    f"mask={realization + 1} MSE={metrics['mse_norm']:.6f}",
                    flush=True,
                )
        manifest["stations"][station] = {
            "dataset_sha256": sha256(data_path),
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "checkpoint_sha256": sha256(checkpoint),
            "mask_uniforms_sha256": sha256(
                station_dir / "mask_uniforms.npz"
            ),
            "archived_copy_sha256": sha256(station_dir / "mask_uniforms.npz"),
            "test_windows": len(origins),
            "train_end_exclusive": train_end,
            "validation_end_exclusive": validation_end,
            "training_tp_mean": training_mean,
            "training_tp_scale": training_scale,
            "clean_reference": str(Path(clean_reference.iloc[0].report)),
            "clean_metric_matches_main_seed_2021": True,
        }
        write_json(output / "manifest.json", manifest)

    write_csv(output / "metrics_by_mask.csv", rows)
    summaries = summarize(rows, output)
    make_analysis(summaries, output)
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["evaluations"] = len(rows)
    manifest["summary_rows"] = len(summaries)
    write_json(output / "manifest.json", manifest)
    write_json(
        output / "verification.json",
        {
            "evaluations_checked": len(rows),
            "summary_rows": len(summaries),
            "clean_metric_matches_main_seed_2021": True,
            "external_masks_reused": mask_source is not None,
            "all_outputs_finite": all(
                np.isfinite(row[metric]) for row in rows for metric in METRICS
            ),
            "complete": len(rows) == 123 and len(summaries) == 15,
        },
    )
    print(f"Complete: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
