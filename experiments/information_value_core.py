"""Independently retrain the five information sets proposed in the PDF review.

Run from the repository root with ``python -B -m experiments.information_value_core``.
No previous experiment reports are read. The script is self-contained and does not read previous reports.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ts_benchmark.baselines.phoconet.models.core import Model
from ts_benchmark.baselines.phoconet.models.online_conditioning import (
    Model as OnlineConditionedModel,
)


ROOT = Path(__file__).resolve().parents[1]
STATIONS = ("juzizhou", "sanjiaozhou", "laodaohe")
AUXILIARIES = ("WT", "pH", "DO", "Cond", "Turb", "PI", "AN", "TN")
ALPHA_INIT = dict(zip(STATIONS, (0.0, -0.5, 3.0)))
BACKBONES = {
    "core": Model,
    "online_conditioned": OnlineConditionedModel,
}


@dataclass(frozen=True)
class InformationSet:
    target_history: bool
    auxiliary_history: bool
    query: bool


CONDITIONS = {
    "T": InformationSet(True, False, False),
    "TH": InformationSet(True, True, False),
    "Q": InformationSet(False, False, True),
    "TQ": InformationSet(True, False, True),
    "THQ": InformationSet(True, True, True),
}
# Positive difference means the added information reduces error.
CONTRASTS = {
    "query_given_T": ("T", "TQ"),
    "query_given_TH": ("TH", "THQ"),
    "target_history_given_Q": ("Q", "TQ"),
    "auxiliary_history_given_TQ": ("TQ", "THQ"),
    "auxiliary_history_given_T": ("T", "TH"),
}
METRICS = ("mse_norm", "mae_norm", "rmse_norm", "endpoint_mse_norm",
           "endpoint_mae_norm", "endpoint_rmse_norm")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Refusing to write an empty result table")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_frame(path: Path, protocol: str) -> pd.DataFrame:
    """Read long or wide CSV without interpolation or fitting on future rows."""
    frame = pd.read_csv(path)
    if "date" not in frame:
        raise ValueError(f"{path}: missing date column")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    if {"data", "cols"}.issubset(frame.columns):
        frame = frame.pivot(index="date", columns="cols", values="data")
    else:
        frame = frame.set_index("date")
    auxiliaries = AUXILIARIES if protocol == "full8" else AUXILIARIES[:5]
    frame = frame.loc[:, ["TP", *auxiliaries]].sort_index().astype(float)
    if frame.index.has_duplicates or frame.index.hasnans:
        raise ValueError(f"{path}: duplicate or missing timestamps")
    if len(frame) < 2 or not np.isfinite(frame.to_numpy()).all():
        raise ValueError(f"{path}: insufficient data or non-finite measurements")
    if not (np.diff(frame.index.asi8) == pd.Timedelta(hours=4).value).all():
        raise ValueError(f"{path}: expected a regular four-hour grid; no implicit filling")
    return frame


def split_origins(n_rows: int, history: int, horizon: int) -> dict[str, np.ndarray]:
    """Every target block stays inside its split; observed history may precede it."""
    train_end, validation_end = int(n_rows * 0.7), int(n_rows * 0.8)
    bounds = {"train": (history, train_end), "validation": (train_end, validation_end),
              "test": (validation_end, n_rows)}
    result = {name: np.arange(start, stop - horizon + 1, dtype=np.int64)
              for name, (start, stop) in bounds.items()}
    if any(len(values) == 0 for values in result.values()):
        raise ValueError("Not enough observations for the requested history, horizon and splits")
    return result


def fit_scaler(values: np.ndarray, train_end: int) -> tuple[np.ndarray, np.ndarray]:
    mean, scale = values[:train_end].mean(axis=0), values[:train_end].std(axis=0)
    if scale[0] < 1e-12:
        raise ValueError("Training TP is constant; normalized target errors are undefined")
    return mean, np.where(scale < 1e-12, 1.0, scale)


def calendar_marks(index: pd.DatetimeIndex) -> np.ndarray:
    """Four hourly calendar markers, held available in every information set."""
    return np.column_stack((index.hour / 23 - .5, index.dayofweek / 6 - .5,
                            (index.day - 1) / 30 - .5,
                            (index.dayofyear - 1) / 365 - .5)).astype(np.float32)


class Windows(Dataset):
    def __init__(self, values: np.ndarray, marks: np.ndarray, origins: np.ndarray,
                 history: int, horizon: int, treatment: str = "clean",
                 perturbation_seed: int = 0):
        self.values, self.marks, self.origins = values, marks, origins
        self.history, self.horizon = history, horizon
        self.treatment, self.perturbation_seed = treatment, perturbation_seed
        if treatment not in {"clean", "shuffle"}:
            if not treatment.startswith("lag_"):
                raise ValueError(f"Unknown query treatment: {treatment}")
            lag = int(treatment[4:])
            if lag <= 0 or lag > history:
                raise ValueError("Query lag must be positive and no longer than history")
        if treatment == "shuffle" and horizon < 2:
            raise ValueError("Query shuffling requires at least two steps")

    def __len__(self) -> int:
        return len(self.origins)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        origin = int(self.origins[item])
        lag = int(self.treatment[4:]) if self.treatment.startswith("lag_") else 0
        query = self.values[origin - lag:origin - lag + self.horizon, 1:].copy()
        if self.treatment == "shuffle":
            # Sample-keyed randomness: reproducible across evaluation batch sizes.
            rng = np.random.default_rng(np.random.SeedSequence([self.perturbation_seed, origin]))
            permutation = rng.permutation(self.horizon)
            if np.array_equal(permutation, np.arange(self.horizon)):
                permutation = np.roll(permutation, 1)
            query = query[permutation]  # One common permutation preserves sensor co-occurrence.
        return {
            "history": torch.from_numpy(self.values[origin - self.history:origin].copy()),
            "query": torch.from_numpy(query),
            "marks": torch.from_numpy(self.marks[origin - self.history:origin].copy()),
            "target": torch.from_numpy(self.values[origin:origin + self.horizon, 0].copy()),
            "origin": torch.tensor(origin, dtype=torch.long),
        }


def model_config(channels: int, history: int, horizon: int, width: int,
                 patch: int, attention_dropout: float = .15) -> SimpleNamespace:
    return SimpleNamespace(
        enc_in=channels, series_dim=1, seq_len=history, horizon=horizon, period=patch,
        d_model=width, d_ff=width, n_heads=4, dropout=0.0, attn_dropout=attention_dropout,
        activation="gelu", stable_len=2, revin=0, ia_layers=1, ca_layers=1,
        attn_mode="full", layer_order="int_coint", fusion_mode="dual",
        alpha_mode="learnable", use_history_target=True, use_history_exog=True,
        use_future_exog=True, infer_use_future=True, future_exog_mode="normal",
        q_hidden=32, q_gate_bias=-2.0, q_mode="aligned",
    )


class ControlledModel(nn.Module):
    """Mask before normalization; never accept query TP or unobserved TP statistics."""
    def __init__(self, config: SimpleNamespace, information: InformationSet,
                 normalization: str = "safe-revin", alpha_init: float = 0.0,
                 backbone_cls: type[nn.Module] = Model):
        super().__init__()
        if normalization not in {"safe-revin", "train"}:
            raise ValueError("Unknown normalization")
        self.information, self.normalization = information, normalization
        self.backbone = backbone_cls(config)
        # Initialize ALL branches before creating the optimizer, even without Q.
        self.backbone.init_fusion_components(config.enc_in - 1, alpha_init)

    def forward(self, history: torch.Tensor, query: torch.Tensor,
                marks: torch.Tensor) -> torch.Tensor:
        if query.shape[-1] != history.shape[-1] - 1:
            raise ValueError("Query must contain auxiliary variables only, without TP")
        available = torch.ones(history.shape[-1], dtype=torch.bool, device=history.device)
        available[0] = self.information.target_history
        available[1:] = self.information.auxiliary_history
        masked = torch.where(available, history, torch.zeros_like(history))
        query = query if self.information.query else torch.zeros_like(query)
        if self.normalization == "safe-revin":
            mean = masked.mean(dim=1, keepdim=True).detach()
            scale = (masked.var(dim=1, keepdim=True, unbiased=False) + 1e-5).sqrt().detach()
            # Missing channels use training-standardized mean=0, scale=1, not sqrt(epsilon).
            mean = torch.where(available, mean, torch.zeros_like(mean))
            scale = torch.where(available, scale, torch.ones_like(scale))
            masked = (masked - mean) / scale
        prediction = self.backbone(masked, x_mark_enc=marks, exog_future=query)[..., 0]
        if self.normalization == "safe-revin":
            prediction = prediction * scale[..., 0] + mean[..., 0]
        return prediction


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def state_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def loader(dataset: Windows, batch_size: int, seed: int, shuffle: bool = False) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, generator=generator, drop_last=False)


def predict(model: ControlledModel, data: DataLoader, device: torch.device
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    predictions, actuals, origins = [], [], []
    with torch.no_grad():
        for batch in data:
            output = model(batch["history"].to(device), batch["query"].to(device),
                           batch["marks"].to(device))
            predictions.append(output.cpu().numpy())
            actuals.append(batch["target"].numpy())
            origins.append(batch["origin"].numpy())
    predicted, actual = np.concatenate(predictions), np.concatenate(actuals)
    if not np.isfinite(predicted).all():
        raise RuntimeError("Non-finite model output")
    return actual, predicted, np.concatenate(origins)


def error_metrics(actual: np.ndarray, predicted: np.ndarray, target_scale: float) -> dict:
    error = predicted.astype(np.float64) - actual.astype(np.float64)
    values = {}
    for prefix, residual in (("", error), ("endpoint_", error[:, -1])):
        mse, mae = float(np.mean(residual ** 2)), float(np.mean(abs(residual)))
        values.update({prefix + "mse_norm": mse, prefix + "mae_norm": mae,
                       prefix + "rmse_norm": float(np.sqrt(mse)),
                       prefix + "mae_raw": mae * target_scale,
                       prefix + "rmse_raw": float(np.sqrt(mse)) * target_scale})
    values["mean_window_rmse_norm"] = float(np.sqrt(np.mean(error ** 2, axis=1)).mean())
    return values


def fit(model: ControlledModel, training: DataLoader, validation: DataLoader,
        args: argparse.Namespace, device: torch.device, directory: Path) -> dict:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best, best_epoch, stale, history = float("inf"), 0, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        squared_error, count = 0.0, 0
        for batch in training:
            optimizer.zero_grad(set_to_none=True)
            predicted = model(batch["history"].to(device), batch["query"].to(device),
                              batch["marks"].to(device))
            target = batch["target"].to(device)
            loss = (predicted - target).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            optimizer.step()
            squared_error += loss.item() * target.numel()
            count += target.numel()
        actual, predicted, _ = predict(model, validation, device)
        score = float(np.mean((predicted.astype(np.float64) - actual) ** 2))
        history.append({"epoch": epoch, "train_mse": squared_error / count, "validation_mse": score})
        if score < best:
            best, best_epoch, stale = score, epoch, 0
            torch.save(model.state_dict(), directory / "best.pt")
        else:
            stale += 1
        if stale >= args.patience:
            break
    write_csv(directory / "training.csv", history)
    model.load_state_dict(torch.load(directory / "best.pt", map_location=device, weights_only=True))
    return {"best_epoch": best_epoch, "best_validation_mse": best,
            "epochs_completed": len(history), "alpha": model.backbone.get_alpha()}


def paired_interval(differences: np.ndarray, seed: int = 4096) -> tuple[float, float]:
    """Paired seed bootstrap conditional on this fixed test period, not window iid CI."""
    if len(differences) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    sampled = rng.choice(differences, (10000, len(differences)), replace=True).mean(axis=1)
    low, high = np.quantile(sampled, [.025, .975])
    return float(low), float(high)


def summarize(rows: list[dict], output: Path, smoke: bool) -> None:
    write_csv(output / "metrics_by_run.csv", rows)
    clean = [row for row in rows if row["treatment"] == "clean"]
    summaries = []
    for station in sorted({row["station"] for row in clean}):
        for condition in CONDITIONS:
            group = [row for row in clean if row["station"] == station and row["condition"] == condition]
            summary = {"station": station, "condition": condition, "n_seeds": len(group)}
            for metric in METRICS:
                values = np.array([row[metric] for row in group])
                summary[metric + "_mean"] = float(values.mean())
                summary[metric + "_std"] = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
            summaries.append(summary)
    write_csv(output / "summary.csv", summaries)
    indexed = {(row["station"], row["seed"], row["condition"]): row for row in clean}
    pairs = []
    for station, seed in sorted({(row["station"], row["seed"]) for row in clean}):
        for contrast, (before, after) in CONTRASTS.items():
            left, right = indexed[station, seed, before], indexed[station, seed, after]
            for metric in METRICS:
                difference = left[metric] - right[metric]
                pairs.append({"station": station, "seed": seed, "contrast": contrast,
                              "metric": metric, "before": left[metric], "after": right[metric],
                              "reduction": difference,
                              "reduction_pct": 100 * difference / left[metric] if left[metric] else float("nan")})
    write_csv(output / "paired_by_seed.csv", pairs)
    paired_summary = []
    for station, contrast, metric in sorted({(r["station"], r["contrast"], r["metric"]) for r in pairs}):
        group = [r for r in pairs if (r["station"], r["contrast"], r["metric"]) == (station, contrast, metric)]
        differences = np.array([r["reduction"] for r in group])
        low, high = paired_interval(differences)
        paired_summary.append({"station": station, "contrast": contrast, "metric": metric,
                               "n_seeds": len(group), "mean_reduction": float(differences.mean()),
                               "seed_bootstrap_ci_low": low, "seed_bootstrap_ci_high": high,
                               "improved_seeds": int((differences > 0).sum())})
    write_csv(output / "paired_summary.csv", paired_summary)
    perturbations = []
    for row in rows:
        if row["treatment"] == "clean":
            continue
        reference = indexed[row["station"], row["seed"], "THQ"]
        perturbations.append({"station": row["station"], "seed": row["seed"],
                              "treatment": row["treatment"], "perturbation_seed": row["perturbation_seed"],
                              "mse_norm": row["mse_norm"], "clean_mse_norm": reference["mse_norm"],
                              "mse_increase": row["mse_norm"] - reference["mse_norm"],
                              "endpoint_mse_increase": row["endpoint_mse_norm"] - reference["endpoint_mse_norm"]})
    write_csv(output / "query_perturbations.csv", perturbations)
    report = ["# Paper information-value experiment", "",
              "SMOKE TEST ONLY: these numbers are not scientific evidence." if smoke else
              "Independent retraining; all comparisons are within this run's manifest.", "",
              "T = historical TP; H = historical auxiliary variables; Q = observed query auxiliary variables.",
              "Calendar markers remain available in all five conditions.", "",
              "| Station | Information | Seeds | MSE mean | MSE sample SD | Endpoint MSE |",
              "|---|---|---:|---:|---:|---:|"]
    for row in summaries:
        sd = f"{row['mse_norm_std']:.6f}" if row["n_seeds"] > 1 else "NA"
        report.append(f"| {row['station']} | {row['condition']} | {row['n_seeds']} | "
                      f"{row['mse_norm_mean']:.6f} | {sd} | {row['endpoint_mse_norm_mean']:.6f} |")
    report.extend(["", "## Interpretation", "",
                   "Positive reduction in paired_summary.csv favors added information; negative values must be retained.",
                   "Query value is tested by T -> TQ and TH -> THQ, not by Q -> TQ alone.",
                   "Shuffle/lag conditions evaluate the SAME trained THQ checkpoint, without retraining.",
                   "Shuffling is a distribution-shift diagnostic, not a causal proof of information usefulness.",
                   "95% seed-bootstrap intervals describe optimization variability on a fixed test period only.",
                   "Overlapping windows and shuffle realizations are NOT independent training replicates.",
                   "Single-seed SD and intervals are NA. Test-period/generalization uncertainty is not estimated here.",
                   "Global RMSE = sqrt(global MSE); mean_window_rmse_norm is separately provided for aggregation audits.",
                   "Normalization is recorded in manifest.json; compare safe-revin with train in a separate output folder."])
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=STATIONS, default=list(STATIONS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[2021, 2022, 2023, 2024, 2025])
    parser.add_argument("--protocol", choices=("full8", "online5"), default="full8")
    parser.add_argument("--normalization", choices=("safe-revin", "train"), default="safe-revin")
    parser.add_argument("--backbone", choices=tuple(BACKBONES), default="core")
    parser.add_argument("--history", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--patch", type=int, default=24)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--lags", nargs="+", type=int, default=[1, 6])
    parser.add_argument("--perturbation-seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset" / "forecasting")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true", help="Five tiny training runs; not publishable")
    parser.add_argument("--dry-run", action="store_true", help="Validate data and print plan; write nothing")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.datasets, args.seeds = args.datasets[:1], args.seeds[:1]
        args.epochs, args.width, args.batch_size = 1, 16, 8
    if min(args.history, args.horizon, args.patch, args.width, args.epochs,
           args.patience, args.batch_size, args.threads) <= 0:
        parser.error("Lengths, widths, counts and patience must be positive")
    if args.horizon < 2 or args.history < args.horizon or args.history % args.patch or args.width % 4:
        parser.error("Require H >= F >= 2, H divisible by patch and width divisible by four")
    if not np.isfinite(args.lr) or args.lr <= 0 or any(lag <= 0 or lag > args.history for lag in args.lags):
        parser.error("Learning rate must be finite and positive; lags must lie in [1,H]")
    for name in ("datasets", "seeds", "lags", "perturbation_seeds"):
        values = getattr(args, name)
        if len(set(values)) != len(values):
            parser.error(f"Duplicate {name} entries")
    if any(seed < 0 or seed >= 2 ** 32 for seed in args.seeds + args.perturbation_seeds):
        parser.error("Seeds must lie in [0, 2**32)")
    if args.dry_run and args.summarize_only:
        parser.error("Choose dry-run or summarize-only, not both")
    if args.output is None:
        experiment = (
            "information_value"
            if args.backbone == "core"
            else "information_value_online_conditioned"
        )
        args.output = ROOT / "result" / (
            experiment + ("_smoke" if args.smoke else "")
        )
    return args


def main() -> int:
    args = parse_args()
    backbone_cls = BACKBONES[args.backbone]
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    torch.set_num_threads(args.threads)
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
                if key not in {"output", "dry_run", "summarize_only"}}
    source_paths = [Path(__file__), *sorted((ROOT / "ts_benchmark" / "baselines" / "core").rglob("*.py"))]
    architecture = (
        "Online-conditioned core: full ChDiv CI+CD backbone plus zero-initialized "
        "position-aligned query residual; all branches retained"
        if args.backbone == "online_conditioned"
        else "paper temporal patch attention + axial variable attention; all branches retained"
    )
    manifest = {"settings": settings, "resolved_device": str(device), "torch_version": torch.__version__,
                "architecture": architecture,
                "conditions": {key: asdict(value) for key, value in CONDITIONS.items()},
                "sources": {str(path.relative_to(ROOT)): file_hash(path) for path in source_paths},
                "datasets": {}}
    prepared = {}
    for station in args.datasets:
        path = args.data_dir / f"{station}.csv"
        frame = load_frame(path, args.protocol)
        origins = split_origins(len(frame), args.history, args.horizon)
        mean, scale = fit_scaler(frame.to_numpy(), int(len(frame) * .7))
        values = ((frame.to_numpy() - mean) / scale).astype(np.float32)
        if args.smoke:
            origins = {name: indices[:16 if name == "train" else 8] for name, indices in origins.items()}
        prepared[station] = (values, calendar_marks(frame.index), origins, mean, scale, frame.index)
        manifest["datasets"][station] = {
            "sha256": file_hash(path), "columns": list(frame.columns), "rows": len(frame),
            "train_end_exclusive": int(len(frame) * .7), "validation_end_exclusive": int(len(frame) * .8),
            "scaler_mean": mean.tolist(), "scaler_scale": scale.tolist(),
            "windows": {name: len(indices) for name, indices in origins.items()},
            "alpha_init": ALPHA_INIT[station],
        }
    count = len(args.datasets) * len(args.seeds) * len(CONDITIONS)
    print(f"{count} independent training runs; normalization={args.normalization}; device={device}", flush=True)
    if args.dry_run:
        print(json.dumps(manifest["datasets"], indent=2))
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Output manifest differs: use a new --output; no existing results were overwritten")
    else:
        if args.summarize_only or any(args.output.iterdir()):
            raise ValueError("Expected an empty new output directory or an existing matching manifest")
        write_json(manifest_path, manifest)
    rows = []
    for station in args.datasets:
        values, marks, origins, mean, scale, dates = prepared[station]
        for seed in args.seeds:
            initial_hash = None
            for condition, information in CONDITIONS.items():
                directory = args.output / station / f"seed_{seed}" / condition
                completion = directory / "completed.json"
                if completion.exists():
                    completed = json.loads(completion.read_text(encoding="utf-8"))
                    if any(not (directory / name).exists() for name in completed["artifacts"]):
                        raise ValueError(f"Missing artifact in completed run: {directory}")
                    fingerprint = completed["initial_state_sha256"]
                    print(f"[resume] {station}/{seed}/{condition}", flush=True)
                else:
                    if args.summarize_only:
                        raise ValueError(f"Incomplete matrix: {directory}")
                    print(f"[train] {station}/{seed}/{condition}", flush=True)
                    directory.mkdir(parents=True, exist_ok=True)
                    seed_everything(seed)
                    config = model_config(values.shape[1], args.history, args.horizon, args.width, args.patch)
                    model = ControlledModel(
                        config,
                        information,
                        args.normalization,
                        ALPHA_INIT[station],
                        backbone_cls=backbone_cls,
                    ).to(device)
                    fingerprint = state_hash(model)
                    training = Windows(values, marks, origins["train"], args.history, args.horizon)
                    validation = Windows(values, marks, origins["validation"], args.history, args.horizon)
                    training_result = fit(model, loader(training, args.batch_size, seed, True),
                                          loader(validation, args.batch_size, seed), args, device, directory)
                    evaluations = [("clean", 0)]
                    if condition == "THQ":
                        evaluations += [("shuffle", s) for s in args.perturbation_seeds]
                        evaluations += [(f"lag_{lag}", 0) for lag in args.lags]
                    run_rows, artifacts = [], ["best.pt", "training.csv"]
                    for treatment, perturbation_seed in evaluations:
                        data = Windows(values, marks, origins["test"], args.history, args.horizon,
                                       treatment, perturbation_seed)
                        actual, predicted, positions = predict(model, loader(data, args.batch_size, seed), device)
                        filename = f"predictions_{treatment}_{perturbation_seed}.npz"
                        target_positions = positions[:, None] + np.arange(args.horizon)
                        np.savez_compressed(directory / filename, actual_norm=actual, predicted_norm=predicted,
                                            actual_raw=actual * scale[0] + mean[0],
                                            predicted_raw=predicted * scale[0] + mean[0], origins=positions,
                                            target_timestamps=dates.asi8[target_positions],
                                            inference_timestamps=dates.asi8[positions + args.horizon - 1])
                        artifacts.append(filename)
                        run_rows.append({"station": station, "seed": seed, "condition": condition,
                                         "treatment": treatment, "perturbation_seed": perturbation_seed,
                                         "test_windows": len(actual), "parameters": sum(p.numel() for p in model.parameters()),
                                         **training_result, **error_metrics(actual, predicted, float(scale[0]))})
                    completed = {"initial_state_sha256": fingerprint, "artifacts": artifacts, "metrics": run_rows}
                    write_json(completion, completed)
                if initial_hash is not None and initial_hash != fingerprint:
                    raise RuntimeError("Paired conditions have different initial model weights")
                initial_hash = fingerprint
                rows.extend(completed["metrics"])
    summarize(rows, args.output, args.smoke)
    print(f"Complete: {args.output / 'report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
