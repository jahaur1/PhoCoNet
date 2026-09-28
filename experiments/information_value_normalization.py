"""Run the five controlled information sets with scale-consistent Q inputs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from experiments import information_value_core as base
from ts_benchmark.baselines.phoconet.models.online_conditioning import (
    Model as OnlineConditionedModel,
)


ROOT = Path(__file__).resolve().parents[1]


class QNormControlledModel(base.ControlledModel):
    """Apply historical-H window statistics to Q when both are available."""

    def prepare_inputs(
        self,
        history: torch.Tensor,
        query: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if query.shape[-1] != history.shape[-1] - 1:
            raise ValueError(
                "Query must contain auxiliary variables only, without TP"
            )
        available = torch.ones(
            history.shape[-1], dtype=torch.bool, device=history.device
        )
        available[0] = self.information.target_history
        available[1:] = self.information.auxiliary_history
        masked = torch.where(available, history, torch.zeros_like(history))
        query = query if self.information.query else torch.zeros_like(query)

        if self.normalization == "safe-revin":
            mean = masked.mean(dim=1, keepdim=True).detach()
            scale = (
                masked.var(dim=1, keepdim=True, unbiased=False) + 1e-5
            ).sqrt().detach()
            mean = torch.where(available, mean, torch.zeros_like(mean))
            scale = torch.where(available, scale, torch.ones_like(scale))
            masked = (masked - mean) / scale
            if self.information.query:
                # If H is unavailable, its fallback mean=0 and scale=1 leave
                # Q unchanged. If H is available, Q and historical H now use
                # the same per-window coordinate system.
                query = (query - mean[..., 1:]) / scale[..., 1:]
        else:
            mean = torch.zeros_like(masked[:, :1])
            scale = torch.ones_like(masked[:, :1])
        return masked, query, mean, scale

    def forward(
        self,
        history: torch.Tensor,
        query: torch.Tensor,
        marks: torch.Tensor,
    ) -> torch.Tensor:
        masked, query, mean, scale = self.prepare_inputs(history, query)
        prediction = self.backbone(
            masked, x_mark_enc=marks, exog_future=query
        )[..., 0]
        if self.normalization == "safe-revin":
            prediction = prediction * scale[..., 0] + mean[..., 0]
        return prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=base.STATIONS,
        default=list(base.STATIONS),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[2021])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "dataset" / "forecasting",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "result"
        / "information_value_qnorm_single_run",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.datasets = args.datasets[:1]
        args.seeds = args.seeds[:1]
        args.output = (
            ROOT / "result" / "information_value_qnorm_smoke"
        )
    if args.threads <= 0:
        parser.error("threads must be positive")
    if len(set(args.datasets)) != len(args.datasets):
        parser.error("Duplicate datasets")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Duplicate seeds")
    if any(seed < 0 or seed >= 2**32 for seed in args.seeds):
        parser.error("Seeds must lie in [0, 2**32)")
    return args


def main() -> int:
    args = parse_args()
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    torch.set_num_threads(args.threads)
    history, horizon, patch = 96, 24, 24
    width = 16 if args.smoke else 128
    epochs = 1 if args.smoke else 100
    batch_size = 8 if args.smoke else 64
    run_args = argparse.Namespace(
        lr=0.001,
        epochs=epochs,
        patience=10,
        batch_size=batch_size,
    )
    settings = {
        "datasets": list(args.datasets),
        "seeds": list(args.seeds),
        "protocol": "full8",
        "normalization": "safe-revin-qnorm",
        "backbone": "online_conditioned",
        "history": history,
        "horizon": horizon,
        "patch": patch,
        "width": width,
        "epochs": epochs,
        "patience": run_args.patience,
        "batch_size": batch_size,
        "lr": run_args.lr,
        "lags": [1, 6],
        "perturbation_seeds": [11, 22, 33],
        "device": args.device,
        "threads": args.threads,
        "data_dir": str(args.data_dir),
        "smoke": args.smoke,
    }
    source_paths = [
        Path(__file__),
        Path(base.__file__),
        *sorted(
            (ROOT / "ts_benchmark" / "baselines" / "phoconet").rglob(
                "*.py"
            )
        ),
    ]
    manifest = {
        "settings": settings,
        "resolved_device": str(device),
        "torch_version": torch.__version__,
        "architecture": (
            "Online-conditioned core with history-window-consistent normalization of Q; "
            "all branches retained"
        ),
        "conditions": {
            key: asdict(value) for key, value in base.CONDITIONS.items()
        },
        "sources": {
            str(path.relative_to(ROOT)): base.file_hash(path)
            for path in source_paths
        },
        "datasets": {},
    }
    prepared = {}
    for station in args.datasets:
        path = args.data_dir / f"{station}.csv"
        frame = base.load_frame(path, "full8")
        origins = base.split_origins(len(frame), history, horizon)
        mean, scale = base.fit_scaler(
            frame.to_numpy(), int(len(frame) * 0.7)
        )
        values = ((frame.to_numpy() - mean) / scale).astype(np.float32)
        if args.smoke:
            origins = {
                name: indices[: 16 if name == "train" else 8]
                for name, indices in origins.items()
            }
        prepared[station] = (
            values,
            base.calendar_marks(frame.index),
            origins,
            mean,
            scale,
            frame.index,
        )
        manifest["datasets"][station] = {
            "sha256": base.file_hash(path),
            "columns": list(frame.columns),
            "rows": len(frame),
            "train_end_exclusive": int(len(frame) * 0.7),
            "validation_end_exclusive": int(len(frame) * 0.8),
            "scaler_mean": mean.tolist(),
            "scaler_scale": scale.tolist(),
            "windows": {
                name: len(indices) for name, indices in origins.items()
            },
            "alpha_init": base.ALPHA_INIT[station],
        }

    count = len(args.datasets) * len(args.seeds) * len(base.CONDITIONS)
    print(
        f"{count} QNorm training runs; device={device}", flush=True
    )
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError(
                "Output manifest differs: use a new output directory"
            )
    else:
        if args.summarize_only or any(args.output.iterdir()):
            raise ValueError(
                "Expected an empty output directory or matching manifest"
            )
        base.write_json(manifest_path, manifest)

    rows = []
    for station in args.datasets:
        values, marks, origins, mean, scale, dates = prepared[station]
        for seed in args.seeds:
            initial_hash = None
            for condition, information in base.CONDITIONS.items():
                directory = args.output / station / f"seed_{seed}" / condition
                completion = directory / "completed.json"
                if completion.exists():
                    completed = json.loads(
                        completion.read_text(encoding="utf-8")
                    )
                    if any(
                        not (directory / name).exists()
                        for name in completed["artifacts"]
                    ):
                        raise ValueError(
                            f"Missing artifact in completed run: {directory}"
                        )
                    fingerprint = completed["initial_state_sha256"]
                    print(
                        f"[resume] {station}/{seed}/{condition}",
                        flush=True,
                    )
                else:
                    if args.summarize_only:
                        raise ValueError(f"Incomplete matrix: {directory}")
                    print(
                        f"[train] {station}/{seed}/{condition}",
                        flush=True,
                    )
                    directory.mkdir(parents=True, exist_ok=True)
                    base.seed_everything(seed)
                    config = base.model_config(
                        values.shape[1], history, horizon, width, patch
                    )
                    model = QNormControlledModel(
                        config,
                        information,
                        "safe-revin",
                        base.ALPHA_INIT[station],
                        backbone_cls=OnlineConditionedModel,
                    ).to(device)
                    fingerprint = base.state_hash(model)
                    training = base.Windows(
                        values,
                        marks,
                        origins["train"],
                        history,
                        horizon,
                    )
                    validation = base.Windows(
                        values,
                        marks,
                        origins["validation"],
                        history,
                        horizon,
                    )
                    training_result = base.fit(
                        model,
                        base.loader(training, batch_size, seed, True),
                        base.loader(validation, batch_size, seed),
                        run_args,
                        device,
                        directory,
                    )
                    evaluations = [("clean", 0)]
                    if condition == "THQ":
                        evaluations += [
                            ("shuffle", perturbation_seed)
                            for perturbation_seed in (11, 22, 33)
                        ]
                        evaluations += [("lag_1", 0), ("lag_6", 0)]
                    run_rows = []
                    artifacts = ["best.pt", "training.csv"]
                    for treatment, perturbation_seed in evaluations:
                        dataset = base.Windows(
                            values,
                            marks,
                            origins["test"],
                            history,
                            horizon,
                            treatment,
                            perturbation_seed,
                        )
                        actual, predicted, positions = base.predict(
                            model,
                            base.loader(dataset, batch_size, seed),
                            device,
                        )
                        filename = (
                            f"predictions_{treatment}_{perturbation_seed}.npz"
                        )
                        target_positions = positions[:, None] + np.arange(
                            horizon
                        )
                        np.savez_compressed(
                            directory / filename,
                            actual_norm=actual,
                            predicted_norm=predicted,
                            actual_raw=actual * scale[0] + mean[0],
                            predicted_raw=predicted * scale[0] + mean[0],
                            origins=positions,
                            target_timestamps=dates.asi8[target_positions],
                            inference_timestamps=dates.asi8[
                                positions + horizon - 1
                            ],
                        )
                        artifacts.append(filename)
                        run_rows.append(
                            {
                                "station": station,
                                "seed": seed,
                                "condition": condition,
                                "treatment": treatment,
                                "perturbation_seed": perturbation_seed,
                                "test_windows": len(actual),
                                "parameters": sum(
                                    p.numel() for p in model.parameters()
                                ),
                                **training_result,
                                **base.error_metrics(
                                    actual, predicted, float(scale[0])
                                ),
                            }
                        )
                    completed = {
                        "initial_state_sha256": fingerprint,
                        "artifacts": artifacts,
                        "metrics": run_rows,
                    }
                    base.write_json(completion, completed)
                if initial_hash is not None and initial_hash != fingerprint:
                    raise RuntimeError(
                        "Paired conditions have different initial weights"
                    )
                initial_hash = fingerprint
                rows.extend(completed["metrics"])

    base.summarize(rows, args.output, args.smoke)
    print(f"Complete: {args.output / 'report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
