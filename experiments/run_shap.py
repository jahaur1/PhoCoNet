"""Generate final-model GradientSHAP figures for PhoCoNet.

The script retrains the frozen final configuration with deterministic seed
2021, restores the validation-selected checkpoint, and verifies its complete
test prediction array against the archived main experiment before computing
attributions.  Attributions use multi-baseline expected gradients with
Gauss--Legendre integration, a GradientSHAP approximation.  Calendar markers
are held fixed, so the results explain water-quality inputs only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.config import DATASETS
from experiments.run_main import (
    params_for as final_params_for,
)
from ts_benchmark.baselines.phoconet.phoconet import (
    PhoCoNet,
)
from ts_benchmark.baselines.utils import get_time_mark
from ts_benchmark.data.utils import read_data
from ts_benchmark.utils.random_utils import fix_all_random_seed


OUTPUT = ROOT / "result" / "phoconet_shap_seed2021"
MAIN_RESULT = ROOT / "result" / "phoconet_main"
HISTORY = 96
HORIZON = 24
SEED = 2021


def train_final(station: str, checkpoint: Path, num_epochs: int) -> tuple:
    fix_all_random_seed(SEED)
    series = read_data(str(ROOT / "dataset" / "forecasting" / f"{station}.csv"))
    target_cols = ["TP"]
    exog_cols = [column for column in series.columns if column != "TP"]
    params = {
        **final_params_for(station, HORIZON, num_epochs),
        "robustness_checkpoint_cache": str(checkpoint.resolve()),
    }
    params.update(
        {
            "norm": True,
            "fusion_method": "",
            "parallel_strategy": None,
            "use_amp": 0,
            "task_name": "short_term_forecast",
        }
    )
    model = PhoCoNet(**params)
    tv_end = int(len(series) * 0.8)
    fit = series.iloc[:tv_end]
    model.forecast_fit(
        fit[target_cols].copy(),
        covariates={"exog": fit[exog_cols].copy()},
        train_ratio_in_tv=0.875,
    )
    if model.check_point is None:
        raise RuntimeError("training produced no validation-selected checkpoint")
    model.model.load_state_dict(model.check_point["Model"])
    model.model.eval()
    return model, series, target_cols, exog_cols


def normalized_values(model, series: pd.DataFrame, target_cols: list[str], exog_cols: list[str]) -> np.ndarray:
    target = model.scaler1.transform(series[target_cols].to_numpy())
    exog = model.scaler2.transform(series[exog_cols].to_numpy())
    return np.concatenate((target, exog), axis=1).astype(np.float32)


def windows_for_origins(
    model,
    series: pd.DataFrame,
    target_cols: list[str],
    exog_cols: list[str],
    origins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    normalized = normalized_values(model, series, target_cols, exog_cols)
    all_marks = get_time_mark(
        series.index.to_numpy()[None, :], timeenc=1, freq=model.config.freq
    )[0]
    values, marks = [], []
    for origin in origins:
        history = normalized[origin - HISTORY : origin].copy()
        future = np.zeros((HORIZON, normalized.shape[1]), dtype=np.float32)
        future[:, 1:] = normalized[origin : origin + HORIZON, 1:]
        values.append(np.concatenate((history, future), axis=0))
        marks.append(all_marks[origin - HISTORY : origin].copy())
    return np.stack(values), np.stack(marks).astype(np.float32)


def differentiable_forward(
    module, history: torch.Tensor, marks: torch.Tensor, future_exog: torch.Tensor
) -> torch.Tensor:
    """Numerically identical final-Q forward with RevIN statistics differentiable.

    The production model detaches its per-window RevIN mean and standard
    deviation.  Detachment leaves forward predictions unchanged but breaks the
    path derivative required by integrated/expected gradients.  This equivalent
    graph retains those derivatives only for attribution.
    """
    original_history = history
    if not module.use_future_exog:
        future_exog = None
    if module.future_exog_mode != "normal":
        raise ValueError("final GradientSHAP requires future_exog_mode='normal'")
    history = module._remove_history_target(history)
    history = module._remove_history_exog(history)
    series_dim = int(getattr(module.configs, "series_dim", 1))
    history = module._mask_exog_columns(history, base_offset=series_dim, which="history")
    future_exog = module._mask_exog_columns(
        future_exog, base_offset=0, which="future"
    )

    if module.revin:
        mean = history.mean(dim=1, keepdim=True)
        std = torch.sqrt(history.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
        normalized = (history - mean) / std
    else:
        mean = std = None
        normalized = history

    base_embedding = module.patch_embed(normalized, marks)
    if module.fusion_mode == "dual":
        local = module._path1_gated_overwrite(normalized.clone(), future_exog)
        local_embedding = module.patch_embed(local, marks)
        global_embedding = module._path2_embedding_enhance(
            base_embedding, future_exog
        )
        if module.alpha_mode == "fixed":
            alpha = module.alpha_fixed
        elif module.fusion_alpha_logit is not None:
            alpha = torch.sigmoid(module.fusion_alpha_logit)
        else:
            alpha = 0.5
        embedding = alpha * local_embedding + (1.0 - alpha) * global_embedding
    elif module.fusion_mode == "only_overwrite":
        local = module._path1_gated_overwrite(normalized.clone(), future_exog)
        embedding = module.patch_embed(local, marks)
    elif module.fusion_mode == "only_embedding":
        embedding = module._path2_embedding_enhance(base_embedding, future_exog)
    else:
        raise ValueError(f"unsupported final-Q fusion mode: {module.fusion_mode}")

    if module.attn_mode == "none":
        encoded = embedding
    elif module.attn_mode == "parallel":
        encoded = (
            module.temporal_encoder(embedding)[0]
            + module.covariate_encoder(embedding)[0]
        ) / 2.0
    else:
        encoded = module.encoder(embedding)[0]
    encoded = module._refine_encoded_channels(encoded[:, : module.c_in])
    prediction = module.decoder(encoded).transpose(-1, -2)
    if module.revin:
        prediction = prediction * std + mean
    prediction = prediction[:, -module.pred_len :, :]
    correction = module.q_correction(original_history, future_exog, prediction)
    target_channels = module.target_channels
    return torch.cat(
        (
            prediction[..., :target_channels] + correction,
            prediction[..., target_channels:],
        ),
        dim=-1,
    )


def forward_scalar(module, values: torch.Tensor, marks: torch.Tensor) -> torch.Tensor:
    history = values[:, :HISTORY, :]
    future_exog = values[:, HISTORY:, 1:]
    output = differentiable_forward(module, history, marks, future_exog)
    return output[:, :, 0].mean(dim=1)


def predict_all(
    model,
    series: pd.DataFrame,
    target_cols: list[str],
    exog_cols: list[str],
    origins: np.ndarray,
) -> np.ndarray:
    values, marks = windows_for_origins(model, series, target_cols, exog_cols, origins)
    device = next(model.model.parameters()).device
    predictions = []
    with torch.no_grad():
        for start in range(0, len(values), 64):
            x = torch.as_tensor(values[start : start + 64], device=device)
            m = torch.as_tensor(marks[start : start + 64], device=device)
            out = model.model(
                x[:, :HISTORY], x_mark_enc=m, exog_future=x[:, HISTORY:, 1:]
            )[:, :, :1]
            predictions.append(out.detach().cpu().numpy())
    pred_norm = np.concatenate(predictions, axis=0)
    return model.scaler1.inverse_transform(pred_norm.reshape(-1, 1)).reshape(pred_norm.shape)


def verify_against_main(
    station: str,
    model,
    series: pd.DataFrame,
    target_cols: list[str],
    exog_cols: list[str],
) -> dict:
    archived = np.load(
        MAIN_RESULT / station / "h24" / "seed_2021" / "predictions.npz"
    )
    tv_end = int(len(series) * 0.8)
    origins = np.arange(tv_end, len(series) - HORIZON + 1, dtype=int)
    reproduced = predict_all(model, series, target_cols, exog_cols, origins)
    expected = archived["predicted"]
    difference = np.abs(reproduced.astype(np.float64) - expected.astype(np.float64))
    result = {
        "windows": int(len(origins)),
        "max_abs_prediction_difference": float(difference.max()),
        "mean_abs_prediction_difference": float(difference.mean()),
        "allclose_rtol_1e-6_atol_1e-6": bool(
            np.allclose(reproduced, expected, rtol=1e-6, atol=1e-6)
        ),
    }
    if not result["allclose_rtol_1e-6_atol_1e-6"]:
        raise RuntimeError(f"{station}: SHAP checkpoint does not reproduce main predictions: {result}")
    return result


def expected_gradients(
    module,
    samples: np.ndarray,
    marks: np.ndarray,
    backgrounds: np.ndarray,
    integration_steps: int,
) -> tuple[np.ndarray, dict]:
    """Multi-baseline integrated gradients (expected-gradients/GradientSHAP)."""
    device = next(module.parameters()).device
    nodes, weights = np.polynomial.legendre.leggauss(integration_steps)
    alpha = torch.as_tensor((nodes + 1.0) / 2.0, dtype=torch.float32, device=device)
    quad_weight = torch.as_tensor(weights / 2.0, dtype=torch.float32, device=device)
    bg = torch.as_tensor(backgrounds, dtype=torch.float32, device=device)
    attributions = np.zeros_like(samples, dtype=np.float32)
    completeness_residuals = []
    completeness_deltas = []
    module.eval()

    for index, (sample_np, mark_np) in enumerate(zip(samples, marks)):
        sample = torch.as_tensor(sample_np, dtype=torch.float32, device=device)
        mark = torch.as_tensor(mark_np, dtype=torch.float32, device=device)
        b_count = bg.shape[0]
        bg_grid = bg[:, None].expand(-1, integration_steps, -1, -1)
        sample_grid = sample[None, None].expand(b_count, integration_steps, -1, -1)
        alpha_grid = alpha[None, :, None, None]
        interpolated = (bg_grid + alpha_grid * (sample_grid - bg_grid)).reshape(
            b_count * integration_steps, *sample.shape
        )
        interpolated = interpolated.clone().detach().requires_grad_(True)
        repeated_marks = mark[None].expand(b_count * integration_steps, -1, -1)
        output = forward_scalar(module, interpolated, repeated_marks)
        gradients = torch.autograd.grad(output.sum(), interpolated)[0]
        gradients = gradients.reshape(b_count, integration_steps, *sample.shape)
        integrated = (gradients * quad_weight[None, :, None, None]).sum(dim=1)
        attribution_by_baseline = (sample[None] - bg) * integrated
        attribution = attribution_by_baseline.mean(dim=0)
        attributions[index] = attribution.detach().cpu().numpy()

        with torch.no_grad():
            fx = forward_scalar(module, sample[None], mark[None])[0]
            bg_marks = mark[None].expand(b_count, -1, -1)
            fb = forward_scalar(module, bg, bg_marks).mean()
            residual = torch.abs(attribution.sum() - (fx - fb))
            completeness_residuals.append(float(residual.cpu()))
            completeness_deltas.append(float(torch.abs(fx - fb).cpu()))

    diagnostic = {
        "method": "multi-baseline expected gradients with Gauss-Legendre integration",
        "background_count": int(len(backgrounds)),
        "integration_steps": int(integration_steps),
        "mean_absolute_completeness_residual": float(np.mean(completeness_residuals)),
        "max_absolute_completeness_residual": float(np.max(completeness_residuals)),
        "aggregate_relative_completeness_error": float(
            np.sum(completeness_residuals)
            / max(np.sum(completeness_deltas), 1e-8)
        ),
    }
    return attributions, diagnostic


def group_attributions(
    attributions: np.ndarray,
    samples: np.ndarray,
    target_cols: list[str],
    exog_cols: list[str],
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, dict]:
    feature_names = target_cols + exog_cols
    labels = [f"{name} (history)" for name in feature_names]
    labels += [f"{name} (online)" for name in exog_cols]
    grouped_shap = []
    grouped_values = []
    absolute_importance = []
    for channel in range(len(feature_names)):
        grouped_shap.append(attributions[:, :HISTORY, channel].sum(axis=1))
        grouped_values.append(samples[:, :HISTORY, channel].mean(axis=1))
        absolute_importance.append(np.abs(attributions[:, :HISTORY, channel]).sum(axis=1).mean())
    for channel in range(1, len(feature_names)):
        grouped_shap.append(attributions[:, HISTORY:, channel].sum(axis=1))
        grouped_values.append(samples[:, HISTORY:, channel].mean(axis=1))
        absolute_importance.append(np.abs(attributions[:, HISTORY:, channel]).sum(axis=1).mean())
    grouped_shap_array = np.stack(grouped_shap, axis=1)
    grouped_value_array = np.stack(grouped_values, axis=1)
    importance = np.asarray(absolute_importance, dtype=float)
    importance /= max(importance.sum(), 1e-12)
    history_mass = float(np.abs(attributions[:, :HISTORY]).sum())
    online_mass = float(np.abs(attributions[:, HISTORY:, 1:]).sum())
    sources = {
        "history_share": history_mass / max(history_mass + online_mass, 1e-12),
        "online_share": online_mass / max(history_mass + online_mass, 1e-12),
    }
    return labels, grouped_shap_array, grouped_value_array, importance, sources


def plot_station(
    station: str,
    labels: list[str],
    grouped_shap: np.ndarray,
    grouped_values: np.ndarray,
    importance: np.ndarray,
    out: Path,
) -> None:
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        grouped_shap,
        grouped_values,
        feature_names=labels,
        max_display=len(labels),
        show=False,
        plot_type="dot",
        plot_size=None,
    )
    plt.title("")
    plt.xlabel("GradientSHAP value for mean 24-step TP estimate")
    plt.tight_layout()
    plt.savefig(out / f"shap_summary_phoconet_{station}.png", dpi=300, bbox_inches="tight")
    plt.close()

    order = np.argsort(importance)
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = ["#4472C4" if labels[index].endswith("(history)") else "#ED7D31" for index in order]
    ax.barh(np.arange(len(labels)), importance[order], color=colors)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels([labels[index] for index in order])
    ax.set_xlabel("Normalized mean absolute GradientSHAP contribution")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / f"shap_importance_phoconet_{station}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_station(
    station: str,
    num_windows: int,
    background_count: int,
    integration_steps: int,
    num_epochs: int,
) -> pd.DataFrame:
    out = OUTPUT / station
    out.mkdir(parents=True, exist_ok=True)
    candidates = sorted(
        (ROOT / "result/phoconet_sensor_noise/model_cache/PhoCoNet" / station).glob(
            f"h{HORIZON}_seed{SEED}_*.pt"
        )
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one frozen PhoCoNet checkpoint for {station}, found {len(candidates)}"
        )
    checkpoint = candidates[0]
    print(f"[train] {station}", flush=True)
    started = time.time()
    model, series, target_cols, exog_cols = train_final(station, checkpoint, num_epochs)
    verification = verify_against_main(station, model, series, target_cols, exog_cols)
    print(
        f"[verified] {station} max prediction difference="
        f"{verification['max_abs_prediction_difference']:.3e}",
        flush=True,
    )

    tv_end = int(len(series) * 0.8)
    test_origins = np.arange(tv_end, len(series) - HORIZON + 1)
    selected = test_origins[np.linspace(0, len(test_origins) - 1, num_windows).round().astype(int)]
    train_end = int(tv_end * 0.875)
    background_origins = np.linspace(
        HISTORY, train_end - HORIZON, background_count
    ).round().astype(int)
    samples, marks = windows_for_origins(model, series, target_cols, exog_cols, selected)
    backgrounds, _ = windows_for_origins(
        model, series, target_cols, exog_cols, background_origins
    )
    device = next(model.model.parameters()).device
    with torch.no_grad():
        check_values = torch.as_tensor(samples[:4], device=device)
        check_marks = torch.as_tensor(marks[:4], device=device)
        actual = model.model(
            check_values[:, :HISTORY],
            x_mark_enc=check_marks,
            exog_future=check_values[:, HISTORY:, 1:],
        )
        equivalent = differentiable_forward(
            model.model,
            check_values[:, :HISTORY],
            check_marks,
            check_values[:, HISTORY:, 1:],
        )
        equivalent_difference = float(torch.max(torch.abs(actual - equivalent)).cpu())
    if equivalent_difference > 1e-6:
        raise RuntimeError(
            f"differentiable attribution graph changes predictions by {equivalent_difference}"
        )
    print(
        f"[explain] {station} windows={len(selected)} backgrounds={len(backgrounds)} "
        f"steps={integration_steps}",
        flush=True,
    )
    attributions, diagnostic = expected_gradients(
        model.model, samples, marks, backgrounds, integration_steps
    )
    labels, grouped_shap, grouped_values, importance, sources = group_attributions(
        attributions, samples, target_cols, exog_cols
    )
    plot_station(station, labels, grouped_shap, grouped_values, importance, out)

    np.savez_compressed(
        out / "gradient_shap_attributions.npz",
        attributions=attributions,
        samples=samples,
        time_markers=marks,
        origins=selected,
        background_origins=background_origins,
        grouped_shap=grouped_shap,
        grouped_values=grouped_values,
        labels=np.asarray(labels),
    )
    pd.DataFrame(grouped_shap, columns=labels).to_csv(out / "grouped_shap_values.csv", index=False)
    pd.DataFrame(grouped_values, columns=labels).to_csv(out / "grouped_feature_values.csv", index=False)
    importance_frame = pd.DataFrame({"feature_source": labels, "importance": importance})
    importance_frame.to_csv(out / "importance.csv", index=False)
    metadata = {
        "station": station,
        "model": "PhoCoNet",
        "seed": SEED,
        "history": HISTORY,
        "horizon": HORIZON,
        "explained_output": "mean normalized TP estimate over 24 steps",
        "explained_inputs": "water-quality inputs; calendar markers held fixed",
        "selected_test_origins": selected.tolist(),
        "background_train_origins": background_origins.tolist(),
        "checkpoint_verification": verification,
        "attribution_diagnostic": diagnostic,
        "equivalent_forward_max_abs_difference": equivalent_difference,
        "source_share": sources,
        "elapsed_seconds": time.time() - started,
    }
    (out / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    importance_frame.insert(0, "station", station)
    importance_frame["history_share"] = sources["history_share"]
    importance_frame["online_share"] = sources["online_share"]
    print(
        f"[done] {station} history={sources['history_share']:.1%} "
        f"online={sources['online_share']:.1%} "
        f"completeness={diagnostic['aggregate_relative_completeness_error']:.2%}",
        flush=True,
    )
    return importance_frame


def cross_station(frames: list[pd.DataFrame]) -> None:
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(OUTPUT / "cross_station_importance.csv", index=False)
    pivot = combined.pivot(index="feature_source", columns="station", values="importance")
    pivot["mean"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("mean", ascending=True)
    pivot.to_csv(OUTPUT / "cross_station_importance_pivot.csv")

    fig, ax = plt.subplots(figsize=(10, 8))
    stations = list(DATASETS)
    y = np.arange(len(pivot))
    width = 0.22
    for offset, station in enumerate(stations):
        ax.barh(y + (offset - 1) * width, pivot[station], height=width, label=station)
    ax.set_yticks(y)
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Normalized mean absolute GradientSHAP contribution")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT / "shap_importance_cross_station.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    sources = combined.groupby("station", sort=False)[["history_share", "online_share"]].first()
    sources.to_csv(OUTPUT / "cross_station_source_share.csv")
    lines = [
        "# PhoCoNet GradientSHAP summary",
        "",
        "Attributions are multi-baseline expected gradients with deterministic "
        "Gauss--Legendre integration. They approximate SHAP values and explain "
        "the mean normalized TP estimate over the 24-step block. Calendar markers "
        "are fixed and are not included in the importance totals.",
        "",
        "| Station | History share | Online-window share | Top three inputs |",
        "|---|---:|---:|---|",
    ]
    for station in DATASETS:
        station_rows = combined[combined.station == station].sort_values("importance", ascending=False)
        top = ", ".join(
            f"{row.feature_source} ({row.importance:.1%})"
            for row in station_rows.head(3).itertuples()
        )
        source = sources.loc[station]
        lines.append(
            f"| {station} | {source.history_share:.1%} | {source.online_share:.1%} | {top} |"
        )
    (OUTPUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stations", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--num-windows", type=int, default=32)
    parser.add_argument("--background-count", type=int, default=8)
    parser.add_argument("--integration-steps", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=100)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frames = [
        run_station(
            station,
            args.num_windows,
            args.background_count,
            args.integration_steps,
            args.num_epochs,
        )
        for station in args.stations
    ]
    if tuple(args.stations) == DATASETS:
        cross_station(frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
