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
from matplotlib.font_manager import FontProperties
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

STATION_DISPLAY = {
    "juzizhou": "Juzizhou",
    "sanjiaozhou": "Sanjiaozhou",
    "laodaohe": "Laodaohe",
}
HISTORY_COLOR = "#4C78A8"
ONLINE_COLOR = "#F28E2B"


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
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, dict, pd.DataFrame]:
    """Aggregate time-step attributions into one row per water-quality variable.

    The beeswarm x position is the signed sum across every available time step
    for a variable. Its colour is the variable's mean normalized value over
    the same input window. Source-specific absolute masses are retained in a
    separate table so history/online provenance is not lost by aggregation.
    """
    feature_names = target_cols + exog_cols
    grouped_shap = []
    grouped_values = []
    absolute_importance = []
    source_rows = []
    for channel, name in enumerate(feature_names):
        history_attr = attributions[:, :HISTORY, channel]
        signed_shap = history_attr.sum(axis=1)
        history_importance = float(np.abs(history_attr).sum(axis=1).mean())
        online_importance = 0.0

        if channel == 0:
            # Future TP is the output being predicted and is never an input.
            feature_value = samples[:, :HISTORY, channel].mean(axis=1)
        else:
            online_attr = attributions[:, HISTORY:, channel]
            signed_shap = signed_shap + online_attr.sum(axis=1)
            online_importance = float(np.abs(online_attr).sum(axis=1).mean())
            feature_value = samples[:, :, channel].mean(axis=1)

        grouped_shap.append(signed_shap)
        grouped_values.append(feature_value)
        absolute_importance.append(history_importance + online_importance)
        source_rows.extend(
            [
                {"variable": name, "source": "History", "absolute_mass": history_importance},
                {"variable": name, "source": "Online window", "absolute_mass": online_importance},
            ]
        )

    grouped_shap_array = np.stack(grouped_shap, axis=1)
    grouped_value_array = np.stack(grouped_values, axis=1)
    importance = np.asarray(absolute_importance, dtype=float)
    importance /= max(importance.sum(), 1e-12)
    source_breakdown = pd.DataFrame(source_rows)
    source_breakdown["importance"] = source_breakdown["absolute_mass"] / max(
        source_breakdown["absolute_mass"].sum(), 1e-12
    )
    history_mass = float(np.abs(attributions[:, :HISTORY]).sum())
    online_mass = float(np.abs(attributions[:, HISTORY:, 1:]).sum())
    sources = {
        "history_share": history_mass / max(history_mass + online_mass, 1e-12),
        "online_share": online_mass / max(history_mass + online_mass, 1e-12),
    }
    return (
        feature_names,
        grouped_shap_array,
        grouped_value_array,
        importance,
        sources,
        source_breakdown,
    )


def plot_station(
    station: str,
    labels: list[str],
    grouped_shap: np.ndarray,
    grouped_values: np.ndarray,
    importance: np.ndarray,
    sources: dict,
    out: Path,
) -> None:
    display_name = STATION_DISPLAY.get(station, station)
    num_explained = len(grouped_shap)
    display_count = min(200, num_explained)
    display_indices = np.linspace(0, num_explained - 1, display_count).round().astype(int)

    def render_beeswarm(
        shap_values: np.ndarray,
        feature_values: np.ndarray,
        title_suffix: str,
        marker_size: float,
        alpha: float,
        filenames: list[str],
    ) -> None:
        plt.figure(figsize=(9.4, 5.9))
        shap.summary_plot(
            shap_values,
            feature_values,
            feature_names=labels,
            max_display=len(labels),
            show=False,
            plot_type="dot",
            plot_size=None,
        )
        figure = plt.gcf()
        figure.set_size_inches(9.4, 5.9)
        axis = figure.axes[0]
        for collection in axis.collections:
            if hasattr(collection, "set_sizes"):
                collection.set_sizes([marker_size])
                collection.set_alpha(alpha)
                collection.set_edgecolor("white")
                collection.set_linewidth(0.20)
        axis.set_title(
            f"{display_name} — variable-level GradientSHAP\n{title_suffix}",
            pad=11,
            fontsize=14,
            weight="bold",
        )
        axis.set_xlabel("GradientSHAP value (impact on mean 24-step TP estimate)")
        axis.grid(axis="y", color="#E6E6E6", linewidth=0.7, linestyle="--")
        axis.set_axisbelow(True)
        if len(figure.axes) > 1:
            figure.axes[-1].set_ylabel("Normalized feature value", labelpad=10)
        plt.tight_layout()
        for filename in filenames:
            figure.savefig(out / filename, dpi=300, bbox_inches="tight")
        plt.close(figure)

    render_beeswarm(
        grouped_shap,
        grouped_values,
        f"All {num_explained} test windows",
        marker_size=23,
        alpha=0.55,
        filenames=[f"shap_summary_phoconet_{station}_all{num_explained}.png"],
    )
    render_beeswarm(
        grouped_shap[display_indices],
        grouped_values[display_indices],
        f"{display_count}-window display sample; statistics use all {num_explained}",
        marker_size=38,
        alpha=0.72,
        filenames=[
            f"shap_summary_phoconet_{station}_display{display_count}.png",
            f"shap_summary_phoconet_{station}.png",
        ],
    )

    order = np.argsort(importance)
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    ax.barh(np.arange(len(labels)), importance[order], color="#5B8FF9")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels([labels[index] for index in order])
    ax.set_xlabel("Normalized mean absolute GradientSHAP contribution")
    ax.set_title(f"{display_name} — variable importance", pad=12, weight="semibold")
    ax.grid(axis="x", color="#EAEAEA", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / f"shap_importance_phoconet_{station}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    shares = [sources["history_share"], sources["online_share"]]
    y = np.array([1, 0])
    ax.barh(y, [1, 1], color="#EEF1F5", height=0.34)
    ax.barh(y, shares, color=[HISTORY_COLOR, ONLINE_COLOR], height=0.34)
    for row, share, color in zip(y, shares, [HISTORY_COLOR, ONLINE_COLOR]):
        ax.text(
            share + 0.018,
            row,
            f"{share:.1%}",
            va="center",
            ha="left",
            color=color,
            fontsize=12,
            weight="bold",
        )
    ax.set_yticks(y, ["History", "Online window"])
    ax.set_xlim(0, 1.12)
    ax.set_xticks([])
    ax.set_title(f"{display_name} — attribution source", pad=10, fontsize=14, weight="bold")
    ax.spines[:].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=11)
    fig.tight_layout()
    fig.savefig(out / f"shap_source_share_phoconet_{station}.png", dpi=300, bbox_inches="tight")
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
    if num_windows <= 0 or num_windows >= len(test_origins):
        selected = test_origins
    else:
        selected = test_origins[
            np.linspace(0, len(test_origins) - 1, num_windows).round().astype(int)
        ]
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
    labels, grouped_shap, grouped_values, importance, sources, source_breakdown = group_attributions(
        attributions, samples, target_cols, exog_cols
    )
    plot_station(station, labels, grouped_shap, grouped_values, importance, sources, out)

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
    importance_frame = pd.DataFrame({"variable": labels, "importance": importance})
    importance_frame.to_csv(out / "importance.csv", index=False)
    source_breakdown.to_csv(out / "importance_by_variable_and_source.csv", index=False)
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
    pivot = combined.pivot(index="variable", columns="station", values="importance")
    pivot["mean"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("mean", ascending=True)
    pivot.to_csv(OUTPUT / "cross_station_importance_pivot.csv")

    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    stations = [station for station in DATASETS if station in pivot.columns]
    y = np.arange(len(pivot))
    width = min(0.28, 0.72 / len(stations))
    for offset, station in enumerate(stations):
        centered_offset = (offset - (len(stations) - 1) / 2) * width
        ax.barh(y + centered_offset, pivot[station], height=width, label=station)
    ax.set_yticks(y)
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Normalized mean absolute GradientSHAP contribution")
    ax.set_title("Variable importance across stations", pad=12, weight="semibold")
    ax.legend(frameon=False)
    ax.grid(axis="x", color="#EAEAEA", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT / "shap_importance_cross_station.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    sources = combined.groupby("station", sort=False)[["history_share", "online_share"]].first()
    sources = sources.reindex(stations)
    sources.to_csv(OUTPUT / "cross_station_source_share.csv")

    chinese_names = {
        "juzizhou": "桔子洲",
        "sanjiaozhou": "三角洲",
        "laodaohe": "捞刀河",
    }
    regular_font = FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc", size=12)
    bold_font = FontProperties(fname=r"C:\Windows\Fonts\msyhbd.ttc", size=12)
    title_font = FontProperties(fname=r"C:\Windows\Fonts\msyhbd.ttc", size=16)
    note_font = FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc", size=9)
    table_rows = [
        [
            chinese_names.get(station, station),
            f"{sources.loc[station, 'history_share']:.1%}",
            f"{sources.loc[station, 'online_share']:.1%}",
        ]
        for station in stations
    ]

    fig, ax = plt.subplots(figsize=(7.4, 2.6))
    ax.axis("off")
    table = ax.table(
        cellText=table_rows,
        colLabels=["数据集", "历史信息占比", "在线窗口占比"],
        cellLoc="center",
        colLoc="center",
        colWidths=[0.28, 0.36, 0.36],
        bbox=[0.04, 0.20, 0.92, 0.58],
    )
    table.auto_set_font_size(False)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_linewidth(2.0)
        if row == 0:
            cell.set_facecolor("#34495E")
            cell.get_text().set_color("white")
            cell.get_text().set_fontproperties(bold_font)
        else:
            cell.get_text().set_fontproperties(regular_font)
            if column == 0:
                cell.set_facecolor("#F3F5F7")
                cell.get_text().set_fontproperties(bold_font)
                cell.get_text().set_color("#273444")
            elif column == 1:
                cell.set_facecolor("#E8F1FA")
                cell.get_text().set_color(HISTORY_COLOR)
                cell.get_text().set_fontproperties(bold_font)
            else:
                cell.set_facecolor("#FFF0E2")
                cell.get_text().set_color(ONLINE_COLOR)
                cell.get_text().set_fontproperties(bold_font)

    fig.text(0.5, 0.90, "信息来源贡献占比", ha="center", va="center", fontproperties=title_font)
    fig.text(
        0.5,
        0.08,
        "注：占比基于全部 435 个测试窗口的绝对 GradientSHAP 归因总量。",
        ha="center",
        va="center",
        color="#667085",
        fontproperties=note_font,
    )
    for filename in ["shap_source_share_cross_station.png", "shap_source_share_table.png"]:
        fig.savefig(OUTPUT / filename, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
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
    for station in stations:
        station_rows = combined[combined.station == station].sort_values("importance", ascending=False)
        top = ", ".join(
            f"{row.variable} ({row.importance:.1%})"
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
    parser.add_argument(
        "--num-windows",
        type=int,
        default=0,
        help="number of evenly spaced test windows; 0 (default) uses every test window",
    )
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
    if len(frames) > 1:
        cross_station(frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
