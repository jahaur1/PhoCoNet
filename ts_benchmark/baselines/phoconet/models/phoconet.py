"""PhoCoNet: position-free block soft sensing under assay delay."""

from __future__ import annotations

import torch
from torch import nn

from ts_benchmark.baselines.phoconet.models.joint_refinement import (
    Model as IsolatedJointModel,
)
from ts_benchmark.baselines.phoconet.models.online_conditioning import (
    AlignedQVariableModulation,
    Model as PhoCoNetQModel,
)


class Model(IsolatedJointModel):
    """Fuse historical context and current online measurements.

    The target correction conditions on current online measurements, their
    changes from the latest historical auxiliary observations, four verified-TP
    history summaries, and the provisional TP estimate. It deliberately omits
    an output-position coordinate from the gated correction and bounded
    variable-modulation context.
    """

    def __init__(self, configs) -> None:
        super().__init__(configs)
        hidden = int(getattr(configs, "q_hidden", 32))
        gate_bias = float(getattr(configs, "q_gate_bias", -2.0))
        feature_dim = 2 * self.q_exog_dim + 5 * self.target_channels

        original_norm = self.q_feature_norm
        original_delta = self.q_delta
        original_gate = self.q_gate
        original_modulation = self.aligned_q_variable_modulation

        # Rebuild only the correction modules whose dimensions previously
        # included the scalar position coordinate. fork_rng prevents this
        # replacement from changing the subsequent training random stream.
        with torch.random.fork_rng(devices=[]):
            self.q_feature_norm = nn.LayerNorm(feature_dim)
            self.q_delta = nn.Sequential(
                nn.Linear(feature_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, self.target_channels),
            )
            self.q_gate = nn.Sequential(
                nn.Linear(feature_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, self.target_channels),
                nn.Sigmoid(),
            )
            nn.init.zeros_(self.q_delta[-1].weight)
            nn.init.zeros_(self.q_delta[-1].bias)
            nn.init.zeros_(self.q_gate[-2].weight)
            nn.init.constant_(self.q_gate[-2].bias, gate_bias)

            if self.use_aligned_q_variable_modulation:
                self.aligned_q_variable_modulation = AlignedQVariableModulation(
                    variable_count=self.q_exog_dim,
                    context_dim=5 * self.target_channels,
                    target_channels=self.target_channels,
                    attention_dim=int(
                        getattr(configs, "q_aligned_modulation_dim", 32)
                    ),
                    hidden_dim=int(
                        getattr(configs, "q_aligned_modulation_hidden", 32)
                    ),
                    max_modulation=float(
                        getattr(configs, "q_aligned_modulation_max", 0.25)
                    ),
                    output_scale=float(
                        getattr(configs, "q_aligned_modulation_output_scale", 0.01)
                    ),
                )
            else:
                self.aligned_q_variable_modulation = None

        # Preserve every common initialization coordinate and remove only the
        # final column that represented position. This isolates the structural
        # dimension change from an unrelated reinitialization change.
        with torch.no_grad():
            self.q_feature_norm.weight.copy_(original_norm.weight[:-1])
            self.q_feature_norm.bias.copy_(original_norm.bias[:-1])
            self.q_delta[0].weight.copy_(original_delta[0].weight[:, :-1])
            self.q_delta[0].bias.copy_(original_delta[0].bias)
            self.q_delta[-1].weight.copy_(original_delta[-1].weight)
            self.q_delta[-1].bias.copy_(original_delta[-1].bias)
            self.q_gate[0].weight.copy_(original_gate[0].weight[:, :-1])
            self.q_gate[0].bias.copy_(original_gate[0].bias)
            self.q_gate[-2].weight.copy_(original_gate[-2].weight)
            self.q_gate[-2].bias.copy_(original_gate[-2].bias)

            if self.aligned_q_variable_modulation is not None:
                if original_modulation is None:
                    raise RuntimeError("Missing source modulation parameters")
                compact = self.aligned_q_variable_modulation
                compact.value_projection.weight.copy_(
                    original_modulation.value_projection.weight
                )
                compact.context_projection.weight.copy_(
                    original_modulation.context_projection.weight[:, :-1]
                )
                compact.variable_identity.copy_(
                    original_modulation.variable_identity
                )
                compact.score.weight.copy_(original_modulation.score.weight)
                compact.modulation_norm.weight.copy_(
                    original_modulation.modulation_norm.weight[:-1]
                )
                compact.modulation_norm.bias.copy_(
                    original_modulation.modulation_norm.bias[:-1]
                )
                compact.modulation_head[0].weight.copy_(
                    original_modulation.modulation_head[0].weight[:, :-1]
                )
                compact.modulation_head[0].bias.copy_(
                    original_modulation.modulation_head[0].bias
                )
                compact.modulation_head[-1].weight.copy_(
                    original_modulation.modulation_head[-1].weight
                )
                compact.modulation_head[-1].bias.copy_(
                    original_modulation.modulation_head[-1].bias
                )

    def q_correction(
        self,
        x_enc: torch.Tensor,
        exog_future: torch.Tensor | None,
        prediction: torch.Tensor,
    ) -> torch.Tensor:
        """Return the position-free target correction [B, F, target_dim]."""
        if exog_future is None:
            return prediction.new_zeros(
                (prediction.shape[0], self.pred_len, self.target_channels)
            )
        if exog_future.shape[1] != self.pred_len:
            raise ValueError(
                "PhoCoNet expects one online-observation vector per "
                f"output step: got {exog_future.shape[1]}, expected {self.pred_len}"
            )
        if exog_future.shape[-1] != self.q_exog_dim:
            raise ValueError(
                "online-observation dimension does not match historical "
                f"auxiliaries: got {exog_future.shape[-1]}, "
                f"expected {self.q_exog_dim}"
            )

        target_history = x_enc[..., : self.target_channels]
        exog_history = x_enc[..., self.target_channels :]
        horizon = exog_future.shape[1]

        if self.q_mode == "mean":
            branch_q = exog_future.mean(dim=1, keepdim=True).expand_as(
                exog_future
            )
        elif self.q_mode == "shifted":
            branch_q = torch.roll(exog_future, shifts=horizon // 2, dims=1)
        elif self.q_mode == "zero":
            branch_q = torch.zeros_like(exog_future)
        else:
            branch_q = exog_future

        target_summary = torch.cat(
            (
                target_history.mean(dim=1),
                target_history.std(dim=1, unbiased=False),
                target_history[:, -1],
                target_history[:, -1] - target_history[:, 0],
            ),
            dim=-1,
        ).unsqueeze(1).expand(-1, horizon, -1)
        last_exog = exog_history[:, -1:, :].expand(-1, horizon, -1)
        exog_change = branch_q - last_exog
        base_target = prediction[..., : self.target_channels]
        target_context = torch.cat((target_summary, base_target), dim=-1)

        features = self.q_feature_norm(
            torch.cat(
                (branch_q, exog_change, target_summary, base_target),
                dim=-1,
            )
        )
        correction = self.q_gate(features) * self.q_delta(features)
        if self.aligned_q_variable_modulation is not None:
            correction = correction * self.aligned_q_variable_modulation(
                branch_q,
                exog_change,
                target_context,
            )
        if self.aligned_q_variable_residual is not None:
            correction = correction + self.aligned_q_variable_residual(
                branch_q,
                exog_change,
                target_context,
            )
        return correction

    def forecast(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        exog_future: torch.Tensor | None = None,
    ) -> torch.Tensor:
        prediction = super(PhoCoNetQModel, self).forecast(
            x_enc, x_mark_enc, exog_future
        )
        correction = self.q_correction(x_enc, exog_future, prediction)
        prediction = prediction.clone()
        prediction[..., : self.target_channels] += correction
        return prediction


__all__ = ["Model"]
