"""Per-variable H--Q fusion followed by one target-conditioned interaction."""

from __future__ import annotations

import torch
import torch.nn as nn

from ts_benchmark.baselines.phoconet.models.late_interaction import (
    TargetConditionedLateInteraction,
)
from ts_benchmark.baselines.phoconet.models.online_conditioning import (
    Model as PhoCoNetQModel,
)


class Model(PhoCoNetQModel):
    """Isolate every variable before a single late cross-variable fusion."""

    def __init__(self, configs) -> None:
        super().__init__(configs)
        self.late_aggregation = str(
            getattr(configs, "late_aggregation", "attention")
        )
        if self.late_aggregation not in {"attention", "uniform"}:
            raise ValueError("late_aggregation must be attention or uniform")

        for module in (self.q_feature_norm, self.q_delta, self.q_gate):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        with torch.random.fork_rng(devices=[]):
            self.late_interaction = TargetConditionedLateInteraction(
                sequence_length=self.seq_len,
                horizon=self.pred_len,
                variable_count=self.q_exog_dim,
                target_channels=self.target_channels,
                interaction_dim=int(getattr(configs, "late_dim", 32)),
                num_heads=int(getattr(configs, "late_heads", 4)),
                hidden_dim=int(getattr(configs, "late_hidden", 32)),
                output_scale=float(
                    getattr(configs, "late_output_scale", 1e-2)
                ),
                gate_init=float(getattr(configs, "late_gate_init", 0.1)),
                use_auxiliary_prediction=True,
            )

        # These modules are created lazily with the other fusion components.
        self.isolated_local_gate: nn.Module | None = None
        self.isolated_future_projection: nn.Module | None = None

    def init_fusion_components(self, exog_dim: int, alpha_init: float = 0.0):
        """Create only variable-wise fusion; no layer receives the full Q vector."""
        if self._fusion_initialized or exog_dim <= 0:
            self._fusion_initialized = True
            return
        if exog_dim != self.q_exog_dim:
            raise ValueError("future auxiliary dimension mismatch")
        device = next(self.parameters()).device
        self.isolated_local_gate = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid(),
        ).to(device)
        self.isolated_future_projection = nn.Sequential(
            nn.Linear(1, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        ).to(device)
        # This gate is applied to one channel token at a time. It therefore
        # changes within-variable fusion without exchanging variables.
        self.future_exog_gate = nn.Sequential(
            nn.Linear(2 * self.d_model, self.d_model),
            nn.Sigmoid(),
        ).to(device)
        self.fusion_alpha_logit = nn.Parameter(
            torch.tensor(float(alpha_init), device=device)
        )
        self.exog_dim = int(exog_dim)
        self._fusion_initialized = True

    def _path1_gated_overwrite(
        self,
        x_enc: torch.Tensor,
        exog_future: torch.Tensor | None,
    ) -> torch.Tensor:
        if exog_future is None or self.isolated_local_gate is None:
            return x_enc
        future_len = min(exog_future.shape[1], self.pred_len, self.seq_len)
        if future_len <= 0:
            return x_enc
        history_tail = x_enc[:, -future_len:, :].clone()
        historical_auxiliary = history_tail[
            :, :, self.target_channels : self.target_channels + self.q_exog_dim
        ]
        current_auxiliary = exog_future[:, -future_len:, :]
        gate_input = torch.stack(
            (historical_auxiliary, current_auxiliary), dim=-1
        )
        gate = self.isolated_local_gate(gate_input).squeeze(-1)
        fused_auxiliary = (
            gate * current_auxiliary
            + (1.0 - gate) * historical_auxiliary
        )
        history_tail = torch.cat(
            (
                history_tail[:, :, : self.target_channels],
                fused_auxiliary,
                history_tail[
                    :,
                    :,
                    self.target_channels + self.q_exog_dim :,
                ],
            ),
            dim=-1,
        )
        x_enc = torch.cat((x_enc, history_tail), dim=1)
        return x_enc[:, -self.seq_len :, :]

    def _path2_embedding_enhance(
        self,
        x_enc_emb: torch.Tensor,
        exog_future: torch.Tensor,
    ) -> torch.Tensor:
        if (
            self.isolated_future_projection is None
            or self.future_exog_gate is None
        ):
            return x_enc_emb
        current_summary = exog_future.mean(dim=1).unsqueeze(-1)
        auxiliary_embedding = self.isolated_future_projection(current_summary)
        auxiliary_embedding = auxiliary_embedding.unsqueeze(2).expand(
            -1, -1, x_enc_emb.shape[2], -1
        )
        injected = torch.cat(
            (
                torch.zeros_like(x_enc_emb[:, : self.target_channels]),
                auxiliary_embedding,
                torch.zeros_like(
                    x_enc_emb[
                        :,
                        self.target_channels + self.q_exog_dim :,
                    ]
                ),
            ),
            dim=1,
        )
        gate = self.future_exog_gate(torch.cat((x_enc_emb, injected), dim=-1))
        # The target and calendar channels receive an exact zero update.
        return x_enc_emb + gate * injected

    def forecast(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        exog_future: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # The parent of PhoCoNetQ invokes the overridden isolated paths.
        prediction = super(PhoCoNetQModel, self).forecast(
            x_enc, x_mark_enc, exog_future
        )
        if exog_future is None:
            return prediction
        correction = self.late_interaction(
            x_enc[..., : self.target_channels],
            x_enc[..., self.target_channels :],
            exog_future,
            prediction[..., : self.target_channels],
            aggregation=self.late_aggregation,
            auxiliary_prediction=prediction[
                ...,
                self.target_channels : self.target_channels + self.q_exog_dim,
            ],
        )
        prediction = prediction.clone()
        prediction[..., : self.target_channels] += correction
        return prediction

    def get_late_attention(self) -> torch.Tensor | None:
        return self.late_interaction.get_last_attention()

    def get_late_gate(self) -> torch.Tensor | None:
        return self.late_interaction.get_last_gate()


__all__ = ["Model"]
