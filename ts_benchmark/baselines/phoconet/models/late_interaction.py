"""PhoCoNet with isolated variable encoders and one late interaction."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ts_benchmark.baselines.phoconet.models.online_conditioning import (
    Model as PhoCoNetQModel,
)


class TargetConditionedLateInteraction(nn.Module):
    """Keep variables separate until TP queries them immediately before output."""

    def __init__(
        self,
        *,
        sequence_length: int,
        horizon: int,
        variable_count: int,
        target_channels: int = 1,
        interaction_dim: int = 32,
        num_heads: int = 4,
        hidden_dim: int = 32,
        output_scale: float = 1e-2,
        gate_init: float = 0.1,
        epsilon: float = 1e-5,
        use_auxiliary_prediction: bool = False,
    ) -> None:
        super().__init__()
        if min(
            sequence_length,
            horizon,
            variable_count,
            target_channels,
            interaction_dim,
            num_heads,
            hidden_dim,
        ) < 1:
            raise ValueError("all late-interaction dimensions must be positive")
        if interaction_dim % num_heads:
            raise ValueError("interaction_dim must be divisible by num_heads")
        if output_scale <= 0.0 or epsilon <= 0.0:
            raise ValueError("output_scale and epsilon must be positive")
        if not 0.0 < gate_init < 1.0:
            raise ValueError("gate_init must lie strictly between zero and one")

        self.sequence_length = int(sequence_length)
        self.horizon = int(horizon)
        self.variable_count = int(variable_count)
        self.target_channels = int(target_channels)
        self.interaction_dim = int(interaction_dim)
        self.epsilon = float(epsilon)
        self.use_auxiliary_prediction = bool(use_auxiliary_prediction)

        # The same temporal projection is applied independently to every
        # auxiliary variable. It cannot exchange information across variables.
        self.auxiliary_history_encoder = nn.Sequential(
            nn.Linear(sequence_length, interaction_dim),
            nn.GELU(),
            nn.LayerNorm(interaction_dim),
        )
        self.current_encoder = nn.Sequential(
            nn.Linear(
                4 if self.use_auxiliary_prediction else 3,
                interaction_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(interaction_dim),
        )
        self.variable_identity = nn.Parameter(
            torch.empty(variable_count, interaction_dim)
        )
        nn.init.normal_(self.variable_identity, mean=0.0, std=0.02)

        self.target_history_encoder = nn.Sequential(
            nn.Linear(
                sequence_length * target_channels,
                interaction_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(interaction_dim),
        )
        self.query_encoder = nn.Sequential(
            nn.Linear(
                interaction_dim + target_channels + 1,
                interaction_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(interaction_dim),
        )
        self.variable_interaction = nn.MultiheadAttention(
            interaction_dim,
            num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.output_fusion = nn.Sequential(
            nn.Linear(2 * interaction_dim, hidden_dim),
            nn.GELU(),
        )
        self.delta_head = nn.Linear(hidden_dim, target_channels)
        self.gate_head = nn.Linear(hidden_dim, target_channels)
        nn.init.normal_(
            self.delta_head.weight, mean=0.0, std=float(output_scale)
        )
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(
            self.gate_head.bias,
            math.log(float(gate_init) / (1.0 - float(gate_init))),
        )
        self._last_attention: torch.Tensor | None = None
        self._last_gate: torch.Tensor | None = None

    def forward(
        self,
        target_history: torch.Tensor,
        auxiliary_history: torch.Tensor,
        current_q: torch.Tensor,
        base_target: torch.Tensor,
        *,
        aggregation: str = "attention",
        auxiliary_prediction: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if aggregation not in {"attention", "uniform"}:
            raise ValueError("aggregation must be attention or uniform")
        if target_history.shape[1:] != (
            self.sequence_length,
            self.target_channels,
        ):
            raise ValueError("target history shape mismatch")
        if auxiliary_history.shape[1:] != (
            self.sequence_length,
            self.variable_count,
        ):
            raise ValueError("auxiliary history shape mismatch")
        if current_q.shape[1:] != (self.horizon, self.variable_count):
            raise ValueError("current Q shape mismatch")
        if base_target.shape[1:] != (self.horizon, self.target_channels):
            raise ValueError("base target shape mismatch")

        auxiliary_mean = auxiliary_history.mean(dim=1, keepdim=True)
        auxiliary_std = auxiliary_history.std(
            dim=1, keepdim=True, unbiased=False
        )
        normalized_history = (
            auxiliary_history - auxiliary_mean
        ) / (auxiliary_std + self.epsilon)
        history_tokens = self.auxiliary_history_encoder(
            normalized_history.transpose(1, 2)
        )

        last_auxiliary = auxiliary_history[:, -1:, :]
        current_features = torch.stack(
            tuple(
                [
                    (current_q - auxiliary_mean) / (
                        auxiliary_std + self.epsilon
                    ),
                    (current_q - last_auxiliary) / (
                        auxiliary_std + self.epsilon
                    ),
                    current_q,
                ]
                + (
                    [
                        (auxiliary_prediction - auxiliary_mean)
                        / (auxiliary_std + self.epsilon)
                    ]
                    if self.use_auxiliary_prediction
                    and auxiliary_prediction is not None
                    else []
                )
            ),
            dim=-1,
        )
        if self.use_auxiliary_prediction and auxiliary_prediction is None:
            raise ValueError("auxiliary prediction is required")
        current_tokens = self.current_encoder(current_features)
        variable_tokens = (
            history_tokens.unsqueeze(1)
            + current_tokens
            + self.variable_identity.view(
                1, 1, self.variable_count, self.interaction_dim
            )
        )

        target_mean = target_history.mean(dim=1, keepdim=True)
        target_std = target_history.std(
            dim=1, keepdim=True, unbiased=False
        )
        normalized_target = (
            target_history - target_mean
        ) / (target_std + self.epsilon)
        target_token = self.target_history_encoder(
            normalized_target.flatten(start_dim=1)
        ).unsqueeze(1).expand(-1, self.horizon, -1)
        position = torch.linspace(
            -1.0,
            1.0,
            self.horizon,
            device=current_q.device,
            dtype=current_q.dtype,
        ).view(1, self.horizon, 1).expand(current_q.shape[0], -1, -1)
        query = self.query_encoder(
            torch.cat((target_token, base_target, position), dim=-1)
        )

        batch_size = current_q.shape[0]
        flat_query = query.reshape(
            batch_size * self.horizon, 1, self.interaction_dim
        )
        flat_variables = variable_tokens.reshape(
            batch_size * self.horizon,
            self.variable_count,
            self.interaction_dim,
        )
        if aggregation == "attention":
            context, attention = self.variable_interaction(
                flat_query,
                flat_variables,
                flat_variables,
                need_weights=True,
                average_attn_weights=False,
            )
            context = context[:, 0]
            attention = attention[:, :, 0, :].reshape(
                batch_size,
                self.horizon,
                -1,
                self.variable_count,
            )
        else:
            context = flat_variables.mean(dim=1)
            attention = current_q.new_full(
                (
                    batch_size,
                    self.horizon,
                    1,
                    self.variable_count,
                ),
                1.0 / self.variable_count,
            )
        context = context.reshape(
            batch_size, self.horizon, self.interaction_dim
        )
        hidden = self.output_fusion(torch.cat((query, context), dim=-1))
        gate = torch.sigmoid(self.gate_head(hidden))
        correction = gate * self.delta_head(hidden)
        self._last_attention = attention.detach()
        self._last_gate = gate.detach()
        return correction

    def get_last_attention(self) -> torch.Tensor | None:
        if self._last_attention is None:
            return None
        return self._last_attention.detach().cpu().clone()

    def get_last_gate(self) -> torch.Tensor | None:
        if self._last_gate is None:
            return None
        return self._last_gate.detach().cpu().clone()


class Model(PhoCoNetQModel):
    """History-only channel-independent backbone followed by one late fusion."""

    def __init__(self, configs) -> None:
        super().__init__(configs)
        self.late_aggregation = str(
            getattr(configs, "late_aggregation", "attention")
        )
        if self.late_aggregation not in {"attention", "uniform"}:
            raise ValueError("late_aggregation must be attention or uniform")

        # The inherited Joint branch is deliberately excluded. The only
        # learned cross-variable exchange occurs in late_interaction.
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
            )

    def forecast(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        exog_future: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Passing None is the architectural isolation: neither dual fusion
        # path can mix current variables before the final interaction.
        prediction = super(PhoCoNetQModel, self).forecast(
            x_enc, x_mark_enc, None
        )
        if exog_future is None:
            return prediction
        correction = self.late_interaction(
            x_enc[..., : self.target_channels],
            x_enc[..., self.target_channels :],
            exog_future,
            prediction[..., : self.target_channels],
            aggregation=self.late_aggregation,
        )
        prediction = prediction.clone()
        prediction[..., : self.target_channels] += correction
        return prediction

    def get_late_attention(self) -> torch.Tensor | None:
        return self.late_interaction.get_last_attention()

    def get_late_gate(self) -> torch.Tensor | None:
        return self.late_interaction.get_last_gate()


__all__ = ["Model", "TargetConditionedLateInteraction"]
