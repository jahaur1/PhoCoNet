"""ChDiv PhoCoNet with a small time-aligned online-observation branch."""

from __future__ import annotations

import torch
import torch.nn as nn

from ts_benchmark.baselines.phoconet.models.core import (
    Model as BaseModel,
)


class AlignedQVariableAttention(nn.Module):
    """Select current online variables for each aligned estimation step.

    Scores are initialized to zero, so the initial softmax is uniform and the
    rescaled online variables exactly reproduce the unweighted Q features.
    Training can then emphasize variables conditionally on the historical TP
    summaries, the base estimate, and the position inside the missing block.
    """

    def __init__(
        self,
        variable_count: int,
        context_dim: int,
        attention_dim: int = 32,
        residual_strength: float = 1.0,
    ) -> None:
        super().__init__()
        if variable_count < 1:
            raise ValueError("variable_count must be positive")
        if context_dim < 1 or attention_dim < 1:
            raise ValueError("context_dim and attention_dim must be positive")
        if not 0.0 < residual_strength <= 1.0:
            raise ValueError("residual_strength must lie in (0, 1]")
        self.variable_count = int(variable_count)
        self.residual_strength = float(residual_strength)
        self.value_projection = nn.Linear(2, attention_dim, bias=False)
        self.context_projection = nn.Linear(
            context_dim, attention_dim, bias=False
        )
        self.variable_identity = nn.Parameter(
            torch.empty(variable_count, attention_dim)
        )
        nn.init.normal_(self.variable_identity, mean=0.0, std=0.02)
        self.score = nn.Linear(attention_dim, 1, bias=False)
        nn.init.zeros_(self.score.weight)
        self._last_attention = None

    def forward(
        self,
        current_q: torch.Tensor,
        q_change: torch.Tensor,
        target_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if current_q.shape != q_change.shape:
            raise ValueError("current_q and q_change must have the same shape")
        if current_q.shape[-1] != self.variable_count:
            raise ValueError(
                "online variable count does not match the attention module"
            )
        variable_features = torch.stack((current_q, q_change), dim=-1)
        variable_tokens = self.value_projection(variable_features)
        variable_tokens = variable_tokens + self.variable_identity.view(
            1, 1, self.variable_count, -1
        )
        query = self.context_projection(target_context).unsqueeze(-2)
        scores = self.score(torch.tanh(variable_tokens + query)).squeeze(-1)
        attention = torch.softmax(scores, dim=-1)
        full_scale = attention * self.variable_count
        scale = 1.0 + self.residual_strength * (full_scale - 1.0)
        self._last_attention = attention.detach()
        return current_q * scale, q_change * scale

    def get_last_attention(self) -> torch.Tensor | None:
        if self._last_attention is None:
            return None
        return self._last_attention.detach().cpu().clone()


class AlignedQVariableResidual(nn.Module):
    """Add a small target-conditioned residual from selected online variables.

    The ordinary aligned-Q correction remains intact.  This branch constructs
    one token per online variable from its current value and recent change,
    selects tokens using the historical target state and output position, and
    maps their weighted context to an additional target correction.  A small
    output initialization and a conservative gate keep the initial model close
    to the no-attention control while allowing attention to learn immediately.
    """

    def __init__(
        self,
        variable_count: int,
        context_dim: int,
        target_channels: int,
        attention_dim: int = 32,
        hidden_dim: int = 32,
        gate_bias: float = -2.0,
        output_scale: float = 0.01,
    ) -> None:
        super().__init__()
        if min(
            variable_count,
            context_dim,
            target_channels,
            attention_dim,
            hidden_dim,
        ) < 1:
            raise ValueError("all aligned residual dimensions must be positive")
        if output_scale <= 0.0:
            raise ValueError("output_scale must be positive")
        self.variable_count = int(variable_count)
        self.value_projection = nn.Linear(2, attention_dim, bias=False)
        self.context_projection = nn.Linear(
            context_dim, attention_dim, bias=False
        )
        self.variable_identity = nn.Parameter(
            torch.empty(variable_count, attention_dim)
        )
        nn.init.normal_(self.variable_identity, mean=0.0, std=0.02)
        self.score = nn.Linear(attention_dim, 1, bias=False)
        nn.init.zeros_(self.score.weight)
        self.residual_norm = nn.LayerNorm(attention_dim + context_dim)
        self.residual_head = nn.Sequential(
            nn.Linear(attention_dim + context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, target_channels),
        )
        nn.init.normal_(
            self.residual_head[-1].weight, mean=0.0, std=float(output_scale)
        )
        nn.init.zeros_(self.residual_head[-1].bias)
        self.gate_logit = nn.Parameter(
            torch.full((target_channels,), float(gate_bias))
        )
        self._last_attention = None

    def forward(
        self,
        current_q: torch.Tensor,
        q_change: torch.Tensor,
        target_context: torch.Tensor,
    ) -> torch.Tensor:
        if current_q.shape != q_change.shape:
            raise ValueError("current_q and q_change must have the same shape")
        if current_q.shape[-1] != self.variable_count:
            raise ValueError(
                "online variable count does not match the residual module"
            )
        variable_features = torch.stack((current_q, q_change), dim=-1)
        variable_tokens = self.value_projection(variable_features)
        variable_tokens = variable_tokens + self.variable_identity.view(
            1, 1, self.variable_count, -1
        )
        query = self.context_projection(target_context).unsqueeze(-2)
        scores = self.score(torch.tanh(variable_tokens + query)).squeeze(-1)
        attention = torch.softmax(scores, dim=-1)
        attended_context = torch.sum(
            attention.unsqueeze(-1) * variable_tokens, dim=-2
        )
        residual_features = self.residual_norm(
            torch.cat((attended_context, target_context), dim=-1)
        )
        self._last_attention = attention.detach()
        return torch.sigmoid(self.gate_logit) * self.residual_head(
            residual_features
        )

    def get_last_attention(self) -> torch.Tensor | None:
        if self._last_attention is None:
            return None
        return self._last_attention.detach().cpu().clone()


class AlignedQVariableModulation(nn.Module):
    """Use selected online variables to bound modulation of the Q correction."""

    def __init__(
        self,
        variable_count: int,
        context_dim: int,
        target_channels: int,
        attention_dim: int = 32,
        hidden_dim: int = 32,
        max_modulation: float = 0.25,
        output_scale: float = 0.01,
    ) -> None:
        super().__init__()
        if min(
            variable_count,
            context_dim,
            target_channels,
            attention_dim,
            hidden_dim,
        ) < 1:
            raise ValueError("all aligned modulation dimensions must be positive")
        if not 0.0 < max_modulation <= 1.0:
            raise ValueError("max_modulation must lie in (0, 1]")
        if output_scale <= 0.0:
            raise ValueError("output_scale must be positive")
        self.variable_count = int(variable_count)
        self.max_modulation = float(max_modulation)
        self.value_projection = nn.Linear(2, attention_dim, bias=False)
        self.context_projection = nn.Linear(
            context_dim, attention_dim, bias=False
        )
        self.variable_identity = nn.Parameter(
            torch.empty(variable_count, attention_dim)
        )
        nn.init.normal_(self.variable_identity, mean=0.0, std=0.02)
        self.score = nn.Linear(attention_dim, 1, bias=False)
        nn.init.zeros_(self.score.weight)
        self.modulation_norm = nn.LayerNorm(attention_dim + context_dim)
        self.modulation_head = nn.Sequential(
            nn.Linear(attention_dim + context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, target_channels),
        )
        nn.init.normal_(
            self.modulation_head[-1].weight,
            mean=0.0,
            std=float(output_scale),
        )
        nn.init.zeros_(self.modulation_head[-1].bias)
        self._last_attention = None
        self._last_scale = None

    def forward(
        self,
        current_q: torch.Tensor,
        q_change: torch.Tensor,
        target_context: torch.Tensor,
    ) -> torch.Tensor:
        if current_q.shape != q_change.shape:
            raise ValueError("current_q and q_change must have the same shape")
        if current_q.shape[-1] != self.variable_count:
            raise ValueError(
                "online variable count does not match the modulation module"
            )
        variable_features = torch.stack((current_q, q_change), dim=-1)
        variable_tokens = self.value_projection(variable_features)
        variable_tokens = variable_tokens + self.variable_identity.view(
            1, 1, self.variable_count, -1
        )
        query = self.context_projection(target_context).unsqueeze(-2)
        scores = self.score(torch.tanh(variable_tokens + query)).squeeze(-1)
        attention = torch.softmax(scores, dim=-1)
        attended_context = torch.sum(
            attention.unsqueeze(-1) * variable_tokens, dim=-2
        )
        features = self.modulation_norm(
            torch.cat((attended_context, target_context), dim=-1)
        )
        scale = 1.0 + self.max_modulation * torch.tanh(
            self.modulation_head(features)
        )
        self._last_attention = attention.detach()
        self._last_scale = scale.detach()
        return scale

    def get_last_attention(self) -> torch.Tensor | None:
        if self._last_attention is None:
            return None
        return self._last_attention.detach().cpu().clone()

    def get_last_scale(self) -> torch.Tensor | None:
        if self._last_scale is None:
            return None
        return self._last_scale.detach().cpu().clone()


class TargetVariableAttention(nn.Module):
    """Compact TP-to-auxiliary attention over aligned historical patches.

    The target patch is the only query and the non-target water-quality
    patches are the only keys/values.  Consequently, the block cannot update
    auxiliary or calendar tokens and has an exact, interpretable ablation:
    removing it deletes only the historical cross-variable message.
    """

    def __init__(
        self,
        d_model: int,
        bridge_dim: int,
        n_heads: int,
        target_channels: int,
        auxiliary_channels: int,
        dropout: float = 0.0,
        scale_init: float = 0.01,
        use_variable_identity: bool = True,
        adaptive_gate_init: float | None = None,
    ) -> None:
        super().__init__()
        if bridge_dim < 1:
            raise ValueError("q_variable_dim must be positive")
        if n_heads < 1 or bridge_dim % n_heads != 0:
            raise ValueError(
                "q_variable_heads must be positive and divide q_variable_dim"
            )
        if target_channels < 1:
            raise ValueError("target_channels must be positive")
        if auxiliary_channels < 1:
            raise ValueError("TargetVariableAttention requires auxiliaries")

        self.target_channels = int(target_channels)
        self.auxiliary_channels = int(auxiliary_channels)
        self.target_norm = nn.LayerNorm(d_model)
        self.auxiliary_norm = nn.LayerNorm(d_model)
        self.target_projection = nn.Linear(d_model, bridge_dim, bias=False)
        self.auxiliary_projection = nn.Linear(d_model, bridge_dim, bias=False)
        self.cross_attention = nn.MultiheadAttention(
            bridge_dim,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output_projection = nn.Linear(bridge_dim, d_model, bias=False)
        self.output_dropout = nn.Dropout(dropout)
        self.residual_scale = nn.Parameter(
            torch.full((d_model,), float(scale_init))
        )
        if adaptive_gate_init is not None:
            gate_init = float(adaptive_gate_init)
            if not 0.0 < gate_init < 1.0:
                raise ValueError(
                    "q_variable_gate_init must lie strictly in (0, 1)"
                )
            self.adaptive_gate = nn.Linear(2 * d_model, d_model)
            nn.init.zeros_(self.adaptive_gate.weight)
            nn.init.constant_(
                self.adaptive_gate.bias,
                torch.logit(torch.tensor(gate_init)).item(),
            )
        else:
            self.adaptive_gate = None

        if use_variable_identity:
            self.variable_identity = nn.Parameter(
                torch.empty(auxiliary_channels, bridge_dim)
            )
            nn.init.normal_(self.variable_identity, mean=0.0, std=0.02)
            self.variable_identity_scale = nn.Parameter(torch.tensor(0.01))
        else:
            self.register_parameter("variable_identity", None)
            self.register_parameter("variable_identity_scale", None)

        self._last_attention = None
        self._last_gate = None

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        if encoded.ndim != 4:
            raise ValueError(
                "TargetVariableAttention expects [batch, channel, patch, feature], "
                f"got {tuple(encoded.shape)}"
            )
        expected_channels = self.target_channels + self.auxiliary_channels
        if encoded.shape[1] != expected_channels:
            raise ValueError(
                "TargetVariableAttention received an unexpected channel count: "
                f"got {encoded.shape[1]}, expected {expected_channels}"
            )

        batch_size, _, patch_count, feature_dim = encoded.shape
        target = encoded[:, : self.target_channels]
        auxiliary = encoded[:, self.target_channels :]

        # Each patch remains aligned in time: TP at patch p queries only the
        # auxiliary-variable tokens from the same patch p.
        target_tokens = target.permute(0, 2, 1, 3).reshape(
            batch_size * patch_count, self.target_channels, feature_dim
        )
        auxiliary_tokens = auxiliary.permute(0, 2, 1, 3).reshape(
            batch_size * patch_count, self.auxiliary_channels, feature_dim
        )
        normalized_target = self.target_norm(target_tokens)
        query = self.target_projection(normalized_target)
        key_value = self.auxiliary_projection(
            self.auxiliary_norm(auxiliary_tokens)
        )
        if self.variable_identity is not None:
            key_value = key_value + (
                self.variable_identity_scale * self.variable_identity
            ).unsqueeze(0)

        context, attention = self.cross_attention(
            query,
            key_value,
            key_value,
            need_weights=True,
            average_attn_weights=False,
        )
        projected_context = self.output_projection(context)
        if self.adaptive_gate is not None:
            gate = torch.sigmoid(
                self.adaptive_gate(
                    torch.cat([normalized_target, projected_context], dim=-1)
                )
            )
            projected_context = gate * projected_context
            self._last_gate = gate.detach()
        else:
            self._last_gate = None
        delta = self.output_dropout(projected_context)
        delta = delta * self.residual_scale
        refined_target = target_tokens + delta
        refined_target = refined_target.reshape(
            batch_size, patch_count, self.target_channels, feature_dim
        ).permute(0, 2, 1, 3)

        self._last_attention = attention.detach()
        return torch.cat([refined_target, auxiliary], dim=1)

    def get_last_attention(self) -> torch.Tensor | None:
        if self._last_attention is None:
            return None
        return self._last_attention.detach().cpu().clone()

    def get_last_gate(self) -> torch.Tensor | None:
        if self._last_gate is None:
            return None
        return self._last_gate.detach().cpu().clone()


class Model(BaseModel):
    """Add a zero-initialized, position-aligned correction from future exog."""

    def __init__(self, configs):
        super().__init__(configs)
        self.target_channels = int(getattr(configs, "series_dim", 1))
        self.q_exog_dim = self.c_in - self.target_channels
        self.q_mode = str(getattr(configs, "q_mode", "aligned"))
        if self.q_mode not in {"aligned", "mean", "shifted", "zero"}:
            raise ValueError(
                "q_mode must be one of aligned, mean, shifted, or zero"
            )
        hidden = int(getattr(configs, "q_hidden", 32))
        gate_bias = float(getattr(configs, "q_gate_bias", -2.0))
        if self.target_channels < 1:
            raise ValueError("PhoCoNetQ requires at least one target channel")
        if self.q_exog_dim < 1:
            raise ValueError("PhoCoNetQ requires online auxiliary variables")
        if hidden < 1:
            raise ValueError("q_hidden must be positive")

        # Per output step: current Q, its change from the last historical
        # auxiliary observation, four historical-target summaries, the base
        # prediction, and a normalized output-position coordinate.
        feature_dim = (
            2 * self.q_exog_dim
            + 5 * self.target_channels
            + 1
        )
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

        # The added branch is an exact identity at initialization. Training
        # must find useful information in Q before it can alter the backbone.
        nn.init.zeros_(self.q_delta[-1].weight)
        nn.init.zeros_(self.q_delta[-1].bias)
        nn.init.zeros_(self.q_gate[-2].weight)
        nn.init.constant_(self.q_gate[-2].bias, gate_bias)

        # Build optional attention only after the shared Q branch.  fork_rng
        # keeps its parameter initialization from shifting the random stream
        # used by the shared model and the subsequent training procedure.  A
        # paired run with the same seed therefore differs only by attention.
        self.use_aligned_q_variable_attention = bool(
            getattr(configs, "q_aligned_variable_attention", False)
        )
        if self.use_aligned_q_variable_attention:
            with torch.random.fork_rng(devices=[]):
                self.aligned_q_variable_attention = AlignedQVariableAttention(
                    variable_count=self.q_exog_dim,
                    context_dim=5 * self.target_channels + 1,
                    attention_dim=int(
                        getattr(configs, "q_aligned_variable_dim", 32)
                    ),
                    residual_strength=float(
                        getattr(configs, "q_aligned_variable_strength", 1.0)
                    ),
                )
        else:
            self.aligned_q_variable_attention = None

        self.use_aligned_q_variable_residual = bool(
            getattr(configs, "q_aligned_variable_residual", False)
        )
        if self.use_aligned_q_variable_residual:
            with torch.random.fork_rng(devices=[]):
                self.aligned_q_variable_residual = AlignedQVariableResidual(
                    variable_count=self.q_exog_dim,
                    context_dim=5 * self.target_channels + 1,
                    target_channels=self.target_channels,
                    attention_dim=int(
                        getattr(configs, "q_aligned_residual_dim", 32)
                    ),
                    hidden_dim=int(
                        getattr(configs, "q_aligned_residual_hidden", 32)
                    ),
                    gate_bias=float(
                        getattr(configs, "q_aligned_residual_gate_bias", -2.0)
                    ),
                    output_scale=float(
                        getattr(configs, "q_aligned_residual_output_scale", 0.01)
                    ),
                )
        else:
            self.aligned_q_variable_residual = None

        self.use_aligned_q_variable_modulation = bool(
            getattr(configs, "q_aligned_variable_modulation", False)
        )
        if self.use_aligned_q_variable_modulation:
            with torch.random.fork_rng(devices=[]):
                self.aligned_q_variable_modulation = AlignedQVariableModulation(
                    variable_count=self.q_exog_dim,
                    context_dim=5 * self.target_channels + 1,
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

        self.use_target_variable_attention = bool(
            getattr(configs, "q_variable_attention", False)
        )
        if self.use_target_variable_attention:
            bridge_dim = int(getattr(configs, "q_variable_dim", 32))
            bridge_heads = int(getattr(configs, "q_variable_heads", 4))
            bridge_dropout = float(
                getattr(configs, "q_variable_dropout", 0.0)
            )
            bridge_scale = float(
                getattr(configs, "q_variable_scale_init", 0.01)
            )
            use_identity = bool(
                getattr(configs, "q_variable_identity", True)
            )
            adaptive_gate_init = getattr(
                configs, "q_variable_gate_init", None
            )
            with torch.random.fork_rng(devices=[]):
                self.target_variable_attention = TargetVariableAttention(
                    d_model=self.d_model,
                    bridge_dim=bridge_dim,
                    n_heads=bridge_heads,
                    target_channels=self.target_channels,
                    auxiliary_channels=self.q_exog_dim,
                    dropout=bridge_dropout,
                    scale_init=bridge_scale,
                    use_variable_identity=use_identity,
                    adaptive_gate_init=adaptive_gate_init,
                )
        else:
            self.target_variable_attention = None

    def _refine_encoded_channels(self, enc_out):
        """Apply the optional historical TP-to-auxiliary attention block."""
        if self.target_variable_attention is None:
            return enc_out
        return self.target_variable_attention(enc_out)

    def get_target_variable_attention(self):
        if self.target_variable_attention is None:
            return None
        return self.target_variable_attention.get_last_attention()

    def get_aligned_q_variable_attention(self):
        if self.aligned_q_variable_modulation is not None:
            return self.aligned_q_variable_modulation.get_last_attention()
        if self.aligned_q_variable_residual is not None:
            return self.aligned_q_variable_residual.get_last_attention()
        if self.aligned_q_variable_attention is not None:
            return self.aligned_q_variable_attention.get_last_attention()
        return None

    def q_correction(self, x_enc, exog_future, prediction):
        """Return an aligned target correction with shape [B, F, target_dim]."""
        if exog_future is None:
            return prediction.new_zeros(
                (prediction.shape[0], self.pred_len, self.target_channels)
            )
        if exog_future.shape[1] != self.pred_len:
            raise ValueError(
                "PhoCoNetQ expects one online-observation vector per output "
                f"step: got {exog_future.shape[1]}, expected {self.pred_len}"
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

        target_mean = target_history.mean(dim=1)
        target_std = target_history.std(dim=1, unbiased=False)
        target_last = target_history[:, -1]
        target_trend = target_history[:, -1] - target_history[:, 0]
        target_summary = torch.cat(
            (target_mean, target_std, target_last, target_trend), dim=-1
        ).unsqueeze(1).expand(-1, horizon, -1)

        last_exog = exog_history[:, -1:, :].expand(-1, horizon, -1)
        exog_change = branch_q - last_exog
        base_target = prediction[..., : self.target_channels]
        position = torch.linspace(
            -1.0,
            1.0,
            horizon,
            device=prediction.device,
            dtype=prediction.dtype,
        ).view(1, horizon, 1).expand(prediction.shape[0], -1, -1)

        target_context = torch.cat(
            (target_summary, base_target, position), dim=-1
        )
        raw_branch_q = branch_q
        raw_exog_change = exog_change
        if self.aligned_q_variable_attention is not None:
            branch_q, exog_change = self.aligned_q_variable_attention(
                branch_q,
                exog_change,
                target_context,
            )

        features = torch.cat(
            (
                branch_q,
                exog_change,
                target_summary,
                base_target,
                position,
            ),
            dim=-1,
        )
        features = self.q_feature_norm(features)
        correction = self.q_gate(features) * self.q_delta(features)
        if self.aligned_q_variable_modulation is not None:
            correction = correction * self.aligned_q_variable_modulation(
                raw_branch_q,
                raw_exog_change,
                target_context,
            )
        if self.aligned_q_variable_residual is not None:
            correction = correction + self.aligned_q_variable_residual(
                raw_branch_q,
                raw_exog_change,
                target_context,
            )
        return correction

    def forecast(self, x_enc, x_mark_enc=None, exog_future=None):
        prediction = super().forecast(x_enc, x_mark_enc, exog_future)
        correction = self.q_correction(x_enc, exog_future, prediction)
        prediction = prediction.clone()
        prediction[..., : self.target_channels] += correction
        return prediction


__all__ = [
    "AlignedQVariableAttention",
    "AlignedQVariableModulation",
    "AlignedQVariableResidual",
    "Model",
]
