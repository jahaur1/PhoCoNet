"""PhoCoNetChDiv: Channel-Diverse Future Exog Injection Variant.

Identical to PhoCoNet Model except for _path2_embedding_enhance:
instead of broadcasting the same future embedding to all channels
(which homogenizes channels and hurts CD), each channel receives
a DISTINCT future embedding via a per-channel projection.

This preserves the channel diversity that CointAttention needs to
discover meaningful cross-channel relationships, even when future
exogenous information is available.
"""

import torch
import torch.nn as nn

from ts_benchmark.baselines.phoconet.layers.Embed import PatchEmbed
from ts_benchmark.baselines.phoconet.layers.SelfAttention_Family import TSMixer, ResAttention
from ts_benchmark.baselines.phoconet.layers.Transformer_EncDec_ci_cd import (
    CointAttention,
    IntAttention,
    TSEncoder,
)


class VariableResidualGate(nn.Module):
    """A single learnable gate around one variable-attention block."""

    def __init__(self, block, gate_init=0.1):
        super().__init__()
        if not 0.0 < gate_init < 1.0:
            raise ValueError("cd_outer_gate_init must lie strictly in (0, 1)")
        self.block = block
        self.gate_logit = nn.Parameter(
            torch.logit(torch.tensor(float(gate_init)))
        )

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        transformed, attention = self.block(
            x, attn_mask=attn_mask, tau=tau, delta=delta
        )
        gate = torch.sigmoid(self.gate_logit)
        return x + gate * (transformed - x), attention

    def get_gate(self):
        return float(torch.sigmoid(self.gate_logit).detach().cpu())


class Model(nn.Module):
    """PhoCoNetChDiv Model — per-channel future exog injection."""

    def __init__(self, configs):
        super(Model, self).__init__()

        self.revin = configs.revin
        self.c_in = configs.enc_in
        self.period = configs.period
        self.seq_len = configs.seq_len
        self.pred_len = configs.horizon
        self.d_model = configs.d_model

        self.use_future_exog = getattr(configs, "use_future_exog", True)
        self.use_history_target = getattr(configs, "use_history_target", True)
        self.use_history_exog = getattr(configs, "use_history_exog", True)
        self.configs = configs
        self.attn_mode = getattr(configs, "attn_mode", "full")
        self.fusion_mode = getattr(configs, "fusion_mode", "dual")
        self.alpha_mode = getattr(configs, "alpha_mode", "learnable")
        self.alpha_fixed = getattr(configs, "alpha_fixed", 0.5)
        self.local_gate_mode = getattr(configs, "local_gate_mode", "adaptive")
        self.local_gate_fixed = float(getattr(configs, "local_gate_fixed", 0.5))
        self.global_gate_mode = getattr(configs, "global_gate_mode", "adaptive")
        self.global_gate_fixed = float(getattr(configs, "global_gate_fixed", 0.5))
        if self.local_gate_mode not in {"adaptive", "fixed"}:
            raise ValueError(
                "local_gate_mode must be 'adaptive' or 'fixed', "
                f"got {self.local_gate_mode!r}"
            )
        if self.global_gate_mode not in {"adaptive", "fixed"}:
            raise ValueError(
                "global_gate_mode must be 'adaptive' or 'fixed', "
                f"got {self.global_gate_mode!r}"
            )
        if not 0.0 <= self.local_gate_fixed <= 1.0:
            raise ValueError("local_gate_fixed must lie in [0, 1]")
        if not 0.0 <= self.global_gate_fixed <= 1.0:
            raise ValueError("global_gate_fixed must lie in [0, 1]")
        self.future_exog_mode = getattr(configs, "future_exog_mode", "normal")
        self.layer_order = getattr(configs, "layer_order", "int_coint")
        self.infer_use_future = getattr(configs, "infer_use_future", False)
        self.exog_drop = list(getattr(configs, "exog_drop", []) or [])
        self.exog_drop_history = list(getattr(configs, "exog_drop_history", []) or [])
        self.exog_drop_future = list(getattr(configs, "exog_drop_future", []) or [])
        self.input_missing_rate = float(getattr(configs, "input_missing_rate", 0.0) or 0.0)
        self.input_noise_level = float(getattr(configs, "input_noise_level", 0.0) or 0.0)
        self.input_noise_scope = str(getattr(configs, "input_noise_scope", "all"))
        if self.input_noise_scope not in {"all", "auxiliary"}:
            raise ValueError(
                "input_noise_scope must be either 'all' or 'auxiliary', "
                f"got {self.input_noise_scope!r}"
            )
        self.input_noise_seed = int(getattr(configs, "input_noise_seed", 20210726))
        self._input_noise_call_index = 0

        # Patch setting
        assert self.seq_len % self.period == 0
        self.num_p = self.seq_len // self.period

        if getattr(configs, "num_p", None) is None:
            configs.num_p = self.num_p

        self.patch_embed = PatchEmbed(configs, num_p=self.num_p)

        # Encoder
        self.temporal_encoder = None
        self.covariate_encoder = None
        layers = self.layers_init(configs)
        if self.attn_mode == "none":
            self.encoder = nn.Identity()
        elif self.attn_mode == "parallel":
            self.encoder = nn.Identity()
        else:
            self.encoder = TSEncoder(layers)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.num_p * configs.d_model, self.pred_len, bias=False),
        )

        # Fusion components (lazy creation)
        self.future_gate = None
        self.future_exog_proj = None
        self.future_exog_gate = None
        self.future_exog_ch_proj = None  # NEW: per-channel projection
        self.future_exog_ch_emb = None   # NEW: per-channel learned embedding
        self.fusion_alpha_logit = None
        self.future_adapter = None
        self.exog_dim = 0
        self._fusion_initialized = False

    # ------------------------------------------------------------------
    # init_fusion_components — adds future_exog_ch_proj
    # ------------------------------------------------------------------
    def init_fusion_components(self, exog_dim, alpha_init=0.0):
        if self._fusion_initialized or exog_dim <= 0:
            self._fusion_initialized = True
            return

        device = next(self.parameters()).device

        # gated_overwrite branch (same as original)
        self.future_gate = nn.Sequential(
            nn.Linear(exog_dim * 2, exog_dim),
            nn.Sigmoid(),
        ).to(device)

        # embedding_concat branch (same as original)
        self.future_exog_proj = nn.Sequential(
            nn.Linear(exog_dim, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        ).to(device)
        self.future_exog_gate = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.Sigmoid(),
        ).to(device)

        # NEW: per-channel future modulation
        # Instead of broadcasting the same future vector to all channels,
        # each channel has a learned embedding that modulates the projected
        # future info differently. This preserves channel diversity for CD.
        # We use a generous max_channels bound; at runtime we slice to the
        # actual channel count from x_enc_emb.shape[1].
        max_channels = 32  # upper bound for any realistic channel count
        self.future_exog_ch_emb = nn.Parameter(
            torch.randn(max_channels, self.d_model, device=device) * 0.02
        )
        self.future_exog_ch_proj = nn.Linear(self.d_model, self.d_model).to(device)

        # Learnable fusion weight (same as original)
        self.fusion_alpha_logit = nn.Parameter(torch.tensor(float(alpha_init), device=device))

        # Future adapter (same as original)
        self.future_adapter = nn.Sequential(
            nn.Linear(exog_dim, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
        ).to(device)

        self.exog_dim = exog_dim
        self._fusion_initialized = True

    # ------------------------------------------------------------------
    # _path2_embedding_enhance — per-channel injection (KEY CHANGE)
    # ------------------------------------------------------------------
    def _path2_embedding_enhance(self, x_enc_emb, exog_future):
        """Branch 2: per-channel future exog injection.

        Instead of broadcasting the same future vector to all channels,
        each channel's future embedding is modulated by a learned
        per-channel embedding (future_exog_ch_emb). This preserves
        channel diversity for cross-channel attention.
        """
        if self.future_exog_proj is None or self.future_exog_gate is None:
            return x_enc_emb

        fut_emb = self.future_exog_proj(exog_future)
        fut_emb_mean = fut_emb.mean(dim=1)  # (B, d_model)

        # Per-channel modulation: project future info, then modulate by
        # a learned per-channel embedding so each channel responds differently
        if self.future_exog_ch_proj is not None and self.future_exog_ch_emb is not None:
            fut_projected = self.future_exog_ch_proj(fut_emb_mean)  # (B, d_model)
            C_total = x_enc_emb.shape[1]  # actual channels in embedding
            ch_emb = self.future_exog_ch_emb[:C_total]  # (C_total, d_model)
            # Modulate: each channel gets a unique combination of future info
            fut_per_ch = fut_projected.unsqueeze(1) * ch_emb.unsqueeze(0)  # (B, C, d_model)
            # Expand across patches: (B, C, 1, d_model) -> (B, C, num_p, d_model)
            fut_per_ch = fut_per_ch.unsqueeze(2).expand_as(x_enc_emb)
        else:
            # Fallback to original broadcast if ch_proj not initialized
            fut_per_ch = fut_emb_mean
            for _ in range(x_enc_emb.ndim - 2):
                fut_per_ch = fut_per_ch.unsqueeze(1)
            fut_per_ch = fut_per_ch.expand_as(x_enc_emb)

        if self.global_gate_mode == "adaptive":
            gate = self.future_exog_gate(
                torch.cat([x_enc_emb, fut_per_ch], dim=-1)
            )
        else:
            gate = self.global_gate_fixed
        x_enc_emb = x_enc_emb + gate * fut_per_ch

        return x_enc_emb

    # ------------------------------------------------------------------
    # All methods below are identical to the original Model
    # ------------------------------------------------------------------
    def build_temporal_encoder(self, configs):
        use_period_norm = getattr(configs, "use_period_norm", True)
        return IntAttention(
            TSMixer(ResAttention(attention_dropout=configs.attn_dropout), configs.d_model, configs.n_heads),
            configs.d_model, configs.d_ff,
            dropout=configs.dropout, stable_len=configs.stable_len,
            activation=configs.activation, stable=use_period_norm, enc_in=self.c_in,
        )

    def build_covariate_encoder(self, configs):
        block = CointAttention(
            TSMixer(ResAttention(attention_dropout=configs.attn_dropout), configs.d_model, configs.n_heads),
            configs.d_model, configs.d_ff,
            dropout=configs.dropout, activation=configs.activation,
            stable=False, enc_in=self.c_in, stable_len=configs.stable_len,
            axial=bool(getattr(configs, "cd_axial", True)),
            pre_norm=bool(getattr(configs, "cd_pre_norm", False)),
            channel_center=bool(getattr(configs, "cd_channel_center", False)),
            residual_scale_init=getattr(configs, "cd_residual_scale_init", None),
            target_only=bool(getattr(configs, "cd_target_only", False)),
            target_channels=int(getattr(configs, "series_dim", 1)),
            channel_identity=bool(getattr(configs, "cd_channel_identity", False)),
            channel_identity_scale_init=float(
                getattr(configs, "cd_channel_identity_scale_init", 0.01)
            ),
        )
        outer_gate_init = getattr(configs, "cd_outer_gate_init", None)
        if outer_gate_init is not None:
            return VariableResidualGate(block, float(outer_gate_init))
        return block

    def layers_init(self, configs):
        layers = []
        ia_layers = getattr(configs, "ia_layers", 1)
        ca_layers = getattr(configs, "ca_layers", 1)

        if self.attn_mode == "none":
            return layers
        if self.attn_mode == "only_int":
            for _ in range(ia_layers):
                layers.append(self.build_temporal_encoder(configs))
            return layers
        if self.attn_mode == "only_coint":
            for _ in range(ca_layers):
                layers.append(self.build_covariate_encoder(configs))
            return layers
        if self.attn_mode == "full":
            if self.layer_order == "int_coint":
                for _ in range(ia_layers):
                    layers.append(self.build_temporal_encoder(configs))
                for _ in range(ca_layers):
                    layers.append(self.build_covariate_encoder(configs))
            elif self.layer_order == "coint_int":
                for _ in range(ca_layers):
                    layers.append(self.build_covariate_encoder(configs))
                for _ in range(ia_layers):
                    layers.append(self.build_temporal_encoder(configs))
            elif self.layer_order == "interleave":
                n = max(ia_layers, ca_layers)
                for i in range(n):
                    if i < ia_layers:
                        layers.append(self.build_temporal_encoder(configs))
                    if i < ca_layers:
                        layers.append(self.build_covariate_encoder(configs))
            return layers
        if self.attn_mode == "parallel":
            self.temporal_encoder = TSEncoder(
                [self.build_temporal_encoder(configs) for _ in range(ia_layers)]
            )
            self.covariate_encoder = TSEncoder(
                [self.build_covariate_encoder(configs) for _ in range(ca_layers)]
            )
            return []
        return layers

    def _remove_history_exog(self, x_enc):
        if self.use_history_exog:
            return x_enc
        series_dim = getattr(self.configs, "series_dim", None)
        if series_dim is None:
            series_dim = getattr(self.configs, "enc_in", 1)
        x_new = x_enc.clone()
        if x_new.shape[-1] > series_dim:
            x_new[:, :, series_dim:] = 0.0
        return x_new

    def _remove_history_target(self, x_enc):
        if self.use_history_target:
            return x_enc
        series_dim = getattr(self.configs, "series_dim", 1)
        x_new = x_enc.clone()
        x_new[:, :, :series_dim] = 0.0
        return x_new

    def _mask_exog_columns(self, tensor, base_offset=0, which="both"):
        if tensor is None or tensor.numel() == 0:
            return tensor
        if which == "history":
            mask = self.exog_drop_history
        elif which == "future":
            mask = self.exog_drop_future
        else:
            mask = list(self.exog_drop_history) + [x for x in self.exog_drop_future
                                                   if x not in self.exog_drop_history]
            if not self.exog_drop and not self.exog_drop_history and not self.exog_drop_future:
                return tensor
        if not mask:
            return tensor
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        for idx in mask:
            col = base_offset + idx
            if 0 <= col < tensor.shape[-1]:
                tensor[..., col] = 0.0
        return tensor

    def _apply_input_perturbation(self, tensor, missing_rate=0.0, noise_level=0.0,
                                   gen=None):
        if tensor is None or tensor.numel() == 0:
            return tensor
        if missing_rate <= 0 and noise_level <= 0:
            return tensor
        x = tensor.clone()
        if missing_rate > 0:
            shape_bt = x.shape[:2]
            mask_bt = torch.rand(shape_bt, generator=gen, device=x.device) < missing_rate
            x = x.masked_fill(mask_bt.unsqueeze(-1), 0.0)
        if noise_level > 0:
            feat_std = x.std(dim=1, keepdim=True, unbiased=False)
            noise = torch.randn(x.shape, generator=gen, device=x.device) * noise_level * feat_std
            x = x + noise
        return x

    def _path1_gated_overwrite(self, x_enc, exog_future):
        if exog_future is None or self.future_gate is None:
            return x_enc
        future_len = min(exog_future.shape[1], self.pred_len, self.seq_len)
        if future_len <= 0:
            return x_enc
        history_tail = x_enc[:, -future_len:, :].clone()
        use_dim = min(exog_future.shape[-1], history_tail.shape[-1])
        if use_dim <= 0:
            return x_enc
        hist_exog = history_tail[:, :, -use_dim:]
        fut_exog = exog_future[:, -future_len:, :use_dim]
        if self.local_gate_mode == "adaptive":
            gate = self.future_gate(torch.cat([hist_exog, fut_exog], dim=-1))
        else:
            gate = self.local_gate_fixed
        fused = gate * fut_exog + (1 - gate) * hist_exog
        history_tail_fixed = torch.cat([
            history_tail[:, :, :history_tail.shape[-1] - use_dim],
            fused
        ], dim=-1)
        x_enc = torch.cat([x_enc, history_tail_fixed], dim=1)
        x_enc = x_enc[:, -self.seq_len:, :]
        return x_enc

    def _refine_encoded_channels(self, enc_out):
        return enc_out

    def forecast(self, x_enc, x_mark_enc=None, exog_future=None):
        if x_mark_enc is None:
            x_mark_enc = torch.zeros(
                (*x_enc.shape[:-1], 4),
                device=x_enc.device, dtype=x_enc.dtype,
            )

        original_c_in = x_enc.shape[-1]

        if not self.use_future_exog:
            exog_future = None

        if exog_future is not None and self.future_exog_mode != "normal":
            if self.future_exog_mode == "shuffled":
                idx = torch.randperm(exog_future.shape[1])
                exog_future = exog_future[:, idx, :]
            elif self.future_exog_mode == "noise":
                exog_future = torch.randn_like(exog_future)
            elif self.future_exog_mode == "last_step":
                exog_future = exog_future[:, -1:, :].expand_as(exog_future)
            elif self.future_exog_mode == "mean_pool":
                exog_future = exog_future.mean(dim=1, keepdim=True).expand_as(exog_future)

        x_enc = self._remove_history_target(x_enc)
        x_enc = self._remove_history_exog(x_enc)

        series_dim_for_mask = getattr(self.configs, "series_dim", None)
        if series_dim_for_mask is None:
            series_dim_for_mask = getattr(self.configs, "enc_in", 1)
        x_enc = self._mask_exog_columns(x_enc, base_offset=series_dim_for_mask, which="history")
        exog_future = self._mask_exog_columns(exog_future, base_offset=0, which="future")

        if (not self.training
                and not bool(getattr(self.configs, "sensor_noise_in_base", False))
                and (self.input_missing_rate > 0 or self.input_noise_level > 0)):
            gen = torch.Generator(device=x_enc.device)
            gen.manual_seed(self.input_noise_seed + self._input_noise_call_index)
            self._input_noise_call_index += 1
            if self.input_noise_scope == "auxiliary":
                series_dim = int(getattr(self.configs, "series_dim", 1))
                if series_dim < x_enc.shape[-1]:
                    x_enc = x_enc.clone()
                    x_enc[..., series_dim:] = self._apply_input_perturbation(
                        x_enc[..., series_dim:],
                        missing_rate=self.input_missing_rate,
                        noise_level=self.input_noise_level,
                        gen=gen,
                    )
            else:
                x_enc = self._apply_input_perturbation(
                    x_enc,
                    missing_rate=self.input_missing_rate,
                    noise_level=self.input_noise_level,
                    gen=gen,
                )
            if exog_future is not None:
                exog_future = self._apply_input_perturbation(
                    exog_future,
                    missing_rate=self.input_missing_rate,
                    noise_level=self.input_noise_level,
                    gen=gen,
                )

        if self.revin:
            mean = x_enc.mean(dim=1, keepdim=True).detach()
            std = torch.sqrt(x_enc.var(dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            x_enc_norm = (x_enc - mean) / std
        else:
            mean = std = None
            x_enc_norm = x_enc

        x_enc_base_emb = self.patch_embed(x_enc_norm, x_mark_enc)

        if self.fusion_mode == "dual":
            x_enc_gated = self._path1_gated_overwrite(x_enc_norm.clone(), exog_future)
            x_enc_gated_emb = self.patch_embed(x_enc_gated, x_mark_enc)
            if exog_future is not None and self.exog_dim > 0:
                x_enc_concat_emb = self._path2_embedding_enhance(x_enc_base_emb, exog_future)
            else:
                x_enc_concat_emb = x_enc_base_emb
            if self.alpha_mode == "fixed":
                alpha = self.alpha_fixed
            elif self.fusion_alpha_logit is not None:
                alpha = torch.sigmoid(self.fusion_alpha_logit)
            else:
                alpha = 0.5
            x_enc_emb = alpha * x_enc_gated_emb + (1 - alpha) * x_enc_concat_emb
        elif self.fusion_mode == "only_overwrite":
            x_enc_gated = self._path1_gated_overwrite(x_enc_norm.clone(), exog_future)
            x_enc_emb = self.patch_embed(x_enc_gated, x_mark_enc)
        elif self.fusion_mode == "only_embedding":
            if exog_future is not None and self.exog_dim > 0:
                x_enc_emb = self._path2_embedding_enhance(x_enc_base_emb, exog_future)
            else:
                x_enc_emb = x_enc_base_emb
        elif self.fusion_mode == "concat":
            x_enc_emb = x_enc_base_emb
        elif self.fusion_mode == "future_adapter":
            if exog_future is not None and self.exog_dim > 0 and self.future_adapter is not None:
                fut_emb = self.future_adapter(exog_future)
                fut_emb_mean = fut_emb.mean(dim=1)
                for _ in range(x_enc_base_emb.ndim - 2):
                    fut_emb_mean = fut_emb_mean.unsqueeze(1)
                x_enc_emb = x_enc_base_emb + fut_emb_mean
            else:
                x_enc_emb = x_enc_base_emb
        else:
            x_enc_emb = x_enc_base_emb

        if self.attn_mode == "none":
            enc_out = x_enc_emb
        elif self.attn_mode == "parallel":
            temporal_out = self.temporal_encoder(x_enc_emb)[0]
            covariate_out = self.covariate_encoder(x_enc_emb)[0]
            enc_out = (temporal_out + covariate_out) / 2
        else:
            enc_out = self.encoder(x_enc_emb)[0]

        enc_out = enc_out[:, :self.c_in, ...]
        enc_out = self._refine_encoded_channels(enc_out)

        dec_out = self.decoder(enc_out).transpose(-1, -2)

        if self.revin:
            dec_out = dec_out * std + mean

        return dec_out

    def forward(self, x_enc, x_mark_enc=None, exog_future=None):
        dec_out = self.forecast(x_enc, x_mark_enc, exog_future)
        return dec_out[:, -self.pred_len:, :]

    def get_alpha(self):
        if self.alpha_mode == "fixed":
            return self.alpha_fixed
        if self.fusion_alpha_logit is not None:
            return torch.sigmoid(self.fusion_alpha_logit).item()
        return None
