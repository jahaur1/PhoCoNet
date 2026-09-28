"""Experimental encoder layers used by the corrected CD ablations.

The original layers remain untouched so published baselines stay reproducible.
``CointAttention`` adds opt-in corrections for cross-channel modelling:

* pre-normalized attention and feed-forward residuals;
* channel-centred queries/keys so shared future context cannot dominate scores;
* exact channel-identity embeddings, independent of future-exogenous injection;
* learnable LayerScale residuals that can safely fall back to the input;
* target-only cross-attention for single-target soft-sensing tasks.
"""

import copy
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class TSEncoder(nn.Module):
    def __init__(self, attn_layers):
        super(TSEncoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)
        return x, attns


def PeriodNorm(x, period_len=6):
    if len(x.shape) == 3:
        x = x.unsqueeze(-2)
    b, c, n, t = x.shape
    x_patch = [x[..., period_len - 1 - i:-i + t] for i in range(0, period_len)]
    x_patch = torch.stack(x_patch, dim=-1)

    mean = x_patch.mean(4)
    mean = F.pad(mean.reshape(b * c, n, -1),
                 mode='replicate', pad=(period_len - 1, 0)).reshape(b, c, n, -1)
    out = x - mean
    return out.squeeze(-2)


class IntAttention(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, stable_len=8, attn_map=False,
                 dropout=0.1, activation="relu", stable=True, enc_in=None):
        super(IntAttention, self).__init__()
        self.stable = stable
        self.stable_len = stable_len
        self.attn_map = attn_map
        d_ff = d_ff or 4 * d_model
        self.attention = attention

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x = self.temporal_attn(x)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.fc1(y)))
        y = self.dropout(self.fc2(y))

        return self.norm2(x + y), None

    def temporal_attn(self, x):
        b, c, n, d = x.shape
        new_x = x.reshape(-1, n, d)

        qk = new_x
        if self.stable:
            with torch.no_grad():
                qk = PeriodNorm(new_x, self.stable_len)
        new_x = self.attention(qk, qk, new_x)[0]
        new_x = new_x.reshape(b, c, n, d)
        return new_x


class PatchSampling(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu",
                 in_p=30, out_p=4, stable=False, stable_len=8):
        super(PatchSampling, self).__init__()

        d_ff = d_ff or 4 * d_model
        self.in_p = in_p
        self.out_p = out_p
        self.stable = stable
        self.stable_len = stable_len

        self.attention = attention
        self.conv1 = nn.Conv1d(
            self.in_p, self.out_p, 1, 1, 0, bias=False)
        self.conv2 = nn.Conv1d(
            self.out_p + 1, self.out_p, 1, 1, 0, bias=False)

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x = self.down_attn(x)
        y = x = self.norm1(new_x)

        y = self.dropout(self.activation(self.fc1(y)))
        y = self.dropout(self.fc2(y))

        return self.norm2(x + y), None

    def down_attn(self, x):
        b, c, n, d = x.shape
        x = x.reshape(-1, n, d)
        new_x = self.conv1(x)
        new_x = self.conv2(torch.cat(
            [new_x, x.mean(-2, keepdim=True)], dim=-2)) + new_x
        new_x = self.attention(new_x, x, x)[0] + self.dropout(new_x)
        return new_x.reshape(b, c, -1, d)


class CointAttention(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, axial=True, stable_len=8,
                 dropout=0.1, activation="relu", stable=True, enc_in=None,
                 pre_norm=False, channel_center=False,
                 residual_scale_init=None, target_only=False,
                 target_channels=1, channel_identity=False,
                 channel_identity_scale_init=0.01):
        super(CointAttention, self).__init__()

        if enc_in is None:
            raise ValueError("enc_in is required for cross-channel attention")
        if target_channels < 1 or target_channels > enc_in:
            raise ValueError(
                f"target_channels must be in [1, {enc_in}], got {target_channels}"
            )
        if target_only and axial:
            raise ValueError("target-only CD requires full channel attention")

        self.stable = stable
        self.stable_len = stable_len
        self.pre_norm = bool(pre_norm)
        self.channel_center = bool(channel_center)
        self.target_only = bool(target_only)
        self.target_channels = int(target_channels)
        self.channel_identity_enabled = bool(channel_identity)
        d_ff = d_ff or 4 * d_model

        self.axial_func = axial
        self.attention1 = attention
        self.attention2 = copy.deepcopy(attention)

        self.num_rc = math.ceil((enc_in + 4) ** 0.5)
        self.pad_ch = nn.ConstantPad1d(
            (0, self.num_rc ** 2 - (enc_in + 4)), 0)

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.norm0 = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

        self.data_channels = int(enc_in)
        if self.channel_identity_enabled:
            self.channel_identity = nn.Parameter(
                torch.empty(self.data_channels, d_model)
            )
            nn.init.normal_(self.channel_identity, mean=0.0, std=0.02)
            self.channel_identity_scale = nn.Parameter(
                torch.tensor(float(channel_identity_scale_init))
            )
        else:
            self.register_parameter("channel_identity", None)
            self.register_parameter("channel_identity_scale", None)

        if residual_scale_init is None:
            self.register_parameter("attention_scale", None)
            self.register_parameter("ffn_scale", None)
        else:
            scale = float(residual_scale_init)
            self.attention_scale = nn.Parameter(torch.full((d_model,), scale))
            self.ffn_scale = nn.Parameter(torch.full((d_model,), scale))

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        if self.channel_identity is not None:
            if x.shape[1] < self.data_channels:
                raise ValueError(
                    "CD input has fewer channels than its data-channel identity "
                    f"table: input={x.shape[1]}, data_channels={self.data_channels}"
                )
            identity = self.channel_identity_scale * self.channel_identity
            data_source = (
                x[:, :self.data_channels]
                + identity.view(1, self.data_channels, 1, x.shape[-1])
            )
            attention_source = torch.cat(
                [data_source, x[:, self.data_channels:]],
                dim=1,
            )
        else:
            attention_source = x

        if self.pre_norm:
            attention_source = self.norm0(attention_source)

        qk = attention_source
        if self.channel_center:
            qk = qk - qk.mean(dim=1, keepdim=True)

        if self.axial_func is True:
            try:
                new_x = self.axial_attn(qk, attention_source)
            except RuntimeError as e:
                warnings.warn(
                    f"CointAttention.axial_attn failed ({e!r}), "
                    f"falling back to full_attn. shape={x.shape}, "
                    f"num_rc={self.num_rc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                new_x = self.full_attn(qk, attention_source)
        else:
            new_x = self.full_attn(qk, attention_source)

        attention_delta = self.dropout(new_x)
        if self.attention_scale is not None:
            attention_delta = attention_delta * self.attention_scale
        if self.target_only:
            attention_delta = self._target_updates_only(attention_delta)
        x = x + attention_delta

        if self.pre_norm:
            y = self.norm1(x)
            y = self.dropout(self.activation(self.fc1(y)))
            y = self.dropout(self.fc2(y))
            if self.ffn_scale is not None:
                y = y * self.ffn_scale
            if self.target_only:
                y = self._target_updates_only(y)
            return x + y, None

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.fc1(y)))
        y = self.dropout(self.fc2(y))

        return self.norm2(x + y), None

    def _target_updates_only(self, delta):
        if self.target_channels == delta.shape[1]:
            return delta
        return torch.cat(
            [delta[:, :self.target_channels],
             torch.zeros_like(delta[:, self.target_channels:])],
            dim=1,
        )

    def axial_attn(self, qk, values):
        b, c, n, d = qk.shape
        # Calendar-token counts depend on the frequency encoder (the Full-8
        # benchmark currently emits six, not the four assumed at init time).
        # Derive the square channel grid from the actual runtime token count;
        # otherwise 15 tokens are padded as if there were 13 and the einops
        # reconstruction necessarily fails before silently using full_attn.
        num_rc = math.ceil(c ** 0.5)
        pad_count = num_rc ** 2 - c
        new_qk = rearrange(qk, 'b c n d -> (b n) c d')
        new_values = rearrange(values, 'b c n d -> (b n) c d')
        new_qk = F.pad(new_qk, (0, 0, 0, pad_count)).reshape(-1, num_rc, d)
        new_values = F.pad(
            new_values, (0, 0, 0, pad_count)
        ).reshape(-1, num_rc, d)
        new_x = self.attention1(new_qk, new_qk, new_values)[0]
        new_x = rearrange(new_x, '(b r) c d -> (b c) r d', r=num_rc)
        new_x = self.attention2(new_x, new_x, new_x)[0] + new_x

        new_x = rearrange(new_x, '(b n c) r d -> b (r c) n d', b=b, n=n)
        return new_x[:, :c, ...]

    def full_attn(self, qk, values):
        b, c, n, d = qk.shape
        new_qk = rearrange(qk, 'b c n d -> (b n) c d')
        new_values = rearrange(values, 'b c n d -> (b n) c d')
        if self.target_only:
            target_q = new_qk[:, :self.target_channels]
            target_out = self.attention1(target_q, new_qk, new_values)[0]
            new_x = torch.cat(
                [target_out, torch.zeros_like(new_values[:, self.target_channels:])],
                dim=1,
            )
        else:
            new_x = self.attention1(new_qk, new_qk, new_values)[0]
        new_x = rearrange(new_x, '(b n) c d -> b c n d', b=b, n=n)
        return new_x[:, :c, :]
