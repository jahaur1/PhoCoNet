"""Target-directed cross-variable encoder used by PhoCoNet."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class TSEncoder(nn.Module):
    def __init__(self, attn_layers):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(
                x, attn_mask=attn_mask, tau=tau, delta=delta
            )
            attns.append(attn)
        return x, attns


class CointAttention(nn.Module):
    """Pre-normalized full-channel attention that updates target channels only."""

    def __init__(
        self,
        attention,
        d_model,
        enc_in,
        d_ff=None,
        dropout=0.1,
        activation="relu",
        target_channels=1,
        residual_scale_init=0.01,
        channel_identity_scale_init=0.01,
    ):
        super().__init__()
        if target_channels < 1 or target_channels > enc_in:
            raise ValueError(
                f"target_channels must be in [1, {enc_in}], got {target_channels}"
            )

        d_ff = d_ff or 4 * d_model
        self.attention1 = attention
        self.data_channels = int(enc_in)
        self.target_channels = int(target_channels)

        self.norm0 = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

        self.channel_identity = nn.Parameter(
            torch.empty(self.data_channels, d_model)
        )
        nn.init.normal_(self.channel_identity, mean=0.0, std=0.02)
        self.channel_identity_scale = nn.Parameter(
            torch.tensor(float(channel_identity_scale_init))
        )
        scale = float(residual_scale_init)
        self.attention_scale = nn.Parameter(torch.full((d_model,), scale))
        self.ffn_scale = nn.Parameter(torch.full((d_model,), scale))

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        if x.shape[1] < self.data_channels:
            raise ValueError(
                "cross-variable input has fewer channels than expected: "
                f"input={x.shape[1]}, data_channels={self.data_channels}"
            )

        identity = self.channel_identity_scale * self.channel_identity
        data_source = (
            x[:, :self.data_channels]
            + identity.view(1, self.data_channels, 1, x.shape[-1])
        )
        attention_source = torch.cat(
            [data_source, x[:, self.data_channels:]], dim=1
        )
        attention_source = self.norm0(attention_source)

        queries_and_keys = attention_source - attention_source.mean(
            dim=1, keepdim=True
        )
        attention_delta = self._target_attention(
            queries_and_keys, attention_source
        )
        attention_delta = self.dropout(attention_delta) * self.attention_scale
        x = x + attention_delta

        y = self.norm1(x)
        y = self.dropout(self.activation(self.fc1(y)))
        y = self.dropout(self.fc2(y)) * self.ffn_scale
        return x + self._target_updates_only(y), None

    def _target_attention(self, queries_and_keys, values):
        batch_size, _, num_patches, _ = queries_and_keys.shape
        qk = rearrange(queries_and_keys, "b c n d -> (b n) c d")
        value = rearrange(values, "b c n d -> (b n) c d")
        target_out = self.attention1(
            qk[:, :self.target_channels], qk, value
        )[0]
        full_out = torch.cat(
            [
                target_out,
                torch.zeros_like(values[:, self.target_channels:]),
            ],
            dim=1,
        )
        return rearrange(
            full_out,
            "(b n) c d -> b c n d",
            b=batch_size,
            n=num_patches,
        )

    def _target_updates_only(self, delta):
        return torch.cat(
            [
                delta[:, :self.target_channels],
                torch.zeros_like(delta[:, self.target_channels:]),
            ],
            dim=1,
        )
