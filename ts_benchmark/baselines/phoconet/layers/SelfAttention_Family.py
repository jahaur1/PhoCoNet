"""Attention modules used by the final PhoCoNet architecture."""

from math import sqrt

import torch
import torch.nn as nn


class TSMixer(nn.Module):
    def __init__(self, attention, d_model, n_heads):
        super().__init__()
        self.attention = attention
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        batch_size, query_len, _ = queries.shape
        key_len = keys.shape[1]
        queries = self.q_proj(queries).reshape(
            batch_size, query_len, self.n_heads, -1
        )
        keys = self.k_proj(keys).reshape(
            batch_size, key_len, self.n_heads, -1
        )
        values = self.v_proj(values).reshape(
            batch_size, key_len, self.n_heads, -1
        )
        output, attention = self.attention(queries, keys, values)
        output = output.reshape(batch_size, query_len, -1)
        return self.out(output), attention


class ResAttention(nn.Module):
    def __init__(self, attention_dropout=0.1, scale=None):
        super().__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        scale = self.scale or 1.0 / sqrt(queries.shape[-1])
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        attention = self.dropout(torch.softmax(scale * scores, dim=-1))
        output = torch.einsum("bhls,bshd->blhd", attention, values)
        return output.contiguous(), attention
