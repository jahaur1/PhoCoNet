"""Patch embedding used by PhoCoNet."""

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, configs, num_p=1):
        super().__init__()
        self.num_patches = num_p
        self.patch_size = configs.seq_len // num_p
        self.proj = nn.Sequential(
            nn.Linear(self.patch_size, configs.d_model, bias=False),
            nn.Dropout(configs.dropout),
        )

    def forward(self, values, time_markers):
        combined = torch.cat([values, time_markers], dim=-1)
        combined = combined.transpose(-1, -2)
        patches = combined.reshape(
            *combined.shape[:-1],
            self.num_patches,
            self.patch_size,
        )
        return self.proj(patches)
