"""Patch embedding used by PhoCoNet."""

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, configs, num_p=1):
        super().__init__()
        self.num_patches = num_p
        self.patch_size = int(getattr(configs, "patch_len", configs.period))
        self.patch_stride = int(getattr(configs, "patch_stride", self.patch_size))
        self.proj = nn.Sequential(
            nn.Linear(self.patch_size, configs.d_model, bias=False),
            nn.Dropout(configs.dropout),
        )

    def forward(self, values, time_markers):
        combined = torch.cat([values, time_markers], dim=-1)
        combined = combined.transpose(-1, -2)
        patches = combined.unfold(
            dimension=-1,
            size=self.patch_size,
            step=self.patch_stride,
        )
        final_start = combined.shape[-1] - self.patch_size
        regular_last_start = (patches.shape[-2] - 1) * self.patch_stride
        if regular_last_start != final_start:
            final_patch = combined[..., -self.patch_size :].unsqueeze(-2)
            patches = torch.cat((patches, final_patch), dim=-2)
        if patches.shape[-2] != self.num_patches:
            raise RuntimeError(
                f"patch count mismatch: expected {self.num_patches}, "
                f"got {patches.shape[-2]}"
            )
        return self.proj(patches)
