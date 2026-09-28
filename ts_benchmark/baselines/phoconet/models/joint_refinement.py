"""Variable-wise early fusion with the proven Joint MLP as the sole mixer."""

from __future__ import annotations

import torch

from ts_benchmark.baselines.phoconet.models.isolated_fusion import (
    Model as IsolatedLateModel,
)
from ts_benchmark.baselines.phoconet.models.online_conditioning import (
    Model as PhoCoNetQModel,
)


class Model(IsolatedLateModel):
    """Use isolated H--Q paths and retain the original final Joint correction."""

    def __init__(self, configs) -> None:
        super().__init__(configs)
        for module in (self.q_feature_norm, self.q_delta, self.q_gate):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        for parameter in self.late_interaction.parameters():
            parameter.requires_grad_(False)

    def forecast(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        exog_future: torch.Tensor | None = None,
    ) -> torch.Tensor:
        prediction = super(PhoCoNetQModel, self).forecast(
            x_enc, x_mark_enc, exog_future
        )
        correction = PhoCoNetQModel.q_correction(
            self, x_enc, exog_future, prediction
        )
        prediction = prediction.clone()
        prediction[..., : self.target_channels] += correction
        return prediction


__all__ = ["Model"]
