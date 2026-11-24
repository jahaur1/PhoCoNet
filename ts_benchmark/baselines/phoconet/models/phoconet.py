import torch

from ts_benchmark.baselines.phoconet.models.gate_free import Model as GateFreeModel


class Model(GateFreeModel):
    def __init__(self, configs):
        super().__init__(configs)
        self.use_global_aux_projection = False
        self.target_only_dpf = (
            self.attn_mode == "none"
            and self.target_variable_attention is None
            and self.fusion_mode != "future_adapter"
            and configs.dropout == 0
        )
        self.register_load_state_dict_pre_hook(self._discard_legacy_late_interaction)

    def _build_late_interaction(self, configs):
        return None

    @staticmethod
    def _discard_legacy_late_interaction(module, state_dict, prefix, *_):
        legacy_prefix = f"{prefix}late_interaction."
        for key in tuple(state_dict):
            if key.startswith(legacy_prefix):
                del state_dict[key]

    def _path1_gated_overwrite(self, x_enc, exog_future):
        if not self.target_only_dpf or x_enc.shape[-1] != self.target_channels:
            return super()._path1_gated_overwrite(x_enc, exog_future)

        if exog_future is None:
            return x_enc
        tail_length = min(exog_future.shape[1], self.pred_len, self.seq_len)
        if tail_length == 0:
            return x_enc
        return torch.cat((x_enc[:, tail_length:], x_enc[:, -tail_length:]), dim=1)

    def _fuse_local_auxiliary(self, historical_auxiliary, current_auxiliary):
        # The auxiliary-output fallback preserves the old rolling behavior.
        return historical_auxiliary + current_auxiliary


__all__ = ["Model"]
