import torch
from torch import nn
from ts_benchmark.baselines.phoconet.models.phoconet2 import Model as PhoCoNetModel


class Model(PhoCoNetModel):







    def __init__(self, configs):
        super().__init__(configs)
        self.local_gate_mode = self.global_gate_mode = "fixed"
        self.local_gate_fixed = self.global_gate_fixed = .5
        self.q_use_change = bool(getattr(configs, "q_use_change", True))
        self.q_history_summary = getattr(configs, "q_history_summary", "full")
        self.use_global_aux_projection = bool(getattr(configs, "use_global_aux_projection", True))
        if self.q_history_summary not in {"full", "last"}:
            raise ValueError("q_history_summary must be full or last")
        if (not self.q_use_change or self.q_history_summary != "full") and self.aligned_q_variable_residual is not None:
            raise ValueError("Compact correction inputs require q_aligned_variable_residual=False")


        d, t = self.q_exog_dim, self.target_channels
        context = (list(range(5*t)) if self.q_history_summary == "full"
                   else [*range(2*t, 3*t), *range(4*t, 5*t)])
        features = [*range(d), *(range(d, 2*d) if self.q_use_change else []),
                    *(2*d + i for i in context)]
        self.q_feature_coordinates = tuple(features)
        self.q_context_coordinates = tuple(context)
        if len(features) != 2*d + 5*t:
            self._slice_norm(self.q_feature_norm, features)
            self._slice_linear(self.q_delta[0], features)
            self._slice_linear(self.q_gate[0], features)
        modulation = self.aligned_q_variable_modulation
        if modulation is not None:
            modulation.use_q_change = self.q_use_change
            if not self.q_use_change:
                self._slice_linear(modulation.value_projection, [0])
            if len(context) != 5*t:
                self._slice_linear(modulation.context_projection, context)
                attention_dim = modulation.value_projection.out_features
                mod_features = [*range(attention_dim), *(attention_dim+i for i in context)]
                self._slice_norm(modulation.modulation_norm, mod_features)
                self._slice_linear(modulation.modulation_head[0], mod_features)
            modulation.attention_mode = getattr(configs, "q_modulation_attention", "learned")
            modulation.condition_on_target = bool(getattr(configs, "q_modulation_context", True))
            modulation.use_variable_identity = bool(getattr(configs, "q_modulation_identity", True))
            if modulation.attention_mode not in {"learned", "uniform"}:
                raise ValueError("q_modulation_attention must be learned or uniform")
            if modulation.attention_mode == "uniform":
                modulation.score.requires_grad_(False)
                modulation.context_projection.requires_grad_(False)
            elif not modulation.condition_on_target:
                modulation.context_projection.requires_grad_(False)
            if not modulation.use_variable_identity:
                modulation.variable_identity.requires_grad_(False)

    @staticmethod
    def _slice_linear(module, columns):
        module.weight = nn.Parameter(module.weight.detach()[:, columns].clone(),
                                     requires_grad=module.weight.requires_grad)
        module.in_features = len(columns)

    @staticmethod
    def _slice_norm(module, columns):
        for name in ("weight", "bias"):
            original = getattr(module, name)
            setattr(module, name, nn.Parameter(original.detach()[columns].clone(),
                                               requires_grad=original.requires_grad))
        module.normalized_shape = (len(columns),)

    def init_fusion_components(self, exog_dim, alpha_init=0.0):


        super().init_fusion_components(exog_dim, alpha_init)


        self.isolated_local_gate = None
        self.future_exog_gate = None
        if not self.use_global_aux_projection:
            self.isolated_future_projection = None


__all__ = ["Model"]
