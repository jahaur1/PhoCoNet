from ts_benchmark.baselines.phoconet.models.gate_free import Model as GateFreeModel


class Model(GateFreeModel):
    def __init__(self, configs):
        super().__init__(configs)
        self.use_global_aux_projection = False

    def _fuse_local_auxiliary(self, historical_auxiliary, current_auxiliary):
        return historical_auxiliary + current_auxiliary


__all__ = ["Model"]
