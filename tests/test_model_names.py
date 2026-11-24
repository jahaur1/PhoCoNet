from types import SimpleNamespace

import torch

from ts_benchmark.baselines.phoconet import PhoCoNet, PhoCoNet2
from ts_benchmark.baselines.phoconet.phoconet import PHOCONET2_HYPER_PARAMS
from ts_benchmark.baselines.phoconet.models.phoconet import Model
from ts_benchmark.baselines.phoconet.models.phoconet2 import Model as LegacyModel
from ts_benchmark.models.model_loader import get_model_info


def config():
    return SimpleNamespace(**{**PHOCONET2_HYPER_PARAMS, "enc_in":9, "series_dim":1})


def test_public_names_and_removed_modules():
    torch.set_num_threads(2)
    current = PhoCoNet(enc_in=9, series_dim=1)
    original = PhoCoNet2(enc_in=9, series_dim=1)
    assert current.model_name == "PhoCoNet"
    assert original.model_name == "PhoCoNet2"
    assert get_model_info({"model_name":"phoconet.PhoCoNet"}) is PhoCoNet
    assert get_model_info({"model_name":"phoconet.PhoCoNet2"}) is PhoCoNet2
    a, b = current._init_model(), original._init_model()
    assert isinstance(a, Model) and isinstance(b, LegacyModel)
    a.init_fusion_components(8)
    b.init_fusion_components(8)
    assert a.isolated_local_gate is None
    assert a.future_exog_gate is None
    assert a.isolated_future_projection is None
    assert a.late_interaction is None
    assert b.isolated_local_gate is not None
    assert b.future_exog_gate is not None
    assert b.isolated_future_projection is not None
    assert b.late_interaction is not None
    inactive = ("isolated_local_gate.", "future_exog_gate.", "isolated_future_projection.", "late_interaction.")
    assert not any(name.startswith(inactive) for name in a.state_dict())


def test_target_only_dpf_retains_target_computation():
    torch.manual_seed(2021)
    legacy = LegacyModel(config()).eval()
    legacy.init_fusion_components(8)
    torch.manual_seed(2021)
    current = Model(config()).eval()
    current.init_fusion_components(8)
    for name,value in current.state_dict().items():
        torch.testing.assert_close(value,legacy.state_dict()[name],rtol=0,atol=0)
    history, query, marks = torch.randn(2, 96, 9), torch.randn(2, 24, 8), torch.randn(2, 96, 4)
    tp = history[..., :1]
    expected = torch.cat((tp[:, 24:], tp[:, -24:]), dim=1)
    torch.testing.assert_close(current._path1_gated_overwrite(tp, query), expected, rtol=0, atol=0)
    with torch.no_grad():
        legacy.q_delta[-1].weight.normal_(std=.1)
        current.q_delta[-1].weight.copy_(legacy.q_delta[-1].weight)
    embedding_channels = []
    def record_embedding(module, args):
        embedding_channels.append((args[0].shape[-1], args[1]))

    with current.patch_embed.register_forward_pre_hook(record_embedding):
        result = current(history, marks, query)
    assert embedding_channels == [(1, None), (1, None)]
    assert result.shape == (2, 24, 1)
    torch.testing.assert_close(result, legacy(history, marks, query)[..., :1])
    feedback = current.forecast(history, marks, query, include_auxiliary_predictions=True)
    assert feedback.shape == (2, 24, 9)
    torch.testing.assert_close(result, feedback[..., :1])
    result.square().mean().backward()
    assert current.q_gate[-2].weight.grad.abs().max()>0
    assert current.aligned_q_variable_modulation.modulation_head[-1].weight.grad.abs().max()>0


def test_current_checkpoint_roundtrip():
    a, b = Model(config()).eval(), Model(config()).eval()
    a.init_fusion_components(8)
    b.init_fusion_components(8)
    checkpoint = a.state_dict()
    legacy = LegacyModel(config())
    checkpoint.update({name: value for name, value in legacy.state_dict().items()
                       if name.startswith("late_interaction.")})
    b.load_state_dict(checkpoint, strict=True)
    assert "late_interaction.delta_head.weight" in checkpoint
    history, query, marks = torch.randn(2, 96, 9), torch.randn(2, 24, 8), torch.randn(2, 96, 4)
    torch.testing.assert_close(a(history, marks, query), b(history, marks, query), rtol=0, atol=0)
