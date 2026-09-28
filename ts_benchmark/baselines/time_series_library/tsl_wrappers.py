# -*- coding: utf-8 -*-
"""
Wrappers for TSL (Time-Series-Library) models.

Each TSL model has its own ``__init__(self, configs)`` interface and is a
plain ``nn.Module`` subclass.  This module exposes *proxy* classes that
inherit from ``DeepForecastingModelBase`` so the benchmark pipeline can
drive them.

The TSL model class and the framework base class are imported lazily to
avoid a circular import.

Public class names mirror the TSL classes so callers can simply say
``--model-name time_series_library.DLinear`` etc.
"""
from typing import Type

import torch
import torch as _torch_mod
import torch.nn as nn


# Hyper-parameter defaults shared by every TSL model.
def _tsl_hyper_params() -> dict:
    """Default TSL hyper-parameters, expressed in TSL field names.

    The framework later overwrites ``enc_in`` / ``dec_in`` / ``c_out`` /
    ``label_len`` / ``freq`` based on the dataset via
    ``multi_forecasting_hyper_param_tune``.
    """
    return {
        # core architecture
        "task_name": "short_term_forecast",
        "seq_len": 96,
        "label_len": 48,
        "horizon": 24,
        "pred_len": 24,
        "enc_in": 5,         # overwritten per dataset
        "dec_in": 5,
        "c_out": 1,
        "d_model": 128,
        "d_ff": 256,
        "n_heads": 4,
        "e_layers": 2,
        "d_layers": 1,
        # embedding / decomposition
        "embed": "timeF",
        "freq": "h",
        "moving_avg": 25,
        "factor": 1,
        # regularisation
        "dropout": 0.1,
        "activation": "gelu",
        "output_attention": 0,
        # training
        "batch_size": 32,
        "lr": 0.0001,
        "lradj": "type3",
        "loss": "MSE",
        "num_epochs": 100,
        "patience": 10,
        "num_workers": 0,
        "distil": True,
        "use_future_exog": True,
        "covariate_dim": 6,
        # framework-side switch
        "norm": True,
    }


def _make_tsl_base():
    """Construct the ``_TSLBase`` class once both classes are importable."""
    from ts_benchmark.baselines.deep_forecasting_model_base import (
        DeepForecastingModelBase,
    )

    class _TSLBase(DeepForecastingModelBase):
        """Common :class:`DeepForecastingModelBase` glue for every TSL model."""

        TSL_CLASS = None
        MODEL_NAME_KEY = None  # set by per-model wrappers

        def __init__(self, tsl_class: Type[nn.Module], **kwargs):
            hp = _tsl_hyper_params()
            hp.update(kwargs)
            super().__init__(hp)
            self._tsl_class = tsl_class

        def _init_model(self):
            if not hasattr(self.config, "pred_len") and hasattr(self.config, "horizon"):
                self.config.pred_len = self.config.horizon
            return self._tsl_class(self.config)

        def _adjust_lr(self, optimizer, epoch, config):
            from ts_benchmark.baselines.utils import adjust_learning_rate
            adjust_learning_rate(optimizer, epoch, config)

        def _process(self, input, target, input_mark, target_mark, exog_future=None):
            """Default ``_process`` used by DLinear / Informer / FEDformer / TFT.

            Builds a decoder input by appending ``pred_len`` zero rows to the
            last ``label_len`` rows of ``target``.
            """
            device = input.device
            dec_zeros = _torch_mod.zeros(
                (target.shape[0], self.config.pred_len, target.shape[2]),
                device=device,
                dtype=input.dtype,
            )
            label_part = target[:, -self.config.label_len :, :]
            dec_input = _torch_mod.cat([label_part, dec_zeros], dim=1).to(device)

            out = self.model(
                input.to(device),
                input_mark.to(device) if input_mark is not None else None,
                dec_input,
                target_mark.to(device) if target_mark is not None else None,
            )
            return {"output": out}

        @property
        def model_name(self):
            return self.MODEL_NAME_KEY or self._model_name or self.TSL_CLASS.__name__

        @staticmethod
        def required_hyper_params() -> dict:
            # Do not inherit the framework's recommended ``output_chunk_length``
            # value as our horizon.  Each wrapper carries its own time-series
            # defaults via :func:`_tsl_hyper_params`.
            return {}

    # expose ``_model_name`` on the base so the property works for subclasses
    # that don't override it.
    _TSLBase._model_name = "DeepForecastingModelBase"
    return _TSLBase


_TSLBase = None


def _get_base():
    """Return the ``_TSLBase`` class (lazily constructed)."""
    global _TSLBase
    if _TSLBase is None:
        _TSLBase = _make_tsl_base()
    return _TSLBase


# Per-model overrides.
def _series_dim_default() -> int:
    """The number of target columns.  The framework treats the entire input
    matrix as the ``series``, so we treat the first column as the target and
    the rest as covariates.  Override ``series_dim`` via ``--model-hyper-params``
    if the target channel is not the last column."""
    return 1


def _tft_datatype_setup(tft_module, config):
    """Augment :data:`datatype_dict` so the dataset name resolves."""
    from collections import namedtuple

    # ``datatype_dict`` is a module-level variable inside
    # ``models.TemporalFusionTransformer``; import the module rather than
    # trying to read the attribute off the class.
    import importlib

    tft_module_full = importlib.import_module(
        "ts_benchmark.baselines.time_series_library.models.TemporalFusionTransformer"
    )
    tft_dict = tft_module_full.datatype_dict
    TypePos = tft_module_full.TypePos
    observed_len = getattr(config, "enc_in", 6)

    if "default" not in tft_dict:
        tft_dict["default"] = TypePos(static=[], observed=list(range(observed_len)))

    if not hasattr(config, "data") or not config.data:
        config.data = "default"

    data_key = config.data
    if data_key not in tft_dict:
        tft_dict[data_key] = TypePos(static=[], observed=list(range(observed_len)))

    for alias in (
        "juzizhou",
        "sanjiaozhou",
        "laodaohe",
        "juzizhou_5var",
        "sanjiaozhou_5var",
        "laodaohe_5var",
    ):
        if alias not in tft_dict:
            tft_dict[alias] = TypePos(static=[], observed=list(range(observed_len)))


def get_class(name: str) -> Type[nn.Module]:
    """Return a freshly built wrapper for the requested TSL model."""

    Base = _get_base()

    # Per-model ``_tsl_hyper_params`` overrides.  FEDformer's attention
    # expects ``n_heads`` to divide ``d_model``.
    per_model_overrides = {
        "FEDformer": {"n_heads": 8},
        "TemporalFusionTransformer": {"n_heads": 4},
    }

    import importlib

    module = importlib.import_module(
        f"ts_benchmark.baselines.time_series_library.models.{name}"
    )
    tsl_class = getattr(module, name)
    overrides = per_model_overrides.get(name, {})

    class _Wrapper(Base):
        TSL_CLASS = None
        MODEL_NAME_KEY = name

        def __init__(self, **kwargs):
            self.__class__.TSL_CLASS = tsl_class
            merged = dict(overrides)
            merged.update(kwargs)
            super().__init__(tsl_class, **merged)

        def _init_model(self):
            if not hasattr(self.config, "pred_len") and hasattr(self.config, "horizon"):
                self.config.pred_len = self.config.horizon

            # Augment TFT's ``datatype_dict`` so it can look up our dataset.
            if name == "TemporalFusionTransformer":
                _tft_datatype_setup(tsl_class, self.config)

            return self._tsl_class(self.config)

        def _process(self, input, target, input_mark, target_mark, exog_future=None):
            """Default processing path (DLinear / Informer / FEDformer / TFT).

            Subclasses with exotic input shapes (TiDE) override this.
            """
            device = input.device
            dec_zeros = _torch_mod.zeros(
                (target.shape[0], self.config.pred_len, target.shape[2]),
                device=device,
                dtype=input.dtype,
            )
            label_part = target[:, -self.config.label_len :, :]
            dec_input = _torch_mod.cat([label_part, dec_zeros], dim=1).to(device)

            out = self.model(
                input.to(device),
                input_mark.to(device) if input_mark is not None else None,
                dec_input,
                target_mark.to(device) if target_mark is not None else None,
            )
            return {"output": out}

    # TiDE: needs to physically split the input into target + exogenous
    # columns and rebuild a time-mark tensor that spans seq_len + pred_len.
    if name == "TiDE":
        from ts_benchmark.baselines.time_series_library.models import TiDE as _TiDE

        class _TiDEWrapper(_Wrapper):
            TSL_CLASS = _TiDE
            MODEL_NAME_KEY = "TiDE"

            def __init__(self, **kwargs):
                # ``covariate_dim`` is the number of exogenous features fed to
                # TiDE; the framework passes ``[target, exog1, ..., exogK]``
                # so K = (enc_in - series_dim) where series_dim is 1.
                # We update the value once hyper_param_tune runs (it sets
                # enc_in from the dataset) by reading it from the model
                # config.
                kwargs.setdefault("covariate_dim", 4)
                kwargs.setdefault("d_model", 64)
                kwargs.setdefault("d_ff", 64)
                kwargs.setdefault("c_out", 1)
                super().__init__(**kwargs)

            def _init_model(self):
                if (
                    not hasattr(self.config, "pred_len")
                    and hasattr(self.config, "horizon")
                ):
                    self.config.pred_len = self.config.horizon
                series_dim = _series_dim_default()
                enc_in = getattr(self.config, "enc_in", 6) or 6
                self.config.covariate_dim = max(enc_in - series_dim, 1)
                self._tsl_instance = self._tsl_class(self.config)
                return self._tsl_instance

            def _process(self, input, target, input_mark, target_mark, exog_future=None):
                """TiDE-specific processing.

                TiDE's ``forecast`` expects ``x_enc`` as the target-history
                tensor reshaped to ``[B, seq_len]``, exog covariates via
                ``batch_y_mark`` (``[B, seq_len+pred_len, exog_dim]``), and
                the future exog via ``x_dec`` (``[B, pred_len, exog_dim]``).

                The framework provides an ``input`` containing
                ``[target, exog...]`` columns for ``seq_len`` rows and a
                ``target`` containing ``[target, exog...]`` columns for
                ``label_len + pred_len`` rows.  We slice them into the
                pieces TiDE expects.
                """
                device = input.device
                series_dim = _series_dim_default()

                # x_enc: TARGET values over history -> [B, seq_len].
                x_enc = input[:, :, :series_dim].to(device).squeeze(-1)

                # x_dec: exogenous covariates for the FUTURE window only
                # -> [B, pred_len, exog_dim].
                target_full = target.to(device)
                x_dec = target_full[:, -self.config.pred_len :, series_dim:]

                # batch_y_mark: exogenous covariate values for history +
                # future -- shape [B, seq_len + pred_len, exog_dim].
                history_exog = input[:, :, series_dim:].to(device)
                batch_y_mark = _torch_mod.cat(
                    [history_exog, x_dec], dim=1
                ).to(dtype=_torch_mod.float32)

                tide = self._tsl_instance
                tide.task_name = "short_term_forecast"
                tide.use_future_exog = True
                im = input_mark.to(device) if input_mark is not None else None
                out = tide.forecast(x_enc, im, x_dec, batch_y_mark)
                if out.dim() == 2:
                    out = out.unsqueeze(-1)
                return {"output": out}

        _Wrapper = _TiDEWrapper

    # TFT: works similarly to DLinear but its TFTEmbedding requires the
    # decoder to be exactly ``pred_len`` rows (not label_len + pred_len).
    if name == "TemporalFusionTransformer":
        from ts_benchmark.baselines.time_series_library.models import (
            TemporalFusionTransformer as _TFT,
        )

        class _TFTWrapper(_Wrapper):
            TSL_CLASS = _TFT
            MODEL_NAME_KEY = "TemporalFusionTransformer"

            def __init__(self, **kwargs):
                # TFT explicitly counts covariates; the framework passes
                # `[target, exog...]` so the count is ``enc_in - 1``.
                kwargs.setdefault("covariate_dim", 5)
                # ``known_len`` in TFTEmbedding reads covariate_dim; keep
                # them in sync.
                kwargs.setdefault("d_model", 64)
                super().__init__(**kwargs)

            def _init_model(self):
                if (
                    not hasattr(self.config, "pred_len")
                    and hasattr(self.config, "horizon")
                ):
                    self.config.pred_len = self.config.horizon
                enc_in = getattr(self.config, "enc_in", 6) or 6
                self.config.covariate_dim = max(enc_in - 1, 1)
                _tft_datatype_setup(tsl_class, self.config)
                return self._tsl_class(self.config)

            def _process(self, input, target, input_mark, target_mark, exog_future=None):
                """TFT's ``forward`` calls ``series_dim = x_enc.shape[-1] - x_dec.shape[-1]``
                and then ``cov_x_mark = cat([x_enc[:, :, series_dim:], x_dec], dim=-2)``
                which must total ``seq_len + pred_len`` rows.  We therefore
                pass ``x_dec`` with exactly ``pred_len`` rows.
                """
                device = input.device

                x_enc = input.to(device)

                # x_dec: only the FUTURE pred_len rows of ``target``.  We
                # drop the historical label_len rows because TFT's
                # ``TFTEmbedding.forward`` already concatenates
                # ``x_enc[:, :, series_dim:]`` (history) with ``x_dec`` to
                # construct ``cov_x_mark``.
                target_full = target.to(device)
                x_dec = target_full[:, -self.config.pred_len :, :]

                # x_mark_dec mirrors x_dec in the time dimension (only the
                # future pred_len rows of target_mark).
                target_mark_dev = target_mark.to(device) if target_mark is not None else None
                if target_mark_dev is not None:
                    x_mark_dec = target_mark_dev[:, -self.config.pred_len :, :]
                else:
                    x_mark_dec = None

                out = self.model(
                    x_enc,
                    input_mark.to(device) if input_mark is not None else None,
                    x_dec,
                    x_mark_dec,
                )
                if out.dim() == 2:
                    out = out.unsqueeze(-1)
                return {"output": out}

        _Wrapper = _TFTWrapper

    _Wrapper.__name__ = name
    _Wrapper.__qualname__ = name
    _Wrapper.__module__ = "ts_benchmark.baselines.time_series_library"
    return _Wrapper
