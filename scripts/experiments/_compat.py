"""Compatibility shim for the benchmark scripts written against the old flat layout.

The experiment scripts (train_single, compare_*, dineof_runner, sst_data,
s3_chl_data) predate the ``chl_gapfill`` package. Rather than rewrite research
code whose outputs are quoted in the README, this module gives them the old
names — ``C.OUTPUT_DIR``, ``available_sensor_paths()``, ``_load_model()`` … —
backed by the new configuration:

* experiment config: ``$CHL_CONFIG`` (default ``configs/convlstm.yaml``)
* paths: ``.env`` / environment (``CHL_*``, ``DINEOF_EXE``, ``S3_*``)
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import torch

from chl_gapfill import data_sources
from chl_gapfill import dataset as _dataset
from chl_gapfill.config import load_config, paths
from chl_gapfill.evaluate import load_model_from
from chl_gapfill.train import forward_batch  # noqa: F401  (re-exported)
from chl_gapfill.train import split_positions as _split_positions

P = paths()
_cfg_path = os.getenv("CHL_CONFIG", str(P.root / "configs" / "convlstm.yaml"))
cfg = load_config(_cfg_path if Path(_cfg_path).exists() else None)

C = SimpleNamespace(
    DATA_DIR=P.data,
    MODEL_DIR=P.models,
    OUTPUT_DIR=P.outputs_for(cfg),
    MODEL_OUT=P.models / f"{cfg.name}.pth",
    MODEL=cfg.model.name,
    HIDDEN_CHANNELS=cfg.model.hidden_channels,
    KERNEL_SIZE=cfg.model.kernel_size,
    USE_NORM=cfg.model.use_norm,
    INPUT_DAYS=cfg.input_days,
    LOG_TRANSFORM=cfg.log_transform,
    SEED=cfg.seed,
    TRAIN_NC=P.data / cfg.data.train_nc,
    VAR_NAME=cfg.data.var_name,
    LAT_BOUNDS=cfg.region.lat_bounds,
    LON_BOUNDS=cfg.region.lon_bounds,
    TRAIN_END=cfg.split.train_end,
    MASK_MIN_FRAC=cfg.mask.min_frac,
    MASK_MAX_FRAC=cfg.mask.max_frac,
    MASK_N_BLOBS=cfg.mask.n_blobs,
    INPUT_SENSORS=cfg.data.input_sensors or data_sources.input_sensors(),
    FUSION=cfg.data.fusion,
)


def available_sensor_paths(subset=None):
    keys = subset if subset is not None else cfg.data.input_sensors
    return data_sources.available_sensor_paths(P.data, keys, fusion=cfg.data.fusion)


def split_positions(sample_dates):
    return _split_positions(sample_dates, cfg.split, cfg.input_days)


def model_input_channels(ds):
    return ds.per_day_channels if cfg.model.name == "convlstm" else ds.flat_channels


def GapFillDataset(sensor_paths, input_days=None, log_transform=None, deterministic=False,
                   min_coverage=None, seed=None, **kw):
    """Old positional-style constructor -> new explicit one."""
    return _dataset.GapFillDataset(
        sensor_paths,
        target_nc=P.data / cfg.data.train_nc,
        lat_bounds=cfg.region.lat_bounds,
        lon_bounds=cfg.region.lon_bounds,
        train_end=cfg.split.train_end,
        var_name=cfg.data.var_name,
        input_days=cfg.input_days if input_days is None else input_days,
        log_transform=cfg.log_transform if log_transform is None else log_transform,
        deterministic=deterministic,
        min_frac=cfg.mask.min_frac,
        max_frac=cfg.mask.max_frac,
        n_blobs=cfg.mask.n_blobs,
        min_coverage=cfg.data.min_coverage if min_coverage is None else min_coverage,
        seed=cfg.seed if seed is None else seed,
        contiguous_windows=cfg.data.contiguous_windows,
        **kw,
    )


def _load_model(ckpt, ds, device):
    return load_model_from(Path(ckpt), cfg, ds, device)


def train_gapfill(subset=None, model_out=None, min_coverage=0.2, label=""):
    """Old signature -> `chl_gapfill.train.fit`; returns (state_dict_path, best_val)."""
    from chl_gapfill.train import fit

    res = fit(cfg, P, subset=subset, min_coverage=min_coverage, label=label, use_mlflow=False,
              state_dict_out=Path(model_out) if model_out else None)
    return res.state_dict_path, res.best_val


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
