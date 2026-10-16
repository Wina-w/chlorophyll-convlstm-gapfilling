"""Synthetic multi-sensor NetCDF fixtures (8x8 grid, 120 days) — no real data needed."""

from __future__ import annotations

import pytest

from chl_gapfill.config import ExperimentConfig, Paths, reset_paths_cache
from chl_gapfill.synthetic import T, make_synthetic


@pytest.fixture(scope="session")
def synthetic_data(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("chl_data")
    clean = make_synthetic(root)
    return {"root": root, "clean": clean, "T": T}


@pytest.fixture
def cfg() -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "name": "test_convlstm",
            "input_days": 3,
            "region": {"lat_bounds": [23.0, 24.0], "lon_bounds": [118.5, 119.5]},
            "data": {"input_sensors": ["modis_aqua", "viirs_snpp"], "min_coverage": 0.2},
            "model": {"name": "convlstm", "hidden_channels": [8, 4], "kernel_size": 3, "use_norm": True},
            "training": {"lr": 1e-3, "n_epochs": 3, "patience": 3, "batch_size": 8},
            "split": {
                "train_end": "2022-03-21",
                "val_start": "2022-03-22",
                "val_end": "2022-04-10",
                "test_start": "2022-04-11",
                "test_end": "2022-04-30",
                "gap_days": 0,
            },
            "smoke": True,
        }
    )


@pytest.fixture
def paths_for(synthetic_data, tmp_path, monkeypatch) -> Paths:
    monkeypatch.setenv("CHL_DATA_DIR", str(synthetic_data["root"]))
    monkeypatch.setenv("CHL_MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("CHL_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    reset_paths_cache()
    from chl_gapfill.config import paths

    yield paths()
    reset_paths_cache()
