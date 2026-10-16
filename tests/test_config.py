from __future__ import annotations

import json
from pathlib import Path

import pytest

from chl_gapfill.cli import main
from chl_gapfill.config import ExperimentConfig, Paths, apply_overrides, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_defaults_reproduce_published_config():
    cfg = ExperimentConfig()
    assert cfg.region.lat_bounds == (22.95, 24.55) and cfg.region.lon_bounds == (118.10, 119.75)
    assert cfg.model.hidden_channels == [64, 32] and cfg.input_days == 3 and cfg.seed == 42
    assert cfg.training.patience == 12 and cfg.training.batch_size == 16
    assert str(cfg.split.train_end) == "2021-12-31" and str(cfg.split.test_end) == "2022-07-25"
    assert len(cfg.config_hash()) == 12


def test_yaml_configs_load_and_hash_is_stable():
    a = load_config(ROOT / "configs" / "convlstm.yaml")
    b = load_config(ROOT / "configs" / "convlstm.yaml")
    assert a.config_hash() == b.config_hash()
    assert a.config_hash() == ExperimentConfig(name="x").config_hash()  # name excluded from the hash
    assert load_config(ROOT / "configs" / "pconv_unet.yaml").model.name == "pconv_unet"
    smoke = load_config(ROOT / "configs" / "convlstm_smoke.yaml")
    assert smoke.smoke and smoke.model.hidden_channels == [16, 8]


def test_overrides_are_yaml_parsed():
    cfg = load_config(
        ROOT / "configs" / "convlstm.yaml",
        ["training.lr=3e-4", "model.hidden_channels=[32,16]", "smoke=true"],
    )
    assert cfg.training.lr == 3e-4 and cfg.model.hidden_channels == [32, 16] and cfg.smoke
    assert apply_overrides({}, ["a.b.c=1"]) == {"a": {"b": {"c": 1}}}
    with pytest.raises(ValueError):
        apply_overrides({}, ["novalue"])


def test_repeated_set_flags_all_apply(capsys):
    # argparse's default `store` kept only the last --set, silently dropping smoke=false here
    argv = ["config", "-c", str(ROOT / "configs" / "synthetic_smoke.yaml")]
    assert main([*argv, "--set", "smoke=false", "--set", "training.n_epochs=30", "training.lr=3e-4"]) == 0
    cfg = json.loads(capsys.readouterr().out)["config"]
    assert cfg["smoke"] is False and cfg["training"]["n_epochs"] == 30 and cfg["training"]["lr"] == 3e-4


def test_smoke_runs_default_outside_the_published_outputs(tmp_path):
    smoke, real = ExperimentConfig(smoke=True), ExperimentConfig()
    p = Paths(root=tmp_path)
    assert p.outputs_for(real) == tmp_path / "outputs"  # published results
    assert p.outputs_for(smoke) == tmp_path / "outputs_smoke"  # gitignored
    explicit = Paths(root=tmp_path, output_dir=tmp_path / "elsewhere")  # CHL_OUTPUT_DIR wins
    assert explicit.outputs_for(smoke) == explicit.outputs_for(real) == tmp_path / "elsewhere"


def test_mlflow_sqlite_path_and_artifacts_are_anchored(tmp_path):
    p = Paths(root=tmp_path)  # the default sqlite:///mlruns/mlflow.db is relative
    assert p.mlflow_tracking_uri == f"sqlite:///{(tmp_path / 'mlruns' / 'mlflow.db').as_posix()}"
    assert p.mlflow_artifact_root == tmp_path / "mlruns" / "artifacts"
    db = tmp_path / "elsewhere" / "x.db"
    q = Paths(root=tmp_path, mlflow_tracking_uri=f"sqlite:///{db.as_posix()}")
    assert (
        q.mlflow_tracking_uri == f"sqlite:///{db.as_posix()}"
        and q.mlflow_artifact_root == db.parent / "artifacts"
    )
    assert Paths(root=tmp_path, mlflow_tracking_uri="http://localhost:5000").mlflow_artifact_root is None
    pg = Paths(root=tmp_path, mlflow_tracking_uri="postgresql+psycopg2://u@h/db")
    assert pg.mlflow_artifact_root == tmp_path / "mlruns" / "artifacts"


def test_validation_catches_bad_settings():
    with pytest.raises(ValueError):
        ExperimentConfig(region={"lat_bounds": [24.0, 23.0]})
    with pytest.raises(ValueError):
        ExperimentConfig(mask={"min_frac": 0.5, "max_frac": 0.4})
    with pytest.raises(ValueError):
        ExperimentConfig(split={"train_end": "2022-06-01", "val_start": "2022-01-01"})
    with pytest.raises(ValueError, match="gap_days"):
        ExperimentConfig(input_days=5, split={"gap_days": 1})
    with pytest.raises(ValueError):
        ExperimentConfig(model={"name": "transformer"})


@pytest.mark.parametrize(
    "override",
    [
        "training.lrr=3e-4",  # typo'd key: used to be dropped with the config_hash unchanged
        "trainig.lr=3e-4",  # typo'd section
        "data.test_nc=VIIRS_CHL_Daily2023.nc",  # removed field (it was never read)
        "training.lr=-1",
        "training.lr=0",
        "training.grad_clip=0",
        "model.hidden_channels=[]",
        "model.hidden_channels=[33,16]",  # ConvLSTM cells need even widths
        "model.hidden_channels=[12]",  # GroupNorm(8, 12) cannot be built
        "model.kernel_size=4",  # even kernel breaks 'same' padding mid-run
        "data.input_sensors=[modis_aqau]",
        "data.input_sensors=[]",
        "data.input_sensors=[viirs_snpp_n20_gapfilled]",  # the target is not an input
        "seed=-1",
        "name=../elsewhere",  # the name becomes a directory under models/
        "region.lat_bounds=[-100,0]",
        "seed.x=1",  # a value used as a section
    ],
)
def test_unknown_keys_and_invalid_values_are_rejected(override):
    with pytest.raises(ValueError):
        load_config(ROOT / "configs" / "convlstm.yaml", [override])


def test_pconv_rejects_settings_it_would_ignore():
    load_config(ROOT / "configs" / "pconv_unet.yaml")
    for override in ("model.kernel_size=5", "model.use_norm=false", "model.dropout=0.2"):
        with pytest.raises(ValueError, match="convlstm only"):
            load_config(ROOT / "configs" / "pconv_unet.yaml", [override])


def test_config_hash_follows_the_validated_values():
    base = load_config(ROOT / "configs" / "convlstm.yaml")
    a = load_config(ROOT / "configs" / "convlstm.yaml", ["training.lr=3e-4"])  # YAML reads '3e-4' as a str
    b = load_config(ROOT / "configs" / "convlstm.yaml", ["training.lr=0.0003"])
    assert a.training.lr == b.training.lr == 3e-4
    assert a.config_hash() == b.config_hash() != base.config_hash()
    for cfg in ("convlstm_smoke.yaml", "synthetic_smoke.yaml", "pconv_unet.yaml"):
        load_config(ROOT / "configs" / cfg)  # every shipped config passes the stricter schema


def test_cli_reports_a_bad_config_without_a_traceback(capsys):
    assert main(["config", "--set", "training.lrr=3e-4"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error:") and "training.lrr" in err and "Extra inputs are not permitted" in err


def test_flat_params_are_scalar():
    flat = ExperimentConfig().flat_params()
    assert flat["model.name"] == "convlstm" and flat["split.train_end"] == "2021-12-31"
    assert all(isinstance(v, int | float | str | bool) for v in flat.values())
