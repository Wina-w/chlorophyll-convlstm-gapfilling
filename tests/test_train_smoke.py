"""Training smokes on the synthetic fixtures: loop, checkpoints, resume, evaluate, predict, MLflow, CLI."""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pytest
import torch
from netCDF4 import Dataset as NetCDFDataset

from chl_gapfill.cli import main
from chl_gapfill.evaluate import evaluate
from chl_gapfill.predict import predict_day
from chl_gapfill.provenance import sha256
from chl_gapfill.train import fit, load_weights


@pytest.mark.slow
def test_fit_writes_checkpoints_and_resumes(cfg, paths_for):
    res = fit(cfg, paths_for, use_mlflow=False)
    assert res.best_path.exists() and res.last_path.exists() and res.state_dict_path.exists()
    assert (res.run_dir / "config.json").exists() and (res.run_dir / "stats.json").exists()
    assert [h["epoch"] for h in res.history] == [1, 2]  # smoke caps at 2 epochs
    ck = torch.load(res.last_path, map_location="cpu", weights_only=True)  # tensors and plain types only
    assert set(ck) >= {
        "model",
        "optimizer",
        "scheduler",
        "epoch",
        "best_val",
        "history",
        "config",
        "config_hash",
        "bad_epochs",
        "rng",
        "run_id",
    }
    assert set(ck["rng"]) == {"python", "numpy", "torch", "cuda", "loader", "holes"}
    assert ck["config_hash"] == cfg.config_hash()

    cfg3 = cfg.model_copy(
        update={"smoke": False, "training": cfg.training.model_copy(update={"n_epochs": 3})}
    )
    res2 = fit(cfg3, paths_for, use_mlflow=False, resume=res.last_path)
    assert [h["epoch"] for h in res2.history] == [1, 2, 3]
    assert res2.best_val <= res.best_val + 1e-9


def _full(cfg, **training):
    """The synthetic config without the smoke caps (all samples, n_epochs as given)."""
    return cfg.model_copy(update={"smoke": False, "training": cfg.training.model_copy(update=training)})


def _losses(history):
    return [{k: v for k, v in h.items() if k != "sec"} for h in history]  # wall-clock time differs


@pytest.mark.slow
def test_resume_is_bit_exact(cfg, paths_for):
    # 4 epochs straight == 2 epochs, then --resume to 4: same losses, lr, best and weights, bit for bit.
    # Streams: DataLoader shuffle, training holes, and (through dropout) the global torch RNG.
    cfg = cfg.model_copy(update={"model": cfg.model.model_copy(update={"dropout": 0.2})})
    straight = fit(_full(cfg, n_epochs=4, patience=10), paths_for, use_mlflow=False, label="straight")
    half = fit(_full(cfg, n_epochs=2, patience=10), paths_for, use_mlflow=False, label="halves")
    resumed = fit(
        _full(cfg, n_epochs=4, patience=10),
        paths_for,
        use_mlflow=False,
        label="halves",
        resume=half.last_path,
    )
    assert _losses(resumed.history) == _losses(straight.history)  # exact float equality
    assert resumed.best_val == straight.best_val and resumed.epochs_run == 4
    for a, b in ((straight.last_path, resumed.last_path), (straight.best_path, resumed.best_path)):
        wa, wb = (torch.load(x, map_location="cpu", weights_only=True)["model"] for x in (a, b))
        assert wa.keys() == wb.keys() and all(torch.equal(wa[k], wb[k]) for k in wa)
    with pytest.raises(ValueError, match="training.lr"):  # resuming into a different experiment
        fit(_full(cfg, n_epochs=4, patience=10, lr=5e-4), paths_for, use_mlflow=False, resume=half.last_path)


@pytest.mark.slow
def test_resume_keeps_the_early_stopping_counter(cfg, paths_for):
    frozen = dict(lr=1e-12, patience=2)  # val loss cannot improve: best at epoch 1, stop at epoch 3
    straight = fit(_full(cfg, n_epochs=10, **frozen), paths_for, use_mlflow=False, label="es")
    assert straight.epochs_run == 3
    first = fit(_full(cfg, n_epochs=2, **frozen), paths_for, use_mlflow=False, label="es2")
    resumed = fit(
        _full(cfg, n_epochs=10, **frozen), paths_for, use_mlflow=False, label="es2", resume=first.last_path
    )
    assert resumed.epochs_run == 3  # a reset counter would train on to epoch 4
    again = fit(
        _full(cfg, n_epochs=10, **frozen), paths_for, use_mlflow=False, label="es2", resume=resumed.last_path
    )
    assert again.epochs_run == 3 and len(again.history) == 3  # already early-stopped: nothing to train


@pytest.mark.slow
def test_same_seed_same_result(cfg, paths_for):
    a = fit(cfg, paths_for, use_mlflow=False, label="a")
    b = fit(cfg, paths_for, use_mlflow=False, label="b")
    assert abs(a.best_val - b.best_val) < 1e-6
    assert [h["train"] for h in a.history] == pytest.approx([h["train"] for h in b.history], abs=1e-6)


@pytest.mark.slow
def test_evaluate_and_predict_on_smoke_model(cfg, paths_for, monkeypatch):
    import chl_gapfill.figures as figures

    shown, draw = {}, figures.gapfill_figure
    monkeypatch.setattr(
        figures, "gapfill_figure", lambda ds, fs, *a: draw(ds, shown.setdefault("fs", fs), *a)
    )
    res = fit(cfg, paths_for, use_mlflow=False)
    result = evaluate(cfg, res.best_path, paths_for, n_boot=30, make_figures=True)
    assert (paths_for.outputs / "metrics.json").exists()
    # the one-day sample map is captioned with that day's hole RMSE, not the test-period one
    fs = shown["fs"]
    assert (paths_for.outputs / "gapfill_sample.png").exists() and isinstance(fs["date"], date)
    day = np.sqrt(np.mean((fs["pred"][fs["hole"]] - fs["target"][fs["hole"]]) ** 2))
    assert fs["rmse"]["model"] == pytest.approx(day)
    assert fs["rmse"]["model"] != pytest.approx(result["chlorophyll_units"]["model"]["rmse"])
    assert set(result["log10_space_ci"]) == {
        "model",
        "model_ablate_single",
        "interp_fusion",
        "interp_single",
        "persistence",
    }
    assert result["n_test_days"] == 20 and result["config_hash"] == cfg.config_hash()
    matched = result["log10_space_matched"]
    n = matched["model"]["n"]
    assert 0 < n <= result["log10_space"]["persistence"]["n"] <= result["log10_space"]["model"]["n"]
    assert all(matched[k]["n"] == n for k in result["log10_space_ci"])  # one pixel set for every method
    for v in result["verdicts"].values():  # paired test on the matched pixels
        d = v["log_rmse_diff"]
        assert d["lo"] <= d["point"] <= d["hi"] and v["resolved"] == (d["hi"] < 0 or d["lo"] > 0)
        assert v["model_better"] == (d["point"] < 0) and v["n"] == n
    obs = result["observed_slice"]
    assert 0 < obs["frac_of_hole_px_viirs_observed"] <= obs["frac_of_hole_px"] <= 1
    for name in ("observed_slice", "independence_slice"):  # model vs persistence on the same pixels, twice
        for truth in ("target", "observation"):
            pr = result[name][f"paired_vs_{truth}"]
            d = pr["diff"]["model_minus_persistence"]
            assert (
                pr["n"] > 0
                and d["lo"] <= d["point"] <= d["hi"]
                and d["resolved"] == (d["hi"] < 0 or d["lo"] > 0)
            )
    indep = result["independence_slice"]
    assert indep["paired_vs_target"]["n"] == indep["log10_space_matched"]["model"]["n"]
    prov = result["provenance"]  # what the numbers were computed from
    assert prov["checkpoint"] == {
        "file": res.best_path.name,
        "sha256": sha256(res.best_path),
        "bytes": res.best_path.stat().st_size,
    }
    assert prov["config_hash"] == cfg.config_hash() and prov["data"]["test_days"]["n"] == 20
    assert prov["data"]["target"]["file"] == cfg.data.train_nc and set(prov["data"]["sensors"]) == {
        "modis_aqua",
        "viirs_snpp",
    }
    assert result["bootstrap"] == {"n_boot": 30, "block_days": 1, "unit": "test day"}
    assert (paths_for.outputs / "validation_reliability.png").exists()
    # legacy bare state_dict loads through the same path
    from chl_gapfill.train import build_gapfill_datasets, build_model_for

    _, ds, _ = build_gapfill_datasets(cfg, paths_for)
    m = build_model_for(cfg, ds)
    load_weights(m, res.state_dict_path, torch.device("cpu"))

    info = predict_day(cfg, res.best_path, date(2022, 4, 20), p=paths_for, png=True)
    assert info["date"] == "2022-04-20" and np.isfinite(info["hole_rmse"])
    with NetCDFDataset(info["out"]) as nc:
        assert nc.variables["chlor_a_filled"].shape == (8, 8)
        assert nc.status == "experimental" and nc.config_hash == cfg.config_hash()
        assert nc.variables["hole"][:].sum() > 0
    assert (paths_for.outputs / "filled_2022-04-20.png").exists()
    op = predict_day(cfg, res.best_path, date(2022, 4, 20), p=paths_for, synthetic_hole=False)
    assert op["hole_frac"] == 0.0 and 0 < op["observed_frac"] < 1
    with NetCDFDataset(op["out"]) as nc:
        filled, model = nc.variables["chlor_a_filled"][:], nc.variables["chlor_a_model"][:]
        observed = nc.variables["observed"][:] == 1
    assert np.array_equal(filled[~observed], model[~observed])  # the model fills only what nobody saw
    # a pixel only VIIRS S-NPP saw keeps the VIIRS value (it shares the target grid in the synthetic set)
    from chl_gapfill.align import load_sensor_aligned

    t = int(np.where((ds.dates == (2022, 4, 20)).all(1))[0][0])
    snpp, m_snpp = load_sensor_aligned(paths_for.data / "viirs_snpp.nc", ds.dates, ds.lat, ds.lon)
    _, m_modis = load_sensor_aligned(paths_for.data / "modis_aqua.nc", ds.dates, ds.lat, ds.lon)
    only = (m_snpp[t] > 0) & (m_modis[t] == 0)
    assert only.any() and np.allclose(filled[only], snpp[t][only], rtol=1e-5)
    assert not np.allclose(filled[only], model[only], rtol=1e-3)  # the old file had model values here


@pytest.mark.slow
def test_mlflow_tracking(cfg, paths_for, tmp_path, monkeypatch):
    pytest.importorskip("mlflow")
    import mlflow

    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)  # MLflow's own default artifact root would be ./mlruns here
    res = fit(cfg, paths_for, use_mlflow=True)
    run = mlflow.get_run(res.run_id)
    assert run.data.params["model.name"] == "convlstm" and run.data.params["config_hash"] == cfg.config_hash()
    assert "best_val_loss" in run.data.metrics and run.info.status == "FINISHED"
    hist = mlflow.MlflowClient().get_metric_history(res.run_id, "val_loss")
    assert [h.step for h in hist] == [1, 2]
    # artifacts sit next to the tracking DB (tmp_path/mlflow.db), never in the working directory
    assert (tmp_path / "artifacts" / cfg.experiment / res.run_id / "artifacts" / "best.pt").exists()
    assert not (cwd / "mlruns").exists()
    # a resumed run is a new MLflow run that points back and carries the whole loss curve
    res2 = fit(_full(cfg, n_epochs=3), paths_for, use_mlflow=True, resume=res.last_path)
    assert mlflow.get_run(res2.run_id).data.tags["resumed_from_run"] == res.run_id
    hist2 = mlflow.MlflowClient().get_metric_history(res2.run_id, "val_loss")
    assert sorted(h.step for h in hist2) == [1, 2, 3]


@pytest.mark.slow
def test_cli_train_evaluate_predict(cfg, paths_for, tmp_path, capsys):
    cfg_path = tmp_path / "cfg.yaml"
    import yaml

    cfg_path.write_text(yaml.safe_dump(cfg.model_dump(mode="json")), encoding="utf-8")
    assert main(["config", "-c", str(cfg_path)]) == 0
    assert json.loads(capsys.readouterr().out)["config_hash"] == cfg.config_hash()
    assert main(["train", "-c", str(cfg_path), "--smoke", "--no-mlflow"]) == 0
    best = paths_for.models / "test_convlstm" / "best.pt"
    assert best.exists()
    assert main(["evaluate", "-c", str(cfg_path), "--n-boot", "20", "--no-figures", "--block-days", "5"]) == 0
    blocked = json.loads((paths_for.outputs / "metrics.json").read_text())
    assert blocked["bootstrap"]["block_days"] == 5
    assert main(["evaluate", "-c", str(cfg_path), "--n-boot", "20", "--no-figures"]) == 0
    daily = json.loads((paths_for.outputs / "metrics.json").read_text())
    for table in (
        "log10_space_ci",
        "chlorophyll_units_ci",
        "log10_space_matched_ci",
    ):  # every CI uses the blocks
        assert blocked[table]["model"]["rmse"] != daily[table]["model"]["rmse"], table
    assert main(["predict", "-c", str(cfg_path), "--date", "2022-04-15"]) == 0
    assert main(["evaluate", "-c", str(cfg_path), "--ckpt", str(tmp_path / "missing.pt")]) == 2
    with pytest.raises(SystemExit):
        main(["--help"])


@pytest.mark.slow
def test_smoke_run_leaves_published_outputs_alone(cfg, paths_for, tmp_path, monkeypatch):
    # the README quick start: smoke config, no CHL_OUTPUT_DIR -> outputs_smoke/, never outputs/
    import yaml

    from chl_gapfill.config import reset_paths_cache

    monkeypatch.setenv("CHL_ROOT", str(tmp_path))
    monkeypatch.delenv("CHL_OUTPUT_DIR")
    reset_paths_cache()
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg.model_dump(mode="json")), encoding="utf-8")
    assert main(["train", "-c", str(cfg_path), "--no-mlflow"]) == 0
    assert main(["evaluate", "-c", str(cfg_path), "--n-boot", "20"]) == 0
    assert main(["predict", "-c", str(cfg_path), "--date", "2022-04-20", "--png"]) == 0
    smoke_out = tmp_path / "outputs_smoke"
    assert (smoke_out / "metrics.json").exists() and (smoke_out / "filled_2022-04-20.png").exists()
    assert not (tmp_path / "outputs").exists()
