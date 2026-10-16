"""Pieces of the DINEOF benchmark that run without dineof.exe: its log parsing and the input designs."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

EXPERIMENTS = Path(__file__).resolve().parents[1] / "scripts" / "experiments"

LOG = """
 number of EOF modes you want to compute          20
 EOF mode    Expected Error    Iterations made   Convergence achieved
     19              0.0393                  1             0.2706E-02
     20              0.0387                  1             0.2628E-02
  Minimum reached in cross-validation
  Number of optimal EOF modes:           20
number of eigenvalues retained for the reconstruction  20
         expected error calculated by cross-validation          0.0387
"""


@pytest.fixture
def experiments(monkeypatch):
    monkeypatch.syspath_prepend(str(EXPERIMENTS))  # the scripts import their siblings
    return importlib.import_module


def test_dineof_log_gives_the_mode_search(experiments):
    parse_log = experiments("dineof_runner").parse_log
    # the published chl run: its cross-validation chose the cap, so the search was cut short
    assert parse_log(LOG) == {"nev": 20, "n_modes": 20, "cv_error": 0.0387, "cap_hit": True}
    below = LOG.replace("optimal EOF modes:           20", "optimal EOF modes:           6")
    assert parse_log(below)["cap_hit"] is False and parse_log(below)["n_modes"] == 6
    assert parse_log("dineof.exe crashed") == {
        "nev": None,
        "n_modes": None,
        "cv_error": None,
        "cap_hit": None,
    }


def test_dineof_designs_differ_only_in_what_dineof_sees(experiments):
    design_cubes = experiments("compare_dineof_chl").design_cubes
    rng = np.random.default_rng(0)
    truth = rng.normal(size=(6, 4, 4)).astype("float32")
    obs = np.where(rng.random((6, 4, 4)) < 0.3, truth + 0.2, np.nan).astype("float32")
    hole = np.zeros((4, 4), bool)
    hole[1:3, 1:3] = True
    obs[2][hole] = 1.0  # a sensor saw part of the test hole: no design may keep it
    cubes = design_cubes(truth, obs, {2: hole})
    assert set(cubes) == {"obs", "target_at_obs", "clean_cube"}
    assert all(np.isnan(c[2][hole]).all() for c in cubes.values())
    rest = np.ones(truth.shape, bool)
    rest[2][hole] = False
    # obs: the sensor values with their real gaps; target_at_obs: the truth on exactly those pixels
    np.testing.assert_array_equal(cubes["obs"][rest], obs[rest])
    np.testing.assert_array_equal(np.isfinite(cubes["target_at_obs"]), np.isfinite(cubes["obs"]))
    seen = np.isfinite(cubes["obs"])
    np.testing.assert_array_equal(cubes["target_at_obs"][seen], truth[seen])
    # clean_cube: the leaky original design, the truth everywhere but the hole
    np.testing.assert_array_equal(cubes["clean_cube"][rest], truth[rest])
    assert np.isfinite(obs).mean() < 0.5 < np.isfinite(cubes["clean_cube"]).mean()


def test_run_dineof_writes_short_relative_names_and_reads_the_mode_search(experiments, tmp_path, monkeypatch):
    runner = experiments("dineof_runner")
    exe = tmp_path / "dineof.exe"
    exe.touch()
    work = tmp_path / "work"
    cube = np.random.default_rng(0).normal(size=(5, 3, 3)).astype("float32")
    cube[1, 0, 0] = np.nan

    def fake_dineof(cmd, cwd, **kw):
        init = Path(cmd[1]).read_text(encoding="utf-8")
        # names relative to its cwd: DINEOF cuts long paths short
        assert tmp_path.as_posix() not in init and "data = ['dineof_in.nc#chl']" in init
        # what the binary does: write the reconstruction relative to its working directory
        runner.write_dineof_nc(
            Path(cwd) / "Output" / "dineof_out.nc", np.nan_to_num(cube), np.ones((3, 3)), "chl"
        )
        return type("Done", (), {"returncode": 0, "stdout": LOG, "stderr": ""})()

    monkeypatch.setattr(runner.subprocess, "run", fake_dineof)
    filled, info = runner.run_dineof(cube, np.ones((3, 3)), work, var="chl", exe=exe)
    np.testing.assert_array_equal(filled, np.nan_to_num(cube))
    assert (info["n_modes"], info["cap_hit"], info["returncode"]) == (20, True, 0)
    assert (work / "dineof.log").exists()


def test_per_satellite_figure_marks_persistence_as_a_subset(experiments, tmp_path):
    figure = experiments("compare_sensors")._figure

    def ci(x):
        return {"point": x, "lo": x - 0.01, "hi": x + 0.01}

    result = {
        "methods": {
            "persistence": {"logRMSE": ci(0.156), "n": 6552},
            "single:modis_aqua": {"logRMSE": ci(0.186), "n": 14625},
            "fusion": {"logRMSE": ci(0.176), "n": 14625},
        },
        "persistence_px": {"methods": {"fusion": {"rmse": ci(0.127)}}},
    }
    ax = figure(result, list(result["methods"]), tmp_path / "per_satellite.png").axes[0]
    assert (tmp_path / "per_satellite.png").exists()
    # persistence is scored on fewer, easier pixels: its bar must not read as the best model
    assert ax.patches[0].get_hatch() == "//" and not ax.patches[2].get_hatch()
    assert [t.get_text().splitlines()[-1] for t in ax.get_xticklabels()] == ["n=6552", "n=14625", "n=14625"]
    assert any("fusion there: 0.127" in t.get_text() for t in ax.texts)


@pytest.mark.slow
def test_pconv_can_train_on_blob_targets_and_scatter_is_unchanged(experiments):
    sst = experiments("compare_dineof_sst")
    obs = np.ones((40, 40), bool)
    obs[:, :6] = False  # land / cloud: never a target
    scatter = sst.training_targets(obs, np.random.default_rng(0), "scatter")
    blob = sst.training_targets(obs, np.random.default_rng(0), "blob")
    # "scatter" is the draw of the published runs: a random half of the observed pixels
    idx = np.argwhere(obs)
    sel = np.random.default_rng(0).random(len(idx)) < 0.5
    published = np.zeros_like(obs)
    published[idx[sel][:, 0], idx[sel][:, 1]] = True
    np.testing.assert_array_equal(scatter, published)
    assert blob.any() and not (blob & ~obs).any()

    def edges(m):
        return int((m[:, 1:] != m[:, :-1]).sum() + (m[1:] != m[:-1]).sum())

    assert edges(blob) * 5 < edges(scatter)  # contiguous holes, like the blob hold-out, not salt and pepper

    import torch

    cube = np.random.default_rng(1).normal(size=(3, 32, 32)).astype("float32")
    model = sst.train_pconv(cube, np.ones((32, 32)), None, torch.device("cpu"), epochs=2, masks="blob")
    assert np.isfinite(sst.predict_pconv(model, cube, torch.device("cpu"))).all()
