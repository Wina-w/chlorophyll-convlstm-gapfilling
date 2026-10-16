"""run_all.py: experiment scripts must import the way they are launched, and failures must fail the run."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from chl_gapfill.config import reset_paths_cache

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "scripts" / "experiments"


@pytest.fixture
def run_all(tmp_path, monkeypatch):
    # CHL_ROOT elsewhere: no local .env (DINEOF_EXE, S3_*) leaks into the stage selection, and no
    # CHL_DATA_DIR / CHL_OUTPUT_DIR from the developer's shell moves the expected paths
    monkeypatch.setenv("CHL_ROOT", str(tmp_path))
    monkeypatch.setenv("CHL_MODEL_DIR", str(tmp_path / "models"))
    for var in (
        "CHL_DATA_DIR",
        "CHL_OUTPUT_DIR",
        "DINEOF_EXE",
        "S3_CHL_MAT_DIR",
        "S3_SST_BASELINE_DIR",
        "S3_SST_DINEOF_L4_DIR",
    ):
        monkeypatch.delenv(var, raising=False)
    reset_paths_cache()
    spec = importlib.util.spec_from_file_location("run_all", ROOT / "run_all.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    reset_paths_cache()


def test_every_experiment_script_imports():
    # the scripts import their siblings (`from _compat import ...`); run_all used to fail on all of them
    names = sorted(p.stem for p in EXPERIMENTS.glob("*.py"))
    code = "import importlib, sys; [importlib.import_module(m) for m in sys.argv[1:]]"
    r = subprocess.run(
        [sys.executable, "-B", "-c", code, *names],
        cwd=EXPERIMENTS,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert r.returncode == 0, r.stderr[-2000:]


def test_script_stage_sees_its_siblings_and_reports_failure(run_all, tmp_path, monkeypatch):
    (tmp_path / "_helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "ok.py").write_text("from _helper import VALUE\nassert VALUE == 1\n", encoding="utf-8")
    (tmp_path / "boom.py").write_text("raise RuntimeError('stage failed')\n", encoding="utf-8")
    monkeypatch.setattr(run_all, "EXPERIMENTS", tmp_path)
    assert run_all._script("ok") is True  # ModuleNotFoundError under the old runpy.run_path
    assert run_all._script("boom") is False


def test_quick_run_skips_missing_inputs_and_exits_nonzero_on_failure(run_all, tmp_path, monkeypatch):
    calls = []

    def fake_run(label, cmd):
        calls.append(cmd[2:])
        return False  # every stage that runs fails

    monkeypatch.setattr(run_all, "_run", fake_run)
    (tmp_path / "models").mkdir()
    shipped = tmp_path / "models" / "convlstm_latest.pth"
    shipped.touch()
    assert run_all.main(["--quick"]) == 1  # used to exit 0 whatever failed
    # only evaluate ran (published days, then all days), on the existing checkpoint; compare_sensors
    # and the DINEOF stages lack inputs
    evaluate = ["chl_gapfill.cli", "evaluate", "-c", run_all.CFG, "--ckpt", str(shipped)]
    all_days = [
        "--set",
        "data.min_coverage=0",
        "--no-figures",
        "--out",
        str(tmp_path / "outputs" / "all_days"),
    ]
    assert calls == [evaluate, evaluate + all_days]

    monkeypatch.setattr(run_all, "_run", lambda label, cmd: True)
    assert run_all.main(["--quick"]) == 0
