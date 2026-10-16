"""Reproduce every experiment end-to-end (thin wrapper over the `chl-gapfill` CLI).

1. train      configs/convlstm.yaml            -> models/convlstm_fusion/best.pt
2. evaluate   bootstrap CIs, baselines, figures -> outputs/metrics.json; every test day, the
              cloudiest included                -> outputs/all_days/metrics.json
3. train_single.py   one genuine model per satellite (scripts/experiments)
4. compare_sensors.py, compare_dineof_*.py     (need DINEOF_EXE / S3_* in .env; skipped otherwise)

  python run_all.py                                         # everything
  python run_all.py --quick                                 # skip retraining, evaluate an existing model
  python run_all.py --quick --ckpt models/convlstm_latest.pth

Exits 1 if any stage fails; a stage skipped for missing inputs is not a failure.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from chl_gapfill.config import load_config, paths

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT / "scripts" / "experiments"
CFG = "configs/convlstm.yaml"
SENSORS = ("modis_aqua", "viirs_snpp", "viirs_noaa20")  # = train_single.SENSORS

# external inputs (Paths fields, set from the env vars of the same name in upper case)
DINEOF_NEEDS = {
    "compare_dineof_chl": ("dineof_exe",),
    "compare_dineof_s3chl": ("dineof_exe", "s3_chl_mat_dir"),
    "compare_dineof_sst": ("dineof_exe", "s3_sst_baseline_dir", "s3_sst_dineof_l4_dir"),
}


def _run(label: str, cmd: list[str]) -> bool:
    print("\n" + "=" * 78 + f"\n>>> {label}\n" + "=" * 78, flush=True)
    rc = subprocess.call(cmd, cwd=ROOT)
    if rc:
        print(f"[run_all] {label} FAILED (exit {rc})", flush=True)
    return rc == 0


def _cli(*args: str) -> bool:
    return _run(f"chl-gapfill {' '.join(args)}", [sys.executable, "-m", "chl_gapfill.cli", *args])


def _script(name: str, *args: str) -> bool:
    # a separate `python <script>` puts the script's folder on sys.path, which the experiment scripts
    # need for `from _compat import ...` (runpy.run_path in this process did not)
    return _run(f"{name}.py", [sys.executable, str(EXPERIMENTS / f"{name}.py"), *args])


def _skip(name: str, reason: str) -> None:
    print(f"\n[run_all] skip {name}.py: {reason}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="skip retraining; evaluate an existing checkpoint")
    ap.add_argument("--ckpt", default=None, help="checkpoint to evaluate (default: see --quick)")
    args = ap.parse_args(argv)

    p, cfg = paths(), load_config(ROOT / CFG)
    ckpt = args.ckpt
    if ckpt is None and args.quick:
        # no training run here: the last run's best.pt, else the original published checkpoint
        found = [c for c in (p.models / cfg.name / "best.pt", p.models / "convlstm_latest.pth") if c.exists()]
        ckpt = str(found[0]) if found else None  # none: evaluate reports the missing checkpoint
    ckpt_args = ("--ckpt", ckpt) if ckpt else ()

    ok = []
    if not args.quick:
        ok.append(_cli("train", "-c", CFG))
    ok.append(_cli("evaluate", "-c", CFG, *ckpt_args))
    # the same evaluation on every test day: the published table keeps days >= 20 % observed
    all_days = ("--set", "data.min_coverage=0", "--no-figures", "--out", str(p.outputs_for(cfg) / "all_days"))
    ok.append(_cli("evaluate", "-c", CFG, *ckpt_args, *all_days))
    if not args.quick:
        ok.append(_script("train_single"))
    singles = [p.models / f"convlstm_single_{s}.pth" for s in SENSORS] + [
        p.models / "convlstm_fusion_shared.pth"
    ]
    if missing := [c.name for c in singles if not c.exists()]:
        _skip("compare_sensors", f"missing {', '.join(missing)} (run without --quick to train them)")
    else:
        ok.append(_script("compare_sensors"))
    for name, needs in DINEOF_NEEDS.items():
        if unset := [n.upper() for n in needs if not (getattr(p, n) and getattr(p, n).exists())]:
            _skip(name, f"{', '.join(unset)} not set or not found (.env)")
        else:
            ok.append(_script(name, *([ckpt] if ckpt and name == "compare_dineof_chl" else [])))

    failed = ok.count(False)
    print(
        f"\n[run_all] done: {len(ok) - failed}/{len(ok)} stages ok. See outputs/ for metrics.json, "
        "per_satellite.json, dineof_vs_ml_*.json and figures."
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
