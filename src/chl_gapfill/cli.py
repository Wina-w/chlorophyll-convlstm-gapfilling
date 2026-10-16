"""`chl-gapfill` command line: download | align | train | evaluate | predict | config."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from chl_gapfill import __version__
from chl_gapfill.config import load_config, paths


def _cfg(args):
    overrides = list(args.set or [])
    if getattr(args, "smoke", False):
        overrides.append("smoke=true")
    return load_config(args.config, overrides)


def cmd_config(args) -> int:
    cfg = _cfg(args)
    p = paths()
    print(
        json.dumps(
            {
                "config_hash": cfg.config_hash(),
                "config": cfg.model_dump(mode="json"),
                "paths": {
                    "data": str(p.data),
                    "models": str(p.models),
                    "outputs": str(p.outputs_for(cfg)),
                    "mlflow": p.mlflow_tracking_uri,
                    "dineof_exe": str(p.dineof_exe),
                },
            },
            indent=2,
            default=str,
        )
    )
    return 0


def cmd_download(args) -> int:
    from chl_gapfill.download import download_all

    cfg = _cfg(args)
    p = paths()
    ok, total = download_all(cfg, p, keys=args.sensor or None, force=args.force)
    if not (target := p.data / cfg.data.train_nc).exists():
        print(
            f"error: the target {target} is missing, and training needs it. Run `chl-gapfill download` "
            "again later, fetch it by hand (data/README.md), or use synthetic data "
            "(python scripts/make_synthetic_data.py)",
            file=sys.stderr,
        )
    return 0 if ok == total else 1


def cmd_align(args) -> int:
    from chl_gapfill import data_sources
    from chl_gapfill.align import coverage_report

    cfg = _cfg(args)
    p = paths()
    keys = data_sources.input_sensors() + data_sources.reference_sensors()
    present = {
        k: data_sources.local_path(p.data, k) for k in keys if data_sources.local_path(p.data, k).exists()
    }
    coverage_report(
        present,
        p.data / cfg.data.train_nc,
        var_name=cfg.data.var_name,
        lat_bounds=cfg.region.lat_bounds,
        lon_bounds=cfg.region.lon_bounds,
    )
    return 0


def cmd_train(args) -> int:
    from chl_gapfill.train import fit, fit_nextday

    cfg = _cfg(args)
    if cfg.task == "nextday":
        out = fit_nextday(cfg)
        print(f"nextday model -> {out}")
        return 0
    res = fit(
        cfg,
        subset=args.subset,
        min_coverage=args.min_coverage,
        label=args.label or "",
        resume=Path(args.resume) if args.resume else None,
        use_mlflow=not args.no_mlflow,
        state_dict_out=Path(args.state_dict_out) if args.state_dict_out else None,
    )
    print(f"run_id={res.run_id} best_val={res.best_val:.6f} epochs={res.epochs_run} best={res.best_path}")
    return 0


def cmd_evaluate(args) -> int:
    from chl_gapfill.evaluate import evaluate

    cfg = _cfg(args)
    ckpt = Path(args.ckpt) if args.ckpt else paths().models / cfg.name / "best.pt"
    if not ckpt.exists():
        print(f"checkpoint not found: {ckpt}", file=sys.stderr)
        return 2
    evaluate(
        cfg,
        ckpt,
        n_boot=args.n_boot,
        block_days=args.block_days,
        make_figures=not args.no_figures,
        out_dir=Path(args.out) if args.out else None,
    )
    return 0


def cmd_predict(args) -> int:

    cfg = _cfg(args)
    ckpt = Path(args.ckpt) if args.ckpt else paths().models / cfg.name / "best.pt"
    if not ckpt.exists():
        print(f"checkpoint not found: {ckpt}", file=sys.stderr)
        return 2
    day = date.fromisoformat(args.date) if args.date else None
    try:
        info = _predict(cfg, ckpt, day, args)
    except KeyError as exc:
        print(f"error: {exc.args[0]}", file=sys.stderr)
        return 1
    print(json.dumps(info, indent=2))
    return 0


def _predict(cfg, ckpt, day, args):
    from chl_gapfill.predict import predict_day

    return predict_day(
        cfg,
        ckpt,
        day,
        out=Path(args.out) if args.out else None,
        png=args.png,
        synthetic_hole=not args.no_hole,
    )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="chl-gapfill", description=__doc__)
    ap.add_argument("--version", action="version", version=f"chl-gapfill {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(sp, smoke=False):
        sp.add_argument("-c", "--config", default=None, help="configs/*.yaml (default: built-in defaults)")
        sp.add_argument(
            "--set",
            action="extend",  # repeated --set flags accumulate instead of replacing each other
            nargs="+",
            metavar="KEY=VALUE",
            help="override, repeatable: --set training.lr=3e-4 model.hidden_channels=[32,16]",
        )
        if smoke:
            sp.add_argument("--smoke", action="store_true", help="2 epochs on a few batches: proves the loop")

    s = sub.add_parser("config", help="print the effective configuration and paths")
    common(s)
    s.set_defaults(func=cmd_config)

    s = sub.add_parser(
        "download", help="fetch the target and the sensors from NOAA CoastWatch ERDDAP (data.start..data.end)"
    )
    common(s)
    s.add_argument("--sensor", nargs="*")
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_download)

    s = sub.add_parser("align", help="per-sensor coverage report on the canonical grid")
    common(s)
    s.set_defaults(func=cmd_align)

    s = sub.add_parser("train", help="train (gapfill or nextday task)")
    common(s, smoke=True)
    s.add_argument("--resume", default=None, help="checkpoint (last.pt) to continue from")
    s.add_argument("--no-mlflow", action="store_true")
    s.add_argument("--subset", nargs="*", default=None, help="restrict input sensors (single-sensor models)")
    s.add_argument("--min-coverage", type=float, default=None)
    s.add_argument("--label", default=None, help="suffix for the run directory")
    s.add_argument("--state-dict-out", default=None, help="also write a bare state_dict here (.pth)")
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("evaluate", help="hole-pixel skill vs baselines with bootstrap CIs")
    common(s)
    s.add_argument("--ckpt", default=None)
    s.add_argument("--n-boot", type=int, default=1000)
    s.add_argument(
        "--block-days", type=int, default=1, help="moving-block bootstrap over runs of this many test days"
    )
    s.add_argument("--no-figures", action="store_true")
    s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("predict", help="fill one day and write a NetCDF (+ png)")
    common(s)
    s.add_argument("--ckpt", default=None)
    s.add_argument("--date", default=None, help="YYYY-MM-DD (default: a mid test day)")
    s.add_argument("--out", default=None)
    s.add_argument("--png", action="store_true")
    s.add_argument("--no-hole", action="store_true", help="operational fill: keep all real observations")
    s.set_defaults(func=cmd_predict)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, FileNotFoundError) as exc:  # bad config (pydantic) or missing data: no traceback
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
