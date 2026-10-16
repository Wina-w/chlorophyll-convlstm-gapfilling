"""Write a small synthetic multi-sensor dataset (8x8, 120 days) for smoke runs and CI.

    python scripts/make_synthetic_data.py --out data_synth
    CHL_DATA_DIR=$PWD/data_synth chl-gapfill train -c configs/synthetic_smoke.yaml --no-mlflow

Same generator as the test fixtures (`chl_gapfill.synthetic`) so a fresh clone, or the
Docker image, can run the whole train -> evaluate -> predict loop without downloading anything.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from chl_gapfill.synthetic import H, T, W, make_synthetic


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_synth")
    args = ap.parse_args()
    root = Path(args.out)
    make_synthetic(root)
    print(f"synthetic data written to {root} (target + viirs_snpp + modis_aqua, {T} days, {H}x{W})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
