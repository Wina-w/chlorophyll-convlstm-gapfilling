"""Train a genuine single-sensor gap-filling model per satellite.

The fusion model in ``models/convlstm_latest.pth`` is trained on all three input
sensors. ``evaluate.py``'s ``model_ablate_single`` merely *zeroes* the other
channels of that fusion model at inference — which is a train/test mismatch, not
an honest measure of how well one satellite alone can fill cloud gaps.

This script trains a dedicated ConvLSTM for each satellite (MODIS-Aqua,
VIIRS S-NPP, VIIRS NOAA-20) on that sensor ALONE, so ``compare_sensors.py`` can
report each satellite's true standalone skill and quantify the fusion gain.

All models are trained with ``min_coverage=0.0`` so every dataset shares the same
eligible-day axis (and therefore the same deterministic evaluation holes, which
are seeded by the target-day index).

Run:
    python scripts/experiments/train_single.py
"""
import sys
import pathlib


from _compat import C, cfg, P  # noqa: F401
from _compat import train_gapfill

SENSORS = ["modis_aqua", "viirs_snpp", "viirs_noaa20"]
MIN_COVERAGE = 0.0   # shared eligible-day axis across all single/fusion datasets


def single_model_path(sensor):
    return C.MODEL_DIR / f"convlstm_single_{sensor}.pth"


def fusion_shared_path():
    """Fusion model retrained on the shared (min_coverage=0) axis, so the
    per-satellite comparison is strictly apples-to-apples with the singles."""
    return C.MODEL_DIR / "convlstm_fusion_shared.pth"


def main():
    C.MODEL_DIR.mkdir(exist_ok=True)
    # fusion on the shared axis (for a fair per-satellite comparison table)
    print("\n==== training FUSION model on shared axis ====")
    train_gapfill(subset=SENSORS, model_out=fusion_shared_path(),
                  min_coverage=MIN_COVERAGE, label="fusion(shared)")
    for s in SENSORS:
        print(f"\n==== training SINGLE-SENSOR model: {s} ====")
        train_gapfill(subset=[s], model_out=single_model_path(s),
                      min_coverage=MIN_COVERAGE, label=s)
    print("\n[train_single] done. models:",
          [fusion_shared_path().name] + [single_model_path(s).name for s in SENSORS])


if __name__ == "__main__":
    main()
