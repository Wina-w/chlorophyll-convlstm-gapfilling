"""Per-satellite comparison: genuine single-sensor models vs the fusion model.

Evaluates, on ONE identical set of held-out days and identical synthetic holes:
  * a dedicated ConvLSTM trained on MODIS-Aqua alone,
  * one on VIIRS S-NPP alone,
  * one on VIIRS NOAA-20 alone,
  * the fusion ConvLSTM (all three), and
  * the persistence baseline,
all scored on hole pixels in log10 space with day-bootstrap 95% CIs.

Because every dataset is built with ``min_coverage=0.0`` it shares the same
eligible-day axis, and the deterministic hole for a day is seeded by that day's
index — so all models literally see the same holes. Answers "how well does each
satellite fill cloud gaps on its own, and how much does fusing them help?".

Whether fusion beats a single sensor is decided by a paired day bootstrap of
RMSE(single) - RMSE(fusion) on the same pixels (resolved when its CI excludes 0),
not by whether two marginal CIs overlap. Persistence only fills the hole pixels
with a recent observation, so it is compared with fusion on those pixels.

Run (after `python scripts/experiments/train_single.py`):
    python scripts/experiments/compare_sensors.py
"""
import sys
import json
import pathlib


import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _compat import C, cfg, P  # noqa: F401
from _compat import GapFillDataset
from _compat import available_sensor_paths, split_positions, forward_batch, model_input_channels
from chl_gapfill.evaluate import bootstrap_ci, compare_same_pixels, persistence_fill
from chl_gapfill.provenance import file_record, provenance
from _compat import _load_model
from train_single import SENSORS, single_model_path, fusion_shared_path, MIN_COVERAGE


def _dataset(subset):
    paths = available_sensor_paths(subset)
    return GapFillDataset(paths, input_days=C.INPUT_DAYS, log_transform=C.LOG_TRANSFORM,
                          deterministic=True, min_coverage=MIN_COVERAGE, seed=C.SEED)


def _eval_model(ckpt, ds, test, device):
    """Per-day hole-pixel (log10) predictions & truths for one model."""
    model = _load_model(ckpt, ds, device)
    pday_p, pday_t = [], []
    with torch.no_grad():
        for pos in test:
            s = ds[int(pos)]
            hole = s["hole"][0].numpy() > 0.5
            if hole.sum() == 0:
                continue
            batch = {k: v.unsqueeze(0) for k, v in s.items()}
            pred_log = ds.target_to_log(forward_batch(model, batch, device)[0, 0].cpu().numpy())
            true_log = ds.target_to_log(s["target"][0].numpy())
            pday_p.append(np.asarray(pred_log[hole], "float64"))
            pday_t.append(np.asarray(true_log[hole], "float64"))
    return pday_p, pday_t


def _persistence(ds, test):
    """Temporal carry-forward baseline on the same holes (log10 space)."""
    pday_p, pday_t = [], []
    for pos in test:
        s = ds[int(pos)]
        hole = s["hole"][0].numpy() > 0.5
        if hole.sum() == 0:
            continue
        true_log = ds.target_to_log(s["target"][0].numpy())
        pday_p.append(np.asarray(persistence_fill(ds, s, hole)[hole], "float64"))
        pday_t.append(np.asarray(true_log[hole], "float64"))
    return pday_p, pday_t


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fusion_ds = _dataset(SENSORS)
    _, _, test = split_positions(fusion_ds.sample_dates)
    print(f"[compare] shared axis: {len(fusion_ds)} eligible days, {len(test)} test days, "
          f"H={fusion_ds.H} W={fusion_ds.W}")

    entries = []  # (label, per_day_pred, per_day_true)
    checkpoints = {}
    # persistence
    pp, pt = _persistence(fusion_ds, test)
    entries.append(("persistence", pp, pt))
    # single-sensor models
    for s in SENSORS:
        ck = single_model_path(s)
        if not ck.exists():
            print(f"[compare][skip] missing {ck.name} (run train_single.py)"); continue
        ds = _dataset([s])
        entries.append((f"single:{s}", *_eval_model(ck, ds, test, device)))
        checkpoints[f"single:{s}"] = file_record(ck)
    # fusion model
    fck = fusion_shared_path()
    if fck.exists():
        entries.append(("fusion", *_eval_model(fck, fusion_ds, test, device)))
        checkpoints["fusion"] = file_record(fck)
    else:
        print(f"[compare][skip] missing {fck.name}")

    result = {"n_test_days": int(len(test)), "min_coverage": MIN_COVERAGE, "methods": {}}
    for label, p, t in entries:
        ci = bootstrap_ci(p, t, n_boot=1000, seed=C.SEED)
        result["methods"][label] = {"logRMSE": ci["rmse"], "logR2": ci["r2"],
                                    "n": ci.get("n", 0)}
    truth = entries[0][2]
    for label, _, t in entries[1:]:  # every model is scored on the same holes and truth
        assert all(np.array_equal(a, b) for a, b in zip(t, truth)), f"{label}: different holes"
    preds = {label: p for label, p, _ in entries}
    if "fusion" in preds:
        models = {k: v for k, v in preds.items() if k != "persistence"}
        result["paired_vs_fusion"] = compare_same_pixels(models, truth, "fusion", n_boot=1000, seed=C.SEED)
        result["persistence_px"] = compare_same_pixels(
            {"fusion": preds["fusion"], "persistence": preds["persistence"]}, truth, "fusion",
            n_boot=1000, seed=C.SEED)
    result["provenance"] = {
        **provenance(cfg, fusion_ds, C.TRAIN_NC, available_sensor_paths(SENSORS),
                     test_dates=[fusion_ds.sample_dates[int(i)] for i in test]),
        "checkpoint": checkpoints,  # one per model
    }

    C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(C.OUTPUT_DIR / "per_satellite.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # ---- table ----
    print("\n[compare] per-satellite hole-pixel skill (log10 space, 95% CI):")
    print(f"  {'method':22s} {'logRMSE':>8s}  {'95% CI':>17s} {'logR2':>7s}  {'n':>8s}")
    for label, _, _ in entries:
        m = result["methods"][label]
        r, r2 = m["logRMSE"], m["logR2"]["point"]
        print(f"  {label:22s} {r['point']:8.3f}  [{r['lo']:.3f},{r['hi']:.3f}]  {r2:7.2f}  {m['n']:8d}")
    for key in ("paired_vs_fusion", "persistence_px"):
        if key not in result:
            continue
        print(f"[compare] {key} (n={result[key]['n']}): paired day bootstrap, minus fusion:")
        for name, d in result[key]["diff"].items():
            print(f"  {name:40s} {d['point']:+.4f} [{d['lo']:+.4f},{d['hi']:+.4f}]"
                  + ("" if d["resolved"] else "  (unresolved)"))

    _figure(result, [e[0] for e in entries], C.OUTPUT_DIR / "per_satellite.png")
    print(f"[compare] wrote {C.OUTPUT_DIR/'per_satellite.json'} and per_satellite.png")


def _figure(result, labels, out):
    """Bar chart with 95% CIs. Persistence is scored on fewer (easier) pixels than the models,
    so its bar is hatched and annotated with the fusion model's score on those same pixels."""
    pts = [result["methods"][l]["logRMSE"]["point"] for l in labels]
    los = [result["methods"][l]["logRMSE"]["lo"] for l in labels]
    his = [result["methods"][l]["logRMSE"]["hi"] for l in labels]
    ns = [result["methods"][l]["n"] for l in labels]
    err = [[p - lo for p, lo in zip(pts, los)], [hi - p for p, hi in zip(pts, his)]]
    colors = ["#888"] + ["#4C9F70"] * sum(l.startswith("single") for l in labels) + ["#2C6FBB"]
    colors = colors[:len(labels)]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(range(len(labels)), pts, yerr=err, capsize=4, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l.replace("single:", "").replace("_", "\n") + f"\nn={n}" for l, n in zip(labels, ns)],
                       fontsize=9)
    if "persistence_px" in result:
        bars[0].set_hatch("//"); bars[0].set_alpha(0.5)
        f = result["persistence_px"]["methods"]["fusion"]["rmse"]["point"]
        ax.annotate(f"only the pixels it fills;\nfusion there: {f:.3f}", (0, his[0]), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8)
        ax.set_ylim(0, max(his) * 1.2)
    ax.set_ylabel("hole-pixel logRMSE (lower is better)")
    ax.set_title("Per-satellite single-sensor models vs fusion (95% CI)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    return fig


if __name__ == "__main__":
    main()
