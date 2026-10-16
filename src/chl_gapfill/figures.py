"""Matplotlib figures for evaluation and prediction (headless)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

VMIN, VMAX, TICKS = 0.03, 10, [0.03, 0.1, 0.4, 2, 10]


def gapfill_figure(ds, fs: dict, result: dict, out: Path) -> Path:
    hole = fs["hole"]
    cols = ds.n_sensors + 3
    fig, axs = plt.subplots(1, cols, figsize=(4 * cols, 4.2))
    for si in range(ds.n_sensors):
        obs = np.where(fs["m_t"][si], ds.log_to_chl(fs["v_t_log"][si], ds.log_transform), np.nan)
        axs[si].imshow(obs, cmap="jet", norm=LogNorm(vmin=VMIN, vmax=VMAX))
        axs[si].set_title(f"{ds.sensor_keys[si]}\n(day-t obs, {np.isfinite(obs).mean():.0%})", fontsize=9)
    axs[ds.n_sensors].imshow(
        np.where(hole, np.nan, fs["target"]), cmap="jet", norm=LogNorm(vmin=VMIN, vmax=VMAX)
    )
    axs[ds.n_sensors].set_title("target with hole\n(white = to fill)", fontsize=9)
    axs[ds.n_sensors + 1].imshow(fs["pred"], cmap="jet", norm=LogNorm(vmin=VMIN, vmax=VMAX))
    axs[ds.n_sensors + 1].set_title(f"{result['model']} filled", fontsize=9)
    err = np.where(hole, fs["pred"] - fs["target"], np.nan)
    im = axs[ds.n_sensors + 2].imshow(err, cmap="RdBu_r", vmin=-1, vmax=1)
    axs[ds.n_sensors + 2].set_title("error on hole\n(pred - target)", fontsize=9)
    fig.colorbar(im, ax=axs[ds.n_sensors + 2], fraction=0.046)
    rmse = fs["rmse"]  # the day shown, not the test-period RMSE in metrics.json
    fig.suptitle(
        f"Multi-sensor cloud gap-filling, {fs['date']}  |  this day's hole RMSE (chl): "
        f"model={rmse['model']:.3f} vs interp_fusion={rmse['interp_fusion']:.3f}"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def validation_figures(acc_log: dict, per_day_rmse: dict, result: dict, out_dir: Path) -> list[Path]:
    unit = "log10(chl)" if result["log_transform"] else "chl"
    p = np.concatenate(acc_log["model"]["p"])
    t = np.concatenate(acc_log["model"]["t"])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.hexbin(t, p, gridsize=40, cmap="viridis", mincnt=1)
    lim = [min(t.min(), p.min()), max(t.max(), p.max())]
    ax.plot(lim, lim, "r--", lw=1)
    ax.set_xlabel(f"truth ({unit})")
    ax.set_ylabel(f"model prediction ({unit})")
    ax.set_title(f"Reliability on hole pixels  (R²={result['log10_space']['model']['r2']:.2f}, n={len(t)})")
    fig.tight_layout()
    f1 = out_dir / "validation_reliability.png"
    fig.savefig(f1, dpi=120)
    plt.close(fig)

    order = ["persistence", "interp_single", "interp_fusion", "model_ablate_single", "model"]
    data = [np.array([r for r in per_day_rmse[k] if np.isfinite(r)]) for k in order]
    labels = [o.replace("_", "\n") for o in order]
    labels[0] += "\n(only px it fills)"  # no value where no sensor saw the pixel in the window
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot(data, tick_labels=labels, showmeans=True)
    ax.set_ylabel("per-day hole RMSE (chl units)")
    ax.set_title("Per-day skill distribution (each box = 1 point per test day)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    f2 = out_dir / "validation_perday_rmse.png"
    fig.savefig(f2, dpi=120)
    plt.close(fig)
    return [f1, f2]


def predict_figure(ds, sample: dict, pred, target, hole, model_name: str, rmse: float, out: Path) -> Path:
    n_s, T = ds.n_sensors, ds.input_days
    x, mflat = sample["x"].numpy(), sample["mask_flat"].numpy()
    base = (T - 1) * n_s
    cols = n_s + 3
    fig, axs = plt.subplots(1, cols, figsize=(4 * cols, 4.4))
    for si in range(n_s):
        m = mflat[base + si] > 0.5
        obs = np.where(m, ds.log_to_chl(ds.sensor_denorm_log(x[base + si], si), ds.log_transform), np.nan)
        axs[si].imshow(obs, cmap="jet", norm=LogNorm(vmin=VMIN, vmax=VMAX))
        axs[si].set_title(f"{ds.sensor_keys[si]}\nday-t obs ({m.mean():.0%})", fontsize=9)
    axs[n_s].imshow(np.where(hole, np.nan, target), cmap="jet", norm=LogNorm(vmin=VMIN, vmax=VMAX))
    axs[n_s].set_title("target + hole\n(white = to fill)", fontsize=9)
    im = axs[n_s + 1].imshow(pred, cmap="jet", norm=LogNorm(vmin=VMIN, vmax=VMAX))
    axs[n_s + 1].set_title(f"{model_name} filled", fontsize=9)
    cb = fig.colorbar(im, ax=axs[n_s + 1], fraction=0.046)
    cb.set_ticks(TICKS)
    err = np.where(hole, pred - target, np.nan)
    im = axs[n_s + 2].imshow(err, cmap="RdBu_r", vmin=-1, vmax=1)
    axs[n_s + 2].set_title(f"error on hole\nRMSE={rmse:.3f}", fontsize=9)
    fig.colorbar(im, ax=axs[n_s + 2], fraction=0.046)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
