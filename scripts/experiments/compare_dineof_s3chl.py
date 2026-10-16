"""Larger-grid chlorophyll: DINEOF vs ML on my Sentinel-3 OLCI chlorophyll.

Companion to the 20x20 VIIRS comparison (compare_dineof_chl.py). Here the
field is my S3 OLCI CHL_NN product (200x200 at ~0.1 deg, 30 days), so DINEOF
and the ML inpainter run on a genuinely large common grid — matching the "make the
ML grid as big as DINEOF" request, using local data (ERDDAP is unavailable).

Chlorophyll is log-normal, so everything is in log10 space (as my
S3_CHL_L4_DINEOF_Processor.m does). Hold-out uses contiguous cloud BLOBS (the
realistic, honest design — see the SST recheck), scored on real observed pixels.

The U-Net trains on scattered observed pixels (as in the published run) although it is
scored on blobs; --train-masks blob trains it on blob-shaped targets instead
(written as dineof_vs_ml_s3chl_trainblob.json / .png).

Run:
    python scripts/experiments/compare_dineof_s3chl.py [--train-masks blob]
"""
import argparse
import sys
import json
import pathlib


import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from _compat import C, cfg, P  # noqa: F401
from s3_chl_data import load_s3_chl_cube, sea_mask, DOWNSAMPLE
from chl_gapfill.evaluate import bootstrap_ci, skill
from dineof_runner import run_dineof
from compare_dineof_sst import train_pconv, predict_pconv, make_cv_blobs

WORK = C.OUTPUT_DIR / "dineof_work" / "s3chl"
SEED = C.SEED


def main(train_masks="scatter"):
    # the published files are scatter-trained; a blob-trained run gets its own names
    out_name = "dineof_vs_ml_s3chl" + ("" if train_masks == "scatter" else f"_train{train_masks}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dates, lat, lon, cube = load_s3_chl_cube()
    seam = sea_mask(cube)
    print(f"[s3chl-cmp] grid {cube.shape[1]}x{cube.shape[2]} ({DOWNSAMPLE}x downsample), "
          f"{len(dates)} days, sea={seam.mean():.0%}, per-day obs={np.isfinite(cube).mean():.1%}")

    logcube = np.where(np.isfinite(cube) & (cube > 0), np.log10(cube), np.nan).astype("float32")

    # realistic contiguous-blob hold-out on observed pixels
    cv = make_cv_blobs(logcube, seam, SEED)
    input_log = logcube.copy(); input_log[cv] = np.nan
    print(f"[s3chl-cmp] held-out observed pixels (blob): {int(cv.sum())}")

    # --- DINEOF in log10 space ---
    WORK.mkdir(parents=True, exist_ok=True)
    print("[s3chl-cmp] running DINEOF (dineof.exe) on the holed log10 CHL series ...")
    dineof_log, info = run_dineof(input_log, seam, WORK, var="chl",
                                  params=dict(nev=min(25, len(dates) - 3), neini=1, toliter=1e-2))
    print(f"[s3chl-cmp] DINEOF done (code {info['returncode']})")

    # --- ML (PartialConv U-Net) in normalised log10 space ---
    obs = np.isfinite(input_log)
    mu, sd = float(np.nanmean(input_log[obs])), float(np.nanstd(input_log[obs]))
    input_n = (input_log - mu) / sd
    print("[s3chl-cmp] training PartialConv U-Net (self-supervised inpainting) ...")
    model = train_pconv(input_n, seam, cv, device, masks=train_masks)
    ml_log = predict_pconv(model, input_n, device) * sd + mu

    # --- score on held-out pixels, log10 (primary) + chl units ---
    result = {"grid": [int(cube.shape[1]), int(cube.shape[2])], "downsample": DOWNSAMPLE,
              "n_days": len(dates), "n_heldout_px": int(cv.sum()), "cv_mode": "blob",
              "ml_train_masks": train_masks, "space": "log10(chl)", "methods": {},
              "dineof": {k: info[k] for k in ("nev", "n_modes", "cv_error", "cap_hit")}}
    dlog, mlog, tlog, dchl, mchl, tchl = [], [], [], [], [], []
    for t in range(len(cube)):
        sel = cv[t]
        if sel.sum() == 0:
            continue
        tl = logcube[t][sel]
        dlog.append(dineof_log[t][sel].astype("float64")); mlog.append(ml_log[t][sel].astype("float64"))
        tlog.append(tl.astype("float64"))
        dchl.append((10.0 ** dineof_log[t][sel]).astype("float64"))
        mchl.append((10.0 ** ml_log[t][sel]).astype("float64"))
        tchl.append((10.0 ** tl).astype("float64"))
    for name, pl, pc in [("dineof", dlog, dchl), ("ml_pconv", mlog, mchl)]:
        ci = bootstrap_ci(pl, tlog, n_boot=1000, seed=SEED)
        chl = skill(np.concatenate(pc), np.concatenate(tchl))
        result["methods"][name] = {"logRMSE": ci["rmse"], "logR2": ci["r2"], "chlRMSE": chl["rmse"], "n": ci.get("n", 0)}

    C.OUTPUT_DIR.mkdir(exist_ok=True)
    with open(C.OUTPUT_DIR / f"{out_name}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n[s3chl-cmp] DINEOF vs ML on identical blob holes (log10, 95% CI, n={int(cv.sum())}):")
    print(f"  {'method':10s} {'logRMSE':>8s}  {'95% CI':>17s} {'logR2':>7s} {'chlRMSE':>8s}")
    for name in ["dineof", "ml_pconv"]:
        m = result["methods"][name]; r = m["logRMSE"]
        print(f"  {name:10s} {r['point']:8.3f}  [{r['lo']:.3f},{r['hi']:.3f}] {m['logR2']['point']:7.2f} {m['chlRMSE']:8.3f}")

    _figure(cube, logcube, input_log, dineof_log, ml_log, cv, seam, result, out_name)
    print(f"[s3chl-cmp] wrote {out_name}.json and .png")
    return result


def _figure(cube, logcube, input_log, dineof_log, ml_log, cv, seam, result, out_name="dineof_vs_ml_s3chl"):
    t = int(np.argmax(cv.reshape(len(cube), -1).sum(1)))
    # The S3 OLCI .mat field is read by h5py transposed -> the in-memory array is
    # [lon(W->E), lat(S->N)]. flipud(a.T) renders it as [lat(N->S), lon(W->E)],
    # i.e. correct geography (Taiwan in its true position). Display-only.
    geo = lambda a: np.flipud(np.asarray(a, dtype="float64").T)
    land = geo(seam) < 0.5
    vmin, vmax = 0.03, 3

    def show(ax, logfld, title):
        chl = np.power(10.0, geo(logfld))
        f = np.where(land, np.nan, chl)
        im = ax.imshow(f, cmap="jet", norm=LogNorm(vmin=vmin, vmax=vmax))
        ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
        return im
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.6))
    show(axs[0], np.where(np.isfinite(input_log[t]), input_log[t], np.nan),
         f"input (real gaps)\nobs={np.isfinite(input_log[t]).mean():.0%}")
    show(axs[1], dineof_log[t],
         f"DINEOF fill\nlogRMSE (all days)={result['methods']['dineof']['logRMSE']['point']:.3f}")
    im = show(axs[2], ml_log[t],
              f"ML PConv-UNet fill\nlogRMSE (all days)={result['methods']['ml_pconv']['logRMSE']['point']:.3f}")
    err = np.full(cube.shape[1:], np.nan)
    err[cv[t]] = dineof_log[t][cv[t]] - logcube[t][cv[t]]
    ime = axs[3].imshow(geo(err), cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    axs[3].set_title("DINEOF error at\nheld-out px (log10)", fontsize=9)
    axs[3].set_xticks([]); axs[3].set_yticks([])
    fig.colorbar(im, ax=axs[2], fraction=0.046); fig.colorbar(ime, ax=axs[3], fraction=0.046)
    fig.suptitle("Sentinel-3 OLCI chlorophyll gap-filling: DINEOF vs ML (200x200, blob hold-out, "
                 f"U-Net trained on {result.get('ml_train_masks', 'scatter')} targets)")
    fig.tight_layout(); fig.savefig(C.OUTPUT_DIR / f"{out_name}.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train-masks", default="scatter", choices=["scatter", "blob"],
                    help="U-Net training targets: scattered pixels (the published run) or cloud blobs")
    main(train_masks=ap.parse_args().train_masks)
