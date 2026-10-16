"""Sentinel-3 SST: DINEOF vs a deep-learning gap-filler on my Sentinel-3 SST series,
on the SAME grid and the SAME held-out pixels (gold-standard cross-validation).

This is the most direct DINEOF-vs-machine-learning benchmark, because my production
DINEOF processing is for SST, and here BOTH methods run on an identical
cropped+downsampled slice of that DINEOF grid.

Setup (leave-out cross-validation on truly-observed pixels)
  * Field  : Enhanced_Production_Output_v12_0_Baseline_QC — real strict-QC SST
             observations with real cloud gaps.
  * Truth  : a deterministic random subset of *actually observed* sea pixels
             (independent measurements, not interpolated), hidden from BOTH
             methods, then reconstructed and compared to the real value.
  * DINEOF : the compiled dineof.exe (GHER code), re-run on the holed series (NOT the
             ready-made L4, because that L4 already "saw" these pixels).
  * ML     : a PartialConv U-Net (Liu et al. 2018) trained self-supervised on the
             remaining observed pixels. (Spatial inpainting is used because the
             series is only 30 days — too short for a temporal ConvLSTM; this is a
             documented limitation, not the ceiling of the ML approach.)

The U-Net's training targets are scattered observed pixels, also when it is scored on blob
holes (as in the published runs), so it never trains on a contiguous hole; --train-masks blob
trains it on blob-shaped targets instead (written as *_trainblob.json / .png).

Run:
    python scripts/experiments/compare_dineof_sst.py [scatter|blob|both] [--train-masks blob]
"""
import argparse
import sys
import json
import pathlib


import numpy as np
import torch
from torch import nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _compat import C, cfg, P  # noqa: F401
from chl_gapfill.model import PartialConvUNet
from sst_data import load_sst_cube, sea_mask, REGION, DOWNSAMPLE
from chl_gapfill.evaluate import bootstrap_ci, skill
from dineof_runner import run_dineof

WORK = C.OUTPUT_DIR / "dineof_work" / "sst"
CV_FRAC = 0.04          # fraction of observed pixels held out for scoring
SEED = C.SEED


def make_cv(cube, seam, frac, seed):
    """Boolean (T,H,W): True at observed sea pixels held out for scoring (scatter)."""
    rng = np.random.default_rng(seed)
    obs = np.isfinite(cube) & (seam[None] > 0.5)
    cv = obs & (rng.random(cube.shape) < frac)
    return cv


def make_cv_blobs(cube, seam, seed, n_blobs=3, min_frac=0.06, max_frac=0.18):
    """Boolean (T,H,W): held-out = observed sea pixels inside random elliptical
    "cloud" blobs. Contiguous blobs are far harder for spatial inpainting than
    scattered pixels (a held-out pixel's same-day neighbours are also hidden), so
    this is the honest robustness check that the ML win is not just near-neighbour
    copying."""
    from chl_gapfill.dataset import make_cloud_mask
    H, W = cube.shape[1:]
    obs = np.isfinite(cube) & (seam[None] > 0.5)
    cv = np.zeros(cube.shape, dtype=bool)
    for t in range(len(cube)):
        if obs[t].sum() < 40:
            continue
        rng = np.random.default_rng(seed + 1000 + t)
        blob = make_cloud_mask(H, W, rng, min_frac, max_frac, n_blobs) > 0.5
        cv[t] = obs[t] & blob
    return cv


def training_targets(obs, rng, masks="scatter"):
    """Observed pixels hidden from the U-Net and predicted in one training step: a random half
    of them ("scatter", what the published runs used, also in blob mode) or those under random
    cloud blobs shaped like the blob hold-out ("blob"), so training matches a blob test."""
    if masks == "blob":
        from chl_gapfill.dataset import make_cloud_mask
        return obs & (make_cloud_mask(*obs.shape, rng, 0.06, 0.18, 3) > 0.5)
    idx = np.argwhere(obs)
    sel = rng.random(len(idx)) < 0.5
    tgt = np.zeros_like(obs)
    tgt[idx[sel][:, 0], idx[sel][:, 1]] = True
    return tgt


def train_pconv(input_cube_n, seam, cv, device, epochs=400, seed=SEED, masks="scatter"):
    """Self-supervised PartialConv U-Net inpainting on the (normalised) gappy SST.

    Held-out CV pixels are already NaN in input_cube_n, so they are never seen as
    input nor used as a training target -> no leakage. `masks` picks the training
    targets (see training_targets)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = PartialConvUNet(1, base=32, depth=3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    days = [d for d in range(len(input_cube_n))
            if np.isfinite(input_cube_n[d]).sum() > 60]
    seam_t = torch.from_numpy(seam[None, None]).float().to(device)
    best, beststate, patience, bad = 1e9, None, 40, 0
    for ep in range(epochs):
        model.train()
        rng.shuffle(days)
        tot = 0.0
        for d in days:
            fld = input_cube_n[d]
            obs = np.isfinite(fld)
            idx = np.argwhere(obs)
            if len(idx) < 40:
                continue
            # split observed pixels into visible (input) and target (to predict)
            tgt = training_targets(obs, rng, masks)
            if masks == "blob" and not tgt.any():
                continue  # the blobs missed every observed pixel this day
            vis = obs & ~tgt
            x = np.where(vis, np.nan_to_num(fld), 0.0).astype("float32")
            vmask = vis.astype("float32")
            tmask = tgt.astype("float32")
            target = np.nan_to_num(fld).astype("float32")
            xb = torch.from_numpy(x[None, None]).to(device)
            mb = torch.from_numpy(vmask[None, None]).to(device)
            tb = torch.from_numpy(target[None, None]).to(device)
            tmb = torch.from_numpy(tmask[None, None]).to(device)
            opt.zero_grad()
            pred = model(xb, mb)
            loss = lossf(pred * tmb, tb * tmb) * (tmb.numel() / (tmb.sum() + 1e-6))
            loss.backward(); opt.step()
            tot += float(loss.detach())
        tot /= max(len(days), 1)
        if tot < best - 1e-6:
            best, beststate, bad = tot, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if beststate:
        model.load_state_dict(beststate)
    model.eval()
    return model


def predict_pconv(model, input_cube_n, device):
    """Reconstruct every day's full field from its observed pixels."""
    H, W = input_cube_n.shape[1:]
    out = np.full_like(input_cube_n, np.nan)
    with torch.no_grad():
        for d in range(len(input_cube_n)):
            fld = input_cube_n[d]
            vis = np.isfinite(fld)
            if vis.sum() < 5:
                continue
            x = np.where(vis, np.nan_to_num(fld), 0.0).astype("float32")
            xb = torch.from_numpy(x[None, None]).to(device)
            mb = torch.from_numpy(vis.astype("float32")[None, None]).to(device)
            out[d] = model(xb, mb)[0, 0].cpu().numpy()
    return out


def main(cv_mode="scatter", train_masks="scatter"):
    # the published files are scatter-trained; a blob-trained run gets its own names
    suffix = ("" if cv_mode == "scatter" else f"_{cv_mode}") + (
        "" if train_masks == "scatter" else f"_train{train_masks}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dates, lat, lon, cube = load_sst_cube("baseline")
    seam = sea_mask(cube)
    print(f"[sst-cmp:{cv_mode}] grid {cube.shape[1]}x{cube.shape[2]} ({DOWNSAMPLE}x downsample), "
          f"{len(dates)} days, sea={seam.mean():.0%}, per-day obs={np.isfinite(cube).mean():.1%}")

    # --- hold out real observations for scoring ---
    if cv_mode == "blob":
        cv = make_cv_blobs(cube, seam, SEED)
    else:
        cv = make_cv(cube, seam, CV_FRAC, SEED)
    truth = cube[cv]
    input_cube = cube.copy(); input_cube[cv] = np.nan
    print(f"[sst-cmp:{cv_mode}] held-out observed pixels for scoring: {int(cv.sum())}")

    # --- DINEOF on the holed series (physical degC) ---
    work = WORK.parent / f"sst{suffix or '_scatter'}"
    work.mkdir(parents=True, exist_ok=True)
    print("[sst-cmp] running DINEOF (dineof.exe) on the holed SST series ...")
    dineof_filled, info = run_dineof(input_cube, seam, work, var="sst",
                                     params=dict(nev=min(25, len(dates) - 3), neini=1,
                                                 toliter=1e-2))
    print(f"[sst-cmp] DINEOF done (code {info['returncode']})")

    # --- ML (PartialConv U-Net) on the same holed series, normalised ---
    obs_all = np.isfinite(input_cube)
    mu, sd = float(np.nanmean(input_cube[obs_all])), float(np.nanstd(input_cube[obs_all]))
    input_n = (input_cube - mu) / sd
    print("[sst-cmp] training PartialConv U-Net (self-supervised inpainting) ...")
    model = train_pconv(input_n, seam, cv, device, masks=train_masks)
    ml_filled = predict_pconv(model, input_n, device) * sd + mu

    # --- score both on the held-out observed pixels ---
    result = {"grid": [int(cube.shape[1]), int(cube.shape[2])], "downsample": DOWNSAMPLE,
              "region": REGION, "n_days": len(dates), "n_heldout_px": int(cv.sum()),
              "cv_mode": cv_mode, "ml_train_masks": train_masks, "units": "degC", "methods": {},
              "dineof": {k: info[k] for k in ("nev", "n_modes", "cv_error", "cap_hit")}}
    # collect per-day arrays for block bootstrap
    d_pred, m_pred, tru = [], [], []
    for t in range(len(cube)):
        sel = cv[t]
        if sel.sum() == 0:
            continue
        d_pred.append(dineof_filled[t][sel].astype("float64"))
        m_pred.append(ml_filled[t][sel].astype("float64"))
        tru.append(cube[t][sel].astype("float64"))
    for name, preds in [("dineof", d_pred), ("ml_pconv", m_pred)]:
        ci = bootstrap_ci(preds, tru, n_boot=1000, seed=SEED)
        pt = skill(np.concatenate(preds), np.concatenate(tru))
        result["methods"][name] = {"rmse": ci["rmse"], "r2": ci["r2"], "bias": pt["bias"], "n": pt["n"]}

    C.OUTPUT_DIR.mkdir(exist_ok=True)
    with open(C.OUTPUT_DIR / f"dineof_vs_ml_sst{suffix}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n[sst-cmp:{cv_mode}] DINEOF vs ML on identical held-out SST pixels "
          f"(degC, 95% CI, n={int(cv.sum())}):")
    print(f"  {'method':10s} {'RMSE':>7s}  {'95% CI':>15s} {'R2':>6s} {'bias':>7s} {'n':>7s}")
    for name in ["dineof", "ml_pconv"]:
        m = result["methods"][name]; r = m["rmse"]
        print(f"  {name:10s} {r['point']:7.3f}  [{r['lo']:.3f},{r['hi']:.3f}] {m['r2']['point']:6.2f} "
              f"{m['bias']:7.3f} {m['n']:7d}")

    _figure(cube, input_cube, dineof_filled, ml_filled, cv, seam, result, suffix)
    print(f"[sst-cmp] wrote dineof_vs_ml_sst{suffix}.json and .png")
    return result


def _figure(cube, input_cube, dineof_filled, ml_filled, cv, seam, result, suffix=""):
    t = int(np.argmax(cv.reshape(len(cube), -1).sum(1)))   # day with most held-out px
    land = seam < 0.5
    vmin, vmax = np.nanpercentile(cube, 2), np.nanpercentile(cube, 98)

    # SST arrays are [lat(S->N), lon(W->E)]; origin="lower" renders them north-up,
    # west-left (correct geography: Taiwan appears in its true position).
    def show(ax, fld, title):
        f = np.where(land, np.nan, fld)
        im = ax.imshow(f, cmap="turbo", vmin=vmin, vmax=vmax, origin="lower")
        ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
        return im
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.4))
    show(axs[0], np.where(np.isfinite(input_cube[t]), input_cube[t], np.nan),
         f"input (real gaps)\nday obs={np.isfinite(input_cube[t]).mean():.0%}")
    show(axs[1], dineof_filled[t],
         f"DINEOF fill\nRMSE (all days)={result['methods']['dineof']['rmse']['point']:.3f}°C")
    im = show(axs[2], ml_filled[t],
              f"ML (PConv-UNet) fill\nRMSE (all days)={result['methods']['ml_pconv']['rmse']['point']:.3f}°C")
    # held-out error for DINEOF
    err = np.full(cube.shape[1:], np.nan)
    err[cv[t]] = dineof_filled[t][cv[t]] - cube[t][cv[t]]
    ime = axs[3].imshow(err, cmap="RdBu_r", vmin=-2, vmax=2, origin="lower")
    axs[3].set_title("DINEOF error at\nheld-out pixels (°C)", fontsize=9)
    axs[3].set_xticks([]); axs[3].set_yticks([])
    fig.colorbar(im, ax=axs[2], fraction=0.046)
    fig.colorbar(ime, ax=axs[3], fraction=0.046)
    fig.suptitle(f"Sentinel-3 SST cloud gap-filling: DINEOF vs ML "
                 f"(same grid, same held-out pixels; hold-out={result.get('cv_mode','scatter')}, "
                 f"U-Net trained on {result.get('ml_train_masks', 'scatter')} targets)")
    fig.tight_layout(); fig.savefig(C.OUTPUT_DIR / f"dineof_vs_ml_sst{suffix}.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", nargs="?", default="both", choices=["scatter", "blob", "both"],
                    help="hold-out design (default: both)")
    ap.add_argument("--train-masks", default="scatter", choices=["scatter", "blob"],
                    help="U-Net training targets: scattered pixels (the published runs) or cloud blobs")
    args = ap.parse_args()
    for m in ["scatter", "blob"] if args.mode == "both" else [args.mode]:
        main(cv_mode=m, train_masks=args.train_masks)
