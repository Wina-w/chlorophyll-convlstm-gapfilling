"""Chlorophyll: DINEOF vs the ConvLSTM on the same synthetic holes, given the same observations.

DINEOF (Beckers & Rixen 2003; Alvera-Azcarate et al. 2005) runs through ``dineof_runner``
(the open-source GHER code, compiled locally). Everything is in log10 space and scored only on
the held-out hole pixels of the test days.

What each method gets
---------------------
ConvLSTM : days t-2..t of the raw MODIS-Aqua, VIIRS S-NPP and VIIRS NOAA-20 observations with
           their real cloud gaps; on day t the evaluation hole is removed from every sensor.
DINEOF   : one (days, 20, 20) cube per input design, every test day's hole removed:

  obs            the same real observations: the mean of the sensors that saw a pixel, each
                 shifted by its mean offset to the target over the training days. The like-for-like
                 comparison.
  target_at_obs  the target's values on exactly the pixels the sensors saw: the same gaps, but values
                 already on the target's scale. Separates coverage from sensor calibration.
  clean_cube     the gap-free target with only the test holes removed (the original design). DINEOF
                 then sees 99 % of the truth, and the truth is itself NOAA's DINEOF product: a leaky
                 upper bound, not a comparison.

DINEOF runs without its temporal filter (alpha = 0): it builds the EOF modes from the whole record,
future days included, and fills day t from day t's own data through those modes.

The target (``Merged_VIIRS_CHL_Daily.nc``) is NOAA's DINEOF gap-filled VIIRS S-NPP + NOAA-20 product
(ERDDAP ``nesdisVHNnoaaSNPPnoaa20chlaGapfilledDaily``). On a hole pixel no sensor saw that day its
"truth" is a DINEOF fill, so the hole pixels a sensor did see are also scored against that withheld
observation (offset as above): a truth NOAA's product is not in.

Every comparison uses the same pixels for every method, with a paired day bootstrap of
RMSE(method) - RMSE(ConvLSTM). DINEOF chooses its number of EOF modes by its own cross-validation
up to ``nev``; when it chooses the cap, the run is repeated with twice the cap (up to MAX_NEV).

Run (needs DINEOF_EXE in .env; the checkpoint defaults to models/<config name>.pth):
    python scripts/experiments/compare_dineof_chl.py [checkpoint]
"""
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
from _compat import GapFillDataset
from _compat import available_sensor_paths, split_positions, forward_batch
from chl_gapfill.evaluate import compare_same_pixels, observed_log, persistence_fill, sensor_offsets
from chl_gapfill.provenance import provenance
from _compat import _load_model
from dineof_runner import run_dineof

WORK = C.OUTPUT_DIR / "dineof_work" / "chl_same_info"
NEV, MAX_NEV = 50, 200
N_BOOT = 1000
DESIGNS = {
    "obs": "the real sensor observations (offset to the target's level), real gaps, test holes removed",
    "target_at_obs": "the target's values on the pixels the sensors saw (same gaps), test holes removed",
    "clean_cube": "the gap-free target with only the test holes removed: leaky upper bound, not a comparison",
}


def design_cubes(truth, obs, holes):
    """{design: (T,H,W) DINEOF input}, NaN = gap to fill; every test hole removed."""
    cubes = {
        "obs": obs.copy(),
        "target_at_obs": np.where(np.isfinite(obs), truth, np.nan),
        "clean_cube": truth.copy(),
    }
    for cube in cubes.values():
        for t, hole in holes.items():
            cube[t][hole] = np.nan
    return {k: v.astype("float32") for k, v in cubes.items()}


def run_with_mode_search(cube, seamask, work):
    """DINEOF with nev = NEV; while its cross-validation picks the cap, again with twice the cap."""
    runs, nev = [], NEV
    while True:
        filled, info = run_dineof(cube, seamask, work / f"nev{nev}", var="chl",
                                  params=dict(nev=min(nev, cube.shape[0] - 7), neini=1, toliter=1e-2))
        runs.append({k: info[k] for k in ("nev", "n_modes", "cv_error", "cap_hit")})
        print(f"[chl-cmp]   nev={info['nev']}: {info['n_modes']} modes, CV error {info['cv_error']}")
        if not info["cap_hit"] or nev >= MAX_NEV:
            return filled, runs
        nev *= 2


def main(ckpt=None):
    ckpt = pathlib.Path(ckpt) if ckpt else C.MODEL_OUT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sensor_paths = available_sensor_paths()
    ds = GapFillDataset(sensor_paths, input_days=C.INPUT_DAYS,
                        log_transform=C.LOG_TRANSFORM, deterministic=True, seed=C.SEED)
    _, _, test = split_positions(ds.sample_dates)
    print(f"[chl-cmp] grid {ds.H}x{ds.W}, {len(ds.dates)} days, {len(test)} test days")

    truth = ds.target_to_log(ds.clean_norm).astype("float32")             # (T,H,W) log10 chl
    offsets = sensor_offsets(ds, cfg.split.train_end)
    obs = np.stack([observed_log(ds, t, offsets)[0] for t in range(len(ds.dates))]).astype("float32")

    positions = {}  # test position -> (day index, hole)
    for pos in test:
        hole = ds[int(pos)]["hole"][0].numpy() > 0.5
        if hole.any():
            positions[int(pos)] = (int(ds.sample_index[int(pos)]), hole)
    holes = dict(positions.values())
    cubes = design_cubes(truth, obs, holes)
    seamask = np.ones((ds.H, ds.W), dtype="float32")   # 20x20 ocean crop, all sea

    filled, designs = {}, {}
    for name, cube in cubes.items():
        print(f"[chl-cmp] DINEOF on the '{name}' cube ({np.isfinite(cube).mean():.1%} observed) ...")
        filled[name], runs = run_with_mode_search(cube, seamask, WORK / name)
        designs[name] = {"input": DESIGNS[name], "observed_frac": float(np.isfinite(cube).mean()), "runs": runs}

    # per test day, on the hole pixels: truth, the withheld observation and every method
    model = _load_model(ckpt, ds, device)
    days = {k: [] for k in ["target", "observation", "convlstm", "persistence", *(f"dineof_{d}" for d in cubes)]}
    with torch.no_grad():
        for pos, (t, hole) in positions.items():
            s = ds[pos]
            batch = {k: v.unsqueeze(0) for k, v in s.items()}
            days["convlstm"].append(ds.target_to_log(forward_batch(model, batch, device)[0, 0].cpu().numpy())[hole])
            days["persistence"].append(persistence_fill(ds, s, hole)[hole])
            days["target"].append(truth[t][hole])
            days["observation"].append(obs[t][hole])
            for d in cubes:
                days[f"dineof_{d}"].append(filled[d][t][hole])

    def score(names, truth_key, observed_only=False):
        per_day = {k: days[k] for k in names}
        tr = days[truth_key]
        if observed_only:  # keep the hole pixels some sensor saw that day
            tr = [np.where(np.isfinite(o), t, np.nan) for o, t in zip(days["observation"], tr)]
        return compare_same_pixels(per_day, tr, "convlstm", n_boot=N_BOOT, seed=C.SEED)

    dineofs = [f"dineof_{d}" for d in cubes]
    result = {
        "grid": [ds.H, ds.W], "n_days": len(ds.dates), "n_test_days": len(positions),
        "space": "log10(chl)",
        "target": "NOAA's DINEOF gap-filled VIIRS S-NPP+NOAA-20 chlorophyll "
                  "(ERDDAP nesdisVHNnoaaSNPPnoaa20chlaGapfilledDaily)",
        "bootstrap": {"n_boot": N_BOOT, "unit": "test day", "seed": C.SEED},
        "sensor_offsets_log10": offsets,
        "dineof": {"params": {"alpha": 0.0, "neini": 1, "toliter": 1e-2, "nev_start": NEV, "nev_max": MAX_NEV},
                   "designs": designs},
        "all_hole_px_vs_target": score(["convlstm", *dineofs], "target"),
        "persistence_px_vs_target": score(["convlstm", "persistence", "dineof_obs"], "target"),
        "observed_px_vs_target": score(["convlstm", *dineofs], "target", observed_only=True),
        "observed_px_vs_observation": score(["convlstm", *dineofs], "observation"),
        "provenance": provenance(cfg, ds, C.TRAIN_NC, sensor_paths, ckpt,
                                 [ds.sample_dates[p] for p in positions]),
    }

    C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(C.OUTPUT_DIR / "dineof_vs_ml_chl.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    for key in ["all_hole_px_vs_target", "persistence_px_vs_target", "observed_px_vs_target",
                "observed_px_vs_observation"]:
        r = result[key]
        print(f"\n[chl-cmp] {key} (n={r['n']}, {r['n_days']} days), log10 RMSE [95% CI], "
              "paired diff vs ConvLSTM:")
        for name, m in r["methods"].items():
            ci, d = m["rmse"], r["diff"].get(f"{name}_minus_convlstm")
            diff = (f"  diff {d['point']:+.3f} [{d['lo']:+.3f},{d['hi']:+.3f}]"
                    + ("" if d["resolved"] else " (unresolved)")) if d else ""
            print(f"  {name:22s} {ci['point']:.3f} [{ci['lo']:.3f},{ci['hi']:.3f}]  R2 {m['r2']:5.2f}{diff}")

    _figure(ds, truth, cubes["obs"], filled["obs"], days, positions, result)
    print(f"[chl-cmp] wrote {C.OUTPUT_DIR/'dineof_vs_ml_chl.json'} and dineof_vs_ml_chl.png")
    return result


def _figure(ds, truth, obs_in, dineof_obs, days, positions, result):
    i = max(range(len(positions)), key=lambda k: list(positions.values())[k][1].sum())
    t, hole = list(positions.values())[i]
    to_chl = (lambda a: np.power(10.0, a)) if ds.log_transform else (lambda a: a)

    def fill(values):
        out = np.full((ds.H, ds.W), np.nan)
        out[hole] = values
        return to_chl(out)

    r = result["all_hole_px_vs_target"]["methods"]
    panels = [
        (to_chl(obs_in[t]), f"DINEOF input ('obs'): sensor data\nwhite = cloud or test hole ({hole.mean():.0%} of grid)"),
        (to_chl(truth[t]), "target (NOAA DINEOF product)"),
        (fill(dineof_obs[t][hole]), f"DINEOF fill ('obs' input)\nlogRMSE={r['dineof_obs']['rmse']['point']:.3f}"),
        (fill(days["convlstm"][i]), f"ConvLSTM fill\nlogRMSE={r['convlstm']['rmse']['point']:.3f}"),
    ]
    fig, axs = plt.subplots(1, 4, figsize=(16, 4.2))
    for ax, (field, title) in zip(axs, panels):
        im = ax.imshow(field, cmap="jet", norm=LogNorm(vmin=0.03, vmax=10))
        ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=axs[3], fraction=0.046)
    fig.suptitle("Chlorophyll gap-filling with the same observations: DINEOF vs ConvLSTM "
                 "(scores: all test days)")
    fig.tight_layout(); fig.savefig(C.OUTPUT_DIR / "dineof_vs_ml_chl.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
