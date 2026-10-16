"""Inference for one target day: fill the hole and write a NetCDF (plus a figure).

    chl-gapfill predict -c configs/convlstm.yaml --ckpt models/convlstm_fusion/best.pt \
        --date 2022-06-12 --out outputs/filled_2022-06-12.nc [--png]

The synthetic hole is deterministic (seeded by the day index) so the same date
always yields the same evaluation problem; `--no-hole` fills only the REAL
cloud gaps of the target day instead (an operational fill).

``chlor_a_filled`` keeps every real observation the model saw (the mean of the sensors
that observed a pixel, `observed` = 1) and takes the model only where nothing did;
``chlor_a_model`` is the raw model output everywhere.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import torch
from netCDF4 import Dataset as NetCDFDataset

from chl_gapfill.config import ExperimentConfig, Paths, paths
from chl_gapfill.evaluate import day_t_observations, load_model_from, skill
from chl_gapfill.train import build_gapfill_datasets, forward_batch, split_positions


def _operational_sample(ds, pos: int):
    """Sample with NO synthetic hole: every real observation is kept."""
    return ds.sample(pos, hole=np.zeros((ds.H, ds.W), dtype="float32"))


def predict_day(
    cfg: ExperimentConfig,
    ckpt: Path,
    target_day: date | None,
    *,
    p: Paths | None = None,
    out: Path | None = None,
    png: bool = False,
    synthetic_hole: bool = True,
) -> dict:
    p = p or paths()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, ds, _ = build_gapfill_datasets(cfg, p)
    if target_day is None:
        _, _, test = split_positions(ds.sample_dates, cfg.split, cfg.input_days)
        pos = int(test[len(test) // 2]) if len(test) else len(ds) // 2
    else:
        pos = ds.position_of_date(target_day)
    model = load_model_from(ckpt, cfg, ds, device)
    s = ds[pos] if synthetic_hole else _operational_sample(ds, pos)
    with torch.no_grad():
        batch = {k: v.unsqueeze(0) for k, v in s.items()}
        pred = ds.inverse(forward_batch(model, batch, device)[0, 0].cpu().numpy())
    target = ds.inverse(s["target"][0].numpy())
    hole = s["hole"][0].numpy() > 0.5
    _, _, fused_log, observed = day_t_observations(ds, s)
    filled = np.where(observed, ds.log_to_chl(fused_log, ds.log_transform), pred)
    y, m, d = (int(v) for v in ds.sample_dates[pos])
    day = date(y, m, d)
    rmse = skill(pred[hole], target[hole])["rmse"] if hole.any() else float("nan")

    out = Path(out) if out else p.outputs_for(cfg) / f"filled_{day.isoformat()}.nc"
    out.parent.mkdir(parents=True, exist_ok=True)
    with NetCDFDataset(str(out), "w") as nc:
        nc.createDimension("lat", ds.H)
        nc.createDimension("lon", ds.W)
        lat = nc.createVariable("lat", "f8", ("lat",))
        lon = nc.createVariable("lon", "f8", ("lon",))
        lat[:], lon[:] = ds.lat, ds.lon
        lat.units, lon.units = "degrees_north", "degrees_east"
        v = nc.createVariable("chlor_a_filled", "f4", ("lat", "lon"), zlib=True)
        v[:] = filled.astype("float32")
        v.units = "mg m-3"
        v.long_name = "chlorophyll-a: observations where observed = 1, model elsewhere"
        mo = nc.createVariable("chlor_a_model", "f4", ("lat", "lon"), zlib=True)
        mo[:] = pred.astype("float32")
        mo.units, mo.long_name = "mg m-3", "chlorophyll-a, model output for every pixel"
        ob = nc.createVariable("observed", "i1", ("lat", "lon"))
        ob[:] = observed.astype("int8")
        ob.long_name = "1 = seen by at least one input sensor and shown to the model (mean of those sensors)"
        t = nc.createVariable("chlor_a_target", "f4", ("lat", "lon"), zlib=True)
        t[:] = target.astype("float32")
        t.long_name = "clean target (VIIRS gap-filled product)"
        h = nc.createVariable("hole", "i1", ("lat", "lon"))
        h[:] = hole.astype("int8")
        h.long_name = "1 = pixel hidden from the model (synthetic cloud) and scored"
        nc.date = day.isoformat()
        nc.model = cfg.model.name
        nc.checkpoint = Path(ckpt).name
        nc.config_hash = cfg.config_hash()
        nc.hole_rmse_chl = float(rmse)
        nc.sensors = ",".join(ds.sensor_keys)
        nc.status = "experimental"
    info = {
        "date": day.isoformat(),
        "position": pos,
        "hole_rmse": rmse,
        "hole_frac": float(hole.mean()),
        "observed_frac": float(observed.mean()),
        "out": str(out),
    }
    if png:
        from chl_gapfill.figures import predict_figure

        info["png"] = str(
            predict_figure(ds, s, pred, target, hole, cfg.model.name, rmse, out.with_suffix(".png"))
        )
    print(f"[predict] {day}  pos={pos}  hole={hole.mean():.0%}  hole RMSE={rmse:.3f}  -> {out}")
    return info
