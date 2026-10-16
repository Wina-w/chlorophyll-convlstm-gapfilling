"""Load my Sentinel-3 SST for the DINEOF-vs-ML comparison.

Reads the daily SLSTR SST NetCDFs my processing pipeline produced:

  * baseline : ``Enhanced_Production_Output_v12_0_Baseline_QC`` — strict-QC gappy
               L3C observations (real clouds; DINEOF's input).
  * dineof   : ``Enhanced_Production_Output_v12_1_DINEOF_L4`` — my DINEOF
               gap-filled L4 (reference / illustration only).

The native grid is 5065x4453 (~1 km, 105-145E / 5-45N). For a CPU-feasible,
common ML-and-DINEOF grid we crop to a region and block-mean downsample — the
ML models and DINEOF then run on the SAME grid, pixel-for-pixel (the ML grid
matches DINEOF's).
"""
import glob
import pathlib

import numpy as np
from netCDF4 import Dataset as NetCDFDataset

from _compat import P

BASE_DIR = P.s3_sst_baseline_dir  # .env: S3_SST_BASELINE_DIR
L4_DIR = P.s3_sst_dineof_l4_dir   # .env: S3_SST_DINEOF_L4_DIR
VAR = "analysed_sst"

# common comparison grid (crop + downsample factor)
REGION = dict(lat=(21.0, 27.0), lon=(118.0, 124.0))   # Taiwan / East China Sea
DOWNSAMPLE = 4                                          # ~1km -> ~4km


def _date_from_name(p):
    # ..._YYYYMMDD.nc
    s = pathlib.Path(p).stem.split("_")[-1]
    return (int(s[:4]), int(s[4:6]), int(s[6:8]))


def _block_mean(a, f):
    H, W = a.shape
    H2, W2 = H // f, W // f
    a = a[:H2 * f, :W2 * f].reshape(H2, f, W2, f)
    with np.errstate(invalid="ignore"):
        return np.nanmean(a, axis=(1, 3))


def load_sst_cube(which="baseline", region=None, downsample=None):
    """Return (dates, lat, lon, cube) for the cropped+downsampled SST series.

    cube : (T,H,W) float32 in degrees C; NaN where cloudy / land / no data.
    """
    region = region or REGION
    f = downsample or DOWNSAMPLE
    d = BASE_DIR if which == "baseline" else L4_DIR
    files = sorted(glob.glob(str(d / "*.nc")))
    if not files:
        raise FileNotFoundError(f"no SST NetCDFs in {d}")

    with NetCDFDataset(files[0], "r") as nc:
        la = np.asarray(nc.variables["latitude"][:], dtype="float64")
        lo = np.asarray(nc.variables["longitude"][:], dtype="float64")
    li = (la >= region["lat"][0]) & (la <= region["lat"][1])
    lj = (lo >= region["lon"][0]) & (lo <= region["lon"][1])
    lat_c = _block_mean(la[li][:, None], f).ravel() if False else la[li][:(li.sum() // f) * f].reshape(-1, f).mean(1)
    lon_c = lo[lj][:(lj.sum() // f) * f].reshape(-1, f).mean(1)

    dates, frames = [], []
    for fp in files:
        with NetCDFDataset(fp, "r") as nc:
            a = np.asarray(nc.variables[VAR][:], dtype="float32")
        a = np.squeeze(a)[np.ix_(li, lj)]
        a[~np.isfinite(a)] = np.nan
        frames.append(_block_mean(a, f).astype("float32"))
        dates.append(_date_from_name(fp))
    cube = np.stack(frames, 0)
    return np.array(dates), lat_c, lon_c, cube


def sea_mask(cube):
    """1 where a pixel is ever observed (the fillable sea domain), else 0."""
    return np.isfinite(cube).any(0).astype("float32")


if __name__ == "__main__":
    dates, lat, lon, cube = load_sst_cube("baseline")
    m = sea_mask(cube)
    print(f"[sst] baseline cube {cube.shape} days={len(dates)} "
          f"lat {lat.min():.2f}..{lat.max():.2f} lon {lon.min():.2f}..{lon.max():.2f}")
    print(f"[sst] per-day finite={np.isfinite(cube).mean():.1%}  sea domain={m.mean():.1%} "
          f"({int(m.sum())} px)  SST {np.nanmin(cube):.1f}..{np.nanmax(cube):.1f} degC")
