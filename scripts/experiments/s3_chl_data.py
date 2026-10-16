"""Load my Sentinel-3 OLCI neural-network chlorophyll for a
LARGER-GRID DINEOF-vs-ML comparison (ERDDAP is unavailable; this is local data).

The in-repo VIIRS chlorophyll is a small 20x20 Taiwan-Strait crop. To compare
DINEOF and ML on a bigger field we use my S3 OLCI CHL_NN daily
products (directory given by ``S3_CHL_MAT_DIR``), a 2001x2001 grid over
15-35N / 115-135E, 30 days (Apr 2025).

Data-integrity note: the exported ``.nc`` files are empty (all fill value) and the
``.mat`` coordinate vectors are degenerate, so we read the real chlorophyll field
from ``.mat`` group ``enhanced_nn_data/grid_data/daily_mean_chl_nn`` (HDF5 v7.3 via
h5py) and reconstruct the grid linearly from the file's geospatial extent.
"""
import glob
import pathlib

import numpy as np
import h5py

from _compat import P

MAT_DIR = P.s3_chl_mat_dir  # .env: S3_CHL_MAT_DIR
FIELD = "enhanced_nn_data/grid_data/daily_mean_chl_nn"
LAT_RANGE = (15.0, 35.0)      # from file geospatial_lat_min/max
LON_RANGE = (115.0, 135.0)
DOWNSAMPLE = 10               # 2001 -> 200 (~0.1 deg)


def _date_from_name(p):
    s = pathlib.Path(p).stem.split("_")[-1]
    return (int(s[:4]), int(s[4:6]), int(s[6:8]))


def _block_mean_valid(a, f):
    """Block-mean downsample, treating <=0 / >=1e30 / non-finite as missing."""
    a = np.where(np.isfinite(a) & (a > 0) & (a < 1e30), a, np.nan)
    H, W = a.shape
    H2, W2 = H // f, W // f
    a = a[:H2 * f, :W2 * f].reshape(H2, f, W2, f)
    with np.errstate(invalid="ignore"):
        return np.nanmean(a, axis=(1, 3)).astype("float32")


def load_s3_chl_cube(downsample=None):
    """Return (dates, lat, lon, cube) — cube (T,H,W) chlorophyll mg/m^3, NaN gaps."""
    f = downsample or DOWNSAMPLE
    files = sorted(glob.glob(str(MAT_DIR / "*.mat")))
    if not files:
        raise FileNotFoundError(f"no CHL .mat files in {MAT_DIR}")
    dates, frames = [], []
    for fp in files:
        with h5py.File(fp, "r") as h:
            a = np.asarray(h[FIELD]).astype("float32")
        frames.append(_block_mean_valid(a, f))
        dates.append(_date_from_name(fp))
    cube = np.stack(frames, 0)
    H, W = cube.shape[1:]
    lat = np.linspace(LAT_RANGE[1], LAT_RANGE[0], H).astype("float64")   # north-up
    lon = np.linspace(LON_RANGE[0], LON_RANGE[1], W).astype("float64")
    return np.array(dates), lat, lon, cube


def sea_mask(cube):
    return np.isfinite(cube).any(0).astype("float32")


if __name__ == "__main__":
    dates, lat, lon, cube = load_s3_chl_cube()
    m = sea_mask(cube)
    print(f"[s3chl] cube {cube.shape} days={len(dates)} "
          f"lat {lat.min():.1f}..{lat.max():.1f} lon {lon.min():.1f}..{lon.max():.1f}")
    print(f"[s3chl] per-day valid={np.isfinite(cube).mean():.1%} sea={m.mean():.1%} "
          f"chl {np.nanmin(cube):.3f}..{np.nanmax(cube):.3f} mg/m3")
