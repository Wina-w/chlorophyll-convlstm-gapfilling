"""Synthetic multi-sensor NetCDF set (8x8 grid, 120 days): the test fixtures, smoke runs and CI.

Lives in the package (not in tests/) so `scripts/make_synthetic_data.py` also works where
the tests and pytest are not installed, e.g. inside the Docker image.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
from netCDF4 import Dataset as NetCDFDataset
from netCDF4 import date2num

H, W, T = 8, 8, 120
LAT = np.linspace(24.0, 23.0, H)  # descending like the VIIRS product
LON = np.linspace(118.5, 119.5, W)
START = date(2022, 1, 1)


def _field(t: int, rng: np.random.Generator) -> np.ndarray:
    """Log-normal-ish chlorophyll with a coastal gradient and a slow seasonal drift."""
    yy, xx = np.mgrid[0:H, 0:W]
    base = 0.3 + 0.6 * (xx / W) + 0.2 * np.sin(2 * np.pi * t / 60)
    return (base * np.exp(rng.normal(0, 0.15, (H, W)))).astype("float32")


def write_series(path, *, lat, lon, cube, var="chlor_a", altitude=False, start=START):
    with NetCDFDataset(str(path), "w") as nc:
        nc.createDimension("time", cube.shape[0])
        if altitude:
            nc.createDimension("altitude", 1)
        nc.createDimension("latitude", len(lat))
        nc.createDimension("longitude", len(lon))
        tv = nc.createVariable("time", "f8", ("time",))
        tv.units = "days since 1970-01-01 00:00:00"
        stamps = [
            datetime(start.year, start.month, start.day) + timedelta(days=i) for i in range(cube.shape[0])
        ]
        tv[:] = date2num(stamps, tv.units)
        nc.createVariable("latitude", "f8", ("latitude",))[:] = lat
        nc.createVariable("longitude", "f8", ("longitude",))[:] = lon
        dims = (
            ("time", "altitude", "latitude", "longitude") if altitude else ("time", "latitude", "longitude")
        )
        v = nc.createVariable(var, "f4", dims, fill_value=-9999.0)
        v[:] = cube[:, None] if altitude else cube


def make_synthetic(root: Path) -> np.ndarray:
    """Write the clean target and two gappy sensors into `root`; returns the clean (T, H, W) cube."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    clean = np.stack([_field(t, rng) for t in range(T)])
    write_series(root / "Merged_VIIRS_CHL_Daily.nc", lat=LAT, lon=LON, cube=clean, altitude=True)

    # sensor A: same grid, 40 % cloud blobs, missing every 7th day entirely
    a = clean * np.exp(rng.normal(0, 0.05, clean.shape)).astype("float32")
    for t in range(T):
        if t % 7 == 3:
            a[t] = np.nan
        else:
            cy, cx = rng.integers(0, H), rng.integers(0, W)
            yy, xx = np.mgrid[0:H, 0:W]
            a[t][(yy - cy) ** 2 + (xx - cx) ** 2 < rng.uniform(2, 12)] = np.nan
    write_series(root / "viirs_snpp.nc", lat=LAT, lon=LON, cube=a, altitude=True)

    # sensor B: coarser 5x5 grid (regridding path), biased x1.3, 30 % random gaps, no altitude dim, ascending lat
    lat_b = np.linspace(23.0, 24.0, 5)
    lon_b = np.linspace(118.5, 119.5, 5)
    b = np.zeros((T, 5, 5), "float32")
    for t in range(T):
        coarse = clean[t].reshape(4, 2, 4, 2).mean(axis=(1, 3))  # 4x4 block means
        full = np.pad(coarse, ((0, 1), (0, 1)), mode="edge")
        b[t] = full[::-1] * 1.3
        b[t][rng.random((5, 5)) < 0.3] = np.nan
    write_series(root / "modis_aqua.nc", lat=lat_b, lon=lon_b, cube=b, var="chlorophyll", altitude=False)
    return clean
