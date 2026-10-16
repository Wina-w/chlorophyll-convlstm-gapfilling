"""Align multi-sensor chlorophyll products onto one common grid and date axis.

Every input sensor lives on its own grid (MODIS-Aqua ~4 km, OLCI-DINEOF 2 km,
VIIRS 4 km) and its own set of available days. To fuse them, we resample each
sensor by nearest-neighbour onto the **canonical grid** — the study-region crop
of the clean VIIRS target — and index every sensor to the target's daily axis,
inserting fully-masked days where a sensor has no observation.

Real cloud gaps are preserved (a cell is invalid if its nearest source cell is
missing), which is exactly what the gap-filling model must learn to fill.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from netCDF4 import Dataset as NetCDFDataset
from netCDF4 import num2date
from scipy.interpolate import RegularGridInterpolator

_CHL_VAR_CANDIDATES = ("chlor_a", "chlorophyll", "chl", "chl_a", "CHL")
_COORD_NAMES = ("latitude", "longitude", "lat", "lon", "time", "altitude")


def _read_coords(nc):
    lat_name = "latitude" if "latitude" in nc.variables else "lat"
    lon_name = "longitude" if "longitude" in nc.variables else "lon"
    return (
        lat_name,
        np.asarray(nc.variables[lat_name][:], dtype="float64"),
        lon_name,
        np.asarray(nc.variables[lon_name][:], dtype="float64"),
    )


def _find_chl_var(nc, coord_names=_COORD_NAMES):
    for name in _CHL_VAR_CANDIDATES:
        if name in nc.variables:
            return name
    for name, v in nc.variables.items():
        if name not in coord_names and v.ndim >= 2:
            return name
    raise KeyError("no chlorophyll-like variable found")


def _dates(nc):
    t = nc.variables["time"]
    d = num2date(t[:], units=t.units)
    return np.array([(x.year, x.month, x.day) for x in d])


def _clean(field):
    """Fill values / non-finite / non-positive -> NaN (a gap)."""
    f = np.asarray(field, dtype="float32")
    f[~np.isfinite(f)] = np.nan
    f[f <= 0] = np.nan
    return f


def _bounds_mask(coords, bounds):
    return (coords >= bounds[0]) & (coords <= bounds[1])


def load_target(
    path: str | Path,
    *,
    var_name: str = "chlor_a",
    lat_bounds: tuple[float, float],
    lon_bounds: tuple[float, float],
):
    """Load the clean target on its canonical (region-cropped) grid.

    Returns
    -------
    dates : (T,3) int array of (year, month, day)
    lat, lon : 1-D canonical coordinate vectors
    clean : (T, H, W) float32 clean chlorophyll (target; gap-free by construction)
    """
    with NetCDFDataset(str(path), "r") as nc:
        _, lat, _, lon = _read_coords(nc)
        var = var_name if var_name in nc.variables else _find_chl_var(nc)
        arr = np.asarray(nc.variables[var][:])  # (T, [alt,] H, W)
        dates = _dates(nc)
    arr = np.squeeze(arr)
    li = _bounds_mask(lat, lat_bounds)
    lj = _bounds_mask(lon, lon_bounds)
    if not li.any() or not lj.any():
        raise ValueError(f"region {lat_bounds} x {lon_bounds} selects no cells of {path}")
    clean = arr[:, li][:, :, lj].astype("float32")
    return dates, lat[li], lon[lj], clean


def _regrid_nearest(src_lat, src_lon, src_field, dst_lat, dst_lon):
    """Nearest-neighbour resample a single (H,W) field onto the destination grid.

    RegularGridInterpolator needs strictly-ascending axes, so descending
    latitude/longitude are flipped first. NaN gaps are carried through."""
    lat, lon, fld = src_lat, src_lon, src_field
    if lat[0] > lat[-1]:
        lat, fld = lat[::-1], fld[::-1, :]
    if lon[0] > lon[-1]:
        lon, fld = lon[::-1], fld[:, ::-1]
    rgi = RegularGridInterpolator((lat, lon), fld, method="nearest", bounds_error=False, fill_value=np.nan)
    glat, glon = np.meshgrid(dst_lat, dst_lon, indexing="ij")
    return rgi(np.stack([glat.ravel(), glon.ravel()], axis=-1)).reshape(glat.shape)


def load_sensor_aligned(path: str | Path, dates_c, lat_c, lon_c):
    """Align one sensor file to the canonical grid and daily axis.

    Returns
    -------
    data : (T, H, W) float32, NaN where the sensor is cloudy or has no day
    mask : (T, H, W) float32 in {0,1}; 1 = valid observation
    """
    T, H, W = len(dates_c), len(lat_c), len(lon_c)
    data = np.full((T, H, W), np.nan, dtype="float32")

    index = {tuple(d): i for i, d in enumerate(dates_c)}
    with NetCDFDataset(str(path), "r") as nc:
        _, s_lat, _, s_lon = _read_coords(nc)
        var = _find_chl_var(nc)
        s_dates = _dates(nc)
        raw = np.squeeze(np.asarray(nc.variables[var][:]))  # (Ts, H, W)
        if raw.ndim == 2:
            raw = raw[None]
        for ti, d in enumerate(s_dates):
            ci = index.get(tuple(d))
            if ci is None:
                continue
            fld = _clean(raw[ti])
            data[ci] = _regrid_nearest(s_lat, s_lon, fld, lat_c, lon_c)

    mask = np.isfinite(data).astype("float32")
    return data, mask


def coverage_report(paths_by_key: dict, target_path, *, var_name, lat_bounds, lon_bounds) -> dict:
    """Per-sensor gap statistics on the canonical grid (a sanity check)."""
    dates_c, lat_c, lon_c, _ = load_target(
        target_path, var_name=var_name, lat_bounds=lat_bounds, lon_bounds=lon_bounds
    )
    print(f"[align] canonical grid H={len(lat_c)} W={len(lon_c)}  days={len(dates_c)}")
    report = {}
    for key, path in paths_by_key.items():
        try:
            _, mask = load_sensor_aligned(path, dates_c, lat_c, lon_c)
            seen = mask.reshape(len(dates_c), -1).max(1) > 0
            report[key] = {
                "days_seen": int(seen.sum()),
                "days_total": int(len(dates_c)),
                "valid_frac": float(mask.mean()),
                "valid_on_seen": float(mask[seen].mean()) if seen.any() else 0.0,
            }
            r = report[key]
            print(
                f"[align] {key:26s} days={r['days_seen']:4d}/{r['days_total']}  "
                f"valid={r['valid_frac']:5.1%} (on observed days {r['valid_on_seen']:5.1%})"
            )
        except Exception as e:  # noqa: BLE001
            report[key] = {"error": f"{type(e).__name__}: {str(e)[:80]}"}
            print(f"[align] {key:26s} ERROR {report[key]['error']}")
    return report
