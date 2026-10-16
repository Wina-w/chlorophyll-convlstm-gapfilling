"""Download multi-sensor daily chlorophyll-a subsets from NOAA CoastWatch ERDDAP.

The target and every input/reference sensor in ``data_sources`` are fetched with a
single code path: query the dataset's metadata (dimension order, variable names, time
coverage), then request a griddap NetCDF subset cropped to the study region and the
configured period ``data.start..data.end`` (yearly chunks; federated datasets time
out on multi-year requests). Nothing needs the target on disk first. A sensor that
fails is reported and skipped so the others are still fetched, and
``chl-gapfill download`` then exits 1.
"""

from __future__ import annotations

import time

import numpy as np
import requests
import xarray as xr
from netCDF4 import Dataset as NetCDFDataset

from chl_gapfill import data_sources as ds
from chl_gapfill.config import ExperimentConfig, Paths

TIMEOUT = 120
RETRIES = 4
BACKOFF = 5
_SESSION = requests.Session()
# ERDDAP rejects the default python-requests user-agent with HTTP 403.
_SESSION.headers.update(
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    }
)


def _get(url, stream=False):
    last = None
    for attempt in range(1, RETRIES + 1):
        try:
            r = _SESSION.get(url, timeout=TIMEOUT, stream=stream)
            if r.status_code == 200:
                return r
            last = f"HTTP {r.status_code}"
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {str(e)[:80]}"
        if attempt < RETRIES:
            time.sleep(BACKOFF * attempt)
    raise RuntimeError(f"GET failed after {RETRIES} tries ({last}): {url[:120]}")


def _info(dataset_id):
    tab = _get(f"{ds.ERDDAP_BASE}/info/{dataset_id}/index.json").json()["table"]
    cols = tab["columnNames"]
    rt, vn, an, va = (cols.index(c) for c in ("Row Type", "Variable Name", "Attribute Name", "Value"))
    dims, variables, tstart, tend = [], [], None, None
    for row in tab["rows"]:
        if row[rt] == "dimension":
            dims.append(row[vn])
        elif row[rt] == "variable":
            variables.append(row[vn])
        if row[an] == "time_coverage_start":
            tstart = row[va]
        elif row[an] == "time_coverage_end":
            tend = row[va]
    return dims, [v for v in variables if v not in dims], tstart, tend


def _axis(dataset_id, name):
    text = _get(f"{ds.ERDDAP_BASE}/griddap/{dataset_id}.csv0?{name}").text
    if text.lstrip().startswith("<"):
        raise RuntimeError(f"axis {name} returned HTML (server error)")
    return [line.strip() for line in text.splitlines() if line.strip()]


def _clip_iso(requested, coverage, is_start):
    if coverage is None:
        return requested
    return max(requested, coverage) if is_start else min(requested, coverage)


def range_endpoints(values, lo, hi):
    """Endpoint strings selecting the in-range block, preserving the axis orientation."""
    arr = np.array([float(v) for v in values])
    sel = np.where((arr >= lo) & (arr <= hi))[0]
    if len(sel) == 0:
        k = int(np.argmin(np.abs(arr - (lo + hi) / 2)))
        return values[k], values[k]
    return values[sel[0]], values[sel[-1]]


def period(cfg: ExperimentConfig) -> tuple[str, str]:
    """The configured download period as ERDDAP ISO times (whole days, inclusive)."""
    return f"{cfg.data.start.isoformat()}T00:00:00Z", f"{cfg.data.end.isoformat()}T23:59:59Z"


def year_chunks(t0, t1):
    y0, y1 = int(t0[:4]), int(t1[:4])
    for y in range(y0, y1 + 1):
        yield (t0 if y == y0 else f"{y}-01-01T00:00:00Z"), (t1 if y == y1 else f"{y}-12-31T23:59:59Z")


def build_query(dims, data_var, t0, t1, la, lo):
    parts = []
    for d in dims:
        if d == "time":
            parts.append(f"[({t0}):1:({t1})]")
        elif d == "latitude":
            parts.append(f"[({la[0]}):1:({la[1]})]")
        elif d == "longitude":
            parts.append(f"[({lo[0]}):1:({lo[1]})]")
        else:
            parts.append("[0]")
    return f"{data_var}" + "".join(parts)


def _fetch_chunk(dataset_id, query, dest):
    with _get(f"{ds.ERDDAP_BASE}/griddap/{dataset_id}.nc?{query}", stream=True) as r, open(dest, "wb") as f:
        for block in r.iter_content(chunk_size=1 << 20):
            f.write(block)
    with NetCDFDataset(str(dest), "r") as nc:
        if len(nc.dimensions.get("time", [])) == 0:
            raise RuntimeError("empty time dimension")


def download_sensor(key: str, cfg: ExperimentConfig, p: Paths, force=False):
    meta = ds.SENSORS[key]
    out = ds.local_path(p.data, key, cfg.data.train_nc)
    if out.exists() and not force:
        print(f"[dl] {key}: {out.name} already present (use --force to refresh)")
        return out
    dataset_id = meta["erddap_id"]
    try:
        dims, data_vars, tstart, tend = _info(dataset_id)
        if not data_vars:
            raise RuntimeError(f"{dataset_id} lists no data variable")
        data_var = meta["var"] if meta["var"] in data_vars else data_vars[0]
        p0, p1 = period(cfg)
        t0, t1 = _clip_iso(p0, tstart, True), _clip_iso(p1, tend, False)
        if t0 > t1:
            raise RuntimeError(f"dataset covers {tstart}..{tend}, outside data.start..data.end")
        la = range_endpoints(_axis(dataset_id, "latitude"), *cfg.region.lat_bounds)
        lo = range_endpoints(_axis(dataset_id, "longitude"), *cfg.region.lon_bounds)
        print(f"[dl] {key}: {dataset_id}  {t0[:10]}..{t1[:10]}  var={data_var}")
        chunks, failed = [], []
        try:
            for cs, ce in year_chunks(t0, t1):
                chunks.append(out.with_name(f".{key}_{cs[:4]}.part.nc"))
                try:
                    _fetch_chunk(dataset_id, build_query(dims, data_var, cs, ce, la, lo), chunks[-1])
                    print(f"[dl]   {cs[:4]}: ok")
                except Exception as e:  # noqa: BLE001
                    failed.append(cs[:4])
                    print(f"[dl]   {cs[:4]}: FAILED ({str(e)[:60]})")
            if failed:
                # a file without those years would look complete to everything downstream
                raise RuntimeError(f"year(s) {', '.join(failed)} failed; nothing written, run download again")
            parts = [xr.open_dataset(f) for f in chunks]
            try:
                combined = (xr.concat(parts, dim="time") if len(parts) > 1 else parts[0]).sortby("time")
                combined.to_netcdf(chunks[0].with_name(f".{key}.part.nc"))
            finally:
                for q in parts:
                    q.close()
            # only a complete file gets the real name (an existing file counts as done next time)
            chunks[0].with_name(f".{key}.part.nc").replace(out)
        finally:
            for f in chunks:
                f.unlink(missing_ok=True)
        print(f"[dl] {key}: wrote {out.name} ({out.stat().st_size / 1e6:.1f} MB, {len(chunks)} yr chunks)")
        return out
    except Exception as e:  # noqa: BLE001
        print(f"[dl] {key}: SKIPPED ({type(e).__name__}: {str(e)[:160]})")
        return None


def download_all(cfg: ExperimentConfig, p: Paths, keys=None, force=False) -> tuple[int, int]:
    keys = keys or [ds.target_sensor(), *ds.input_sensors(), *ds.reference_sensors()]
    p.data.mkdir(parents=True, exist_ok=True)
    print(
        f"[dl] host={ds.ERDDAP_BASE}  region lat{cfg.region.lat_bounds} lon{cfg.region.lon_bounds}  "
        f"period {cfg.data.start}..{cfg.data.end}"
    )
    ok = 0
    for key in keys:
        if key not in ds.SENSORS:
            print(f"[dl] unknown sensor '{key}' (known: {list(ds.SENSORS)})")
            continue
        if download_sensor(key, cfg, p, force=force):
            ok += 1
    print(f"[dl] done: {ok}/{len(keys)} sensors available")
    return ok, len(keys)
