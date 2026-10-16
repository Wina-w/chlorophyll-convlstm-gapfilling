"""`chl-gapfill download` offline: a stub stands in for ERDDAP and serves synthetic NetCDF subsets."""

from __future__ import annotations

import re
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from chl_gapfill import data_sources, download
from chl_gapfill.align import load_sensor_aligned, load_target
from chl_gapfill.cli import main
from chl_gapfill.config import ExperimentConfig, Paths, reset_paths_cache
from chl_gapfill.synthetic import write_series

LAT = np.round(np.arange(25.0, 22.9, -1 / 12), 5)  # descending, like the VIIRS products
LON = np.round(np.arange(118.0, 120.0, 1 / 12), 5)
TARGET_ID = data_sources.SENSORS[data_sources.target_sensor()]["erddap_id"]
NOAA20_ID = data_sources.SENSORS["viirs_noaa20"]["erddap_id"]
SNPP_ID = data_sources.SENSORS["viirs_snpp"]["erddap_id"]


class _Resp:
    def __init__(self, *, json_obj=None, text="", content=b""):
        self._json, self.text, self._content = json_obj, text, content

    def json(self):
        return self._json

    def iter_content(self, chunk_size):
        yield self._content

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeErddap:
    """Answers the info / axis / griddap requests download.py makes and records every URL."""

    def __init__(self, tmp: Path, coverage=None, fail=()):
        self.tmp, self.coverage, self.fail, self.urls = tmp, coverage or {}, set(fail), []

    def __call__(self, url, stream=False):
        self.urls.append(url)
        ds_id = re.search(r"/(?:info|griddap)/([A-Za-z0-9]+)", url).group(1)
        meta = next(v for v in data_sources.SENSORS.values() if v["erddap_id"] == ds_id)
        dims = (
            ["time", "latitude", "longitude"]
            if meta["var"] == "chlorophyll"
            else ["time", "altitude", "latitude", "longitude"]
        )
        if "/info/" in url:
            t0, t1 = self.coverage.get(ds_id, ("2015-01-01T12:00:00Z", "2026-01-01T12:00:00Z"))
            rows = [["dimension", d, "", "double", ""] for d in dims]
            rows += [["attribute", "NC_GLOBAL", "time_coverage_start", "String", t0]]
            rows += [["attribute", "NC_GLOBAL", "time_coverage_end", "String", t1]]
            rows += [["variable", "quality_flag", "", "byte", ""], ["variable", meta["var"], "", "float", ""]]
            cols = ["Row Type", "Variable Name", "Attribute Name", "Data Type", "Value"]
            return _Resp(json_obj={"table": {"columnNames": cols, "rows": rows}})
        if ".csv0?" in url:
            return _Resp(text="\n".join(str(v) for v in (LAT if url.endswith("latitude") else LON)))
        assert url.split("?", 1)[1].startswith(meta["var"] + "["), (
            url
        )  # the registry variable, not the first one
        (t0, t1), (la0, la1), (lo0, lo1) = re.findall(r"\[\(([^)]*)\):1:\(([^)]*)\)\]", url)
        if (ds_id, t0[:4]) in self.fail:
            raise RuntimeError("GET failed after 4 tries (HTTP 502)")
        d0, d1 = date.fromisoformat(t0[:10]), date.fromisoformat(t1[:10])
        lat = LAT[(LAT >= min(float(la0), float(la1))) & (LAT <= max(float(la0), float(la1)))]
        lon = LON[(LON >= float(lo0)) & (LON <= float(lo1))]
        n = (d1 - d0).days + 1
        cube = np.random.default_rng(n).uniform(0.1, 2.0, (n, len(lat), len(lon))).astype("float32")
        path = self.tmp / f"{ds_id}_{t0[:10]}.nc"
        write_series(
            path, lat=lat, lon=lon, cube=cube, var=meta["var"], altitude="altitude" in dims, start=d0
        )
        return _Resp(content=path.read_bytes())

    def griddap(self, ds_id):
        return [u for u in self.urls if f"/griddap/{ds_id}.nc?" in u]


def _cfg(start="2021-12-30", end="2022-01-03"):
    return ExperimentConfig.model_validate(
        {
            "region": {"lat_bounds": [23.0, 24.0], "lon_bounds": [118.5, 119.5]},
            "data": {"start": start, "end": end},
        }
    )


def test_download_fetches_the_target_and_takes_the_period_from_the_config(tmp_path, monkeypatch):
    srv = FakeErddap(tmp_path, coverage={NOAA20_ID: ("2022-01-02T12:00:00Z", "2026-01-01T12:00:00Z")})
    monkeypatch.setattr(download, "_get", srv)
    cfg, p = _cfg(), Paths(root=tmp_path, data_dir=tmp_path / "data")
    assert download.download_all(cfg, p) == (5, 5)
    assert f"/info/{TARGET_ID}/" in srv.urls[0]  # the target first; a fresh clone has nothing else
    for key in data_sources.SENSORS:
        assert data_sources.local_path(p.data, key, cfg.data.train_nc).exists(), key
    assert not list(p.data.glob(".*part.nc"))
    # yearly chunks of data.start..data.end, clipped to what the dataset covers
    assert [re.search(r"\[\(([^)]*)\):1:\(([^)]*)\)\]", u).groups() for u in srv.griddap(TARGET_ID)] == [
        ("2021-12-30T00:00:00Z", "2021-12-31T23:59:59Z"),
        ("2022-01-01T00:00:00Z", "2022-01-03T23:59:59Z"),
    ]
    assert (
        len(srv.griddap(NOAA20_ID)) == 1
        and "(2022-01-02T12:00:00Z):1:(2022-01-03T23:59:59Z)" in srv.griddap(NOAA20_ID)[0]
    )
    # the files are what the pipeline reads: target on its region crop, sensors aligned onto it
    dates, lat, lon, clean = load_target(
        p.data / cfg.data.train_nc, var_name="chlor_a", lat_bounds=(23.0, 24.0), lon_bounds=(118.5, 119.5)
    )
    assert [tuple(d) for d in dates] == [
        ((date(2021, 12, 30) + timedelta(i)).timetuple()[:3]) for i in range(5)
    ]
    assert clean.shape == (5, 13, 13) and np.isfinite(clean).all()
    data, mask = load_sensor_aligned(p.data / "modis_aqua.nc", dates, lat, lon)
    assert mask.all() and np.isfinite(data).all()
    _, mask20 = load_sensor_aligned(p.data / "viirs_noaa20.nc", dates, lat, lon)
    assert mask20[:3].sum() == 0 and mask20[3:].all()  # NOAA-20 only from 2022-01-02
    n = len(srv.urls)
    assert download.download_all(cfg, p) == (5, 5) and len(srv.urls) == n  # present files are kept


def test_inputs_do_not_need_the_target_on_disk(tmp_path, monkeypatch):
    srv = FakeErddap(tmp_path)
    monkeypatch.setattr(download, "_get", srv)
    p = Paths(root=tmp_path, data_dir=tmp_path / "data")
    assert download.download_all(_cfg(), p, keys=["modis_aqua"]) == (1, 1)
    assert (p.data / "modis_aqua.nc").exists() and not (p.data / "Merged_VIIRS_CHL_Daily.nc").exists()


def test_a_failed_year_writes_no_partial_file(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "_get", FakeErddap(tmp_path, fail={(SNPP_ID, "2022")}))
    p = Paths(root=tmp_path, data_dir=tmp_path / "data")
    assert download.download_all(_cfg(), p) == (4, 5)
    assert not (p.data / "viirs_snpp.nc").exists()  # a file missing 2022 would look complete
    assert not list(p.data.glob(".*part.nc"))


def test_cli_explains_a_missing_target(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(download, "_get", FakeErddap(tmp_path, fail={(TARGET_ID, "2022")}))
    monkeypatch.setenv("CHL_DATA_DIR", str(tmp_path / "data"))
    reset_paths_cache()
    try:
        rc = main(["download", "--set", "data.start=2022-01-01", "data.end=2022-01-02"])
    finally:
        reset_paths_cache()
    assert rc == 1
    out, err = capsys.readouterr()
    assert "viirs_snpp_n20_gapfilled: SKIPPED" in out and "4/5 sensors available" in out
    assert "error: the target" in err and "data/README.md" in err and "make_synthetic_data.py" in err


def test_download_period_is_validated():
    with pytest.raises(ValueError, match="after data.end"):
        _cfg(start="2022-02-01", end="2022-01-01")
