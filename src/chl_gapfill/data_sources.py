"""Sensor registry for multi-sensor chlorophyll-a cloud gap-filling.

Each entry describes a daily chlorophyll-a source over the Taiwan Strait study
region. All ERDDAP sources live on the same NOAA CoastWatch ERDDAP host that the
original VIIRS training file came from, so they share the CF-NetCDF conventions
and can be fetched by ``chl_gapfill.download`` with a single code path.

Roles
-----
``target``    Clean, (near) gap-free field used as ground truth for supervised
              inpainting: the VIIRS SNPP+NOAA-20 DINEOF gap-filled product, saved
              as ``DataConfig.train_nc``.
``input``     Single-sensor products *with real cloud gaps*, fed to the model.
              Each sensor is cloudy at different times, so fusing them fills gaps.
``reference`` Gap-filled products kept for later comparison, NOT model input.

My own Sentinel-3 products used by the DINEOF benchmarks are not
listed here: their locations are machine-specific and come from ``Paths``
(``S3_SST_BASELINE_DIR``, ``S3_SST_DINEOF_L4_DIR``, ``S3_CHL_MAT_DIR``).
"""

from __future__ import annotations

from pathlib import Path

ERDDAP_BASE = "https://coastwatch.pfeg.noaa.gov/erddap"

SENSORS = {
    "viirs_snpp_n20_gapfilled": dict(
        erddap_id="nesdisVHNnoaaSNPPnoaa20chlaGapfilledDaily",
        var="chlor_a",
        role="target",
        note="VIIRS SNPP+NOAA-20 DINEOF gap-filled, Science Quality, 9 km; saved as data.train_nc.",
    ),
    "modis_aqua": dict(
        erddap_id="erdMH1chla1day",
        var="chlorophyll",
        role="input",
        note="MODIS-Aqua L3SMI Science Quality 4km. Real cloud gaps. Coverage ends ~2022-07 (sensor ageing).",
    ),
    "viirs_snpp": dict(
        erddap_id="nesdisVHNSQchlaDaily",
        var="chlor_a",
        role="input",
        note="VIIRS S-NPP raw (non-gap-filled) Science Quality 4km. Real gaps.",
    ),
    "viirs_noaa20": dict(
        erddap_id="nesdisVHNnoaa20chlaDaily",
        var="chlor_a",
        role="input",
        note="VIIRS NOAA-20 raw (non-gap-filled). Real gaps; from 2021-08.",
    ),
    "olci_s3_dineof": dict(
        erddap_id="noaacwNPPN20S3ASCIDINEOF2kmDaily",
        var="chlor_a",
        role="reference",
        note="VIIRS + Copernicus Sentinel-3 OLCI DINEOF gap-filled, 2km. "
        "Reference only: not used by the model or the benchmarks.",
    ),
}


def sensors_by_role(role: str) -> list[str]:
    return [k for k, v in SENSORS.items() if v["role"] == role]


def input_sensors() -> list[str]:
    return sensors_by_role("input")


def reference_sensors() -> list[str]:
    return sensors_by_role("reference")


def target_sensor() -> str | None:
    keys = sensors_by_role("target")
    return keys[0] if keys else None


def local_path(data_dir: str | Path, key: str, target_file: str = "Merged_VIIRS_CHL_Daily.nc") -> Path:
    """Local NetCDF path a sensor is downloaded to (the target is saved as `target_file`)."""
    data_dir = Path(data_dir)
    if key == target_sensor():
        return data_dir / target_file
    return data_dir / f"{key}.nc"


def available_sensor_paths(
    data_dir: str | Path, keys: list[str] | None = None, *, fusion: bool = True
) -> dict[str, Path]:
    """Input sensor files for a run, in the order given (registry order for ``keys=None``).

    Explicitly requested sensors must all be on disk; ``keys=None`` uses every registry input
    that is. Never returns an empty dict: training on no sensors is the separate, explicit
    ``data.self_supervised`` mode, not a silent fallback for a typo or a missing download.
    """
    valid = input_sensors()
    if keys is not None and not keys:
        raise ValueError(f"empty input sensor list; valid: {valid}")
    if keys is not None and (unknown := [k for k in keys if k not in valid]):
        raise ValueError(f"unknown input sensor(s) {unknown}; valid: {valid}")
    wanted = valid if keys is None else list(keys)
    if not fusion:
        wanted = wanted[:1]
    found = {key: local_path(data_dir, key) for key in wanted}
    if keys is not None and (missing := [str(p) for p in found.values() if not p.exists()]):
        raise FileNotFoundError(
            f"input sensor file(s) not found: {missing}; fetch them with `chl-gapfill download`"
        )
    out = {key: p for key, p in found.items() if p.exists()}
    if not out:
        raise FileNotFoundError(
            f"no input sensor files in {data_dir} (looked for {[p.name for p in found.values()]}); fetch "
            "them with `chl-gapfill download`, or set data.self_supervised: true for the target-only toy task"
        )
    return out
