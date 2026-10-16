from __future__ import annotations

from datetime import date

import numpy as np
import pytest
from scipy import ndimage

from chl_gapfill.align import load_sensor_aligned, load_target
from chl_gapfill.cli import main
from chl_gapfill.data_sources import available_sensor_paths
from chl_gapfill.dataset import GapFillDataset, _stats_from_field, make_cloud_mask
from chl_gapfill.evaluate import observed_log, sensor_offsets
from chl_gapfill.synthetic import H, T, W
from chl_gapfill.train import build_gapfill_datasets


def _target_args(synthetic_data):
    return dict(
        path=synthetic_data["root"] / "Merged_VIIRS_CHL_Daily.nc",
        var_name="chlor_a",
        lat_bounds=(23.0, 24.0),
        lon_bounds=(118.5, 119.5),
    )


def test_load_target_crops_region_and_squeezes_altitude(synthetic_data):
    dates, lat, lon, clean = load_target(**_target_args(synthetic_data))
    assert clean.shape == (T, H, W) and lat.shape == (H,) and lon.shape == (W,)
    assert tuple(dates[0]) == (2022, 1, 1) and tuple(dates[-1]) == (2022, 4, 30)
    np.testing.assert_allclose(clean, synthetic_data["clean"], rtol=1e-6)
    # a tighter crop
    _, lat2, _, clean2 = load_target(**{**_target_args(synthetic_data), "lat_bounds": (23.4, 23.8)})
    assert clean2.shape[1] == len(lat2) < H
    with pytest.raises(ValueError):
        load_target(**{**_target_args(synthetic_data), "lat_bounds": (50.0, 51.0)})


def test_sensor_alignment_preserves_gaps_and_regrids(synthetic_data):
    dates, lat, lon, _ = load_target(**_target_args(synthetic_data))
    a, ma = load_sensor_aligned(synthetic_data["root"] / "viirs_snpp.nc", dates, lat, lon)
    assert a.shape == (T, H, W) and ma.shape == (T, H, W)
    assert ma[3].sum() == 0  # the fully missing day stays missing
    assert 0.5 < ma[0].mean() < 1.0  # cloud blob preserved
    b, mb = load_sensor_aligned(synthetic_data["root"] / "modis_aqua.nc", dates, lat, lon)
    assert b.shape == (T, H, W)  # 5x5 source regridded to the canonical 8x8
    assert 0.5 < mb.mean() < 0.9
    assert np.nanmean(b) > np.nanmean(a)  # the x1.3 bias survives (per-sensor stats must handle it)


def test_stats_use_train_period_only(synthetic_data):
    dates, _, _, clean = load_target(**_target_args(synthetic_data))
    m1, s1 = _stats_from_field(clean, dates, True, date(2022, 2, 1))
    polluted = clean.copy()
    polluted[40:] *= 100.0  # after train_end
    m2, s2 = _stats_from_field(polluted, dates, True, date(2022, 2, 1))
    assert m1 == m2 and s1 == s2
    # a sensor absent in the train period falls back to the whole record
    absent = clean.copy()
    absent[:40] = np.nan
    m3, _ = _stats_from_field(absent, dates, True, date(2022, 2, 1))
    assert np.isfinite(m3)


def test_cloud_mask_fraction_and_contiguity():
    rng = np.random.default_rng(1)
    for _ in range(20):
        m = make_cloud_mask(20, 20, rng, 0.10, 0.40, 3)
        assert 0.10 <= m.mean() <= 0.6  # target reached (may overshoot on the last blob)
        n_components = ndimage.label(m)[1]
        assert 1 <= n_components <= 3 * 4  # contiguous blobs, never scattered pixels


def _dataset(synthetic_data, **kw):
    root = synthetic_data["root"]
    return GapFillDataset(
        {"modis_aqua": root / "modis_aqua.nc", "viirs_snpp": root / "viirs_snpp.nc"},
        target_nc=root / "Merged_VIIRS_CHL_Daily.nc",
        lat_bounds=(23.0, 24.0),
        lon_bounds=(118.5, 119.5),
        train_end=date(2022, 3, 21),
        input_days=3,
        seed=42,
        **kw,
    )


def test_gapfill_dataset_sample_contract(synthetic_data):
    ds = _dataset(synthetic_data, deterministic=True)
    assert ds.n_sensors == 2 and ds.per_day_channels == 4 and ds.flat_channels == 6
    assert len(ds) > 100 and ds.sample_index.min() >= 2  # needs input_days-1 history
    s = ds[10]
    assert s["seq"].shape == (3, 4, H, W) and s["x"].shape == (6, H, W)
    for k in ("mask", "target", "hole", "valid"):
        assert s[k].shape == (1, H, W)
    hole = s["hole"][0].numpy()
    assert 0.1 <= hole.mean() <= 0.6
    # the hole is hidden from EVERY sensor on the target day
    last = s["seq"][-1].numpy()
    assert (last[1][hole > 0.5] == 0).all() and (last[3][hole > 0.5] == 0).all()
    # earlier days keep their real coverage
    assert s["seq"][0, 1].numpy().sum() > 0
    assert np.isfinite(s["target"].numpy()).all()


def test_deterministic_masks_repeat_and_random_ones_do_not(synthetic_data):
    det = _dataset(synthetic_data, deterministic=True)
    a, b = det[5]["hole"].numpy(), det[5]["hole"].numpy()
    assert np.array_equal(a, b)
    rnd = _dataset(synthetic_data, deterministic=False)
    assert not np.array_equal(rnd[5]["hole"].numpy(), rnd[5]["hole"].numpy())


def test_per_sensor_stats_and_inverse(synthetic_data):
    ds = _dataset(synthetic_data, deterministic=True)
    m_modis, _ = ds.sensor_stats["modis_aqua"]
    m_viirs, _ = ds.sensor_stats["viirs_snpp"]
    assert m_modis > m_viirs  # the x1.3 bias shows in the per-sensor mean, not in the target stats
    x = np.array([0.0, 1.0])
    assert np.allclose(np.log10(ds.inverse(x)), ds.target_to_log(x))
    assert ds.position_of_date(date(2022, 4, 15)) == int(
        np.where((ds.sample_dates == (2022, 4, 15)).all(1))[0][0]
    )
    with pytest.raises(KeyError):
        ds.position_of_date(date(2030, 1, 1))


def test_sensor_offsets_and_observed_field(synthetic_data):
    ds = _dataset(synthetic_data, deterministic=True)
    off = sensor_offsets(ds, date(2022, 3, 21))
    assert abs(off["viirs_snpp"]) < 0.01  # an unbiased sensor is not shifted
    assert off["modis_aqua"] < -0.08  # the x1.3 bias (log10 1.3 = 0.11) is taken out
    t = int(ds.sample_index[5])
    per = np.stack(
        [
            np.where(ds.masks[si][t] > 0.5, ds.sensor_denorm_log(ds.values[si][t], si) + off[key], np.nan)
            for si, key in enumerate(ds.sensor_keys)
        ]
    )
    count = np.isfinite(per).sum(0)
    field, seen = observed_log(ds, t, off)
    np.testing.assert_allclose(field, np.where(count > 0, np.nansum(per, 0) / np.maximum(count, 1), np.nan))
    assert np.array_equal(seen, count > 0) and np.array_equal(seen, np.isfinite(field))
    modis, modis_seen = observed_log(ds, t, off, keys=["modis_aqua"])
    assert np.array_equal(modis_seen, ds.masks[0][t] > 0.5)
    np.testing.assert_allclose(modis[modis_seen], per[0][modis_seen])


def test_min_coverage_filters_days(synthetic_data):
    loose = _dataset(synthetic_data, min_coverage=0.0)
    strict = _dataset(synthetic_data, min_coverage=0.95)
    assert len(strict) < len(loose)


def test_sensor_selection_never_falls_back_silently(synthetic_data, tmp_path):
    root = synthetic_data["root"]  # holds modis_aqua.nc and viirs_snpp.nc, not viirs_noaa20.nc
    assert list(available_sensor_paths(root)) == ["modis_aqua", "viirs_snpp"]  # registry order
    assert list(available_sensor_paths(root, ["viirs_snpp", "modis_aqua"])) == ["viirs_snpp", "modis_aqua"]
    assert list(available_sensor_paths(root, ["viirs_snpp", "modis_aqua"], fusion=False)) == ["viirs_snpp"]
    with pytest.raises(ValueError, match="unknown input sensor"):
        available_sensor_paths(root, ["modis_aqau"])  # a typo used to be dropped -> self-supervised
    with pytest.raises(ValueError, match="unknown input sensor"):
        available_sensor_paths(root, ["viirs_snpp_n20_gapfilled"])  # the target is never an input
    with pytest.raises(ValueError, match="empty"):
        available_sensor_paths(root, [])
    with pytest.raises(FileNotFoundError, match="viirs_noaa20"):
        available_sensor_paths(root, ["modis_aqua", "viirs_noaa20"])  # requested but not downloaded
    with pytest.raises(FileNotFoundError, match="self_supervised"):
        available_sensor_paths(tmp_path)  # nothing on disk at all
    with pytest.raises(ValueError, match="self_supervised=True"):
        _dataset_from(synthetic_data, {})


def _dataset_from(synthetic_data, sensor_paths, **kw):
    root = synthetic_data["root"]
    return GapFillDataset(
        sensor_paths,
        target_nc=root / "Merged_VIIRS_CHL_Daily.nc",
        lat_bounds=(23.0, 24.0),
        lon_bounds=(118.5, 119.5),
        train_end=date(2022, 3, 21),
        **kw,
    )


def test_self_supervised_mode_is_explicit(cfg, paths_for, capsys):
    ss = cfg.model_copy(
        update={"data": cfg.data.model_copy(update={"input_sensors": None, "self_supervised": True})}
    )
    _, ds, sensor_paths = build_gapfill_datasets(ss, paths_for)
    assert sensor_paths == {} and ds.sensor_keys == ["_clean_selfsup"] and ds.n_sensors == 1
    assert "self_supervised" in capsys.readouterr().out
    both = ss.model_dump(mode="json")
    both["data"]["input_sensors"] = ["modis_aqua"]
    with pytest.raises(ValueError, match="no input sensors"):
        type(cfg).model_validate(both)  # the two input modes are exclusive


def test_cli_fails_clearly_on_missing_sensor_or_target(cfg, paths_for, tmp_path, monkeypatch, capsys):
    import yaml

    from chl_gapfill.config import reset_paths_cache

    cfg_path = tmp_path / "cfg.yaml"
    bad = cfg.model_dump(mode="json")
    bad["data"]["input_sensors"] = ["modis_aqua", "viirs_noaa20"]  # viirs_noaa20 not in the synthetic set
    cfg_path.write_text(yaml.safe_dump(bad), encoding="utf-8")
    assert main(["train", "-c", str(cfg_path), "--no-mlflow"]) == 1
    assert "viirs_noaa20.nc" in capsys.readouterr().err
    monkeypatch.setenv("CHL_DATA_DIR", str(tmp_path / "empty"))  # a fresh clone: no target file
    reset_paths_cache()
    assert main(["train", "-c", str(cfg_path), "--no-mlflow"]) == 1
    err = capsys.readouterr().err
    assert "Merged_VIIRS_CHL_Daily.nc not found" in err and "chl-gapfill download" in err


def test_input_windows_never_span_missing_days(synthetic_data, tmp_path):
    import shutil

    from netCDF4 import Dataset as NetCDFDataset

    root = tmp_path / "gapped"
    shutil.copytree(synthetic_data["root"], root)
    with NetCDFDataset(str(root / "Merged_VIIRS_CHL_Daily.nc"), "a") as nc:
        nc.variables["time"][50:] = nc.variables["time"][50:] + 2  # two missing days before index 50
    sensors = {"modis_aqua": root / "modis_aqua.nc", "viirs_snpp": root / "viirs_snpp.nc"}
    kw = dict(
        target_nc=root / "Merged_VIIRS_CHL_Daily.nc",
        lat_bounds=(23.0, 24.0),
        lon_bounds=(118.5, 119.5),
        train_end=date(2022, 3, 21),
        input_days=3,
        min_coverage=0.0,
    )
    calendar = GapFillDataset(sensors, **kw)
    by_index = GapFillDataset(sensors, contiguous_windows=False, **kw)  # the original behaviour
    # the windows of indices 50 and 51 would span 5 and 4 calendar days instead of 3
    assert set(by_index.sample_index) - set(calendar.sample_index) == {50, 51}
    assert set(calendar.sample_index) <= set(by_index.sample_index)


def test_nextday_uses_train_period_stats_and_target_dates(synthetic_data, tmp_path, cfg):
    import shutil

    from netCDF4 import Dataset as NetCDFDataset

    from chl_gapfill.dataset import ChlorophyllDataset
    from chl_gapfill.train import nextday_positions

    root = tmp_path / "polluted"
    shutil.copytree(synthetic_data["root"], root)
    with NetCDFDataset(str(root / "Merged_VIIRS_CHL_Daily.nc"), "a") as nc:
        nc.variables["chlor_a"][40:] = nc.variables["chlor_a"][40:] * 100.0  # after train_end
    kw = dict(lat_bounds=(23.0, 24.0), lon_bounds=(118.5, 119.5), train_end=date(2022, 2, 1))
    clean = ChlorophyllDataset(synthetic_data["root"] / "Merged_VIIRS_CHL_Daily.nc", "chlor_a", 3, True, **kw)
    polluted = ChlorophyllDataset(root / "Merged_VIIRS_CHL_Daily.nc", "chlor_a", 3, True, **kw)
    assert clean.stats == polluted.stats  # later days cannot move the normalisation
    whole = dict(kw, train_end=None)  # the original: statistics over the whole record, test period included
    assert (
        ChlorophyllDataset(root / "Merged_VIIRS_CHL_Daily.nc", "chlor_a", 3, True, **whole).stats
        != ChlorophyllDataset(
            synthetic_data["root"] / "Merged_VIIRS_CHL_Daily.nc", "chlor_a", 3, True, **whole
        ).stats
    )
    tr, va = nextday_positions(clean.dates, len(clean), cfg)  # train_end 03-21, val 03-22..04-10

    def target(i):
        return tuple(int(v) for v in clean.dates[i + 3])

    assert (
        target(tr[-1]) == (2022, 3, 21) and target(va[0]) == (2022, 3, 22) and target(va[-1]) == (2022, 4, 10)
    )


def test_nextday_task_trains(cfg, paths_for):
    from chl_gapfill.train import fit_nextday

    assert fit_nextday(cfg.model_copy(update={"task": "nextday"}), paths_for).exists()
