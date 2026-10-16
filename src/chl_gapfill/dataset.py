"""Datasets for chlorophyll-a modelling.

``GapFillDataset`` (the real objective)
    Multi-sensor cloud gap-filling. For a target day *t* it stacks the previous
    ``input_days`` days of every input sensor (each an observation channel + a
    validity-mask channel), hides a synthetic "cloud" region on day *t* across
    all sensors, and asks the model to reconstruct the clean field there. Ground
    truth is the (gap-free) VIIRS target; the loss/metrics are scored only on the
    hidden pixels. Real cloud gaps in the sensors are preserved and are exactly
    what fusion has to overcome.

``ChlorophyllDataset`` (legacy)
    The original next-day predictor, kept for backward comparison.

Both share the log10 + z-score normalisation (chlorophyll is log-normal) and an
``inverse()`` that returns predictions to chlorophyll units (mg m^-3).
Normalisation statistics are computed on the TRAIN period only and, for the
input sensors, per sensor (each instrument has its own bias).
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import numpy as np
import torch
from netCDF4 import Dataset as NetCDFDataset
from torch.utils.data import Dataset

from chl_gapfill import align

# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------


def _normalise(arr, mean, std, log_transform):
    """log10 (optional) then z-score; NaN gaps are preserved as NaN."""
    a = np.asarray(arr, dtype="float32").copy()
    a[a <= 0] = np.nan
    if log_transform:
        a = np.log10(a)
    return (a - mean) / std


def _stats_from_field(field, dates, log_transform, train_end: date):
    """Mean/std computed on the TRAIN period of a single field (target or sensor).

    Falls back to the whole record if the train period has no finite pixels for
    this field (a sensor that only starts after train_end).
    """
    end = datetime(train_end.year, train_end.month, train_end.day)
    keep = np.array([datetime(*map(int, d)) <= end for d in dates])
    ref = field[keep] if keep.any() else field
    a = np.asarray(ref, dtype="float32").copy()
    a[a <= 0] = np.nan
    if log_transform:
        a = np.log10(a)
    if not np.isfinite(a).any():
        a = np.asarray(field, dtype="float32").copy()
        a[a <= 0] = np.nan
        if log_transform:
            a = np.log10(a)
    mean = float(np.nanmean(a))
    std = float(np.nanstd(a))
    return mean, (std if std > 1e-6 else 1.0)


# ---------------------------------------------------------------------------
# Synthetic cloud masks
# ---------------------------------------------------------------------------
def make_cloud_mask(h, w, rng, min_frac=0.10, max_frac=0.40, n_blobs=3):
    """Random elliptical "cloud" hole mask (1 = hidden).

    Blobs are added until the hidden fraction reaches a target drawn from
    [min_frac, max_frac] (capped at 4*n_blobs attempts), so every sample poses a
    non-trivial but bounded reconstruction problem. Contiguous blobs — not
    scattered pixels — are what make the evaluation honest (see README).
    """
    yy, xx = np.mgrid[0:h, 0:w].astype("float32")
    target = rng.uniform(min_frac, max_frac)
    mask = np.zeros((h, w), dtype="float32")
    for _ in range(max(1, n_blobs) * 4):
        cy, cx = rng.uniform(0, h), rng.uniform(0, w)
        ay = rng.uniform(0.12, 0.4) * h
        ax = rng.uniform(0.12, 0.4) * w
        th = rng.uniform(0, np.pi)
        dy, dx = yy - cy, xx - cx
        u = dx * np.cos(th) + dy * np.sin(th)
        v = -dx * np.sin(th) + dy * np.cos(th)
        mask[(u / ax) ** 2 + (v / ay) ** 2 <= 1.0] = 1.0
        if mask.mean() >= target:
            break
    return mask


# ---------------------------------------------------------------------------
# Multi-sensor gap-filling dataset
# ---------------------------------------------------------------------------
class GapFillDataset(Dataset):
    """See module docstring. All configuration is explicit — no global state.

    Parameters
    ----------
    sensor_paths : {key: path} input sensors; empty only with ``self_supervised=True``
    target_nc    : the clean target file
    lat_bounds, lon_bounds : canonical region crop
    train_end    : last training date (normalisation statistics stop here)
    """

    def __init__(
        self,
        sensor_paths: dict[str, Path],
        *,
        target_nc: Path,
        lat_bounds: tuple[float, float],
        lon_bounds: tuple[float, float],
        train_end: date,
        var_name: str = "chlor_a",
        input_days: int = 3,
        log_transform: bool = True,
        stats=None,
        sensor_stats=None,
        deterministic: bool = False,
        min_frac: float = 0.10,
        max_frac: float = 0.40,
        n_blobs: int = 3,
        min_coverage: float = 0.2,
        seed: int = 42,
        self_supervised: bool = False,
        contiguous_windows: bool = True,
    ):
        if bool(sensor_paths) == self_supervised:
            raise ValueError(
                "no input sensors given; pass self_supervised=True to train on the clean target itself"
                if not sensor_paths
                else "self_supervised=True takes no input sensors"
            )
        self.input_days = input_days
        self.log_transform = log_transform
        self.deterministic = deterministic
        self.min_frac, self.max_frac, self.n_blobs = min_frac, max_frac, n_blobs
        self.seed = seed
        self.train_end = train_end
        # training holes (deterministic=False): one generator seeded once per dataset; train.py
        # checkpoints its state so --resume continues the same hole sequence
        self.hole_rng = np.random.default_rng(seed + 10_007)

        dates, lat, lon, clean = align.load_target(
            target_nc, var_name=var_name, lat_bounds=lat_bounds, lon_bounds=lon_bounds
        )
        self.dates, self.lat, self.lon = dates, lat, lon
        self.H, self.W = clean.shape[1], clean.shape[2]

        if stats is None:
            self.mean, self.std = _stats_from_field(clean, dates, log_transform, train_end)
        else:
            self.mean, self.std = stats
        # The "clean" target has a handful of residual fill / non-positive pixels
        # (~0.02%); make the ground truth strictly gap-free so masked losses on a
        # hole never see a NaN. Fill per-day, falling back to the global mean.
        cn = _normalise(clean, self.mean, self.std, log_transform)
        day_means = np.nanmean(cn.reshape(len(cn), -1), axis=1)
        day_means = np.where(np.isfinite(day_means), day_means, 0.0)
        cn = np.where(np.isnan(cn), day_means[:, None, None], cn)
        self.clean_norm = np.nan_to_num(cn, nan=0.0).astype("float32")

        self.sensor_keys = list(sensor_paths.keys())
        self.values, self.masks = [], []
        self.sensor_stats = {} if sensor_stats is None else dict(sensor_stats)
        for key in self.sensor_keys:
            data, mask = align.load_sensor_aligned(sensor_paths[key], dates, lat, lon)
            if key not in self.sensor_stats:
                self.sensor_stats[key] = _stats_from_field(data, dates, log_transform, train_end)
            s_mean, s_std = self.sensor_stats[key]
            vnorm = _normalise(data, s_mean, s_std, log_transform)
            self.values.append(np.nan_to_num(vnorm, nan=0.0).astype("float32"))
            self.masks.append(mask.astype("float32"))
        if self_supervised:  # explicit toy task: past days of the clean target are the input
            self.sensor_keys = ["_clean_selfsup"]
            self.sensor_stats = {"_clean_selfsup": (self.mean, self.std)}
            self.values = [np.nan_to_num(self.clean_norm, nan=0.0).astype("float32")]
            self.masks = [np.isfinite(self.clean_norm).astype("float32")]
        self.n_sensors = len(self.sensor_keys)

        # eligible target days: enough history AND enough same-day coverage to fill. Windows are
        # array slices, and the target axis has missing days (16 gaps in the real record), so a
        # window must also be `input_days` consecutive calendar days unless contiguous_windows=False
        # (the original behaviour, behind the published per_satellite.json)
        T = len(dates)
        union = np.stack(self.masks, 0).max(0)
        cover = union.reshape(T, -1).mean(1)
        day = np.array([date(*map(int, d)).toordinal() for d in dates])
        span_ok = np.zeros(T, bool)
        span_ok[input_days - 1 :] = day[input_days - 1 :] - day[: T - input_days + 1] == input_days - 1
        elig = [
            t
            for t in range(input_days - 1, T)
            if cover[t] >= min_coverage and (span_ok[t] or not contiguous_windows)
        ]
        self.sample_index = np.array(elig, dtype=int)
        self.sample_dates = dates[self.sample_index]

    # -- config the rest of the pipeline reads -----------------------------
    @property
    def stats(self):
        return (self.mean, self.std)

    @property
    def per_day_channels(self):
        return self.n_sensors * 2  # value + mask per sensor

    @property
    def flat_channels(self):
        return self.input_days * self.n_sensors

    def __len__(self):
        return len(self.sample_index)

    def inverse(self, norm):
        """De-normalise a MODEL/target-space field back to chlorophyll units."""
        out = np.asarray(norm) * self.std + self.mean
        return np.power(10.0, out) if self.log_transform else out

    def sensor_denorm_log(self, norm, si):
        """Undo one sensor's z-score -> log10 space (or physical if no log)."""
        s_mean, s_std = self.sensor_stats[self.sensor_keys[si]]
        return np.asarray(norm) * s_std + s_mean

    def target_to_log(self, norm):
        return np.asarray(norm) * self.std + self.mean

    @staticmethod
    def log_to_chl(log_or_phys, log_transform):
        return np.power(10.0, log_or_phys) if log_transform else np.asarray(log_or_phys)

    def position_of_date(self, d: date) -> int:
        """Sample position for a calendar date (raises KeyError if not eligible)."""
        key = (d.year, d.month, d.day)
        for i, sd in enumerate(self.sample_dates):
            if tuple(int(x) for x in sd) == key:
                return i
        from datetime import date as _date

        all_days = [_date(*map(int, sd)) for sd in self.sample_dates]
        nearest = sorted(all_days, key=lambda x: abs((x - d).days))[:3]
        raise KeyError(
            f"{d.isoformat()} is not an eligible target day (needs the {self.input_days - 1} days before it "
            f"and enough same-day coverage); nearest eligible: {[x.isoformat() for x in nearest]}"
        )

    def _rng(self, i):
        """Per-sample RNG for the synthetic hole.

        deterministic=True  -> seeded by the target-day index, so evaluation holes are
                               identical across runs and models (the same day always
                               poses the same problem).
        deterministic=False -> ``hole_rng``, one dataset-level generator seeded once, so a
                               training run is reproducible for a given seed (the original
                               code used an unseeded generator here and was not).
        """
        if self.deterministic:
            return np.random.default_rng(self.seed + int(self.sample_index[i]))
        return self.hole_rng

    def __getitem__(self, i):
        return self.sample(i)

    def sample(self, i, hole=None):
        """Build one sample; `hole=None` draws a synthetic cloud, an explicit (H,W) mask
        (e.g. all zeros for an operational fill of the real gaps only) is used as-is."""
        t = int(self.sample_index[i])
        days = list(range(t - self.input_days + 1, t + 1))
        if hole is None:
            hole = make_cloud_mask(self.H, self.W, self._rng(i), self.min_frac, self.max_frac, self.n_blobs)
        hole = np.asarray(hole, dtype="float32")
        keep_t = 1.0 - hole

        seq = np.zeros((self.input_days, self.per_day_channels, self.H, self.W), "float32")
        x_flat = np.zeros((self.flat_channels, self.H, self.W), "float32")
        m_flat = np.zeros((self.flat_channels, self.H, self.W), "float32")
        for k, d in enumerate(days):
            is_target_day = d == t
            for si in range(self.n_sensors):
                val = self.values[si][d]
                valid = self.masks[si][d]
                if is_target_day:
                    valid = valid * keep_t
                    val = val * valid
                seq[k, 2 * si] = val
                seq[k, 2 * si + 1] = valid
                x_flat[k * self.n_sensors + si] = val
                m_flat[k * self.n_sensors + si] = valid

        valid_day = np.stack([self.masks[si][t] * keep_t for si in range(self.n_sensors)], 0).max(0)
        target = self.clean_norm[t][None]
        mask_union = m_flat.max(0, keepdims=True)
        return {
            "seq": torch.from_numpy(seq),
            "x": torch.from_numpy(x_flat),
            "mask": torch.from_numpy(mask_union),
            "mask_flat": torch.from_numpy(m_flat),
            "target": torch.from_numpy(target),
            "hole": torch.from_numpy(hole[None]),
            "valid": torch.from_numpy(valid_day[None]),
        }


# ---------------------------------------------------------------------------
# Legacy next-day dataset (bounds-based crop)
# ---------------------------------------------------------------------------
class ChlorophyllDataset(Dataset):
    """Sample i: days i .. i+input_days-1 -> day i+input_days (array positions, see `dates`).

    With `train_end` the gap-fill value and the z-score statistics come from the days up to
    `train_end` only; without it from the whole record, as the original did (test period included).
    """

    def __init__(
        self,
        nc_file,
        var_name,
        input_days,
        log_transform=False,
        stats=None,
        *,
        lat_bounds,
        lon_bounds,
        train_end: date | None = None,
    ):
        with NetCDFDataset(str(nc_file), "r") as nc:
            lat = np.asarray(nc.variables.get("latitude", nc.variables.get("lat"))[:])
            lon = np.asarray(nc.variables.get("longitude", nc.variables.get("lon"))[:])
            arr = np.squeeze(np.asarray(nc.variables[var_name][:]))
            self.dates = align._dates(nc)
        li = (lat >= lat_bounds[0]) & (lat <= lat_bounds[1])
        lj = (lon >= lon_bounds[0]) & (lon <= lon_bounds[1])
        data = arr[:, li][:, :, lj].astype("float32")
        data[~np.isfinite(data)] = np.nan
        data[data <= 0] = np.nan
        if log_transform:
            data = np.log10(data)
        ref = np.ones(len(data), bool)
        if train_end is not None:
            ref = np.array([date(*map(int, d)) <= train_end for d in self.dates])
            if not ref.any():
                raise ValueError(f"no day of {nc_file} is on or before train_end={train_end}")
        gmean = np.nanmean(data[ref])
        data = np.where(np.isnan(data), gmean, data)
        self.log_transform = log_transform
        if stats is None:
            self.mean, self.std = float(np.mean(data[ref])), float(np.std(data[ref]))
        else:
            self.mean, self.std = stats
        self.data = (data - self.mean) / self.std
        self.input_days = input_days

    @property
    def stats(self):
        return (self.mean, self.std)

    def __len__(self):
        return len(self.data) - self.input_days

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.input_days]
        y = self.data[idx + self.input_days][None]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def inverse(self, norm):
        out = np.asarray(norm) * self.std + self.mean
        return np.power(10.0, out) if self.log_transform else out
