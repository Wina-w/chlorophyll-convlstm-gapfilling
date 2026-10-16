"""Evaluate the gap-filling model on held-out synthetic cloud holes and benchmark it
against classical baselines and a single-vs-multi-sensor fusion ablation.

Scoring is done **only on the hidden hole pixels** in real chlorophyll units and,
as the primary metric, in log10 space. Every metric carries a **day bootstrap
95% CI** (resampling whole test days; ``--block-days`` > 1 resamples runs of
consecutive days, for the day-to-day correlation of the errors).

Baselines (same holes, all in log10 space): persistence, interp_single,
interp_fusion, model_ablate_single (fusion model input-ablated at inference —
NOT a genuine single-sensor model; see scripts/experiments/train_single.py).

Each method's own table (``log10_space``) keeps every pixel it can fill, so the
pixel sets differ: persistence has no value where nothing was seen in the window.
The comparison is therefore made on ``log10_space_matched`` (the hole pixels every
method fills), and a verdict is a **paired** day bootstrap of the RMSE difference
model - baseline on those pixels (resolved when its 95% CI excludes 0); whether
two marginal CIs overlap does not test a difference.

The target is NOAA's DINEOF gap-filled VIIRS product, so on a hole pixel no sensor
saw that day the "truth" is itself a DINEOF fill. Two slices check this:
``observed`` = hole pixels some input sensor observed that day; ``independence`` =
hole pixels MODIS-Aqua (not part of the VIIRS product) observed. Each slice compares
model and persistence on the same pixels (paired) twice: against the target, and
against the withheld same-day observation itself (each sensor shifted to the
target's level by its mean training-period offset), a truth NOAA's product is not in.

    chl-gapfill evaluate -c configs/convlstm.yaml --ckpt models/convlstm_fusion/best.pt
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import griddata

from chl_gapfill.config import ExperimentConfig, Paths, paths
from chl_gapfill.provenance import provenance
from chl_gapfill.train import (
    build_gapfill_datasets,
    build_model_for,
    check_split_windows,
    forward_batch,
    load_weights,
    split_positions,
)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def skill(pred, true) -> dict:
    pred, true = np.asarray(pred, dtype=np.float64).ravel(), np.asarray(true, dtype=np.float64).ravel()
    if pred.size != true.size:
        raise ValueError(f"size mismatch: {pred.size} vs {true.size}")
    m = np.isfinite(pred) & np.isfinite(true)
    pred, true = pred[m], true[m]
    if pred.size == 0:
        return {"rmse": float("nan"), "bias": float("nan"), "r2": float("nan"), "n": 0}
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    bias = float(np.mean(pred - true))
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"rmse": rmse, "bias": bias, "r2": r2, "n": int(pred.size)}


def resample_days(rng, n_days: int, block_days: int = 1) -> np.ndarray:
    """Day indices for one bootstrap replicate: whole days with replacement, or with
    `block_days` > 1 a moving-block bootstrap (runs of consecutive test days, cut to n_days).
    block_days=1 draws exactly what the original day bootstrap drew."""
    L = min(max(1, int(block_days)), n_days)
    starts = rng.integers(0, n_days - L + 1, -(-n_days // L))
    return (starts[:, None] + np.arange(L)).ravel()[:n_days]


def bootstrap_ci(per_day_pred, per_day_true, n_boot=1000, seed=0, block_days=1) -> dict:
    """Bootstrap 95% CI by resampling whole test DAYS with replacement.

    Days, not pixels: within-day spatial correlation is strong, so a pixel
    bootstrap would report intervals far tighter than the evidence supports.
    Fewer than two days -> the point estimate is its own interval (`degenerate`).
    """
    if len(per_day_pred) != len(per_day_true):
        raise ValueError("day count mismatch")
    if not per_day_pred:
        raise ValueError("no days supplied")
    rng = np.random.default_rng(seed)
    D = len(per_day_pred)
    point = skill(np.concatenate(per_day_pred), np.concatenate(per_day_true))
    keys = ("rmse", "r2", "bias")
    if D < 2:
        return {
            **{k: {"point": point[k], "lo": point[k], "hi": point[k]} for k in keys},
            "n": point["n"],
            "n_days": D,
            "degenerate": True,
        }
    boot = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = resample_days(rng, D, block_days)
        s = skill(
            np.concatenate([per_day_pred[i] for i in idx]), np.concatenate([per_day_true[i] for i in idx])
        )
        for k in keys:
            boot[k].append(s[k])
    out = {}
    for k in keys:
        arr = np.array(boot[k], dtype="float64")
        arr = arr[np.isfinite(arr)]
        lo, hi = (
            (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
            if arr.size
            else (float("nan"), float("nan"))
        )
        out[k] = {"point": point[k], "lo": lo, "hi": hi}
    out["n"], out["n_days"], out["degenerate"] = point["n"], D, False
    return out


def overlaps(a: dict, b: dict) -> bool:
    """Do two bootstrap intervals overlap? Not a test of their difference: use paired_rmse_diff."""
    return not (a["hi"] < b["lo"] or b["hi"] < a["lo"])


def paired_rmse_diff(per_day_a, per_day_b, per_day_true, n_boot=1000, seed=0, block_days=1) -> dict:
    """RMSE(a) - RMSE(b) on the same pixels, with a 95% CI from resampling the same test days
    for both, so day-to-day difficulty cancels out of the difference."""
    D = len(per_day_true)
    if not D or not (len(per_day_a) == len(per_day_b) == D):
        raise ValueError("need the same, non-zero number of days for both methods")

    def diff(idx):
        t = np.concatenate([per_day_true[i] for i in idx])
        a = np.concatenate([per_day_a[i] for i in idx])
        b = np.concatenate([per_day_b[i] for i in idx])
        return float(np.sqrt(np.mean((a - t) ** 2)) - np.sqrt(np.mean((b - t) ** 2)))

    point = diff(range(D))
    if D < 2:
        return {"point": point, "lo": point, "hi": point, "n_days": D, "degenerate": True}
    rng = np.random.default_rng(seed)
    boot = [diff(resample_days(rng, D, block_days)) for _ in range(n_boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"point": point, "lo": float(lo), "hi": float(hi), "n_days": D, "degenerate": False}


def compare_same_pixels(preds: dict, truth, ref: str, *, n_boot=1000, seed=0, block_days=1) -> dict:
    """Methods {name: per-day predictions} scored on the pixels every one of them fills: each one's
    RMSE [95% CI], R2 and bias, and the paired day bootstrap of RMSE(method) - RMSE(ref) (resolved when
    its CI excludes 0). Days left without a pixel are dropped."""
    keep = [np.isfinite(np.asarray(t, "float64")) for t in truth]
    for per_day in preds.values():
        keep = [k & np.isfinite(np.asarray(p, "float64")) for k, p in zip(keep, per_day, strict=True)]
    days = [i for i, k in enumerate(keep) if k.any()]
    if not days:
        return {"n": 0, "n_days": 0}

    def pick(per_day):
        return [np.asarray(per_day[i], "float64")[keep[i]] for i in days]

    t, p = pick(truth), {k: pick(v) for k, v in preds.items()}
    boot = dict(n_boot=n_boot, seed=seed, block_days=block_days)
    out = {"n": int(sum(x.size for x in t)), "n_days": len(days), "methods": {}, "diff": {}}
    for k, v in p.items():
        ci = bootstrap_ci(v, t, **boot)
        out["methods"][k] = {"rmse": ci["rmse"], "r2": ci["r2"]["point"], "bias": ci["bias"]["point"]}
        if k != ref:
            d = paired_rmse_diff(v, p[ref], t, **boot)
            out["diff"][f"{k}_minus_{ref}"] = {**d, "resolved": d["hi"] < 0 or d["lo"] > 0}
    return out


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def restrict_to_sensors(s, keep, n_s, T):
    """Batched sample with every channel of sensors NOT in `keep` zeroed (values and masks)."""
    seq, x, m = s["seq"].clone(), s["x"].clone(), s["mask_flat"].clone()
    for si in range(n_s):
        if si in keep:
            continue
        seq[:, 2 * si] = 0
        seq[:, 2 * si + 1] = 0
        for k in range(T):
            x[k * n_s + si] = 0
            m[k * n_s + si] = 0
    out = {
        "seq": seq,
        "x": x,
        "mask": m.max(0, keepdim=True).values,
        "mask_flat": m,
        "target": s["target"],
        "hole": s["hole"],
        "valid": s["valid"],
    }
    return {k: v.unsqueeze(0) for k, v in out.items()}


def interp_fill(values, obs_bool, hole_bool):
    """Fill hole pixels by interpolating from observed pixels (linear -> nearest)."""
    H, W = values.shape
    yy, xx = np.mgrid[0:H, 0:W]
    pts = np.stack([yy[obs_bool], xx[obs_bool]], axis=-1)
    tgt = np.stack([yy[hole_bool], xx[hole_bool]], axis=-1)
    out = np.full(hole_bool.sum(), np.nan, dtype="float32")
    if len(pts) >= 3:
        out = griddata(pts, values[obs_bool], tgt, method="linear")
    if len(pts) >= 1:
        nan = ~np.isfinite(out)
        if nan.any():
            out[nan] = griddata(pts, values[obs_bool], tgt[nan], method="nearest")
    return out


def persistence_fill(ds, s, hole):
    """Most recent valid observation in the window, any sensor (log10 space)."""
    n_s, T = ds.n_sensors, ds.input_days
    x, mflat = s["x"].numpy(), s["mask_flat"].numpy()
    persist = np.full((ds.H, ds.W), np.nan, dtype="float32")
    for k in range(T):
        blk = k * n_s
        vv = np.stack([ds.sensor_denorm_log(x[blk + si], si) for si in range(n_s)], 0)
        mm = mflat[blk : blk + n_s] > 0.5
        if k == T - 1:
            mm = mm & ~hole[None]
        dd = mm.sum(0).astype("float32")
        fused = np.divide((vv * mm).sum(0), dd, out=np.zeros_like(dd), where=dd > 0)
        persist = np.where(dd > 0, fused, persist)
    return persist


def day_t_observations(ds, s):
    """The target day as the model saw it: per-sensor values in log10 space (n_s, H, W), their
    validity (n_s, H, W), and the fusion (mean over the sensors that saw a pixel; 0 where none did)
    with its (H, W) observed mask. Each sensor keeps its own calibration (no bias correction)."""
    n_s, T = ds.n_sensors, ds.input_days
    x, mflat = s["x"].numpy(), s["mask_flat"].numpy()
    base = (T - 1) * n_s
    v_t_log = np.stack([ds.sensor_denorm_log(x[base + si], si) for si in range(n_s)], 0)
    m_t = mflat[base : base + n_s] > 0.5
    denom = m_t.sum(0).astype("float32")
    fused_log = np.divide((v_t_log * m_t).sum(0), denom, out=np.zeros_like(denom), where=denom > 0)
    return v_t_log, m_t, fused_log, denom > 0


def sensor_offsets(ds, train_end: date) -> dict:
    """Per sensor, mean(target - observation) in log10 over the training days it observed: the shift
    that puts the sensor on the target's level (a bias correction; nothing else is fitted)."""
    train = np.array([date(*map(int, d)) <= train_end for d in ds.dates])
    truth = ds.target_to_log(ds.clean_norm)
    out = {}
    for si, key in enumerate(ds.sensor_keys):
        seen = (ds.masks[si] > 0.5) & train[:, None, None]
        obs = ds.sensor_denorm_log(ds.values[si], si)
        out[key] = float(np.mean(truth[seen] - obs[seen])) if seen.any() else 0.0
    return out


def observed_log(ds, t: int, offsets: dict, keys=None):
    """Day t's real observations in log10, before any synthetic hole: the mean of the sensors (all, or
    only `keys`) that saw each pixel, each shifted by its offset; NaN where none did. Returns
    (field (H, W), seen (H, W))."""
    num = np.zeros((ds.H, ds.W), "float64")
    den = np.zeros((ds.H, ds.W), "float64")
    for si, key in enumerate(ds.sensor_keys):
        if keys is not None and key not in keys:
            continue
        seen = ds.masks[si][t] > 0.5
        num += np.where(seen, ds.sensor_denorm_log(ds.values[si][t], si) + offsets.get(key, 0.0), 0.0)
        den += seen
    return np.where(den > 0, num / np.maximum(den, 1), np.nan), den > 0


def load_model_from(ckpt: Path, cfg: ExperimentConfig, ds, device):
    model = build_model_for(cfg, ds, dropout=0.0).to(device)
    load_weights(model, Path(ckpt), device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# main evaluation
# ---------------------------------------------------------------------------
def evaluate(
    cfg: ExperimentConfig,
    ckpt: Path,
    p: Paths | None = None,
    *,
    n_boot=1000,
    block_days=1,
    make_figures=True,
    out_dir: Path | None = None,
) -> dict:
    p = p or paths()
    out_dir = Path(out_dir) if out_dir else p.outputs_for(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] device={device} checkpoint={Path(ckpt).name} model={cfg.model.name}")

    _, ds, sensor_paths = build_gapfill_datasets(cfg, p)
    train, val, test = split_positions(ds.sample_dates, cfg.split, cfg.input_days)
    check_split_windows(ds.sample_dates, train, val, test, cfg)
    print(f"[eval] sensors={ds.n_sensors} ({list(ds.sensor_keys)}) test days={len(test)}")
    if len(test) == 0:
        raise ValueError("no test days: check split.test_start/test_end vs the data")
    model = load_model_from(ckpt, cfg, ds, device)

    methods = ["model", "model_ablate_single", "interp_fusion", "interp_single", "persistence"]
    acc = {k: {"p": [], "t": []} for k in methods}
    acc_log = {k: {"p": [], "t": []} for k in methods}
    indep_idx = [i for i, k in enumerate(ds.sensor_keys) if "modis" in k.lower()]
    indep = {k: {"p": [], "t": []} for k in ["model", "persistence"]}
    offsets = sensor_offsets(ds, cfg.split.train_end)
    viirs_idx = [i for i, k in enumerate(ds.sensor_keys) if "viirs" in k.lower()]
    slice_keys = {"observed": None}  # None = every input sensor
    if indep_idx:
        slice_keys["independence"] = [ds.sensor_keys[i] for i in indep_idx]
    slices = {name: {k: [] for k in ("model", "persistence", "target", "observation")} for name in slice_keys}
    n_viirs_seen = n_indep_also_viirs = 0
    matched = {k: [] for k in [*methods, "_truth"]}  # per day, the hole pixels every method fills
    observed = {k: {"p": [], "t": []} for k in methods}
    n_observed = n_hole = 0
    n_s, T, lt = ds.n_sensors, ds.input_days, ds.log_transform
    cover_single, cover_fusion, per_day_rmse = [], [], {k: [] for k in methods}
    figure_sample = None

    with torch.no_grad():
        for pos in test:
            s = ds[int(pos)]
            hole = s["hole"][0].numpy() > 0.5
            if hole.sum() == 0:
                continue
            t_day = int(ds.sample_index[int(pos)])
            target_norm = s["target"][0].numpy()
            target_log, target_chl = ds.target_to_log(target_norm), ds.inverse(target_norm)
            thl, th = target_log[hole], target_chl[hole]

            batch = {k: v.unsqueeze(0) for k, v in s.items()}
            pred_norm = forward_batch(model, batch, device)[0, 0].cpu().numpy()
            pred_log, pred_chl = ds.target_to_log(pred_norm), ds.inverse(pred_norm)

            v_t_log, m_t, fused_log, obs_fusion = day_t_observations(ds, s)
            best = int(np.argmax(m_t.reshape(n_s, -1).sum(1)))
            single_pred_norm = (
                forward_batch(model, restrict_to_sensors(s, {best}, n_s, T), device)[0, 0].cpu().numpy()
            )

            fusion_fill_log = interp_fill(fused_log, obs_fusion, hole)
            single_fill_log = interp_fill(v_t_log[best], m_t[best], hole)
            cover_single.append(float(m_t[best].mean()))
            cover_fusion.append(float(obs_fusion.mean()))
            persist_log = persistence_fill(ds, s, hole)

            preds_log = {
                "model": pred_log[hole],
                "model_ablate_single": ds.target_to_log(single_pred_norm)[hole],
                "interp_fusion": fusion_fill_log,
                "interp_single": single_fill_log,
                "persistence": persist_log[hole],
            }
            for key, plog in preds_log.items():
                pchl = ds.log_to_chl(plog, lt)
                acc_log[key]["p"].append(np.asarray(plog, "float64"))
                acc_log[key]["t"].append(np.asarray(thl, "float64"))
                acc[key]["p"].append(np.asarray(pchl, "float64"))
                acc[key]["t"].append(np.asarray(th, "float64"))
                per_day_rmse[key].append(skill(pchl, th)["rmse"])
            common = np.logical_and.reduce(
                [np.isfinite(np.asarray(v, "float64")) for v in preds_log.values()]
            )
            for key, plog in preds_log.items():
                matched[key].append(np.asarray(plog, "float64")[common])
            matched["_truth"].append(np.asarray(thl, "float64")[common])
            seen = np.stack([ds.masks[si][t_day] for si in range(n_s)], 0).max(0)[hole] > 0.5
            n_observed, n_hole = n_observed + int(seen.sum()), n_hole + int(hole.sum())
            viirs_seen = np.zeros_like(hole)
            for si in viirs_idx:
                viirs_seen |= ds.masks[si][t_day] > 0.5
            n_viirs_seen += int(viirs_seen[hole].sum())
            for name, keys in slice_keys.items():
                obs_log, obs_seen = observed_log(ds, t_day, offsets, keys)
                sel = obs_seen[hole]
                if name == "independence":
                    n_indep_also_viirs += int((sel & viirs_seen[hole]).sum())
                for key, arr in (
                    ("model", pred_log[hole]),
                    ("persistence", persist_log[hole]),
                    ("target", thl),
                    ("observation", obs_log[hole]),
                ):
                    slices[name][key].append(np.asarray(arr, "float64")[sel])
            for key, plog in preds_log.items():
                observed[key]["p"].append(np.asarray(plog, "float64")[seen])
                observed[key]["t"].append(np.asarray(thl, "float64")[seen])

            if indep_idx:
                mi = indep_idx[0]
                modis_seen = ds.masks[mi][t_day] > 0.5
                sub = hole & modis_seen
                if sub.sum() > 0:
                    subhole = sub[hole]
                    indep["model"]["p"].append(np.asarray(pred_log[hole][subhole], "float64"))
                    indep["model"]["t"].append(np.asarray(thl[subhole], "float64"))
                    indep["persistence"]["p"].append(np.asarray(persist_log[hole][subhole], "float64"))
                    indep["persistence"]["t"].append(np.asarray(thl[subhole], "float64"))

            if figure_sample is None and obs_fusion.mean() > 0.35:
                figure_sample = dict(
                    v_t_log=v_t_log,
                    m_t=m_t,
                    pred=pred_chl,
                    target=target_chl,
                    hole=hole,
                    best=best,
                    date=date(*map(int, ds.sample_dates[int(pos)])),
                    rmse={k: per_day_rmse[k][-1] for k in ("model", "interp_fusion")},  # this day's only
                )

    if not acc["model"]["p"]:
        raise ValueError("no test day produced a non-empty hole")

    result = {
        "checkpoint": Path(ckpt).name,
        "model": cfg.model.name,
        "config_hash": cfg.config_hash(),
        "n_sensors": ds.n_sensors,
        "input_sensors": list(sensor_paths),
        "sensor_keys": list(ds.sensor_keys),
        "n_test_days": int(len(test)),
        "log_transform": cfg.log_transform,
        "per_sensor_stats": {k: [float(m), float(sd)] for k, (m, sd) in ds.sensor_stats.items()},
        "mean_daytime_coverage": {
            "single_best_sensor": float(np.mean(cover_single)),
            "fused_all_sensors": float(np.mean(cover_fusion)),
        },
        "chlorophyll_units": {},
        "log10_space": {},
        "chlorophyll_units_ci": {},
        "log10_space_ci": {},
    }
    for key in methods:
        result["chlorophyll_units"][key] = skill(np.concatenate(acc[key]["p"]), np.concatenate(acc[key]["t"]))
        result["log10_space"][key] = skill(
            np.concatenate(acc_log[key]["p"]), np.concatenate(acc_log[key]["t"])
        )
        result["chlorophyll_units_ci"][key] = bootstrap_ci(
            acc[key]["p"], acc[key]["t"], n_boot=n_boot, seed=cfg.seed, block_days=block_days
        )
        result["log10_space_ci"][key] = bootstrap_ci(
            acc_log[key]["p"], acc_log[key]["t"], n_boot=n_boot, seed=cfg.seed, block_days=block_days
        )
    result["bootstrap"] = {"n_boot": n_boot, "block_days": block_days, "unit": "test day"}
    truth = matched["_truth"]
    result["log10_space_matched"] = {
        "note": "every method scored on the same pixels: the hole pixels all methods fill",
        **{key: skill(np.concatenate(matched[key]), np.concatenate(truth)) for key in methods},
    }
    result["log10_space_matched_ci"] = {
        key: bootstrap_ci(matched[key], truth, n_boot=n_boot, seed=cfg.seed, block_days=block_days)
        for key in methods
    }
    result["verdicts"] = {}
    for k in methods[1:]:
        d = paired_rmse_diff(
            matched["model"], matched[k], truth, n_boot=n_boot, seed=cfg.seed, block_days=block_days
        )
        result["verdicts"][f"model_vs_{k}"] = {
            "model_better": d["point"] < 0,
            "resolved": d["hi"] < 0 or d["lo"] > 0,
            "log_rmse_diff": d,  # model - baseline on the matched pixels; < 0 favours the model
            "n": result["log10_space_matched"]["model"]["n"],
        }
    result["independence_slice"] = {
        "note": "hole pixels where MODIS-Aqua (independent instrument) observed",
        "log10_space": {},
    }
    for key in indep:
        result["independence_slice"]["log10_space"][key] = (
            skill(np.concatenate(indep[key]["p"]), np.concatenate(indep[key]["t"]))
            if indep[key]["p"]
            else {"n": 0}
        )
    if indep["model"]["p"]:  # the same comparison on the pixels persistence can fill
        ok = np.isfinite(np.concatenate(indep["persistence"]["p"]))
        result["independence_slice"]["log10_space_matched"] = {
            key: skill(np.concatenate(indep[key]["p"])[ok], np.concatenate(indep[key]["t"])[ok])
            for key in indep
        }
    result["observed_slice"] = {
        "note": "hole pixels some input sensor observed that day; the target there is still NOAA's DINEOF "
        "product, so the slice is also scored against the withheld observation",
        "frac_of_hole_px": n_observed / max(n_hole, 1),
        "frac_of_hole_px_viirs_observed": n_viirs_seen / max(n_hole, 1),
        "log10_space": {
            key: skill(np.concatenate(observed[key]["p"]), np.concatenate(observed[key]["t"]))
            for key in methods
        },
    }
    result["sensor_offsets_log10"] = offsets
    boot = dict(n_boot=n_boot, seed=cfg.seed, block_days=block_days)
    for name, sl in slices.items():
        out = result[f"{name}_slice"]
        if name == "independence":
            out["n_also_viirs_observed"] = n_indep_also_viirs
        pair = {"model": sl["model"], "persistence": sl["persistence"]}
        out["paired_vs_target"] = compare_same_pixels(pair, sl["target"], "persistence", **boot)
        out["paired_vs_observation"] = compare_same_pixels(pair, sl["observation"], "persistence", **boot)
        # how far the "truth" itself sits from the real same-day observation
        out["target_vs_observation"] = skill(np.concatenate(sl["target"]), np.concatenate(sl["observation"]))
    result["provenance"] = provenance(
        cfg, ds, p.data / cfg.data.train_nc, sensor_paths, Path(ckpt), [ds.sample_dates[int(i)] for i in test]
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print_table(result)
    if make_figures:
        from chl_gapfill.figures import gapfill_figure, validation_figures

        if figure_sample is not None:
            gapfill_figure(ds, figure_sample, result, out_dir / "gapfill_sample.png")
        validation_figures(acc_log, per_day_rmse, result, out_dir)
    print(f"[eval] wrote {out_dir / 'metrics.json'}")
    return result


def _fmt_ci(ci):
    return f"{ci['point']:.3f} [{ci['lo']:.3f},{ci['hi']:.3f}]"


def print_table(result: dict) -> None:
    cov = result["mean_daytime_coverage"]
    print(
        f"\n[eval] mean day-t coverage: single best sensor={cov['single_best_sensor']:.1%}"
        f"  fused all sensors={cov['fused_all_sensors']:.1%}"
    )
    print("[eval] hole-pixel skill (point [95% CI]); log10 space is the primary metric:")
    print(f"  {'method':22s} {'n':>6s} {'logRMSE (95% CI)':>26s} {'logR2':>6s}   {'chlRMSE':>8s}")
    for key in ["persistence", "interp_single", "interp_fusion", "model_ablate_single", "model"]:
        lci = result["log10_space_ci"][key]["rmse"]
        lr2 = result["log10_space"][key]["r2"]
        c = result["chlorophyll_units"][key]["rmse"]
        print(f"  {key:22s} {result['log10_space'][key]['n']:6d} {_fmt_ci(lci):>26s} {lr2:6.2f}   {c:8.3f}")
    n = result["log10_space_matched"]["model"]["n"]
    print(f"[eval] on the {n} hole pixels every method fills; paired day bootstrap of model - baseline:")
    print(f"  {'baseline':22s} {'logRMSE':>8s} {'model':>8s} {'diff (95% CI)':>28s}  verdict")
    for key in ["persistence", "interp_single", "interp_fusion", "model_ablate_single"]:
        v = result["verdicts"][f"model_vs_{key}"]
        better = "model" if v["model_better"] else "baseline"
        verdict = f"{better} better" + ("" if v["resolved"] else " (unresolved: CI spans 0)")
        print(
            f"  {key:22s} {result['log10_space_matched'][key]['rmse']:8.3f} "
            f"{result['log10_space_matched']['model']['rmse']:8.3f} {_fmt_ci(v['log_rmse_diff']):>28s}  {verdict}"
        )
    ind = result["independence_slice"]["log10_space"]
    if ind.get("model", {}).get("n", 0):
        print(
            f"[eval] independence slice (MODIS-observed hole px, n={ind['model']['n']}): "
            f"model logRMSE={ind['model']['rmse']:.3f}  persistence logRMSE={ind['persistence']['rmse']:.3f}"
        )
    if m := result["independence_slice"].get("log10_space_matched"):
        print(
            f"[eval]   same slice, pixels persistence fills (n={m['model']['n']}): "
            f"model={m['model']['rmse']:.3f}  persistence={m['persistence']['rmse']:.3f}"
        )
    obs = result["observed_slice"]
    print(
        f"[eval] observed slice ({obs['frac_of_hole_px']:.0%} of hole px, n={obs['log10_space']['model']['n']}): "
        f"model logRMSE={obs['log10_space']['model']['rmse']:.3f}"
    )
    for name in ("observed", "independence"):
        sl = result[f"{name}_slice"]
        for truth in ("target", "observation"):
            if not (pr := sl.get(f"paired_vs_{truth}", {})).get("n"):
                continue
            d, m = pr["diff"]["model_minus_persistence"], pr["methods"]
            print(
                f"[eval]   {name} slice vs {truth}, pixels both fill (n={pr['n']}, {pr['n_days']} days): "
                f"model={m['model']['rmse']['point']:.3f} persistence={m['persistence']['rmse']['point']:.3f} "
                f"diff {_fmt_ci(d)}{'' if d['resolved'] else ' (unresolved)'}"
            )
