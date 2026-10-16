"""Training: configurable, resumable, tracked.

    chl-gapfill train -c configs/convlstm.yaml
    chl-gapfill train -c configs/convlstm.yaml --smoke          # 2 epochs, proves the loop
    chl-gapfill train -c configs/convlstm.yaml --resume models/convlstm_fusion/last.pt
    chl-gapfill train -c configs/convlstm.yaml --set training.lr=3e-4 --set model.hidden_channels=[32,16]

What changed from the original ``src/train.py`` (behaviour is otherwise identical):
* one ``run_epoch`` for train and validation instead of two copies;
* ``best.pt`` / ``last.pt`` carry model + optimizer + scheduler + epoch + history +
  the early-stopping counter + every RNG stream (python, numpy, torch, the DataLoader
  shuffle generator, the training-hole generator), so ``--resume`` continues the
  interrupted run bit for bit on CPU; a plain ``state_dict`` is still written to
  ``models/<name>.pth`` for the evaluation/benchmark scripts;
* per-epoch train/val loss and the final summary go to MLflow (optional);
* the validation split has an optional gap (``split.gap_days``);
* optional gradient clipping and a wall-clock ``--smoke`` mode.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from chl_gapfill import data_sources
from chl_gapfill.config import ExperimentConfig, Paths, SplitConfig, flatten, paths
from chl_gapfill.dataset import ChlorophyllDataset, GapFillDataset
from chl_gapfill.losses import GapFillLoss
from chl_gapfill.model import ConvLSTM, build_model
from chl_gapfill.tracking import Tracker


# ---------------------------------------------------------------------------
# reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def rng_state(loader_gen: torch.Generator, train_ds: GapFillDataset) -> dict:
    """Every random stream a training epoch draws from, in a form `torch.load(weights_only=True)` reads."""
    np_state = np.random.get_state(legacy=False)
    np_state["state"] = {"key": np_state["state"]["key"].tolist(), "pos": np_state["state"]["pos"]}
    return {
        "python": random.getstate(),
        "numpy": np_state,
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loader": loader_gen.get_state(),  # shuffle order
        "holes": train_ds.hole_rng.bit_generator.state,  # synthetic training clouds
    }


def restore_rng_state(state: dict, loader_gen: torch.Generator, train_ds: GapFillDataset) -> None:
    random.setstate(state["python"])
    np_state = dict(state["numpy"])
    np_state["state"] = {
        "key": np.asarray(np_state["state"]["key"], np.uint32),
        "pos": np_state["state"]["pos"],
    }
    np.random.set_state(np_state)
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():  # pragma: no cover
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    loader_gen.set_state(state["loader"].cpu())
    train_ds.hole_rng.bit_generator.state = state["holes"]


# what may differ between a checkpoint and the run that resumes it (extend training, rename)
RESUME_MAY_CHANGE = ("name", "experiment", "smoke", "training.n_epochs", "training.patience")


def check_resume_config(saved: dict, cfg: ExperimentConfig, resume: Path) -> None:
    old, new = flatten(saved), cfg.flat_params()
    # keys only one side has are schema changes between versions, not a different experiment
    diff = sorted(k for k in old.keys() & new.keys() if k not in RESUME_MAY_CHANGE and old[k] != new[k])
    if diff:
        detail = ", ".join(f"{k}: {old[k]} -> {new[k]}" for k in diff)
        raise ValueError(
            f"--resume {resume}: the checkpoint was trained with a different configuration ({detail}); "
            f"only {', '.join(RESUME_MAY_CHANGE)} may change on resume"
        )


# ---------------------------------------------------------------------------
# splits (shared with evaluate/predict)
# ---------------------------------------------------------------------------
def _to_dt(triples):
    return np.array([datetime(int(y), int(m), int(d)) for y, m, d in triples])


def split_positions(sample_dates, split: SplitConfig, input_days: int = 3):
    """train/val/test positions (into a GapFillDataset) by target-day date.

    With ``gap_days > 0`` the first `gap_days` days of the validation and test
    ranges are dropped so their input windows never contain a training target day.
    """
    dt = _to_dt(sample_dates)
    gap = np.timedelta64(int(split.gap_days), "D")

    def d(x: date):
        return np.datetime64(datetime(x.year, x.month, x.day))

    dt64 = dt.astype("datetime64[D]")
    train = np.where(dt64 <= d(split.train_end))[0]
    val = np.where((dt64 >= d(split.val_start) + gap) & (dt64 <= d(split.val_end)))[0]
    test = np.where((dt64 >= d(split.test_start) + gap) & (dt64 <= d(split.test_end)))[0]
    return train, val, test


def assert_no_overlap(sample_dates, train_idx, later_idx, input_days: int) -> None:
    """Raise if any later-split input window contains a target day of the earlier split."""
    if len(train_idx) == 0 or len(later_idx) == 0:
        return
    dt = _to_dt(sample_dates)
    last_train = dt[train_idx].max()
    first_later_window_start = dt[later_idx].min() - np.timedelta64(input_days - 1, "D").astype(
        "timedelta64[s]"
    ).astype(object)
    if last_train >= first_later_window_start:
        raise ValueError(
            f"split overlap: the last earlier-split target {last_train.date()} lies inside the first "
            f"later-split input window (starts {first_later_window_start.date()}); "
            "set split.gap_days >= input_days-1"
        )


def check_split_windows(sample_dates, tr, va, te, cfg: ExperimentConfig) -> None:
    """Both boundaries (train->val, val->test): no input window may reach an earlier split's
    target days. An error with split.gap_days > 0; with the published gap_days=0 the overlap
    (input days only, never targets) is known and reported instead."""
    for name, earlier, later in (("train->val", tr, va), ("val->test", va, te)):
        try:
            assert_no_overlap(sample_dates, earlier, later, cfg.input_days)
        except ValueError as exc:
            if cfg.split.gap_days:
                raise
            print(f"[split][WARN] {name}: {exc} (gap_days=0 as published)")


# ---------------------------------------------------------------------------
# data assembly
# ---------------------------------------------------------------------------
def build_gapfill_datasets(cfg: ExperimentConfig, p: Paths, subset=None, min_coverage=None):
    target_nc = p.data / cfg.data.train_nc
    if not target_nc.exists():
        raise FileNotFoundError(
            f"target file {target_nc} not found (CHL_DATA_DIR={p.data}); fetch it with `chl-gapfill download` "
            "(see data/README.md) or make synthetic data with `python scripts/make_synthetic_data.py`"
        )
    if cfg.data.self_supervised:
        if subset is not None:
            raise ValueError("--subset selects input sensors, but data.self_supervised uses none")
        sensor_paths = {}
        print("[train][WARN] data.self_supervised: the inputs are past days of the clean target (a toy task)")
    else:
        keys = subset if subset is not None else cfg.data.input_sensors
        sensor_paths = data_sources.available_sensor_paths(p.data, keys, fusion=cfg.data.fusion)
        print(f"[train] input sensors: {list(sensor_paths)}")
    common = dict(
        target_nc=target_nc,
        lat_bounds=cfg.region.lat_bounds,
        lon_bounds=cfg.region.lon_bounds,
        train_end=cfg.split.train_end,
        var_name=cfg.data.var_name,
        input_days=cfg.input_days,
        log_transform=cfg.log_transform,
        min_frac=cfg.mask.min_frac,
        max_frac=cfg.mask.max_frac,
        n_blobs=cfg.mask.n_blobs,
        min_coverage=cfg.data.min_coverage if min_coverage is None else min_coverage,
        seed=cfg.seed,
        self_supervised=cfg.data.self_supervised,
        contiguous_windows=cfg.data.contiguous_windows,
    )
    train_ds = GapFillDataset(sensor_paths, deterministic=False, **common)
    val_ds = GapFillDataset(
        sensor_paths, deterministic=True, stats=train_ds.stats, sensor_stats=train_ds.sensor_stats, **common
    )
    return train_ds, val_ds, sensor_paths


def forward_batch(model, batch, device):
    if isinstance(model, ConvLSTM):
        return model(batch["seq"].to(device))
    return model(batch["x"].to(device), batch["mask"].to(device))


def model_input_channels(ds, model_name: str):
    return ds.per_day_channels if model_name == "convlstm" else ds.flat_channels


def build_model_for(cfg: ExperimentConfig, ds, dropout: float | None = None):
    return build_model(
        cfg.model.name,
        model_input_channels(ds, cfg.model.name),
        cfg.model.hidden_channels,
        cfg.model.kernel_size,
        (ds.H, ds.W),
        step=cfg.input_days,
        use_norm=cfg.model.use_norm,
        dropout=cfg.model.dropout if dropout is None else dropout,
    )


# ---------------------------------------------------------------------------
# checkpoints
# ---------------------------------------------------------------------------
def save_checkpoint(
    path: Path, model, optimizer, scheduler, epoch, best_val, history, cfg: ExperimentConfig, extra=None
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": int(epoch),
            "best_val": float(best_val),
            "history": history,
            "config": cfg.model_dump(mode="json"),
            "config_hash": cfg.config_hash(),
            **(extra or {}),
        },
        path,
    )


def load_weights(model, ckpt: Path, device, strict: bool = True):
    """Load either a full checkpoint (dict with 'model') or a bare state_dict (.pth).

    ``weights_only=True``: checkpoints hold tensors and plain types, so loading one cannot run
    code. Strict by default: a key or shape mismatch means the model built from the config is
    not the one that was trained (other hidden_channels, use_norm or input sensors), and scoring
    it would score a partly random model. ``strict=False`` loads what matches and warns.
    """
    state = torch.load(ckpt, map_location=device, weights_only=True)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    own = model.state_dict()
    missing = [k for k in own if k not in state]
    unexpected = [k for k in state if k not in own]
    reshaped = [k for k in own if k in state and tuple(state[k].shape) != tuple(own[k].shape)]
    if missing or unexpected or reshaped:
        detail = f"missing={missing[:6]} unexpected={unexpected[:6]} shape mismatch={reshaped[:6]}"
        if strict:
            raise ValueError(
                f"checkpoint {ckpt} does not match the model built from the config "
                f"(check model.* and the input sensors): {detail}"
            )
        print(f"[ckpt][WARN] partial load of {ckpt}: {detail}")
        state = {k: v for k, v in state.items() if k in own and k not in reshaped}
    model.load_state_dict(state, strict=strict)
    return model


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------
@dataclass
class TrainResult:
    run_id: str
    run_dir: Path
    best_path: Path
    last_path: Path
    state_dict_path: Path
    best_val: float
    epochs_run: int
    history: list[dict] = field(default_factory=list)


def run_epoch(model, loader, loss_fn, device, optimizer=None, grad_clip=None):
    training = optimizer is not None
    model.train(training)
    total, n_batches = 0.0, 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            target = batch["target"].to(device)
            hole = batch["hole"].to(device)
            valid = batch["valid"].to(device)
            pred = forward_batch(model, batch, device)
            loss = loss_fn(pred, target, hole, valid)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            total += loss.item()
            n_batches += 1
    return total / max(n_batches, 1)  # mean over batches, as in the original


def fit(
    cfg: ExperimentConfig,
    p: Paths | None = None,
    *,
    subset=None,
    min_coverage=None,
    label: str = "",
    resume: Path | None = None,
    use_mlflow: bool = True,
    state_dict_out: Path | None = None,
) -> TrainResult:
    p = p or paths()
    set_seed(cfg.seed)
    if cfg.training.num_threads:
        torch.set_num_threads(cfg.training.num_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, val_ds, sensor_paths = build_gapfill_datasets(cfg, p, subset, min_coverage)
    tr, va, te = split_positions(train_ds.sample_dates, cfg.split, cfg.input_days)
    check_split_windows(train_ds.sample_dates, tr, va, te, cfg)
    if len(tr) == 0 or len(va) == 0:
        raise ValueError(f"empty split: train={len(tr)} val={len(va)} — check split dates vs data coverage")
    if cfg.smoke:
        tr, va = (
            tr[: min(len(tr), 4 * cfg.training.batch_size)],
            va[: min(len(va), 2 * cfg.training.batch_size)],
        )
    n_epochs = min(cfg.training.n_epochs, 2) if cfg.smoke else cfg.training.n_epochs
    print(
        f"[train]{(' ' + label) if label else ''} task={cfg.task} model={cfg.model.name} device={device} "
        f"samples train={len(tr)} val={len(va)}  H={train_ds.H} W={train_ds.W} "
        f"sensors={train_ds.n_sensors} ({list(train_ds.sensor_keys)}) smoke={cfg.smoke}"
    )

    g = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(
        Subset(train_ds, tr), batch_size=cfg.training.batch_size, shuffle=True, generator=g
    )
    val_loader = DataLoader(Subset(val_ds, va), batch_size=cfg.training.batch_size, shuffle=False)

    model = build_model_for(cfg, train_ds).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=max(3, cfg.training.patience // 3)
        )
        if cfg.training.lr_schedule
        else None
    )
    lc = cfg.training.loss
    loss_fn = GapFillLoss(w_hole=lc.w_hole, w_valid=lc.w_valid, w_grad=lc.w_grad)

    run_name = f"{cfg.name}{('_' + label) if label else ''}"
    run_dir = p.models / run_name
    best_path, last_path = run_dir / "best.pt", run_dir / "last.pt"
    state_dict_path = Path(state_dict_out) if state_dict_out else p.models / f"{run_name}.pth"

    start_epoch, best_val, bad, history, saved_rng, resumed_from = 1, float("inf"), 0, [], None, ""
    if resume is not None:
        ck = torch.load(resume, map_location=device, weights_only=True)
        check_resume_config(ck.get("config", {}), cfg, resume)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        if scheduler is not None and ck.get("scheduler"):
            scheduler.load_state_dict(ck["scheduler"])
        start_epoch, best_val, history = ck["epoch"] + 1, ck["best_val"], list(ck.get("history", []))
        bad, saved_rng, resumed_from = int(ck.get("bad_epochs", 0)), ck.get("rng"), ck.get("run_id", "")
        print(
            f"[train] resumed from {resume} at epoch {ck['epoch']} (best val {best_val:.6f}, {bad} bad epochs)"
        )
        if saved_rng is None:
            print("[train][WARN] checkpoint predates RNG checkpointing: data order and holes restart")

    tracker = Tracker(
        use_mlflow,
        experiment=cfg.experiment,
        tracking_uri=p.mlflow_tracking_uri,
        artifact_root=p.mlflow_artifact_root,
    )
    run_id = tracker.start(
        run_name,
        tags={
            "model": cfg.model.name,
            "smoke": str(cfg.smoke),
            "resumed": str(resume is not None),
            "resumed_from_run": resumed_from,
            "self_supervised": str(cfg.data.self_supervised),
        },
    )
    tracker.log_params(
        {
            **cfg.flat_params(),
            "config_hash": cfg.config_hash(),
            "sensors": ",".join(sensor_paths),
            "n_train": len(tr),
            "n_val": len(va),
            "H": train_ds.H,
            "W": train_ds.W,
            "n_params": sum(q.numel() for q in model.parameters()),
            "device": str(device),
        }
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg.model_dump(mode="json"), indent=2), encoding="utf-8")
    (run_dir / "stats.json").write_text(
        json.dumps({"target": train_ds.stats, "sensors": train_ds.sensor_stats}), encoding="utf-8"
    )

    if bad >= cfg.training.patience:
        print(f"[train] the checkpoint had already early-stopped ({bad} bad epochs); nothing to train")
        n_epochs = start_epoch - 1

    epoch = start_epoch - 1
    try:
        for h in history:  # a resumed run carries the whole curve, not only the new epochs
            tracker.log_metrics(
                {"train_loss": h["train"], "val_loss": h["val"], "lr": h["lr"]}, step=h["epoch"]
            )
        if saved_rng is not None:  # last thing before the loop: nothing above may draw from a stream
            restore_rng_state(saved_rng, g, train_ds)
        for epoch in range(start_epoch, n_epochs + 1):
            t0 = time.time()
            tr_loss = run_epoch(model, train_loader, loss_fn, device, optimizer, cfg.training.grad_clip)
            val_loss = run_epoch(model, val_loader, loss_fn, device)
            if scheduler is not None:
                scheduler.step(val_loss)
            lr = optimizer.param_groups[0]["lr"]
            history.append(
                {
                    "epoch": epoch,
                    "train": tr_loss,
                    "val": val_loss,
                    "lr": lr,
                    "sec": round(time.time() - t0, 2),
                }
            )
            tracker.log_metrics({"train_loss": tr_loss, "val_loss": val_loss, "lr": lr}, step=epoch)
            print(
                f"[epoch {epoch:4d}] train={tr_loss:.6f}  val={val_loss:.6f}  lr={lr:.1e}  ({time.time() - t0:.1f}s)"
            )

            if val_loss < best_val - 1e-6:
                best_val, bad = val_loss, 0
            else:
                bad += 1
            resume_state = {"bad_epochs": bad, "rng": rng_state(g, train_ds), "run_id": run_id}
            if bad == 0:
                save_checkpoint(
                    best_path, model, optimizer, scheduler, epoch, best_val, history, cfg, resume_state
                )
                torch.save(model.state_dict(), state_dict_path)  # legacy format for the benchmark scripts
            save_checkpoint(
                last_path, model, optimizer, scheduler, epoch, best_val, history, cfg, resume_state
            )
            if bad >= cfg.training.patience:
                print(f"[train] early stopping at epoch {epoch}; best val={best_val:.6f}")
                break
        summary = {
            "best_val": best_val,
            "epochs_run": epoch,
            "n_params": sum(q.numel() for q in model.parameters()),
            "early_stopped": bad >= cfg.training.patience,
            "state_dict": str(state_dict_path),
        }
        (run_dir / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        tracker.log_metrics({"best_val_loss": best_val, "epochs_run": epoch})
        tracker.log_dict(summary, "summary.json")
        for f in ("config.json", "stats.json", "history.json"):
            tracker.log_artifact(run_dir / f)
        if best_path.exists():
            tracker.log_artifact(best_path)
        tracker.end("FINISHED")
    except Exception:
        tracker.end("FAILED")
        raise
    print(f"[train] done. best val={best_val:.6f} -> {best_path} (state_dict: {state_dict_path})")
    return TrainResult(run_id, run_dir, best_path, last_path, state_dict_path, best_val, epoch, history)


# ---------------------------------------------------------------------------
# legacy next-day task (kept for backward comparison; same model and loss as the original,
# but normalised on the training period and split by target day like the gap-filling task)
# ---------------------------------------------------------------------------
def nextday_positions(dates, n: int, cfg: ExperimentConfig):
    """train/val positions of a ChlorophyllDataset by the TARGET day (window start + input_days),
    skipping windows across missing days when data.contiguous_windows is set."""
    dt = _to_dt(dates).astype("datetime64[D]")
    start, target = dt[:n], dt[cfg.input_days : cfg.input_days + n]
    ok = np.ones(n, bool)
    if cfg.data.contiguous_windows:
        ok = target - start == np.timedelta64(cfg.input_days, "D")

    def d(x: date):
        return np.datetime64(x.isoformat())

    tr = np.where(ok & (target <= d(cfg.split.train_end)))[0]
    va = np.where(ok & (target >= d(cfg.split.val_start)) & (target <= d(cfg.split.val_end)))[0]
    return tr, va


def fit_nextday(cfg: ExperimentConfig, p: Paths | None = None) -> Path:
    p = p or paths()
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = ChlorophyllDataset(
        p.data / cfg.data.train_nc,
        cfg.data.var_name,
        cfg.input_days,
        cfg.log_transform,
        lat_bounds=cfg.region.lat_bounds,
        lon_bounds=cfg.region.lon_bounds,
        train_end=cfg.split.train_end,
    )
    tr, va = nextday_positions(ds.dates, len(ds), cfg)
    train_loader = DataLoader(Subset(ds, tr), batch_size=cfg.training.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(ds, va), batch_size=cfg.training.batch_size, shuffle=False)
    H, W = ds.data.shape[1], ds.data.shape[2]
    model = ConvLSTM(
        cfg.input_days, cfg.model.hidden_channels, cfg.model.kernel_size, (H, W), step=cfg.input_days
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    loss_fn = torch.nn.MSELoss()
    best_val, counter = float("inf"), 0
    out = p.models / f"{cfg.name}_nextday.pth"
    out.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, (2 if cfg.smoke else cfg.training.n_epochs) + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                val_loss += loss_fn(model(x.to(device)), y.to(device)).item()
        val_loss /= max(len(val_loader), 1)
        print(f"[epoch {epoch:4d}] val={val_loss:.6f}")
        if val_loss < best_val:
            best_val, counter = val_loss, 0
            torch.save(model.state_dict(), out)
        else:
            counter += 1
            if counter >= cfg.training.patience:
                break
    return out
