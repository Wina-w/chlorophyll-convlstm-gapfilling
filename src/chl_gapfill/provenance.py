"""What a published number was computed from: checkpoint, configuration, data files, versions.

Every results JSON carries this block, so each number traces back to the exact checkpoint
(sha256), the effective configuration and the data window it was computed on. Only file
names are recorded, never machine paths.
"""

from __future__ import annotations

import hashlib
import platform
from datetime import date
from pathlib import Path

import numpy as np

from chl_gapfill import __version__


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def file_record(path: Path) -> dict:
    path = Path(path)
    return {"file": path.name, "sha256": sha256(path), "bytes": path.stat().st_size}


def _iso(d) -> str:
    return date(*map(int, d)).isoformat()


def provenance(cfg, ds, target_nc: Path, sensor_paths: dict, ckpt: Path | None = None, test_dates=()) -> dict:
    """Checkpoint, effective config, data files and window, and versions behind one result."""
    import torch

    test_dates = list(test_dates)
    return {
        "checkpoint": file_record(ckpt) if ckpt else None,
        "config_hash": cfg.config_hash(),
        "config": cfg.model_dump(mode="json"),
        "data": {
            "target": file_record(target_nc),
            "sensors": {key: file_record(path) for key, path in sensor_paths.items()},
            "first_day": _iso(ds.dates[0]),
            "last_day": _iso(ds.dates[-1]),
            "n_days": len(ds.dates),
            "test_days": {"first": _iso(test_dates[0]), "last": _iso(test_dates[-1]), "n": len(test_dates)}
            if test_dates
            else None,
        },
        "versions": {
            "chl_gapfill": __version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
    }
