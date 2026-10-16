"""Optional MLflow tracking behind one small class (no mlflow import elsewhere)."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

from chl_gapfill.config import sqlite_file

log = logging.getLogger(__name__)


class Tracker:
    def __init__(
        self,
        enabled: bool = True,
        experiment: str = "chl-gapfill",
        tracking_uri: str | None = None,
        artifact_root: Path | None = None,
    ):
        self.enabled = enabled
        self.run_id: str | None = None
        self._mlflow = None
        if not enabled:
            return
        try:
            import mlflow
        except ImportError:
            log.warning("mlflow not installed; tracking disabled (pip install 'chl-gapfill[track]')")
            self.enabled = False
            return
        self._mlflow = mlflow
        if tracking_uri:
            if (db := sqlite_file(tracking_uri)) is not None:
                db.parent.mkdir(parents=True, exist_ok=True)
            mlflow.set_tracking_uri(tracking_uri)
        if artifact_root is not None and mlflow.get_experiment_by_name(experiment) is None:
            # an explicit location, or MLflow puts artifacts in ./mlruns of the current directory;
            # an existing experiment keeps the location it was created with
            location = (Path(artifact_root) / experiment).resolve().as_uri()
            mlflow.create_experiment(experiment, artifact_location=location)
        mlflow.set_experiment(experiment)

    def start(self, run_name: str | None, tags: dict[str, str] | None = None) -> str:
        if not self.enabled:
            self.run_id = uuid.uuid4().hex[:16]
            return self.run_id
        self.run_id = self._mlflow.start_run(run_name=run_name, tags=tags).info.run_id
        return self.run_id

    def end(self, status: str = "FINISHED") -> None:
        if self.enabled and self._mlflow.active_run():
            self._mlflow.end_run(status=status)

    def log_params(self, params: dict[str, Any]) -> None:
        if self.enabled:
            self._mlflow.log_params({k: str(v)[:500] for k, v in params.items()})

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if self.enabled:
            self._mlflow.log_metrics(
                {k: float(v) for k, v in metrics.items() if v is not None and v == v}, step=step
            )

    def log_dict(self, obj: dict, name: str) -> None:
        if self.enabled:
            self._mlflow.log_dict(obj, name)

    def log_artifact(self, path: Path, artifact_path: str | None = None) -> None:
        if self.enabled:
            self._mlflow.log_artifact(str(path), artifact_path)
