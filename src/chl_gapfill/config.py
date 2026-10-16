"""Experiment configuration (YAML -> pydantic) and machine paths (environment).

Two kinds of settings, kept apart on purpose:

* ``ExperimentConfig`` — everything that defines *what* was trained: region,
  sensors, windowing, masks, model, optimiser, splits, seed. Loaded from
  ``configs/*.yaml``, validated, hashed (``config_hash``) and logged with every run.
* ``Paths`` — *where* things are on this machine: data/model/output directories
  and the optional external inputs (compiled DINEOF, my Sentinel-3
  products). Read from environment variables / ``.env`` so nothing about a
  particular disk is committed.

The old ``config.py`` module constants map 1:1 onto these fields (see
``docs/MIGRATION.md``); the defaults below reproduce the published experiment.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from chl_gapfill import data_sources

# ---------------------------------------------------------------------------
# machine paths
# ---------------------------------------------------------------------------


def project_root() -> Path:
    if env := os.getenv("CHL_ROOT"):
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


class Paths(BaseModel):
    root: Path = Field(default_factory=project_root)
    data_dir: Path | None = None
    model_dir: Path | None = None
    output_dir: Path | None = None
    mlflow_tracking_uri: str = "sqlite:///mlruns/mlflow.db"
    # optional external inputs for the benchmark scripts (scripts/experiments/)
    dineof_exe: Path | None = None
    s3_sst_baseline_dir: Path | None = None
    s3_sst_dineof_l4_dir: Path | None = None
    s3_chl_mat_dir: Path | None = None

    @property
    def data(self) -> Path:
        return self.data_dir or self.root / "data"

    @property
    def models(self) -> Path:
        return self.model_dir or self.root / "models"

    @property
    def outputs(self) -> Path:
        return self.output_dir or self.root / "outputs"

    def outputs_for(self, cfg: ExperimentConfig) -> Path:
        """Default output directory for a run. `outputs/` holds the published real-data results,
        so smoke runs (2 epochs, usually on synthetic data) default to the gitignored
        `outputs_smoke/` instead; an explicit CHL_OUTPUT_DIR always wins."""
        if self.output_dir is None and cfg.smoke:
            return self.root / "outputs_smoke"
        return self.outputs

    def resolve(self, p: str | Path) -> Path:
        p = Path(p)
        return p if p.is_absolute() else self.root / p

    @model_validator(mode="after")
    def _anchor_sqlite(self) -> Paths:
        # MLflow reads a relative sqlite path against the current directory; anchor it at the
        # project root like the other default paths, so every working directory uses the same DB
        db = sqlite_file(self.mlflow_tracking_uri)
        if db is not None and not db.is_absolute():
            self.mlflow_tracking_uri = f"sqlite:///{(self.root / db).as_posix()}"
        return self

    @property
    def mlflow_artifact_root(self) -> Path | None:
        """Where new MLflow experiments keep their artifacts. With a database backend MLflow's
        default is ./mlruns under the *current directory*, whatever the tracking URI says; keep
        them next to a sqlite DB instead (under <root>/mlruns for another database). None: a
        tracking server or a file store decides."""
        db = sqlite_file(self.mlflow_tracking_uri)
        if db is not None:
            return db.parent / "artifacts"
        if self.mlflow_tracking_uri.split(":", 1)[0].split("+", 1)[0] in ("postgresql", "mysql", "mssql"):
            return self.root / "mlruns" / "artifacts"
        return None


def sqlite_file(uri: str) -> Path | None:
    """The database file of a `sqlite:///<path>` URI (None for other URIs and in-memory DBs)."""
    if not uri.startswith("sqlite:///"):
        return None
    rest = uri[len("sqlite:///") :]
    return None if rest in ("", ":memory:") else Path(rest)


def _env_path(name: str) -> Path | None:
    v = os.getenv(name)
    return Path(v).expanduser() if v else None


@lru_cache(maxsize=1)
def paths() -> Paths:
    root = project_root()
    try:
        from dotenv import load_dotenv

        load_dotenv(root / ".env", override=False)
    except ImportError:  # pragma: no cover
        pass
    return Paths(
        root=root,
        data_dir=_env_path("CHL_DATA_DIR"),
        model_dir=_env_path("CHL_MODEL_DIR"),
        output_dir=_env_path("CHL_OUTPUT_DIR"),
        mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns/mlflow.db"),
        dineof_exe=_env_path("DINEOF_EXE"),
        s3_sst_baseline_dir=_env_path("S3_SST_BASELINE_DIR"),
        s3_sst_dineof_l4_dir=_env_path("S3_SST_DINEOF_L4_DIR"),
        s3_chl_mat_dir=_env_path("S3_CHL_MAT_DIR"),
    )


def reset_paths_cache() -> None:
    paths.cache_clear()


# ---------------------------------------------------------------------------
# experiment configuration
# ---------------------------------------------------------------------------
class _Section(BaseModel):
    # a misspelt key is an error, not a silently ignored setting with an unchanged config_hash
    model_config = ConfigDict(extra="forbid")


class RegionConfig(_Section):
    lat_bounds: tuple[float, float] = (22.95, 24.55)  # 20 rows of the 1/12 deg grid
    lon_bounds: tuple[float, float] = (118.10, 119.75)  # 20 cols

    @field_validator("lat_bounds", "lon_bounds")
    @classmethod
    def _ordered(cls, v: tuple[float, float], info: ValidationInfo) -> tuple[float, float]:
        if v[0] >= v[1]:
            raise ValueError(f"bounds must be (min, max), got {v}")
        lo, hi = (-90.0, 90.0) if info.field_name == "lat_bounds" else (-180.0, 360.0)
        if v[0] < lo or v[1] > hi:
            raise ValueError(f"{info.field_name} must lie within [{lo}, {hi}], got {v}")
        return v


class DataConfig(_Section):
    train_nc: str = Field("Merged_VIIRS_CHL_Daily.nc", min_length=1)  # clean target, relative to Paths.data
    var_name: str = Field("chlor_a", min_length=1)
    input_sensors: list[str] | None = None  # None -> every 'input' sensor in the registry
    fusion: bool = True  # False -> first input sensor only (ablation)
    min_coverage: float = Field(0.2, ge=0.0, le=1.0)  # eligible day needs this much any-sensor cover
    # an input window must be consecutive calendar days (the target axis has missing days);
    # false = the original array-index windows (how the published per_satellite.json was made)
    contiguous_windows: bool = True
    # no input sensors: the model sees past days of the clean target itself. A leaky toy task for
    # experiments without sensor files, never for results; it must be asked for explicitly.
    self_supervised: bool = False
    # the period `chl-gapfill download` fetches for the target and every sensor (= the published files)
    start: date = date(2018, 5, 30)
    end: date = date(2023, 8, 31)

    @field_validator("input_sensors")
    @classmethod
    def _known_sensors(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return v
        valid = data_sources.input_sensors()
        if not v:
            raise ValueError(f"input_sensors is empty; use null for every input sensor ({valid})")
        if unknown := [k for k in v if k not in valid]:
            raise ValueError(f"unknown input sensor(s) {unknown}; valid: {valid}")
        if len(set(v)) != len(v):
            raise ValueError(f"input_sensors has duplicates: {v}")
        return v

    @model_validator(mode="after")
    def _consistent(self) -> DataConfig:
        if self.self_supervised and self.input_sensors is not None:
            raise ValueError("data.self_supervised uses no input sensors; leave data.input_sensors null")
        if self.start > self.end:
            raise ValueError(f"data.start {self.start} is after data.end {self.end}")
        return self


class MaskConfig(_Section):
    min_frac: float = Field(0.10, ge=0.0, le=1.0)
    max_frac: float = Field(0.40, ge=0.0, le=1.0)
    n_blobs: int = Field(3, ge=1)

    @model_validator(mode="after")
    def _order(self) -> MaskConfig:
        if self.min_frac > self.max_frac:
            raise ValueError("mask.min_frac must be <= mask.max_frac")
        return self


class ModelConfig(_Section):
    name: Literal["convlstm", "pconv_unet"] = "convlstm"
    hidden_channels: list[int] = Field([64, 32], min_length=1)
    kernel_size: int = Field(3, ge=1)
    use_norm: bool = True
    dropout: float = Field(0.0, ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def _buildable(self) -> ModelConfig:
        # the checks the model constructors would otherwise fail on mid-run (or not at all)
        hc = self.hidden_channels
        if self.name == "convlstm":
            if self.kernel_size % 2 == 0:
                raise ValueError(f"model.kernel_size must be odd ('same' padding), got {self.kernel_size}")
            if any(h < 2 or h % 2 for h in hc):
                raise ValueError(f"convlstm hidden_channels must be even and >= 2, got {hc}")
            if self.use_norm and any(h >= 8 and h % 8 for h in hc):
                raise ValueError(
                    f"with use_norm, hidden_channels >= 8 must be multiples of 8 (GroupNorm), got {hc}"
                )
        else:
            if (self.kernel_size, self.use_norm, self.dropout) != (3, True, 0.0):
                raise ValueError(
                    "pconv_unet has fixed 3x3 kernels and always normalises: model.kernel_size, "
                    "model.use_norm and model.dropout apply to convlstm only"
                )
            if max(16, hc[0] // 2) % 8:
                raise ValueError(
                    f"pconv_unet base width max(16, hidden_channels[0] // 2) must be a multiple of 8, got {hc}"
                )
        return self


class LossConfig(_Section):
    w_hole: float = Field(1.0, ge=0.0)
    w_valid: float = Field(0.1, ge=0.0)
    w_grad: float = Field(0.05, ge=0.0)


class TrainingConfig(_Section):
    lr: float = Field(1e-3, gt=0.0)
    n_epochs: int = Field(1000, ge=1)
    patience: int = Field(12, ge=1)
    batch_size: int = Field(16, ge=1)
    lr_schedule: bool = True  # ReduceLROnPlateau on the validation loss
    grad_clip: float | None = Field(None, gt=0.0)  # e.g. 1.0; None keeps the original behaviour
    num_threads: int | None = Field(None, ge=1)
    loss: LossConfig = LossConfig()


class SplitConfig(_Section):
    """Date-based split by TARGET day (inclusive bounds, as in the original config).

    `gap_days` removes validation/test target days whose input window would
    overlap the last training target day. The published numbers used 0.
    """

    train_end: date = date(2021, 12, 31)
    val_start: date = date(2022, 1, 1)
    val_end: date = date(2022, 3, 31)
    test_start: date = date(2022, 4, 1)
    test_end: date = date(2022, 7, 25)
    gap_days: int = Field(0, ge=0)

    @model_validator(mode="after")
    def _ordered(self) -> SplitConfig:
        if not (self.train_end < self.val_start <= self.val_end < self.test_start <= self.test_end):
            raise ValueError(
                "split dates must be ordered: train_end < val_start <= val_end < test_start <= test_end"
            )
        return self


class ExperimentConfig(_Section):
    name: str = Field("convlstm_fusion", pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")  # a directory name
    experiment: str = Field("chl-gapfill", min_length=1)  # MLflow experiment
    task: Literal["gapfill", "nextday"] = "gapfill"
    input_days: int = Field(3, ge=1)
    log_transform: bool = True
    seed: int = Field(42, ge=0, lt=2**32)  # numpy's legacy seeding range
    region: RegionConfig = RegionConfig()
    data: DataConfig = DataConfig()
    mask: MaskConfig = MaskConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    split: SplitConfig = SplitConfig()
    smoke: bool = False  # 2 epochs, tiny batch count: proves the loop, not the model

    @model_validator(mode="after")
    def _gap_covers_window(self) -> ExperimentConfig:
        if self.split.gap_days and self.split.gap_days < self.input_days - 1:
            raise ValueError(
                f"split.gap_days={self.split.gap_days} < input_days-1={self.input_days - 1}: "
                "a validation window would still include training target days"
            )
        return self

    # -- helpers --------------------------------------------------------------------
    def config_hash(self) -> str:
        canon = json.dumps(self.model_dump(mode="json", exclude={"name", "smoke"}), sort_keys=True)
        return hashlib.sha256(canon.encode()).hexdigest()[:12]

    def flat_params(self) -> dict[str, Any]:
        return flatten(self.model_dump(mode="json"))

    @property
    def image_size_hint(self) -> str:
        return f"lat{self.region.lat_bounds} lon{self.region.lon_bounds}"


def flatten(config: dict) -> dict[str, Any]:
    """{'training': {'lr': 1e-3}} -> {'training.lr': 1e-3}; lists and None become JSON strings."""
    out: dict[str, Any] = {}

    def walk(prefix: str, obj: Any) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(f"{prefix}.{k}" if prefix else str(k), v)
        else:
            out[prefix] = obj if isinstance(obj, int | float | str | bool) else json.dumps(obj, default=str)

    walk("", config)
    return out


def apply_overrides(data: dict, overrides: list[str]) -> dict:
    """`--set training.lr=3e-4` style dotted overrides (values parsed as YAML)."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item!r}")
        key, raw = item.split("=", 1)
        node = data
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                raise ValueError(f"override {item!r}: {p!r} is a value, not a section")
        node[parts[-1]] = yaml.safe_load(raw)
    return data


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> ExperimentConfig:
    data: dict = {}
    if path is not None:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{path}: top level must be a mapping")
    if overrides:
        data = apply_overrides(data, overrides)
    return ExperimentConfig.model_validate(data)


def to_datetime(d: date) -> datetime:
    return datetime(d.year, d.month, d.day)
