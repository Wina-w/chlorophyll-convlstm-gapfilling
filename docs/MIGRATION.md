# Migration: research scripts → `chl_gapfill` package

The model and its main evaluation (`outputs/metrics.json`) are unchanged; the engineering
around them is not, and the DINEOF and per-satellite comparisons were redesigned (see the end
of this page).
This page maps the old entry points and constants to the new ones so old notes stay usable.

## Entry points

| Before | After |
|---|---|
| edit `config.py`, `python src/train.py` | `chl-gapfill train -c configs/convlstm.yaml` (`--set training.lr=3e-4` to override) |
| `python src/evaluate.py models/x.pth` | `chl-gapfill evaluate -c configs/convlstm.yaml --ckpt models/convlstm_fusion/best.pt` |
| `python src/predict.py ckpt 40` (test position → PNG) | `chl-gapfill predict -c … --date 2022-06-12 --out filled.nc --png` |
| `python src/download.py` | `chl-gapfill download` |
| `python src/align.py` | `chl-gapfill align` |
| `python src/train_single.py` | `python scripts/experiments/train_single.py` |
| `python src/compare_*.py` | `python scripts/experiments/compare_*.py` (need `DINEOF_EXE` / `S3_*` in `.env`) |
| `python run_all.py` | `python run_all.py` (now a thin wrapper over the CLI; `--quick [--ckpt …]` skips retraining, exit 1 if a stage fails) |

## Constants → config fields (`configs/convlstm.yaml` == the old defaults)

| `config.py` | `ExperimentConfig` |
|---|---|
| `LAT_BOUNDS`, `LON_BOUNDS` | `region.lat_bounds`, `region.lon_bounds` |
| `TRAIN_NC`, `VAR_NAME` | `data.train_nc`, `data.var_name` (relative to `CHL_DATA_DIR`) |
| `TEST_NC` | removed: it was never read. The test set is the `split.test_start..test_end` slice of `data.train_nc` |
| `INPUT_SENSORS`, `FUSION` | `data.input_sensors` (null = registry inputs), `data.fusion` |
| `TASK`, `MODEL` | `task`, `model.name` |
| `INPUT_DAYS`, `LOG_TRANSFORM`, `SEED` | `input_days`, `log_transform`, `seed` |
| `MASK_MIN_FRAC/MAX_FRAC/N_BLOBS` | `mask.min_frac/max_frac/n_blobs` |
| `HIDDEN_CHANNELS`, `KERNEL_SIZE`, `USE_NORM`, `DROPOUT` | `model.*` |
| `LR`, `N_EPOCHS`, `PATIENCE`, `BATCH_SIZE`, `LR_SCHEDULE` | `training.*` (+ `training.grad_clip`, `training.loss.*`) |
| `TRAIN_END`, `VAL_RANGE`, `TEST_RANGE` | `split.train_end`, `split.val_start/val_end`, `split.test_start/test_end` (+ `split.gap_days`) |
| `DATA_DIR`, `MODEL_DIR`, `OUTPUT_DIR` | env `CHL_DATA_DIR`, `CHL_MODEL_DIR`, `CHL_OUTPUT_DIR` (`.env`) |
| `MODEL_OUT` (`models/convlstm_latest.pth`) | `models/<name>/best.pt` (full checkpoint) + `models/<name>.pth` (bare state_dict, legacy format) |
| `src/dineof_runner.py: DINEOF_EXE` | env `DINEOF_EXE` |
| `src/sst_data.py: BASE_DIR, L4_DIR` | env `S3_SST_BASELINE_DIR`, `S3_SST_DINEOF_L4_DIR` |
| `src/s3_chl_data.py: MAT_DIR` | env `S3_CHL_MAT_DIR` |

## Behavioural differences (all opt-in or documented)

- **Training masks are now reproducible.** The old training loop drew synthetic holes
  from an unseeded generator, so two runs with the same `SEED` differed. Now a
  dataset-level generator is seeded once (`seed + 10007`). Evaluation holes were and
  are seeded by the day index, so published evaluation numbers are unaffected.
- **Checkpoints carry optimizer/scheduler/epoch/history**, the early-stopping counter and
  every RNG stream (python, numpy, torch, the DataLoader shuffle and the training-hole
  generator) in `best.pt` / `last.pt`, so on CPU `--resume` continues the interrupted run
  bit for bit (a test checks 2 + 2 epochs against 4 straight). Only `training.n_epochs`,
  `training.patience`, `smoke`, `name` and `experiment` may change on resume. The bare
  `.pth` is still written for the benchmark scripts.
- **`split.gap_days`** (default 0 = published behaviour) can enforce a gap so no
  validation/test input window contains a training target day.
- **`training.grad_clip`** (default null = published behaviour).
- Old checkpoints (`models/*.pth`) load unchanged through `load_weights`.
- **`evaluate` adds, without changing any existing number:** the same-pixel table and paired
  verdicts, the `observed` / `independence` slices compared on the same pixels against the target
  and against the withheld observation, `sensor_offsets_log10`, and a `provenance` block
  (checkpoint sha256, effective config, data files and window, versions).
- **`compare_dineof_chl.py` is a different benchmark now.** The old design fed DINEOF the gap-free
  target (a leak); DINEOF now gets the same sensor observations as the ConvLSTM, with the old
  design kept only as a labelled upper bound. Its JSON has a new layout.
- **`compare_sensors.py`** adds paired single-minus-fusion differences and uses consecutive-day
  windows (`data.contiguous_windows`), so it has 112 test days instead of 114.
- `run_all.py` also writes `outputs/all_days/metrics.json` (every test day, `data.min_coverage=0`).
- **`compare_dineof_sst.py` / `compare_dineof_s3chl.py`** take `--train-masks blob` to train the
  U-Net on blob-shaped targets (written to `*_trainblob.*`); the default is the published
  scattered-pixel training.
