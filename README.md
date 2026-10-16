# Chlorophyll-a ConvLSTM — multi-sensor cloud gap-filling

A PyTorch deep-learning framework that **fills cloud gaps in satellite ocean-colour
(chlorophyll-a)** by fusing several polar-orbiting sensors. A ConvLSTM (and a
partial-convolution U-Net baseline) learn the spatiotemporal dynamics of chlorophyll and
reconstruct the field where clouds hide it, scored only on the hidden pixels, against
classical baselines and against DINEOF.

Study region: the **Taiwan Strait** (≈22.96–24.54°N, 118.13–119.71°E), a 20×20 daily grid,
2018-05-30 to 2023-08-31 (1 881 days).

## Results at a glance

log10 RMSE of chlorophyll on synthetic cloud holes, 50 held-out 2022 test days. A verdict
is a **paired day bootstrap** of the RMSE difference on the same pixels (1 000 resamples of
whole test days): *resolved* when its 95 % CI excludes 0. Every number below is in a
committed JSON under [`outputs/`](outputs/), produced by the commands in
[Reproduce](#reproduce-every-number); the full tables are in [`docs/RESULTS.md`](docs/RESULTS.md).

| Comparison (same pixels for both) | Scores | Difference [95 % CI] | Verdict |
|---|---|---|---|
| ConvLSTM vs persistence, truth = target (4 087 px) | 0.109 vs 0.143 | −0.034 [−0.050, −0.020] | ConvLSTM better |
| ConvLSTM vs spatial interpolation of all sensors (4 087 px) | 0.109 vs 0.223 | −0.114 [−0.139, −0.088] | ConvLSTM better |
| ConvLSTM vs persistence, truth = the withheld real observation (3 179 px) | 0.178 vs 0.162 | +0.016 [−0.001, +0.031] | unresolved |
| … on MODIS-observed pixels, truth = the MODIS observation (1 005 px) | 0.179 vs 0.146 | +0.033 [+0.018, +0.046] | persistence better |
| DINEOF vs ConvLSTM, both given the real sensor observations, truth = target (6 792 px) | 0.232 vs 0.137 | +0.095 [+0.069, +0.130] | ConvLSTM better |
| DINEOF given the target's values on those same observed pixels (6 792 px) | 0.132 vs 0.137 | −0.005 [−0.025, +0.013] | unresolved |
| DINEOF vs ConvLSTM, truth = the withheld real observation (4 300 px) | 0.205 vs 0.184 | +0.021 [−0.016, +0.074] | unresolved |
| NOAA-20-only model vs the fusion model (112 days, 14 625 px) | 0.179 vs 0.176 | +0.004 [−0.004, +0.011] | unresolved |
| MODIS-only / S-NPP-only model vs the fusion model (14 625 px) | 0.186 / 0.205 vs 0.176 | +0.010 [+0.003, +0.017] / +0.029 [+0.022, +0.039] | fusion better |

What this says:

- The **target** is NOAA's DINEOF gap-filled VIIRS product, not a raw observation (see
  [Data](#data)). Against it, the ConvLSTM clearly beats persistence, spatial interpolation and
  DINEOF fed the same sensor observations.
- That lead is mostly the ConvLSTM having **learned NOAA's product**. Scored against the withheld
  same-day observations instead, it is no better than persistence (worse on MODIS pixels), and
  DINEOF and the ConvLSTM cannot be told apart. The product itself sits 0.159 log10 RMS from
  those observations, more than the model's error against the product.
- With matched information, **no method is shown to be the better gap-filler on this grid**. The
  earlier headline "DINEOF 0.105 beats ConvLSTM 0.137" came from a leaky design and is withdrawn
  ([details](docs/RESULTS.md#dineof-vs-machine-learning)).
- On my Sentinel-3 series (July 2026 runs) DINEOF beats a purely spatial U-Net under realistic
  contiguous cloud holes: SST 1.34 vs 3.32 °C, OLCI chlorophyll 0.359 vs 0.425 (log10, with a
  one-mode DINEOF). The U-Net was trained on scattered pixels, not on contiguous holes.

## Credits and scope

This builds on an earlier next-day-prediction ConvLSTM prototype co-developed with a PhD
colleague; that prototype is kept locally and is not part of this repository. Its peephole
ConvLSTM cell, kept here in `model.py`, follows the structure of the open-source PyTorch
implementation [automan000/Convolutional_LSTM_PyTorch](https://github.com/automan000/Convolutional_LSTM_PyTorch).
The refactor here (a) fixes real bugs, (b) **pivots the task from next-day prediction to cloud
gap-filling**: the prototype could not beat a persistence baseline at next-day prediction,
because day-to-day chlorophyll is so autocorrelated that "tomorrow = today" is already excellent,
and a learned model is only useful where persistence is *unavailable*, under clouds; and
(c) **extends from a single product to multi-sensor fusion**.

**DINEOF** (Data Interpolating Empirical Orthogonal Functions) is the published method of
Beckers & Rixen (2003) and Alvera-Azcárate et al. (2005); the binary used here is the
open-source GHER code ([aida-alvera/DINEOF](https://github.com/aida-alvera/DINEOF)) compiled
locally. My part of the DINEOF work is:

- my Sentinel-3 SLSTR SST and OLCI chlorophyll processing chains (outside this repo), which
  produce the gappy daily fields and their DINEOF L4 reconstructions used in the Sentinel-3
  benchmarks;
- the Python driver [`scripts/experiments/dineof_runner.py`](scripts/experiments/dineof_runner.py)
  (NetCDF in/out, init file, reading DINEOF's own mode search from its log) and the benchmark
  designs in `scripts/experiments/compare_dineof_*.py`, run with the solver tolerances of my
  Sentinel-3 processors.

## Data

All VIIRS / MODIS sources are daily chlorophyll-a from **NOAA CoastWatch ERDDAP**, fetched by
`chl-gapfill download` ([`src/chl_gapfill/download.py`](src/chl_gapfill/download.py); details
and the file list in [`data/README.md`](data/README.md)).

| Role | Product | Notes |
|---|---|---|
| **Target** | NOAA STAR's **DINEOF gap-filled** VIIRS S-NPP + NOAA-20 chlorophyll, Science Quality, 9 km (`nesdisVHNnoaaSNPPnoaa20chlaGapfilledDaily`; `Merged_VIIRS_CHL_Daily.nc`) | the file's own title is "Chlorophyll (Gap-filled DINEOF), NOAA S-NPP NOAA-20, VIIRS, Science Quality, Global 9km"; method: Liu & Wang (2018) |
| Input (real gaps) | **MODIS-Aqua** (`erdMH1chla1day`) | Science Quality 4 km, ends 2022-07 |
| Input (real gaps) | **VIIRS S-NPP** (`nesdisVHNSQchlaDaily`) | not gap-filled, 4 km |
| Input (real gaps) | **VIIRS NOAA-20** (`nesdisVHNnoaa20chlaDaily`) | not gap-filled, from 2021-08 |

> **Correction.** Earlier versions of this README, including the July 2026 public one, called
> the target a ready-made product rather than a DINEOF reconstruction. That was wrong: its own
> metadata (title above; `chlor_a:long_name = "Chlorophyll Concentration, DINEOF Gap-Filled"`)
> say it is NOAA's DINEOF product. Consequences: on hole pixels that no sensor saw that day
> (37 % of them) the "truth" is a DINEOF fill, and a DINEOF benchmark scored against this target
> is partly circular. Both are handled below.

The coastal raw products are heavily clouded: together the three sensors observe only about a
fifth of the pixel-days of the 1 881-day record (21.8 % of the cube DINEOF gets in
`dineof_vs_ml_chl.json`, test holes removed). Fusion matters because each sensor is clouded at
different times.

## Method and evaluation design

For a target day *t* the model receives days *t*−2..*t* of every input sensor, each as an
observation channel plus a validity-mask channel, with a **synthetic cloud hole** punched into
day *t* across all sensors. It must reconstruct the target inside the hole. The loss (and every
metric) is computed **only on the hole pixels**
([`src/chl_gapfill/losses.py`](src/chl_gapfill/losses.py): masked MSE + light gradient term).
Chlorophyll is log-normal, so training and scoring are in log10 space; each sensor is z-scored
with its own training-period statistics.

Evaluation caveats (all quantified in [`outputs/metrics.json`](outputs/metrics.json)):

- **Synthetic holes.** Each test day gets 3 random ellipses covering 10–40 % of the grid
  (seeded by the day, so every method sees the same holes); real cloud masks from other days
  are not transplanted.
- **The truth is a DINEOF product.** 36.7 % of the scored hole pixels had no observation from
  any sensor that day, so their truth is purely NOAA's DINEOF fill; 56.0 % were seen by VIIRS.
  Where a sensor did see the pixel, the target still differs from that observation by 0.159
  log10 RMS (bias +0.040). Hence the slices scored against the withheld observation itself
  (each sensor shifted by its mean training-period offset to the target, at most 0.013 log10).
- **The cloudiest days are excluded** from the main table (`data.min_coverage: 0.2` keeps 50 of
  the 112 test days). With every test day ([`outputs/all_days/metrics.json`](outputs/all_days/metrics.json)):
  ConvLSTM 0.175 [0.157, 0.192] on 14 625 hole pixels; on the 5 695 pixels every method fills,
  ConvLSTM 0.115 vs persistence 0.152, difference −0.037 [−0.052, −0.023].
- **The "independence slice" is not independent.** It is the hole pixels MODIS-Aqua observed
  (1 354), but its truth is still the VIIRS product and 856 of them (63 %) were also seen by
  VIIRS. Scored against the MODIS observation itself, persistence beats the model
  (table above).
- **Persistence** (the most recent observation in the 3-day window, any sensor) has no value on
  40 % of the hole pixels, so every model-vs-persistence verdict uses the pixels both fill.
- **Split boundary.** The published config uses `split.gap_days: 0`, so the first validation
  input window contains the last training target day (the CLI prints a warning). It makes no
  difference here: a retrain with `--set split.gap_days=2` drops two validation days, keeps the
  same 50 test days and ends with byte-identical weights to the `gap_days: 0` retrain
  (sha256 `6ef5c847…`, [below](#reproducibility-and-compute)).

## Quick start (clone → train → evaluate → predict)

Synthetic 8×8 data, CPU only, no data download. The commands are bash (Linux, macOS, WSL, or Git Bash
on Windows, where the venv activates with `source .venv/Scripts/activate`); CI runs this block verbatim.
On Windows keep the clone's path under about 120 characters (or enable long paths): some MLflow files
inside `.venv` otherwise pass the 260-character path limit and the MLflow test fails.

<!-- quickstart:begin -->
```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -e ".[dev,track]"
python scripts/make_synthetic_data.py --out data_synth           # real data: see data/README.md
CHL_DATA_DIR=$PWD/data_synth chl-gapfill train    -c configs/synthetic_smoke.yaml --no-mlflow
CHL_DATA_DIR=$PWD/data_synth chl-gapfill evaluate -c configs/synthetic_smoke.yaml
CHL_DATA_DIR=$PWD/data_synth chl-gapfill predict  -c configs/synthetic_smoke.yaml --date 2022-04-20 --png
pytest -q                                                         # unit + training smokes on synthetic fixtures
```
<!-- quickstart:end -->

Smoke runs write to `outputs_smoke/` (gitignored), so the published results in `outputs/` stay as they are.
The full test suite takes about a minute on my 16-thread CPU, most of it in the training smokes and
MLflow's first start; `pytest -m "not slow"` skips the training smokes and takes about half that.

With the real data in `data/` (see `data/README.md`), a retrain from scratch:

```bash
chl-gapfill train    -c configs/convlstm.yaml                    # MLflow: mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
chl-gapfill train    -c configs/convlstm.yaml --resume models/convlstm_fusion/last.pt   # after an interruption
chl-gapfill evaluate -c configs/convlstm.yaml --ckpt models/convlstm_fusion.pth --no-figures --out outputs/retrain
chl-gapfill predict  -c configs/convlstm.yaml --date 2022-06-12 --out outputs/retrain/filled_2022-06-12.nc --png
chl-gapfill config   -c configs/convlstm.yaml                    # effective config + config_hash + paths
```

A retrain's results go to `outputs/retrain/`, so the published ones in `outputs/` stay as they are;
`outputs/retrain/metrics.json` is my own retrain ([below](#reproducibility-and-compute)).

### Reproduce every number

The published checkpoint is `models/convlstm_latest.pth` (not in git; see
[Reproducibility](#reproducibility-and-compute)).

```bash
chl-gapfill evaluate -c configs/convlstm.yaml --ckpt models/convlstm_latest.pth        # outputs/metrics.json + figures
chl-gapfill evaluate -c configs/convlstm.yaml --ckpt models/convlstm_latest.pth \
    --set data.min_coverage=0 --no-figures --out outputs/all_days                      # outputs/all_days/metrics.json
python scripts/experiments/compare_dineof_chl.py models/convlstm_latest.pth            # outputs/dineof_vs_ml_chl.json (DINEOF_EXE)
python scripts/experiments/compare_sensors.py                                          # outputs/per_satellite.json
python run_all.py --quick --ckpt models/convlstm_latest.pth                            # all of the above
chl-gapfill predict -c configs/convlstm.yaml --ckpt models/convlstm_latest.pth \
    --date 2022-06-12 --out outputs/filled_2022-06-12.nc --png                         # outputs/filled_2022-06-12.png
```

`compare_sensors.py` needs the per-satellite models from `python scripts/experiments/train_single.py`;
the DINEOF scripts need `DINEOF_EXE` in `.env`, and the Sentinel-3 ones also `S3_SST_BASELINE_DIR`,
`S3_SST_DINEOF_L4_DIR`, `S3_CHL_MAT_DIR`. `run_all.py` skips a stage whose inputs are missing.
`chl-gapfill train -c configs/pconv_unet.yaml` trains the partial-convolution U-Net on the same
data and holes.

## Reproducibility and compute

- **Provenance.** `metrics.json`, `all_days/metrics.json`, `retrain/metrics.json`,
  `dineof_vs_ml_chl.json` and `per_satellite.json` carry a `provenance` block: the checkpoint's
  file name and sha256, the effective config and its `config_hash`, the sha256 of every data
  file, the data window and the test days, and the package versions. The published checkpoint is
  `convlstm_latest.pth`, sha256 `f25ca9295c3aed17867c571d086faaf798d2491f2a600856c4e4e1091571d9ec`,
  config `configs/convlstm.yaml` (`config_hash` 68ad484fba3c), data 2018-05-30 to 2023-08-31
  (1 881 days), test days 2022-04-03 to 2022-07-25. Checkpoints are not in git (`models/` is
  ignored).
- **Determinism.** Fixed `seed`; evaluation holes are seeded by the day. Re-running the evaluation
  gives a byte-identical `metrics.json` on one machine, and the numbers published before this
  revision reproduced to within 3e-8 in this environment. Training is reproducible too (the
  training-hole generator is seeded; `tests/test_train_smoke.py::test_same_seed_same_result`): two
  retrains of the published config on my machine gave byte-identical weights. `--resume` continues
  a run bit for bit on CPU.
- **A retrain gives 0.133.** Retraining from scratch with this package (the commands above; early
  stopping at epoch 45, 6–7 min on my CPU) gives ConvLSTM 0.133 [0.117, 0.149] (R² 0.76) on the
  same 50 days and 6 792 hole pixels ([`outputs/retrain/metrics.json`](outputs/retrain/metrics.json),
  weights sha256 `6ef5c847…`). The verdicts do not change: on the pixels both fill it beats
  persistence against the target, −0.042 [−0.057, −0.028]; against the withheld observation it
  does not (+0.009 [−0.009, +0.025]), and on the MODIS pixels persistence is better
  (+0.025 [+0.007, +0.040]). The tables stay on the published checkpoint.
- **Why 0.137 and not the July release's 0.130.** The July 2026 public README reported ConvLSTM
  0.130 (R² 0.77). That came from an earlier checkpoint (sha256 `8305db95…`) trained while every
  input sensor was z-scored with the *target's* statistics, which shifted MODIS by the VIIRS mean.
  After the fix (each sensor normalised by its own training-period statistics) the model was
  retrained; the retrained checkpoint scores 0.137 [0.120, 0.154]. Both runs use the same 50 days
  and 6 792 hole pixels (their persistence rows are identical, the interpolation rows agree to
  0.001); 0.130 lies inside the CI, and training holes were not yet seeded, so a retrain alone
  moves this number. Re-running the July release's own `evaluate.py` on its checkpoint gives
  0.1297 again ([`outputs/july_release_metrics.json`](outputs/july_release_metrics.json)). The
  July README's "6 912 hole pixels" was a typo for 6 792.
- **Compute: no GPU needed.** CPU-only PyTorch. The published ConvLSTM has 387 681 parameters
  (a test pins it near 387 K), the partial-conv U-Net about 0.25 M; grids are 20×20 (VIIRS),
  200×200 (Sentinel-3 OLCI) and 183×167 (Sentinel-3 SST). One epoch of the published config takes
  about 7 s on my 16-thread CPU. One DINEOF run on the 1 881 × 20 × 20 cube takes 10–20 s.
  [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb) runs the same CLI on a free Colab
  GPU (synthetic data by default); no GPU timing is recorded yet.

## Detailed results

[`docs/RESULTS.md`](docs/RESULTS.md) has every table behind the summary above: each method on
its own pixels and all of them on the same pixels, the slices scored against the target and
against the withheld observation, the per-satellite models, and the three DINEOF benchmarks
(VIIRS chlorophyll with the same information for both, Sentinel-3 OLCI chlorophyll and
Sentinel-3 SST).

## Structure
```
configs/                 convlstm.yaml (= the published experiment), pconv_unet.yaml, *_smoke.yaml
src/chl_gapfill/
  config.py              ExperimentConfig (YAML → pydantic, validated, hashed) + Paths (env / .env)
  data_sources.py        sensor registry (ERDDAP ids, roles)
  download.py            ERDDAP griddap downloader (yearly-chunked, retrying)
  align.py               regrid every sensor onto the common grid + daily axis
  dataset.py             GapFillDataset (multi-sensor, synthetic contiguous holes, train-only stats)
  losses.py              masked reconstruction loss (+ valid + gradient terms)
  model.py               ConvLSTM (peephole, sequence mode) + PartialConvUNet + factory
  train.py               fit(): seeded, best/last checkpoints with optimizer and RNG state, resume, MLflow
  evaluate.py            hole-pixel skill vs baselines, day bootstrap CIs, paired same-pixel verdicts, slices
  provenance.py          checkpoint / config / data hashes written into the results JSONs
  predict.py             one day → NetCDF (+ png); synthetic hole or operational fill
  figures.py, tracking.py, cli.py
scripts/experiments/     train_single, compare_sensors, compare_dineof_{chl,s3chl,sst}, dineof_runner,
                         sst_data, s3_chl_data (DINEOF / Sentinel-3 benchmarks; DINEOF_EXE / S3_* in .env)
scripts/make_synthetic_data.py   8x8 synthetic multi-sensor set for CI and smoke runs
outputs/                 the published results (JSON + figures); outputs/all_days/ = every test day,
                         outputs/retrain/ = a retrain of the published config
tests/                   config, data/alignment, model/losses, split/bootstrap, DINEOF driver, training
                         smokes (resume, determinism, MLflow, CLI, NetCDF output), and the docs' links
                         and quoted results against outputs/*.json
notebooks/train_colab.ipynb      the CLI on a free Colab GPU
docs/RESULTS.md          every results table (this README has the summary)
docs/MIGRATION.md        old config.py constants / scripts → new config fields / commands
Dockerfile, .github/workflows/ci.yml, .env.example
```

## What changed from the prototype

- **Task pivot**: next-day prediction → masked cloud gap-filling.
- **Multi-sensor fusion**: MODIS-Aqua + VIIRS S-NPP + VIIRS NOAA-20 as inputs, each with its own
  cloud mask, regridded to a common grid and daily axis.
- **Reproducible data pipeline**: geographic region bounds instead of hard-coded array indices;
  one ERDDAP downloader for every source.
- **Per-sensor normalisation**: each sensor is z-scored by its own training-period log10
  statistics, not the target's.
- **Honest evaluation**: metrics on hole pixels only; day bootstrap CIs; paired verdicts on the
  same pixels; slices scored against real observations; genuine per-satellite models instead of
  input ablation; a DINEOF benchmark with matched information.
- **Engineering**: `chl_gapfill` package with a validated config, CLI, bit-exact resume, strict
  `weights_only` checkpoint loading, MLflow tracking, tests and CI.
- Kept from the prototype's fixes: device-safe (CPU/GPU), registered peephole weights, log-space
  training, de-normalising `inverse()`.

## References

- Alvera-Azcárate, A., Barth, A., Rixen, M., & Beckers, J.-M. (2005). Reconstruction of incomplete
  oceanographic data sets using empirical orthogonal functions: application to the Adriatic Sea
  surface temperature. *Ocean Modelling*, 9(4), 325–346. https://doi.org/10.1016/j.ocemod.2004.08.001
- Beckers, J.-M., & Rixen, M. (2003). EOF calculations and data filling from incomplete
  oceanographic datasets. *Journal of Atmospheric and Oceanic Technology*, 20(12), 1839–1856.
  https://doi.org/10.1175/1520-0426(2003)020<1839:ECADFF>2.0.CO;2
- Liu, X., & Wang, M. (2018). Gap filling of missing data for VIIRS global ocean color products
  using the DINEOF method. *IEEE Transactions on Geoscience and Remote Sensing*, 56(8), 4464–4476.
  https://doi.org/10.1109/TGRS.2018.2820423
- Liu, G., Reda, F. A., Shih, K. J., Wang, T.-C., Tao, A., & Catanzaro, B. (2018). Image inpainting
  for irregular holes using partial convolutions. *ECCV 2018*. https://arxiv.org/abs/1804.07723
- Shi, X., Chen, Z., Wang, H., Yeung, D.-Y., Wong, W.-K., & Woo, W.-C. (2015). Convolutional LSTM
  network: a machine learning approach for precipitation nowcasting. *NeurIPS 2015*.
  https://arxiv.org/abs/1506.04214
- Data: NOAA CoastWatch ERDDAP (VIIRS, MODIS-Aqua); MODIS-Aqua from NASA OBPG.
