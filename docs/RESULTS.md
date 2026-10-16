# Detailed results

The tables behind the summary in the [README](../README.md#results-at-a-glance). Every number is
in a committed JSON under [`outputs/`](../outputs/); the evaluation caveats are in the README's
[method section](../README.md#method-and-evaluation-design) and the commands that regenerate each
file in [Reproduce every number](../README.md#reproduce-every-number).

## The ConvLSTM and its baselines (VIIRS grid)

### Main table

Held-out **2022** test days (MODIS-Aqua and both VIIRS sensors coexist), hole pixels only;
mean same-day coverage 38.7 % for the best single sensor and 40.6 % fused. Each method's own row
keeps every pixel it can fill, so persistence has fewer. Methods: `persistence` (temporal
carry-forward), `interp_single` / `interp_fusion` (spatial interpolation from the best single
sensor / all sensors), `model_ablate_single` (the fusion model with all but one sensor zeroed at
inference: a train/test mismatch, *not* a single-sensor model), `model` (the multi-sensor
ConvLSTM). From [`outputs/metrics.json`](../outputs/metrics.json):

| Method | n | log10 RMSE [95 % CI] ↓ | log10 R² ↑ | chl RMSE ↓ |
|---|---|---|---|---|
| persistence | 4 087 | 0.143 [0.132, 0.156] | 0.71 | 0.574 |
| interp_single | 6 792 | 0.275 [0.239, 0.310] | −0.03 | 1.83 |
| interp_fusion | 6 792 | 0.260 [0.228, 0.292] | 0.08 | 1.70 |
| model_ablate_single | 6 792 | 0.153 [0.138, 0.168] | 0.68 | 0.494 |
| model (ConvLSTM, fusion) | 6 792 | 0.137 [0.120, 0.154] | 0.74 | 0.496 |

On the 4 087 hole pixels every method fills (`log10_space_matched`, `verdicts`):

| Baseline | baseline log10 RMSE | ConvLSTM | ConvLSTM − baseline [95 % CI] |
|---|---|---|---|
| persistence | 0.143 | 0.109 | −0.034 [−0.050, −0.020] |
| interp_single | 0.244 | 0.109 | −0.135 [−0.186, −0.095] |
| interp_fusion | 0.223 | 0.109 | −0.114 [−0.139, −0.088] |
| model_ablate_single | 0.131 | 0.109 | −0.022 [−0.040, −0.004] |

Against the target, the ConvLSTM beats every baseline on the same pixels; its edge over
persistence was understated before, when the two were compared on different pixel sets. The
ablation row is resolved too but is not evidence for fusion (train/test mismatch); the
per-satellite models below are.

Slices, model vs persistence on the pixels both fill (`observed_slice`, `independence_slice`):

| Slice | Truth | n | ConvLSTM | Persistence | ConvLSTM − persistence [95 % CI] |
|---|---|---|---|---|---|
| hole px some sensor saw | target | 3 179 | 0.100 | 0.137 | −0.037 [−0.051, −0.023] |
| hole px some sensor saw | that observation | 3 179 | 0.178 | 0.162 | +0.016 [−0.001, +0.031] |
| hole px MODIS saw | target | 1 005 | 0.093 | 0.137 | −0.044 [−0.053, −0.033] |
| hole px MODIS saw | the MODIS observation | 1 005 | 0.179 | 0.146 | +0.033 [+0.018, +0.046] |

The earlier claim that the MODIS slice shows the skill "is not merely VIIRS self-copy" does not
hold: against the MODIS observation itself, persistence is better. The ConvLSTM reproduces
NOAA's smooth 9-km DINEOF product; persistence carries the sensors' own 4-km values forward,
which lands closer to the withheld same-day observation.

### Per-satellite: genuine single-sensor models

A *dedicated* ConvLSTM per satellite, all scored on one identical set of days and holes
(`min_coverage=0`, 112 test days; [`outputs/per_satellite.json`](../outputs/per_satellite.json),
figure [`per_satellite.png`](../outputs/per_satellite.png), where the persistence bar is hatched
because it covers only the pixels persistence fills):

| Model | log10 RMSE [95 % CI] ↓ | log10 R² ↑ | single − fusion [95 % CI] |
|---|---|---|---|
| single: VIIRS S-NPP | 0.205 [0.192, 0.220] | 0.52 | +0.029 [+0.022, +0.039] |
| single: MODIS-Aqua | 0.186 [0.172, 0.200] | 0.60 | +0.010 [+0.003, +0.017] |
| single: VIIRS NOAA-20 | 0.179 [0.164, 0.195] | 0.63 | +0.004 [−0.004, +0.011] |
| fusion (all three) | 0.176 [0.161, 0.190] | 0.64 | |

*(14 625 hole pixels.)* Fusion is nominally best: resolved against the MODIS-only and S-NPP-only
models, unresolved against NOAA-20 alone. Persistence fills 6 552 of these pixels; there it scores
0.156 against the fusion model's 0.127, a difference of +0.029 [+0.016, +0.043]. This table now
has 112 test days instead of 114, because input windows must be consecutive calendar days
(`data.contiguous_windows`) and two test days straddled a gap in the record.

**Fusion and coverage.** Fusing these three *early-afternoon polar-orbiting* sensors raises
same-day coverage only from 38.7 % to 40.6 %, because they see nearly the same clouds at
nearly the same local time. Large coverage gains need a *geostationary* sensor (Himawari) or
*microwave*. Here fusion helps mostly **temporally** (different sensors fill different days of
the input window).

## DINEOF vs machine learning

### (a) VIIRS chlorophyll, 20×20, 1 881 days: the same information for both

The original benchmark gave DINEOF the gap-free target with only the 50 test holes removed:
DINEOF saw 99 % of the truth, which is itself NOAA's DINEOF reconstruction, while the ConvLSTM
saw only the gappy sensors. Its "DINEOF 0.105 vs ConvLSTM 0.137" (with the mode search stopped at
its cap of 20) measured that leak, not gap-filling skill. The redesigned
[`compare_dineof_chl.py`](../scripts/experiments/compare_dineof_chl.py) runs DINEOF on three input
cubes, every test hole removed, and scores every method on the same 6 792 hole pixels
([`outputs/dineof_vs_ml_chl.json`](../outputs/dineof_vs_ml_chl.json)):

| DINEOF input | DINEOF | ConvLSTM | DINEOF − ConvLSTM [95 % CI] | EOF modes (cap) |
|---|---|---|---|---|
| the real sensor observations (21.8 % of the cube): what the ConvLSTM gets | 0.232 [0.205, 0.263] | 0.137 [0.120, 0.154] | +0.095 [+0.069, +0.130] | 35 (50) |
| the target's values on exactly those pixels | 0.132 [0.115, 0.148] | 0.137 | −0.005 [−0.025, +0.013] | 47 (50) |
| the gap-free target (99.1 %): the original design, a leaky upper bound | 0.100 [0.080, 0.125] | 0.137 | −0.037 [−0.064, −0.008] | 46 (50) |

On the 4 300 hole pixels some sensor saw, scored against that withheld observation instead of the
target: ConvLSTM 0.184 [0.168, 0.201], DINEOF on the observations 0.205 [0.170, 0.254], a
difference of +0.021 [−0.016, +0.074]; no design differs from the ConvLSTM there.

How to read it: with the same observations, DINEOF is far behind *against the target*, but that
gap is calibration rather than gap-filling. The ConvLSTM learned the mapping from these sensors to
NOAA's product during training; DINEOF can only reconstruct the field it is given. Given the
target's values on the same pixels, DINEOF ties; against the real observations, nothing is
resolved. On this grid no method is shown to be the better gap-filler. DINEOF also has an
advantage the ConvLSTM lacks: it builds its EOFs from the whole record, future days included,
while the ConvLSTM sees days *t*−2..*t* only. No mode search hit its cap (cross-validation chose
35, 47 and 46 modes out of 50).

### (b) Sentinel-3 OLCI chlorophyll, 200×200, 30 days

My Sentinel-3 OLCI CHL_NN product at ~0.1°, all of April 2025, contiguous blob holes on
observed pixels ([`outputs/dineof_vs_ml_s3chl.json`](../outputs/dineof_vs_ml_s3chl.json)):

| Method | log10 RMSE [95 % CI] ↓ | log10 R² ↑ |
|---|---|---|
| ML (partial-conv U-Net, spatial only) | 0.425 [0.396, 0.445] | −0.13 |
| DINEOF | 0.359 [0.324, 0.390] | 0.19 |

Both have little skill on this short, sparse series. DINEOF's cross-validation kept a single EOF
mode (its error rose from mode 2 on), so its fill is one spatial pattern scaled day by day.

### (c) Sentinel-3 SST, 183×167, 30 days

My strict-QC Sentinel-3 SLSTR SST at ~4 km, April 2025; held-out real observations
([`outputs/dineof_vs_ml_sst.json`](../outputs/dineof_vs_ml_sst.json),
[`outputs/dineof_vs_ml_sst_blob.json`](../outputs/dineof_vs_ml_sst_blob.json)):

| SST hold-out | DINEOF RMSE (°C) [95 % CI] | ML PConv U-Net RMSE (°C) [95 % CI] |
|---|---|---|
| scattered pixels (easy) | 0.974 [0.826, 1.094] | 0.364 [0.323, 0.399] |
| contiguous cloud blobs (realistic) | 1.338 [1.073, 1.648] | 3.319 [2.406, 4.133] (R² < 0) |

With *scattered* held-out pixels every hidden pixel has an observed same-day neighbour, so a
purely spatial inpainter "wins" by local copying. With *contiguous cloud-shaped* holes the
spatial U-Net collapses (worse than the mean) while DINEOF, which borrows structure across the
whole series, holds up: evaluation design flips the conclusion. DINEOF kept 6 modes on the
scattered run (its own cross-validation error, 0.987 °C, matches the measured 0.974 °C) and
8 on the blob run.

One confound remains in both Sentinel-3 benchmarks: the U-Net trains on a random half of the
observed pixels, also when it is scored on blobs, so it never sees a contiguous hole in training.
`--train-masks blob` (both scripts) trains it on blob-shaped targets instead; that run has not been
made yet.

The Sentinel-3 numbers come from runs made in July 2026; the inputs are not on the machine used
for this revision, so they were not re-run, and the mode counts come from those runs' DINEOF logs.
The scripts now record the mode search in their JSON.

**Overall.** Under realistic contiguous holes, DINEOF beats a purely spatial U-Net trained on
scattered targets on my 30-day Sentinel-3 series (marginal 95 % CIs do not overlap; no paired
test was run). On the VIIRS
grid, where the truth is NOAA's own DINEOF product, neither DINEOF nor the ConvLSTM is shown to
be better once both get the same information. A temporal model on a longer Sentinel-3 record is
the natural next comparison.
