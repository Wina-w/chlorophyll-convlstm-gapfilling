# data/

Not committed (see `.gitignore`). Everything comes from public NOAA CoastWatch ERDDAP.

| File | Role | Source | How to get it |
|---|---|---|---|
| `Merged_VIIRS_CHL_Daily.nc` | target: NOAA's DINEOF gap-filled VIIRS S-NPP + NOAA-20 chlorophyll, 9 km | ERDDAP `nesdisVHNnoaaSNPPnoaa20chlaGapfilledDaily` | `chl-gapfill download` |
| `modis_aqua.nc` | input (real gaps) | ERDDAP `erdMH1chla1day` | `chl-gapfill download` |
| `viirs_snpp.nc` | input (real gaps) | ERDDAP `nesdisVHNSQchlaDaily` | `chl-gapfill download` |
| `viirs_noaa20.nc` | input (real gaps) | ERDDAP `nesdisVHNnoaa20chlaDaily` | `chl-gapfill download` |
| `olci_s3_dineof.nc` | reference only (not used by the model or the benchmarks) | ERDDAP `noaacwNPPN20S3ASCIDINEOF2kmDaily` | `chl-gapfill download` |

The target is a DINEOF reconstruction (Liu & Wang 2018, IEEE TGRS 56(8)), not a raw
observation. Its own metadata say so: the global `title` is "Chlorophyll (Gap-filled DINEOF),
NOAA S-NPP NOAA-20, VIIRS, Science Quality, Global 9km, 2018-recent, Daily",
`chlor_a:long_name` is "Chlorophyll Concentration, DINEOF Gap-Filled", and the `history`
records the request to `coastwatch.noaa.gov` dataset `noaacwNPPN20VIIRSSCIDINEOFDaily`, which
the pfeg server below republishes as `nesdisVHNnoaaSNPPnoaa20chlaGapfilledDaily`. Where no
sensor saw a pixel, its value is a DINEOF fill; the main README says how the evaluation deals
with that.

`chl-gapfill download -c configs/convlstm.yaml` fetches the target first, then every
sensor, cropped to `region` and limited to `data.start`..`data.end` (2018-05-30 to
2023-08-31, the period of the published target file: 1 881 days), one yearly request at
a time. Files already present are kept (`--force` refetches). A sensor whose request
fails is not written at all, so a file on disk is always complete; the command then
exits 1 and can simply be run again.

The host is `https://coastwatch.pfeg.noaa.gov/erddap`. Several of these datasets are
federated from `https://coastwatch.noaa.gov/erddap`, so a pfeg "502 Proxy Error" means
the upstream server did not answer; try again later. NOAA reprocesses these products
from time to time, so a fresh download can differ slightly from the files behind the
published results.

To work without the network, run `python scripts/make_synthetic_data.py` to get
synthetic data (what CI does).

Point `CHL_DATA_DIR` (in `.env`) at another directory to keep data outside the repo.
