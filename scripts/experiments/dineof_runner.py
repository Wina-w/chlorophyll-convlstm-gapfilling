"""Drive the compiled DINEOF binary (dineof.exe) from Python via NetCDF I/O.

DINEOF (Data Interpolating Empirical Orthogonal Functions; Beckers & Rixen 2003,
Alvera-Azcarate et al. 2005) is the classical EOF-based cloud gap-filler. Its
open-source Fortran code (GHER, University of Liege: github.com/aida-alvera/DINEOF)
compiles to a ``dineof.exe`` that reads/writes NetCDF directly:

    data    = ['input.nc#var']      # (time, lat, lon), NaN at gaps to fill
    mask    = ['input.nc#mask']     # (lat, lon) in {0,1}; 1 = sea, 0 = land
    results = ['Output/out.nc#var'] # filled (time, lat, lon)

This module writes that NetCDF + an init file, runs the binary as a subprocess,
and reads the reconstruction back — so the *real* DINEOF (not a re-implementation)
can be benchmarked against the deep-learning gap-filler on identical holes. It also
reads DINEOF's own mode search from its log: the cap asked for (``nev``), the number
of EOF modes its cross-validation chose and that cross-validation error.

No MATLAB required. Windows resolves the binary's sibling DLLs from the exe's own
directory, so the full path to dineof.exe is sufficient.
"""
import os
import re
import shutil
import subprocess
import pathlib

import numpy as np
from netCDF4 import Dataset as NetCDFDataset

from _compat import P

DINEOF_EXE = P.dineof_exe  # from .env: DINEOF_EXE=/path/to/dineof.exe

DEFAULT_PARAMS = dict(
    nev=10,          # max EOF modes (DINEOF stops at optimal + 3)
    neini=1,         # min EOF modes
    ncv=None,        # Krylov subspace size; default nev+6, capped < T
    tol=1.0e-3,      # convergence tolerance (as in my Sentinel-3 processors)
    nitemax=300,
    toliter=1.0e-2,  # iterative solver tolerance (as in my Sentinel-3 processors)
    rec=1, eof=1, norm=0,
    seed=243435,
)


def write_dineof_nc(path, cube, seamask, var="v"):
    """Write a (T,H,W) cube + (H,W) sea mask to a DINEOF-readable NetCDF.

    Parameters
    ----------
    cube : (T,H,W) float; NaN marks gaps (clouds + synthetic holes) to be filled.
    seamask : (H,W) in {0,1}; 1 = sea (fillable domain), 0 = land (ignored).
    """
    cube = np.asarray(cube, dtype="float32")
    T, H, W = cube.shape
    path = pathlib.Path(path)
    with NetCDFDataset(str(path), "w", format="NETCDF4") as nc:
        nc.createDimension("time", T)
        nc.createDimension("lat", H)
        nc.createDimension("lon", W)
        v = nc.createVariable(var, "f4", ("time", "lat", "lon"), fill_value=-9999.0)
        # DINEOF treats the NetCDF _FillValue as "missing"; write NaN->fill.
        filled = np.where(np.isfinite(cube), cube, -9999.0).astype("float32")
        v[:] = filled
        v.missing_value = -9999.0
        m = nc.createVariable("mask", "f4", ("lat", "lon"))
        m[:] = np.asarray(seamask, dtype="float32")
        t = nc.createVariable("time", "f8", ("time",))
        t[:] = np.arange(T, dtype="float64")
        t.units = "days since 2000-01-01"
    return path


def _write_init(init_path, data_nc, var, out_nc, params, work_dir):
    T = params["_T"]
    ncv = params.get("ncv") or min(params["nev"] + 6, max(3, T - 1))
    lines = [
        "! auto-generated DINEOF init",
        f"data = ['{data_nc}#{var}']",
        f"mask = ['{data_nc}#mask']",
        f"time = '{data_nc}#time'",
        "alpha = 0.0",
        "numit = 3",
        f"nev = {params['nev']}",
        f"neini = {params['neini']}",
        f"ncv = {ncv}",
        f"tol = {params['tol']:.3e}",
        f"nitemax = {params['nitemax']}",
        f"toliter = {params['toliter']:.3e}",
        f"rec = {params['rec']}",
        f"eof = {params['eof']}",
        f"norm = {params['norm']}",
        f"Output = '{work_dir}/Output/'",
        f"results = ['{out_nc}#{var}']",
        f"seed = {params['seed']}",
        f"EOF.U = ['{work_dir}/Output/eof.nc#U']",
        f"EOF.V = '{work_dir}/Output/eof.nc#V'",
        f"EOF.Sigma = '{work_dir}/Output/eof.nc#Sigma'",
        "cloud_size = 500",
    ]
    pathlib.Path(init_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_log(text):
    """DINEOF's mode search from its stdout: the cap asked for (``nev``), the number of EOF modes
    its cross-validation chose (``n_modes``; equal to ``nev`` means the search hit the cap) and the
    cross-validation error of that choice. None for a line the log does not have."""

    def grab(pattern, cast):
        m = re.search(pattern, text)
        return cast(m.group(1)) if m else None

    nev = grab(r"number of EOF modes you want to compute\s+(\d+)", int)
    n_modes = grab(r"Number of optimal EOF modes:\s+(\d+)", int)
    return {
        "nev": nev,
        "n_modes": n_modes,
        "cv_error": grab(r"expected error calculated by cross-validation\s+([-+0-9.Ee]+)", float),
        "cap_hit": None if nev is None or n_modes is None else n_modes >= nev,
    }


def run_dineof(cube, seamask, work_dir, var="v", params=None, exe=DINEOF_EXE):
    """Fill the NaN gaps in `cube` (within `seamask`) using the real DINEOF binary.

    Returns
    -------
    filled : (T,H,W) float32 reconstruction. With the default ``rec=1`` every pixel is the
             truncated EOF reconstruction, observed ones included (``rec=0`` fills only the gaps).
    info : dict with the return code, the paths and the mode search (see ``parse_log``).
    """
    exe = pathlib.Path(exe)
    if not exe.exists():
        raise FileNotFoundError(f"dineof.exe not found at {exe}")
    work_dir = pathlib.Path(work_dir)
    (work_dir / "Output").mkdir(parents=True, exist_ok=True)

    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    p["_T"] = cube.shape[0]

    data_nc = (work_dir / "dineof_in.nc").as_posix()
    out_nc = (work_dir / "Output" / "dineof_out.nc").as_posix()
    init = work_dir / "dineof.init"
    write_dineof_nc(data_nc, cube, seamask, var=var)
    # names relative to the work dir (the binary's cwd): DINEOF cuts long paths short
    _write_init(init, "dineof_in.nc", var, "Output/dineof_out.nc", p, ".")

    proc = subprocess.run([str(exe), str(init)], cwd=str(work_dir),
                          capture_output=True, text=True)
    log = (work_dir / "dineof.log")
    log.write_text((proc.stdout or "") + "\n===STDERR===\n" + (proc.stderr or ""), encoding="utf-8")
    if proc.returncode != 0:
        tail = (proc.stdout or "")[-600:] + (proc.stderr or "")[-600:]
        raise RuntimeError(f"dineof.exe failed (code {proc.returncode}). tail:\n{tail}")

    out_file = pathlib.Path(out_nc)
    if not out_file.exists():
        raise RuntimeError(f"DINEOF produced no output at {out_nc}; see {log}")
    with NetCDFDataset(str(out_file), "r") as nc:
        name = var if var in nc.variables else [k for k in nc.variables
                                                if nc.variables[k].ndim == 3][0]
        filled = np.asarray(nc.variables[name][:], dtype="float32")
        if hasattr(nc.variables[name], "_FillValue"):
            fv = float(nc.variables[name]._FillValue)
            filled = np.where(filled == fv, np.nan, filled)
    info = {"returncode": proc.returncode, "out_nc": out_nc, "log": str(log)}
    return filled, {**info, **parse_log(proc.stdout or "")}


# ---------------------------------------------------------------------------
def _smoke_test():
    """Round-trip DINEOF on the toolbox SmallExample: remove known pixels, fill,
    and check the reconstruction is sane. Verifies the writer + runner + reader."""
    import xarray as xr
    small = (
        (DINEOF_EXE.parent / "SmallExample" / "alboranSST_small.nc") if DINEOF_EXE else pathlib.Path("missing")
    )
    d = xr.open_dataset(small)
    cube = d["SST"].values.astype("float32")          # (T,H,W) with NaN gaps
    seamask = d["mask"].values.astype("float32")      # (H,W) 1=sea
    rng = np.random.default_rng(0)

    # hide a set of currently-OBSERVED sea pixels as a synthetic cross-val set
    observed = np.isfinite(cube)
    held = observed & (rng.random(cube.shape) < 0.05)
    truth = cube[held].copy()
    holed = cube.copy()
    holed[held] = np.nan

    work = pathlib.Path(os.environ.get("TEMP", ".")) / "dineof_smoke"
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    filled, info = run_dineof(holed, seamask, work, var="SST",
                              params=dict(nev=5, neini=1, ncv=10, toliter=1e-3))
    rec = filled[held]
    ok = np.isfinite(rec)
    rmse = float(np.sqrt(np.mean((rec[ok] - truth[ok]) ** 2)))
    print(f"[dineof smoke] held={held.sum()} filled={ok.sum()} "
          f"RMSE={rmse:.3f} degC (should be < ~1.0)  code={info['returncode']}")
    assert ok.mean() > 0.9, "DINEOF left held-out pixels unfilled"
    assert rmse < 2.0, f"DINEOF reconstruction RMSE too high ({rmse})"
    print("[dineof smoke] PASS")


if __name__ == "__main__":
    _smoke_test()
