import os
import warnings
from datetime import timedelta

import numpy as np
from obspy import UTCDateTime


def g2s_fname(date: UTCDateTime, lat: float, lon: float) -> str:
    return f"g2s_{date.strftime('%Y-%m-%dT%H')}_lat{lat:.4f}_lon{lon:.4f}.met"


def parse_g2s_fname(fname: str):
    if not fname.startswith("g2s"):
        return None, None, None

    if fname.endswith(".met"):
        parts = fname.removesuffix(".met").split("_")
    elif fname.endswith(".dat"):
        parts = fname.removesuffix(".dat").split("_")
    else:
        return None, None, None
    date = UTCDateTime(parts[1])
    lat = float(parts[2].replace("lat", ""))
    lon = float(parts[3].replace("lon", ""))
    return date, lat, lon


def parse_ncpag2s_fname(fname):
    if not fname.startswith("ncpag2s_") and fname.endswith(".met"):
        return None, None, None

    parts = fname.removesuffix(".met").split("_")

    date = UTCDateTime(parts[1])
    lat = float(parts[2].replace("lat", ""))
    lon = float(parts[3].replace("lon", "").replace(".dat", ""))

    return date, lat, lon


def find_g2s_grid(date: UTCDateTime, atmos_dir: str):
    if not os.path.exists(atmos_dir):
        raise Warning(f"Atmos dir does not exist: {atmos_dir}")

    for dirname in os.listdir(atmos_dir):
        if not dirname.startswith("grid_"):
            continue
        parts = dirname.removeprefix("grid_").split("_")
        dirdate = UTCDateTime(parts[0])
        if dirdate == date:
            return os.path.join(atmos_dir, dirname)
    warnings.warn(f"Could not find grid for {date} in {atmos_dir}")
    return None


def list_g2s_files(
    atmos_dir: str,
    lat: float,
    lon: float,
    start: UTCDateTime,
    end: UTCDateTime,
    h_resol=None,
    ignore_warnings=False,
):
    files, times = [], []

    for fname in os.listdir(atmos_dir):
        if not fname.endswith(".met"):
            continue
        t, la, lo = parse_g2s_fname(fname)
        if t is None:
            continue
        if (la == lat) and (lo == lon) and (start <= t <= end):
            files.append(fname)
            times.append(t)

    if not files and not ignore_warnings:
        raise Warning(f"No suitable times found in {atmos_dir}!")

    files = np.array(files)
    times = np.array(times)
    order = np.argsort(times)
    files, times = files[order], times[order]

    if h_resol is not None:
        if (h_resol <= 0) and not ignore_warnings:
            raise Warning(f"h_resol = {h_resol} is probably a stupid value :)")

        if (start not in times) and not ignore_warnings:
            raise Warning("Start_time is not found among the files!")
        if (end not in times) and not ignore_warnings:
            raise Warning("End_time is not found among the files!")

        n_times = int((end - start) / (h_resol * 3600)) + 1
        expected = [start + timedelta(hours=i * h_resol) for i in range(n_times)]
        intersection, idx_files, _ = np.intersect1d(times, expected, return_indices=True)
        if (intersection.size != n_times) and not ignore_warnings:
            raise Warning("The available files does not support the current time and resolution parameters!")

        files = files[idx_files]
        times = times[idx_files]

    return files, times
