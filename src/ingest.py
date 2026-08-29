"""
src/ingest.py — AURN + Open-Meteo -> one clean hourly Parquet per station.

This layer fetches raw truth and does NOTHING derived. No lags, no rolling
statistics, no wind components, no calendar columns. Those belong in
src/features.py (fold-independent) or the walk-forward harness (fold-dependent).
See docs/information_contract.md and PROJECT_SPEC Part 10.

Usage:
    python src/ingest.py --site MY1 --start 2018 --end 2025
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd
import rdata

log = logging.getLogger("ingest")

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

DEFRA = "https://uk-air.defra.gov.uk/openair/R_data"
OPENMETEO = "https://archive-api.open-meteo.com/v1/archive"

# --------------------------------------------------------------------------
# DECISION 1 — schema reconciliation by explicit allowlist.
#
# AURN column counts drift by year at MY1: 2019=51, 2020=49, 2024=43, 2026=43.
# pd.concat would take the union and silently fill absentees with NaN, making
# "instrument never existed" indistinguishable from "instrument failed".
# Instead: name what we need, and fail loudly if a year cannot supply it.
# Verified 2026-08-29: every column below is present in 2019, 2020 and 2024.
# --------------------------------------------------------------------------
AURN_KEEP = {
    "date": "date",
    "PM2.5": "pm2_5",       # the forecast target
    "NO2": "no2",           # diurnal diagnostic; watcher feature
    "PM10": "pm10",
    "NOXasNO2": "nox",
    "O3": "o3",
    "ws": "ws_aurn",        # site-level met, kept separate from Open-Meteo
    "wd": "wd_aurn",
    "temp": "temp_aurn",
}

WEATHER_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "pressure_msl",
    "wind_speed_10m",
    "wind_direction_10m",
    "boundary_layer_height",
]

EXPECTED_UNITS = {
    "wind_speed_10m": "m/s",      # NOT optional — see PROJECT_SPEC Part 5
    "temperature_2m": "\u00b0C",
    "pressure_msl": "hPa",
    "boundary_layer_height": "m",
}


# --------------------------------------------------------------------------
# Download helper. Plain Path, never NamedTemporaryFile: Windows holds an
# exclusive lock on an open temp file and rdata cannot then read it.
# --------------------------------------------------------------------------
def _download(url: str, dest: pathlib.Path, force: bool = False) -> pathlib.Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        log.info("cached  %s", dest.name)
        return dest
    log.info("fetch   %s", url)
    dest.write_bytes(urllib.request.urlopen(url, timeout=120).read())
    return dest


def _read_rda(path: pathlib.Path) -> pd.DataFrame:
    """Read a DEFRA .RData file and return its single DataFrame."""
    objects = rdata.read_rda(path)
    df = objects[list(objects)[0]]
    # rdata returns numpy string objects as column labels; normalise to str
    # so that ordinary df["PM2.5"] indexing works.
    df.columns = [str(c) for c in df.columns]
    return df.reset_index(drop=True)     # R indexes from 1


# --------------------------------------------------------------------------
# Station metadata -> coordinates. Never hardcode lat/lon: a wrong coordinate
# fetches real weather from the wrong place and raises nothing.
# --------------------------------------------------------------------------
def site_coords(site: str) -> tuple[float, float, str]:
    meta = _read_rda(_download(f"{DEFRA}/AURN_metadata.RData",
                               RAW / "AURN_metadata.RData"))
    rows = meta[meta["site_id"].astype(str) == site]
    if rows.empty:
        raise ValueError(f"site {site!r} not found in AURN_metadata")
    r = rows.iloc[0]
    return float(r["latitude"]), float(r["longitude"]), str(r["site_name"])


# --------------------------------------------------------------------------
# AURN
# --------------------------------------------------------------------------
def load_aurn_year(site: str, year: int) -> pd.DataFrame:
    df = _read_rda(_download(f"{DEFRA}/{site}_{year}.RData",
                             RAW / f"{site}_{year}.RData"))

    missing = [c for c in AURN_KEEP if c not in df.columns]
    if missing:
        raise KeyError(
            f"{site}_{year}: allowlist columns absent: {missing}. "
            f"Decide explicitly whether to drop the column or the year — "
            f"do not let pandas fill it with NaN."
        )

    df = df[list(AURN_KEEP)].rename(columns=AURN_KEEP)
    # 'date' arrives as float64 Unix epoch seconds, and is genuine UTC
    # (verified W1.3, docs/ingest_checks.md §1). Localise, do not convert.
    df["date"] = pd.to_datetime(df["date"], unit="s", utc=True)
    return df


def load_aurn(site: str, start: int, end: int) -> pd.DataFrame:
    frames = [load_aurn_year(site, y) for y in range(start, end + 1)]
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset="date", keep="first").sort_values("date")
    return df.set_index("date")


# --------------------------------------------------------------------------
# Open-Meteo
# --------------------------------------------------------------------------
def _openmeteo_get(params: dict, retries: int = 4) -> dict:
    url = OPENMETEO + "?" + urllib.parse.urlencode(params, safe=",")
    for attempt in range(retries):
        try:
            payload = json.load(urllib.request.urlopen(url, timeout=120))
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries - 1:
                time.sleep(2 ** attempt * 5)
                continue
            raise
        # Rate-limit failures and genuine bad-variable failures share the same
        # {"error": true} shape. Branch on `reason`, never on `error`.
        if payload.get("error"):
            reason = str(payload.get("reason", "")).lower()
            if "minutely" in reason or "hourly" in reason or "limit" in reason:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt * 5)
                    continue
            raise RuntimeError(f"Open-Meteo refused the request: {reason}")
        return payload
    raise RuntimeError("Open-Meteo: retries exhausted")


def load_weather(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    payload = _openmeteo_get({
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(WEATHER_VARS),
        "timezone": "UTC",
        "wind_speed_unit": "ms",
    })

    units = payload["hourly_units"]
    for var, expected in EXPECTED_UNITS.items():
        if units.get(var) != expected:
            raise AssertionError(
                f"Open-Meteo returned {var} in {units.get(var)!r}, "
                f"expected {expected!r}. Every downstream physical statement "
                f"would be wrong and nothing else would raise."
            )

    df = pd.DataFrame(payload["hourly"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.rename(columns={"time": "date"}).set_index("date")

    # The archive endpoint has been observed returning timestamps up to ~13 h
    # in the future. In Mode B that is future weather, i.e. leakage.
    now = pd.Timestamp.now(tz="UTC")
    future = (df.index > now).sum()
    if future:
        log.warning("dropping %d weather rows dated after now", future)
        df = df[df.index <= now]

    return df


# --------------------------------------------------------------------------
# DECISION 2 — the imputation rule (PROJECT_SPEC Part 6).
#
#   gap  < 2 h  -> linear interpolation, recorded in `imputed`
#   gap >= 2 h  -> left NaN
#
# At hourly resolution "under 2 hours" means a single missing hour and nothing
# more. The obvious one-liner, .interpolate(limit=1), is WRONG: it fills the
# first NaN of *every* run, so a 200-hour outage silently gains a fabricated
# first hour. So: reindex to a complete grid (a missing ROW is otherwise
# invisible), interpolate everything, then put the NaNs back wherever the run
# was 2 or longer.
# --------------------------------------------------------------------------
def impute(df: pd.DataFrame) -> pd.DataFrame:
    full = pd.date_range(df.index.min(), df.index.max(), freq="h", tz="UTC")
    df = df.reindex(full)
    df.index.name = "date"

    out = df.copy()
    flags = pd.DataFrame(False, index=df.index, columns=df.columns)

    for col in df.columns:
        isna = df[col].isna()
        if not isna.any():
            continue
        # Label each consecutive run of NaNs, then measure its length.
        run_id = (isna != isna.shift()).cumsum()
        run_len = isna.groupby(run_id).transform("sum")

        short = isna & (run_len == 1)          # exactly one missing hour
        filled = df[col].interpolate(method="time", limit_area="inside")
        out.loc[short, col] = filled.loc[short]
        flags.loc[short, col] = True

    out["imputed"] = flags.any(axis=1)
    return out


# --------------------------------------------------------------------------
def build(site: str, start: int, end: int) -> pd.DataFrame:
    lat, lon, name = site_coords(site)
    log.info("%s = %s at (%.5f, %.5f)", site, name, lat, lon)

    aurn = load_aurn(site, start, end)
    weather = load_weather(lat, lon, f"{start}-01-01", f"{end}-12-31")

    df = aurn.join(weather, how="left")       # AURN's clock is the spine
    df = impute(df)

    df.attrs["site"] = site
    df.attrs["snapshot_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="MY1")
    ap.add_argument("--start", type=int, default=2018)
    ap.add_argument("--end", type=int, default=2025)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    df = build(args.site, args.start, args.end)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = PROCESSED / f"{args.site}.parquet"
    df.to_parquet(out)

    log.info("wrote %s  rows=%d  cols=%d", out, len(df), df.shape[1])
    log.info("span  %s .. %s", df.index.min(), df.index.max())
    cov = df[["pm2_5", "no2", "boundary_layer_height"]].notna().mean() * 100
    for k, v in cov.items():
        log.info("coverage %-24s %5.1f%%", k, v)
    log.info("imputed hours: %d", int(df["imputed"].sum()))


if __name__ == "__main__":
    main()