"""src/freshness.py - W4b step 3: how fresh is each data source, really?

One run = one snapshot. For each station and each source it records the
timestamp of the latest non-null PM2.5 reading and how many hours old that is,
and APPENDS the rows to docs/freshness_log.csv. Run it several times across
two or three days, at different times of day. The log then shows:

  - each source's typical delay
  - each station's real batch-end hour (HRL's 20:00 is still one observation)
  - when the daily AURN batch actually appears

Sources
  AURN   DEFRA openair .RData files - what the whole project was trained on
  SOS    DEFRA UK-AIR near-real-time REST API (52North), base URL as used by
         the UK-AQ ingest project. Timed out from Claude's sandbox on
         2026-09-29, so this is the untested one
  LAQN   Imperial's London Air API. Codes are the PM2.5 instruments LAQN lists
         at the same sites (looked up 2026-09-29). WARNING: these may be
         different instruments from AURN's. A faster source only helps if its
         values match what the model was trained on - check that before
         switching.

A source that errors or times out is logged as such, not skipped silently.

Run from the repo root:
    python -m src.freshness
"""
from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
import warnings
from pathlib import Path

import pandas as pd
import rdata

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "docs" / "freshness_log.csv"
TMP = ROOT / "data" / "tmp"

STATIONS = ["MY1", "KC1", "BEX", "HRL"]
LAQN_CODES = {"MY1": "MR8", "KC1": "KF1", "BEX": "BQ9", "HRL": "LH0"}
SOS = "https://uk-air.defra.gov.uk/sos-ukair/api/v1"
AURN = "https://uk-air.defra.gov.uk/openair/R_data"

warnings.filterwarnings("ignore", module="rdata")


def get(url: str, timeout: float) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "when-should-you-trust-the-forecast/freshness"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def read_rdata(name: str, timeout: float) -> pd.DataFrame:
    """Download an openair .RData file to a plain Path (not NamedTemporaryFile:
    Windows holds a lock) and return the single object inside it."""
    TMP.mkdir(parents=True, exist_ok=True)
    p = TMP / f"{name}.RData"
    p.write_bytes(get(f"{AURN}/{name}.RData", timeout))
    d = rdata.read_rda(p)
    try:
        p.unlink()
    except OSError:
        pass
    return d[name] if name in d else next(iter(d.values()))


def aurn_latest(site: str, year: int, timeout: float) -> tuple[pd.Timestamp | None, str]:
    df = read_rdata(f"{site}_{year}", timeout)
    df["date"] = pd.to_datetime(df["date"], unit="s", utc=True)
    s = df.set_index("date")["PM2.5"].dropna()
    return (s.index.max() if len(s) else None), "PM2.5"


def laqn_latest(code: str, now: pd.Timestamp, timeout: float) -> tuple[pd.Timestamp | None, str]:
    start = (now - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    end = (now + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    url = (f"https://api.erg.ic.ac.uk/AirQuality/Data/SiteSpecies/SiteCode={code}"
           f"/SpeciesCode=PM25/StartDate={start}/EndDate={end}/Json")
    rows = json.loads(get(url, timeout))["RawAQData"]["Data"]
    rows = rows if isinstance(rows, list) else [rows]
    ok = [r["@MeasurementDateGMT"] for r in rows if r.get("@Value", "") != ""]
    # LAQN labels are GMT, i.e. UTC
    return (pd.Timestamp(ok[-1], tz="UTC") if ok else None), code


def sos_latest(lat: float, lon: float, timeout: float) -> tuple[pd.Timestamp | None, str]:
    near = json.dumps({"center": {"type": "Point", "coordinates": [lon, lat]}, "radius": 0.5})
    stations = json.loads(get(f"{SOS}/stations?near={urllib.parse.quote(near)}", timeout))
    best, label = None, "no PM2.5 timeseries found"
    for st in stations:
        sid = st.get("properties", {}).get("id") or st.get("id")
        ts = json.loads(get(f"{SOS}/timeseries?station={sid}&expanded=true", timeout))
        for t in ts:
            lab = t.get("label", "")
            lv = t.get("lastValue") or {}
            if "2.5" in lab and lv.get("timestamp") is not None:
                when = pd.Timestamp(lv["timestamp"], unit="ms", tz="UTC")
                if best is None or when > best:
                    best, label = when, lab[:80]
    return best, label


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=30, help="seconds per request")
    ap.add_argument("--skip-sos", action="store_true")
    args = ap.parse_args()

    now = pd.Timestamp.now(tz="UTC").floor("s")
    rows: list[dict] = []

    def record(source: str, station: str, fn) -> None:
        try:
            latest, detail = fn()
            lag = round((now - latest).total_seconds() / 3600, 1) if latest is not None else None
            rows.append({"pulled_utc": now, "source": source, "station": station,
                         "latest_obs_utc": latest, "lag_h": lag,
                         "status": "ok" if latest is not None else "no recent data",
                         "detail": detail})
        except Exception as e:                      # log it, never hide it
            rows.append({"pulled_utc": now, "source": source, "station": station,
                         "latest_obs_utc": None, "lag_h": None,
                         "status": f"error: {type(e).__name__}", "detail": str(e)[:80]})

    coords = {}
    if not args.skip_sos:
        try:
            meta = read_rdata("AURN_metadata", args.timeout * 2)
            for s in STATIONS:
                m = meta[meta["site_id"] == s].iloc[0]
                coords[s] = (float(m["latitude"]), float(m["longitude"]))
        except Exception as e:
            print(f"could not read AURN metadata for SOS coordinates: {e}")

    for s in STATIONS:
        record("AURN", s, lambda s=s: aurn_latest(s, now.year, args.timeout * 2))
        record("LAQN", s, lambda s=s: laqn_latest(LAQN_CODES[s], now, args.timeout))
        if not args.skip_sos and s in coords:
            record("SOS", s, lambda s=s: sos_latest(*coords[s], args.timeout))

    snap = pd.DataFrame(rows)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    snap.to_csv(LOG, mode="a", header=not LOG.exists(), index=False)

    pd.set_option("display.width", 160)
    print(f"\nSnapshot at {now} UTC\n")
    print(snap.drop(columns=["pulled_utc"]).to_string(index=False))
    print(f"\nAppended {len(snap)} rows to {LOG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()