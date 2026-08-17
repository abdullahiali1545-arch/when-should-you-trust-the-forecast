"""Phase 0 smoke test — proves both data sources return real data.

Throwaway. src/ingest.py replaces this properly on Session 4.
Run:  python smoke_test.py

Note: this does NOT use pyaurn.importAURN, which is broken.
See docs/ingest_checks.md for the diagnosis.
"""
from pathlib import Path

import pandas as pd
import rdata
import requests
from pyaurn import importMeta

SITE, YEAR = "MY1", 2024

# --- 1. AURN metadata ------------------------------------------------------
# pyaurn still works here: importMeta reads a CSV, not an RData file.
meta = importMeta()
site = meta[meta.site_id == SITE].iloc[0]
print(f"[1] {SITE} = {site.site_name} | {site.location_type} | "
      f"{site.latitude:.4f}, {site.longitude:.4f} | ratified to {site.ratified_to}")

# --- 2. AURN hourly data ---------------------------------------------------
# Downloaded straight from DEFRA and parsed with `rdata` (pure Python, reads RDX3).
# Written to a plain file, not tempfile.NamedTemporaryFile: on Windows that
# holds an exclusive lock and rdata cannot then open it by name.
url = f"https://uk-air.defra.gov.uk/openair/R_data/{SITE}_{YEAR}.RData"
raw_path = Path(f"{SITE}_{YEAR}.RData")
raw_path.write_bytes(requests.get(url, timeout=120).content)

tables = rdata.conversion.convert(rdata.parser.parse_file(raw_path))
raw_path.unlink()  # tidy up

df = tables[f"{SITE}_{YEAR}"].copy()
df.columns = [str(c) for c in df.columns]            # rdata returns np.str_ labels
df["date"] = pd.to_datetime(df["date"], unit="s", utc=True)   # epoch seconds -> UTC
print(f"[2] AURN: {df.shape[0]} rows x {df.shape[1]} cols | "
      f"{df.date.min()} -> {df.date.max()} | "
      f"PM2.5 coverage {100 * df['PM2.5'].notna().mean():.1f}%")

# --- 3. Open-Meteo ERA5 archive, matched to the site's coordinates ---------
# boundary_layer_height is deliberately absent — it errors on this endpoint.
# Investigate on Session 4; see docs/ingest_checks.md.
resp = requests.get(
    "https://archive-api.open-meteo.com/v1/archive",
    params={
        "latitude": float(site.latitude),
        "longitude": float(site.longitude),
        "start_date": f"{YEAR}-01-01",
        "end_date": f"{YEAR}-01-07",
        "hourly": "temperature_2m,relative_humidity_2m,pressure_msl,"
                  "wind_speed_10m,wind_direction_10m",
        "timezone": "UTC",
    },
    timeout=60,
).json()

if "hourly" not in resp:
    raise SystemExit(f"[3] Open-Meteo FAILED: {resp.get('reason', resp)}")

wx = pd.DataFrame(resp["hourly"])
wx["time"] = pd.to_datetime(wx["time"], utc=True)
print(f"[3] Open-Meteo: {wx.shape[0]} rows x {wx.shape[1]} cols | "
      f"{wx.time.min()} -> {wx.time.max()} | "
      f"vars: {[c for c in wx.columns if c != 'time']}")

ok = len(df) > 8000 and len(wx) == 168
print("\nPHASE 0 GATE: PASSED" if ok else "\nPHASE 0 GATE: FAILED")
