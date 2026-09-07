-- ===========================================================================
-- sql/schema.sql — W1.9
--
-- Table definitions for the Postgres deployment. WRITTEN, NOT DEPLOYED.
-- Deployment is Week 5 (PROJECT_SPEC Part 3, change 6): the realistic Week 1
-- failure mode is days lost to database setup with no science done. Parquet
-- carries the project through Week 3.
--
-- Target: PostgreSQL 14+.
-- Load order: stations -> observations -> features.
-- ===========================================================================


-- ---------------------------------------------------------------------------
-- stations — the four sites surviving the W1.6 coverage audit.
--
-- Coordinates come from AURN_metadata.RData and are NEVER hardcoded: a wrong
-- lat/lon fetches real weather from the wrong place and raises nothing.
-- ---------------------------------------------------------------------------
CREATE TABLE stations (
    site_id         TEXT        PRIMARY KEY,          -- 'MY1', 'KC1', 'BEX', 'HRL'
    site_name       TEXT        NOT NULL,             -- 'London Marylebone Road'
    location_type   TEXT        NOT NULL,             -- 'Urban Traffic', etc.
    latitude        DOUBLE PRECISION NOT NULL,
    longitude       DOUBLE PRECISION NOT NULL,
    CONSTRAINT lat_range CHECK (latitude  BETWEEN -90  AND 90),
    CONSTRAINT lon_range CHECK (longitude BETWEEN -180 AND 180)
);

COMMENT ON TABLE stations IS
    'AURN sites kept by src/select_stations.py. Rejections and their coverage
     numbers live in docs/station_selection.md, not here — a dropped station
     documented with a number is evidence; one documented with silence is a hole.';


-- ---------------------------------------------------------------------------
-- observations — hourly raw truth, one row per (station, hour).
--
-- TIMESTAMPTZ, not TIMESTAMP. AURN timestamps are genuine UTC instants
-- (verified W1.3, docs/ingest_checks.md §1). A naive TIMESTAMP column would
-- discard that fact and reintroduce exactly the ambiguity the W1.3 check
-- existed to rule out.
--
-- Every pollutant is NULLABLE. A gap is real information: gaps under 2h are
-- interpolated at ingest and flagged; gaps of 2h or more stay NULL and must
-- never be filled here (PROJECT_SPEC Part 6).
-- ---------------------------------------------------------------------------
CREATE TABLE observations (
    site_id                 TEXT        NOT NULL REFERENCES stations(site_id),
    ts                      TIMESTAMPTZ NOT NULL,

    -- AURN pollutants
    pm2_5                   DOUBLE PRECISION,         -- the forecast target
    no2                     DOUBLE PRECISION,         -- diurnal diagnostic
    pm10                    DOUBLE PRECISION,
    nox                     DOUBLE PRECISION,
    o3                      DOUBLE PRECISION,

    -- AURN site-level met, kept separate from Open-Meteo on purpose:
    -- mixing two weather sources without deciding which is authoritative is
    -- how silent inconsistencies get in.
    ws_aurn                 DOUBLE PRECISION,
    wd_aurn                 DOUBLE PRECISION,
    temp_aurn               DOUBLE PRECISION,

    -- Open-Meteo ERA5 reanalysis. See information_contract.md §6: reanalysis
    -- at prediction time is a mild look-ahead, accepted and disclosed.
    -- wind_speed_10m is metres per second. NOT km/h — see Part 5.
    temperature_2m          DOUBLE PRECISION,
    relative_humidity_2m    DOUBLE PRECISION,
    pressure_msl            DOUBLE PRECISION,
    wind_speed_10m          DOUBLE PRECISION,
    wind_direction_10m      DOUBLE PRECISION,
    boundary_layer_height   DOUBLE PRECISION,         -- NULL 2024-01..2024-06,
                                                      -- archive-wide gap

    imputed                 BOOLEAN     NOT NULL DEFAULT FALSE,

    PRIMARY KEY (site_id, ts)
);

-- The composite primary key already indexes (site_id, ts). This second index
-- serves the other access pattern: "all stations at one time", used by the
-- live scoreboard and by any cross-station comparison.
CREATE INDEX observations_ts_idx ON observations (ts);

COMMENT ON COLUMN observations.imputed IS
    'TRUE where a single missing hour was linearly interpolated at ingest.
     Gaps of 2h or more are left NULL and must never be filled: interpolating
     a long gap and then computing a rolling statistic over it manufactures
     data that was never measured.';


-- ---------------------------------------------------------------------------
-- features — the fold-independent feature table from src/features.py.
--
-- DERIVED DATA, STORED DELIBERATELY. This duplicates what features.py can
-- regenerate from observations, so it can drift out of sync with its own
-- definition: change features.py, and this table silently holds numbers from
-- an older meaning of the same column names.
--
-- Guard: feature_version and generated_at travel with every row, so staleness
-- is detectable by query rather than by memory. Bump feature_version whenever
-- features.py changes what a column MEANS, not merely when it is re-run.
--
-- WHAT IS NOT HERE, AND WHY:
-- No scaler output, no climatology, no residual features, no model
-- disagreement, no distribution distances, no label thresholds. Every one of
-- those fails the fold test (information_contract.md §5) — its value at a
-- fixed timestamp would change if a fold boundary moved — so it is computed
-- inside the walk-forward harness, per fold, and never persisted here.
-- Persisting a fold-dependent column would freeze one fold's view of the data
-- across all folds. That is leakage, and it produces plausible output with no
-- error and no warning.
-- ---------------------------------------------------------------------------
CREATE TABLE features (
    site_id                 TEXT        NOT NULL REFERENCES stations(site_id),
    ts                      TIMESTAMPTZ NOT NULL,     -- forecast origin t

    -- lags: the reading at t and at t-1, -3, -6, -12, -24 hours
    pm2_5_lag_0             DOUBLE PRECISION,
    pm2_5_lag_1             DOUBLE PRECISION,
    pm2_5_lag_3             DOUBLE PRECISION,
    pm2_5_lag_6             DOUBLE PRECISION,
    pm2_5_lag_12            DOUBLE PRECISION,
    pm2_5_lag_24            DOUBLE PRECISION,
    no2_lag_0               DOUBLE PRECISION,
    no2_lag_1               DOUBLE PRECISION,
    no2_lag_3               DOUBLE PRECISION,
    no2_lag_6               DOUBLE PRECISION,
    no2_lag_12              DOUBLE PRECISION,
    no2_lag_24              DOUBLE PRECISION,

    -- rolling means over the row's own past, window ending at t inclusive
    pm2_5_mean_3h           DOUBLE PRECISION,
    pm2_5_mean_6h           DOUBLE PRECISION,
    pm2_5_mean_12h          DOUBLE PRECISION,
    pm2_5_mean_24h          DOUBLE PRECISION,
    no2_mean_3h             DOUBLE PRECISION,
    no2_mean_6h             DOUBLE PRECISION,
    no2_mean_12h            DOUBLE PRECISION,
    no2_mean_24h            DOUBLE PRECISION,

    -- rolling standard deviations: the volatility family, and the ONLY
    -- watcher feature family that survives the fold test
    pm2_5_std_3h            DOUBLE PRECISION,
    pm2_5_std_6h            DOUBLE PRECISION,
    pm2_5_std_12h           DOUBLE PRECISION,
    pm2_5_std_24h           DOUBLE PRECISION,
    no2_std_3h              DOUBLE PRECISION,
    no2_std_6h              DOUBLE PRECISION,
    no2_std_12h             DOUBLE PRECISION,
    no2_std_24h             DOUBLE PRECISION,

    -- rates of change
    pm2_5_delta_1h          DOUBLE PRECISION,
    pm2_5_delta_3h          DOUBLE PRECISION,
    pm2_5_delta_6h          DOUBLE PRECISION,
    no2_delta_1h            DOUBLE PRECISION,
    no2_delta_3h            DOUBLE PRECISION,
    no2_delta_6h            DOUBLE PRECISION,

    -- weather observed at t
    temperature_2m          DOUBLE PRECISION,
    relative_humidity_2m    DOUBLE PRECISION,
    pressure_msl            DOUBLE PRECISION,
    boundary_layer_height   DOUBLE PRECISION,

    -- wind as components. Raw direction is deliberately absent: it is
    -- circular, so 359 and 1 degrees are adjacent in reality and 358 apart
    -- numerically, and a tree splitting on degrees cuts the compass in an
    -- arbitrary place.
    wind_u                  DOUBLE PRECISION,
    wind_v                  DOUBLE PRECISION,

    -- calendar at t, derived in Europe/London. 'Is it rush hour' is a
    -- local-clock question; ts is UTC, and in summer UTC hour 8 is local 9.
    hour                    SMALLINT,
    dow                     SMALLINT,
    month                   SMALLINT,

    -- calendar at the target hour t+6h. Legal despite naming a future hour
    -- (information_contract.md §4): calendars are deterministic, so standing
    -- at 09:00 you can compute 15:00 without waiting. These are the pointer,
    -- not the pattern — remove them and the model cannot tell a February
    -- afternoon from 3am in July.
    target_hour             SMALLINT,
    target_dow              SMALLINT,
    target_month            SMALLINT,
    target_is_weekend       SMALLINT,

    -- THE TARGET, never an input. PM2.5 at t+6h.
    y_t6                    DOUBLE PRECISION,

    imputed                 BOOLEAN,

    -- provenance: makes staleness detectable
    feature_version         TEXT        NOT NULL,
    generated_at            TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY (site_id, ts, feature_version),
    FOREIGN KEY (site_id, ts) REFERENCES observations (site_id, ts)
);

CREATE INDEX features_ts_idx ON features (ts);

COMMENT ON COLUMN features.y_t6 IS
    'PM2.5 at t+6h. The target, not a feature. Any model reading this as an
     input has not leaked — it has cheated outright.';

COMMENT ON COLUMN features.feature_version IS
    'Bump whenever src/features.py changes what a column MEANS. Part of the
     primary key so two versions can coexist and be compared rather than one
     silently overwriting the other.';


-- ---------------------------------------------------------------------------
-- Deliberately absent, and this is the load-bearing part of the design:
--
--   no forecasts table       — F0..F3 predictions are per-fold
--   no residuals table       — residuals depend on which F3 produced them
--   no watcher_scores table  — the watcher is retrained every fold
--   no labels table          — decile edges and the 80th-percentile cut are
--                              FITTED on training folds and applied frozen
--
-- All four are fold-dependent. Persisting any of them would fix one fold's
-- view across every fold, which is leakage of exactly the kind that produces
-- a plausible fold table with no error and no warning.
--
-- Fold-level results are written to results/ as artifacts of a specific run,
-- not to the database as though they were facts about the world.
-- ---------------------------------------------------------------------------
