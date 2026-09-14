"""tests/test_canary.py — does the canary actually detect anything?

A canary that has never failed is untested. This runs canary_check against two
fake pipelines: one honest, one that reads one hour into the future. The honest
one must pass, the leaking one must fail. Both directions, or the machinery is
unproven.

Run from the repo root:   python tests/test_canary.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluate import canary_check  # noqa: E402

T_STAR = pd.Timestamp("2020-01-01 12:00", tz="UTC")


def make_raw(n_hours: int = 48) -> pd.DataFrame:
    """A toy raw series. Values are arbitrary but distinct, so any leak shows."""
    idx = pd.date_range("2020-01-01 00:00", periods=n_hours, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    return pd.DataFrame({"pm2_5": rng.uniform(5, 25, n_hours)}, index=idx)


def _as_predictions(origins, yhat, y_true) -> pd.DataFrame:
    """Shape a toy result like the real harness output."""
    out = pd.DataFrame({"origin": origins, "y_true": y_true})
    for col in ("yhat_F0", "yhat_F1", "yhat_F2", "yhat_F3"):
        out[col] = yhat
    return out


def honest_pipeline(raw: pd.DataFrame) -> pd.DataFrame:
    """Persistence. The prediction at origin t uses y at t — nothing later."""
    origins = raw.index
    yhat = raw["pm2_5"].to_numpy()
    y_true = raw["pm2_5"].shift(-6).to_numpy()
    return _as_predictions(origins, yhat, y_true)


def leaking_pipeline(raw: pd.DataFrame) -> pd.DataFrame:
    """Deliberately broken: the prediction at origin t uses y at t+1.

    This is the shape of a missing purge — a value from after the origin
    reaching a prediction made before it.
    """
    origins = raw.index[:-1]
    yhat = raw["pm2_5"].to_numpy()[1:]          # <- one hour into the future
    y_true = raw["pm2_5"].shift(-6).to_numpy()[:-1]
    return _as_predictions(origins, yhat, y_true)


if __name__ == "__main__":
    raw = make_raw()

    good = canary_check(honest_pipeline, raw, T_STAR, placement="honest-control")
    bad = canary_check(leaking_pipeline, raw, T_STAR, placement="leak-injected")

    print(good)
    print(bad)

    assert good.passed, "canary fired on an honest pipeline — machinery is broken"
    assert not bad.passed, "canary MISSED an injected leak — it proves nothing"
    assert bad.first_divergence == T_STAR - pd.Timedelta(hours=1), (
        f"leak detected at {bad.first_divergence}, expected {T_STAR - pd.Timedelta(hours=1)}"
    )
    print("\nBoth directions confirmed. The canary detects and does not false-alarm.")