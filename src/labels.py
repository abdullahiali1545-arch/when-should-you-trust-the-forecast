"""src/labels.py — stratified error labels for the watcher (W3.1).

The watcher is a classifier, so it needs a yes/no target: was F3 unreliable at
this hour? This module manufactures it.

The obvious approach — rank raw absolute errors, flag the worst 20% — is wrong,
and wrong in a way that would silently invalidate the headline result. Absolute
error scales with concentration, so ranking it produces a class that is almost
exactly "high pollution hour". R1 is literally the rule "route when predicted
concentration is high", so R3 and R1 would then be solving the same task and the
H2/H3 comparison would mean nothing.

So: bin by PREDICTED concentration into deciles, then flag the worst q of
absolute errors WITHIN each bin. The label then means "bad for an hour that
looked like this", which is what reliability detection actually requires.

Bins use predicted, not actual, values because actual values do not exist at
prediction time.

Both the decile edges and the within-bin cut-offs are FITTED PARAMETERS. They
are estimated on out-of-fold residuals from earlier folds only, then applied
frozen to the test fold — the same rule that governs a scaler. The realised
test-fold positive rate will drift from q; that drift is reported, never
corrected. It measures shift in the error process, which is the object of study.

All five edge cases below were pre-registered in PROJECT_SPEC.md before this
file was written.

Convention: label = 1 iff abs_err exceeds the cut-off. Strictly greater, in both
fit and apply. Ties fall on the reliable side.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

N_BINS = 10
Q = 0.80
MIN_BIN = 50
FLOOR_Q = 0.10


@dataclass
class LabelThresholds:
    """Everything apply_labels needs, and nothing it could re-estimate.

    Passed around as data rather than recomputed, so that a test fold cannot
    influence its own thresholds. Printable because these numbers go into the
    session log and the write-up.
    """

    edges: np.ndarray            # bin boundaries, len = n_bins_realised + 1
    cutoffs: np.ndarray          # per-bin absolute-error cut-off
    floor: float                 # relative-error denominator floor
    n_bins_realised: int         # may be fewer than N_BINS if edges collapsed
    fallback_bins: list[int] = field(default_factory=list)
    n_fit_rows: int = 0
    q: float = Q

    def __str__(self) -> str:
        return (
            f"LabelThresholds(n_fit={self.n_fit_rows:,}, "
            f"bins={self.n_bins_realised}, q={self.q}, "
            f"floor={self.floor:.3f}, fallback_bins={self.fallback_bins})\n"
            f"  edges:   {np.round(self.edges, 2)}\n"
            f"  cutoffs: {np.round(self.cutoffs, 2)}"
        )


def _clean(pred: pd.Series, actual: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Drop rows where either side is missing — decision 4.

    A missing residual means the hour is ABSENT, not that the forecast was good.
    Labelling it 0 would teach the watcher that hours with no data were fine.
    """
    ok = pred.notna() & actual.notna()
    return pred[ok], actual[ok]


def fit_label_thresholds(
    pred: pd.Series,
    actual: pd.Series,
    n_bins: int = N_BINS,
    q: float = Q,
    min_bin: int = MIN_BIN,
) -> LabelThresholds:
    """Estimate bin edges and per-bin error cut-offs.

    `pred` and `actual` must be out-of-fold rows from folds STRICTLY EARLIER
    than the test fold this will be applied to. In-sample residuals are
    systematically smaller and differently shaped; thresholds fitted on them
    would understate what counts as a bad error.
    """
    pred, actual = _clean(pred, actual)
    if len(pred) == 0:
        raise ValueError("no usable rows: every row had a missing pred or actual")

    abs_err = (actual - pred).abs()

    # duplicates="drop" — decision 2. Heavily tied predictions can make two
    # decile edges identical; forcing them apart would invent structure the data
    # does not contain. Fewer bins is information, so it is recorded below.
    binned, edges = pd.qcut(pred, q=n_bins, retbins=True,
                            duplicates="drop", labels=False)
    n_realised = len(edges) - 1

    # The global cut-off, used both as the fallback for thin bins and as a
    # sanity reference when reading the per-bin numbers.
    global_cutoff = float(abs_err.quantile(q))

    cutoffs = np.empty(n_realised, dtype=float)
    fallback_bins: list[int] = []
    for b in range(n_realised):
        in_bin = abs_err[binned == b]
        if in_bin.size >= min_bin:
            cutoffs[b] = float(in_bin.quantile(q))
        else:
            # Decision 3. A q-quantile of a handful of points is noise, and a
            # noisy cut-off produces noisy labels the watcher then tries to fit.
            cutoffs[b] = global_cutoff
            fallback_bins.append(b)

    # Decision 5. Floor for the relative-error robustness check, derived from
    # the fitting set rather than picked by hand, so it cannot be tuned after
    # the fact. Frozen exactly as the other thresholds are.
    floor = float(pred.quantile(FLOOR_Q))

    return LabelThresholds(
        edges=edges,
        cutoffs=cutoffs,
        floor=floor,
        n_bins_realised=n_realised,
        fallback_bins=fallback_bins,
        n_fit_rows=len(pred),
        q=q,
    )


def _assign_bins(pred: pd.Series, edges: np.ndarray) -> pd.Series:
    """Bin each prediction, clipping out-of-range values — decision 1.

    pd.cut returns NaN outside the edges; it does not clip. So the outer edges
    are replaced with -inf and +inf, which states the intent in the data rather
    than hiding it in a clip() call elsewhere.

    Clipping rather than dropping: dropping would alter the evaluation set and
    would remove exactly the extrapolated cases the watcher most needs testing
    on. The bias runs in the safe direction — such rows become more likely to be
    labelled unreliable, and they probably are.
    """
    open_edges = edges.astype(float).copy()
    open_edges[0] = -np.inf
    open_edges[-1] = np.inf
    return pd.cut(pred, bins=open_edges, labels=False,
                  include_lowest=True, duplicates="drop")


def apply_labels(
    pred: pd.Series,
    actual: pd.Series,
    thr: LabelThresholds,
) -> pd.Series:
    """Label rows unreliable (1) or reliable (0) using FROZEN thresholds.

    This function consumes thresholds; it never estimates them. It must not call
    .quantile(), .mean(), .std() or any other statistic of its own input — that
    separation is the leakage defence, not a stylistic preference.

    Rows with a missing pred or actual return pd.NA, not 0 (decision 4), so the
    return dtype is nullable Int8.
    """
    bins = _assign_bins(pred, thr.edges)
    abs_err = (actual - pred).abs()

    # Map each row's bin index to that bin's cut-off. Rows whose bin is NaN
    # (only possible if pred is NaN) get a NaN cut-off and fall out below.
    cutoff_per_row = bins.map(lambda b: thr.cutoffs[int(b)] if pd.notna(b) else np.nan)

    labels = abs_err.gt(cutoff_per_row).astype("Int8")
    labels[pred.isna() | actual.isna()] = pd.NA
    return labels


def label_walk_forward(
    oof: pd.DataFrame,
    pred_col: str = "yhat_F3",
    actual_col: str = "y_true",
) -> tuple[pd.Series, pd.DataFrame]:
    """Label every fold using thresholds fitted only on EARLIER folds.

    This is the honest counterpart to the module self-check below. That check
    fits on all rows at once, which is fine for verifying the binning maths and
    useless for anything else: it lets a fold's own errors set the standard its
    errors are judged against.

    Here, fold k is labelled with thresholds estimated on folds 1..k-1 and then
    frozen. The fitting set expands, mirroring the harness's expanding training
    window, and keeping enough rows per bin for the per-bin quantile to mean
    something.

    Fold 1 receives no labels: there are no earlier folds to fit on. This is the
    burn-in Part 10 warns about — the watcher's first scoreable fold is one
    behind F3's. It is reported, not worked around.

    Returns
    -------
    labels : Int8 Series aligned to oof.index; pd.NA for fold 1 and for any row
             with a missing prediction or actual.
    diag   : one row per fold. The realised positive rate is the pre-registered
             drift measurement, so this frame is a deliverable, not debug output.
    """
    labels = pd.Series(pd.NA, index=oof.index, dtype="Int8")
    diag_rows: list[dict] = []

    for k in sorted(oof["fold"].unique()):
        # Strictly earlier folds only. That strictness is the whole leakage
        # defence here: allowing fold k into its own fitting set would let it
        # set the standard it is judged against, and the symptom would be a
        # positive rate of exactly 0.20 in every fold.
        fit = oof[oof["fold"].lt(k)]
        test = oof[oof["fold"].eq(k)]

        if len(fit) == 0:
            diag_rows.append({
                "fold": k, "n_fit": 0, "n_test": len(test),
                "n_labelled": 0, "positive_rate": np.nan,
                "n_bins_realised": 0, "n_fallback_bins": 0,
                "n_clipped_low": 0, "n_clipped_high": 0,
            })
            continue

        thr = fit_label_thresholds(fit[pred_col], fit[actual_col])
        fold_labels = apply_labels(test[pred_col], test[actual_col], thr)

        # Align by index, never by position. test is a filtered view of oof, so
        # its index carries the original row identities.
        labels.loc[test.index] = fold_labels

        # Counted against the RAW edges, because _assign_bins opens them to
        # -inf and +inf before cutting, after which nothing is out of range by
        # construction. Decision 1 clips rather than drops, so these rows are
        # still labelled; this counts how many were judged against a cut-off
        # fitted on predictions unlike their own.
        pred_test = test[pred_col]
        diag_rows.append({
            "fold": k,
            "n_fit": thr.n_fit_rows,
            "n_test": len(test),
            "n_labelled": int(fold_labels.notna().sum()),
            "positive_rate": (float(fold_labels.mean())
                              if fold_labels.notna().any() else np.nan),
            "n_bins_realised": thr.n_bins_realised,
            "n_fallback_bins": len(thr.fallback_bins),
            "n_clipped_low": int(pred_test.lt(thr.edges[0]).sum()),
            "n_clipped_high": int(pred_test.gt(thr.edges[-1]).sum()),
        })

    return labels, pd.DataFrame(diag_rows).set_index("fold")


def relative_error(pred: pd.Series, actual: pd.Series, floor: float) -> pd.Series:
    """The Part 8 robustness definition: absolute error over max(pred, floor).

    The floor stops small denominators exploding. It comes from thr.floor, i.e.
    from the fitting set, never from the rows being scored.
    """
    return (actual - pred).abs() / np.maximum(pred, floor)


if __name__ == "__main__":
    from pathlib import Path

    oof = pd.read_parquet(
        Path(__file__).resolve().parents[1] / "data/oof" / "MY1_oof.parquet"
    )

    # --- mechanical self-check ----------------------------------------------
    # Thresholds applied to the data they were fitted on must give a positive
    # rate of 1 - q. This verifies the binning and clipping only. It is NOT how
    # labels are produced for the watcher.
    thr_all = fit_label_thresholds(oof["yhat_F3"], oof["y_true"])
    rate_all = apply_labels(oof["yhat_F3"], oof["y_true"], thr_all).mean()
    assert np.isclose(rate_all, 1 - Q, atol=0.02), "binning is wrong — check edges"
    print(f"in-sample self-check: {rate_all:.4f} (expect {1 - Q:.2f}) — passed\n")

    # --- the real thing ------------------------------------------------------
    labels, diag = label_walk_forward(oof)
    print(diag.to_string())

    scored = diag["positive_rate"].dropna()
    print(f"\nlabelled: {labels.notna().sum():,} of {len(labels):,} rows")
    print(f"overall positive rate: {labels.mean():.4f}")
    print(f"per-fold rate: min {scored.min():.3f}  max {scored.max():.3f}  "
          f"spread {scored.max() - scored.min():.3f}")

    if np.isclose(scored.std(), 0.0):
        raise SystemExit(
            "every fold has an identical positive rate — thresholds are being "
            "refitted on the fold being labelled"
        )