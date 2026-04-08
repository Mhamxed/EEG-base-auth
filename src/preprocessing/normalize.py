"""
normalize.py
------------
Fit a StandardScaler on enrolment features and apply it to both enrolment
and verification sets.

WHY THIS STEP IS MANDATORY BEFORE ANY CLASSIFIER
-------------------------------------------------
The 486-dim feature vector concatenates three families with very different
dynamic ranges:

    DE  (cols 0–203)    log-variance values, typically in [-2, 4]
    BP  (cols 204–407)  raw mean power, scale depends on signal amplitude
    COV (cols 408–485)  normalised covariance, values in [-1, 1]

Without standardisation:
  • LDA's within-class covariance estimate is dominated by BP (largest scale),
    so DE and COV contribute almost nothing to the discriminant.
  • Cross-stimulus EER comparisons are unfair because stimuli differ in
    average signal amplitude, inflating BP variance for some types.

The scaler is FIT ONLY on enrolment data (sessions 1+2) and then applied
to verification (session 3). Fitting on verification would constitute data
leakage — the scaler must not have seen the test distribution.

Per-stimulus normalisation option
----------------------------------
Set per_stimulus=True to fit a separate scaler for each stimulus type.
This removes between-stimulus amplitude differences, making EER numbers
directly comparable across stimulus types.  Use this when your research
question is "which stimulus gives better biometric performance" — which
is exactly RQ1 of this study.

Outputs written to data/processed/
------------------------------------
    enrol_features_norm.npy     (N_enrol, 486)  z-scored enrolment
    verif_features_norm.npy     (N_verif, 486)  z-scored verification
    scaler_global.pkl           global StandardScaler (fitted on all enrol)
    scalers_per_stimulus.pkl    dict[stimuli_type → StandardScaler]
                                (only when per_stimulus=True)
    normalisation_report.txt    summary of means/stds before and after

Usage
-----
    # as a script
    python -m src.preprocessing.normalize

    # as a module
    from src.preprocessing.normalize import load_normalized

    X_enrol, X_verif, enrol_meta, verif_meta = load_normalized()
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

from sklearn.preprocessing import StandardScaler

from src.utils.config import PROCESSED_PATH


# ─────────────────────────────────────────────────────────────────────────────
# Feature block boundaries (must match extract_features.py)
# ─────────────────────────────────────────────────────────────────────────────
BLOCK_DE  = slice(0,   204)
BLOCK_BP  = slice(204, 408)
BLOCK_COV = slice(408, 486)

BLOCK_NAMES = {
    "DE":  BLOCK_DE,
    "BP":  BLOCK_BP,
    "COV": BLOCK_COV,
}


# ─────────────────────────────────────────────────────────────────────────────
# Core normalisation
# ─────────────────────────────────────────────────────────────────────────────

def normalize(
    proc_path:    Optional[Path] = None,
    per_stimulus: bool           = True,
    verbose:      bool           = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit scaler(s) on enrolment features and transform both sets.

    Parameters
    ----------
    proc_path    : path to data/processed/ (defaults to config.PROCESSED_PATH)
    per_stimulus : if True, fit one scaler per stimulus type on enrolment data.
                   Produces both global and per-stimulus normalised arrays.
    verbose      : print diagnostic summary

    Returns
    -------
    X_enrol_norm : ndarray (N_enrol, 486)
    X_verif_norm : ndarray (N_verif, 486)
    """
    proc_path = Path(proc_path) if proc_path else PROCESSED_PATH

    # ── Load raw features ────────────────────────────────────────────────────
    X_enrol = np.load(proc_path / "enrol_features.npy")   # (N_enrol, 486)
    X_verif = np.load(proc_path / "verif_features.npy")   # (N_verif, 486)

    enrol_meta = pd.read_csv(proc_path / "enrol_meta.csv")
    verif_meta = pd.read_csv(proc_path / "verif_meta.csv")

    assert len(X_enrol) == len(enrol_meta), "Enrolment feature/meta length mismatch"
    assert len(X_verif) == len(verif_meta), "Verification feature/meta length mismatch"

    if verbose:
        print(f"\nLoaded enrol : {X_enrol.shape}  verif : {X_verif.shape}")
        _print_raw_stats(X_enrol, "Enrolment (raw)")

    # ── Global normalisation ─────────────────────────────────────────────────
    global_scaler = StandardScaler()
    global_scaler.fit(X_enrol)

    X_enrol_norm = global_scaler.transform(X_enrol)
    X_verif_norm = global_scaler.transform(X_verif)

    np.save(proc_path / "enrol_features_norm.npy", X_enrol_norm)
    np.save(proc_path / "verif_features_norm.npy", X_verif_norm)

    with open(proc_path / "scaler_global.pkl", "wb") as f:
        pickle.dump(global_scaler, f)

    if verbose:
        _print_raw_stats(X_enrol_norm, "Enrolment (global z-scored)")

    # ── Per-stimulus normalisation ───────────────────────────────────────────
    scalers: Dict[str, StandardScaler] = {}

    if per_stimulus:
        stim_types = enrol_meta["stimuli_type"].unique()

        # Allocate output arrays (same shape, filled stimulus-by-stimulus)
        X_enrol_stim = np.zeros_like(X_enrol)
        X_verif_stim = np.zeros_like(X_verif)

        for st in sorted(stim_types):
            e_mask = enrol_meta["stimuli_type"] == st
            v_mask = verif_meta["stimuli_type"] == st

            X_e = X_enrol[e_mask]
            X_v = X_verif[v_mask]

            if len(X_e) == 0:
                if verbose:
                    print(f"  WARNING: stimulus '{st}' has 0 enrolment trials — skipped")
                continue

            sc = StandardScaler()
            sc.fit(X_e)
            scalers[st] = sc

            X_enrol_stim[e_mask] = sc.transform(X_e)
            if len(X_v) > 0:
                X_verif_stim[v_mask] = sc.transform(X_v)

            if verbose:
                print(f"  [{st:<12}]  enrol={e_mask.sum():>5,}  verif={v_mask.sum():>5,}"
                      f"  mean≈{X_e.mean():.3f}  std≈{X_e.std():.3f}")

        np.save(proc_path / "enrol_features_norm_perstim.npy", X_enrol_stim)
        np.save(proc_path / "verif_features_norm_perstim.npy", X_verif_stim)

        with open(proc_path / "scalers_per_stimulus.pkl", "wb") as f:
            pickle.dump(scalers, f)

        if verbose:
            print(f"\n  Per-stimulus scalers fitted for {len(scalers)} stimulus types")

    # ── Write normalisation report ───────────────────────────────────────────
    report = _build_report(
        X_enrol, X_enrol_norm,
        enrol_meta, scalers if per_stimulus else None
    )
    (proc_path / "normalisation_report.txt").write_text(report)

    print(f"\n✅ Normalisation complete → {proc_path.resolve()}")
    print(f"   enrol_features_norm.npy        {X_enrol_norm.shape}")
    print(f"   verif_features_norm.npy        {X_verif_norm.shape}")
    if per_stimulus:
        print(f"   enrol_features_norm_perstim.npy (same shape, per-stimulus scaled)")
        print(f"   verif_features_norm_perstim.npy (same shape, per-stimulus scaled)")
        print(f"   scalers_per_stimulus.pkl        {len(scalers)} scalers")
    print(f"   scaler_global.pkl")
    print(f"   normalisation_report.txt")

    return X_enrol_norm, X_verif_norm


# ─────────────────────────────────────────────────────────────────────────────
# Load helper
# ─────────────────────────────────────────────────────────────────────────────

def load_normalized(
    proc_path:    Optional[Path] = None,
    per_stimulus: bool           = False,
) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame]:
    """
    Load normalised features and aligned metadata.

    Parameters
    ----------
    proc_path    : path to data/processed/
    per_stimulus : if True, load per-stimulus-normalised arrays instead of
                   global ones.  Use this for cross-stimulus comparisons.

    Returns
    -------
    X_enrol    : ndarray (N_enrol, 486)
    enrol_meta : DataFrame
    X_verif    : ndarray (N_verif, 486)
    verif_meta : DataFrame
    """
    proc_path = Path(proc_path) if proc_path else PROCESSED_PATH

    if per_stimulus:
        enrol_f = proc_path / "enrol_features_norm_perstim.npy"
        verif_f = proc_path / "verif_features_norm_perstim.npy"
    else:
        enrol_f = proc_path / "enrol_features_norm.npy"
        verif_f = proc_path / "verif_features_norm.npy"

    for p in (enrol_f, verif_f):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Run normalize.py first."
            )

    X_enrol    = np.load(enrol_f)
    X_verif    = np.load(verif_f)
    enrol_meta = pd.read_csv(proc_path / "enrol_meta.csv")
    verif_meta = pd.read_csv(proc_path / "verif_meta.csv")

    return X_enrol, enrol_meta, X_verif, verif_meta


def load_scaler(
    proc_path:    Optional[Path] = None,
    per_stimulus: bool           = False,
) -> object:
    """
    Load the fitted scaler(s).

    Returns
    -------
    StandardScaler (global) or dict[str, StandardScaler] (per_stimulus=True)
    """
    proc_path = Path(proc_path) if proc_path else PROCESSED_PATH
    fname = "scalers_per_stimulus.pkl" if per_stimulus else "scaler_global.pkl"
    with open(proc_path / fname, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_raw_stats(X: np.ndarray, label: str) -> None:
    print(f"\n  {label}:")
    for name, sl in BLOCK_NAMES.items():
        block = X[:, sl]
        print(f"    {name}  mean={block.mean():+.4f}  std={block.std():.4f}"
              f"  min={block.min():+.4f}  max={block.max():+.4f}")


def _build_report(
    X_raw:     np.ndarray,
    X_norm:    np.ndarray,
    meta:      pd.DataFrame,
    scalers:   Optional[Dict],
) -> str:
    lines = [
        "ARRC Normalisation Report",
        "=" * 50,
        "",
        "Global statistics before z-scoring:",
    ]
    for name, sl in BLOCK_NAMES.items():
        b = X_raw[:, sl]
        lines.append(
            f"  {name:<4}  mean={b.mean():+.6f}  std={b.std():.6f}"
            f"  min={b.min():+.6f}  max={b.max():+.6f}"
        )

    lines += ["", "Global statistics after z-scoring:"]
    for name, sl in BLOCK_NAMES.items():
        b = X_norm[:, sl]
        lines.append(
            f"  {name:<4}  mean={b.mean():+.6f}  std={b.std():.6f}"
            f"  min={b.min():+.6f}  max={b.max():+.6f}"
        )

    if scalers:
        lines += ["", "Per-stimulus trial counts (enrolment):"]
        for st, sc in sorted(scalers.items()):
            mask = meta["stimuli_type"] == st
            lines.append(
                f"  {st:<14}  {mask.sum():>5,} trials"
                f"  feature_mean={sc.mean_.mean():+.4f}"
                f"  feature_std={sc.scale_.mean():.4f}"
            )

    lines += [
        "",
        "Note: scaler fitted on enrolment data only.",
        "Applying it to verification data does NOT constitute leakage.",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    normalize(per_stimulus=True, verbose=True)
