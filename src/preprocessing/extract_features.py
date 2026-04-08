"""
extract_features.py
-------------------
Feature extraction for ARRC EEG tensors.

Input tensor shape per trial: (12, 17, 14)
    axis 0 → 12 EEG channels
    axis 1 → 17 frequency bands (pre-computed spectral representation)
    axis 2 → 14 time segments

Three feature families extracted and concatenated:

    1. Differential Entropy (DE)
       log-variance across time segments per (channel, band) cell.
       Shape per trial: (12 × 17,) = 204-dim

    2. Band Power (BP)
       mean power across time segments per (channel, band) cell.
       Shape per trial: (12 × 17,) = 204-dim

    3. Riemannian Covariance (COV)
       upper triangle of (12 × 12) channel covariance matrix,
       computed on the (12, 17×14) unfolded signal.
       Shape per trial: (12 × 13 / 2,) = 78-dim

    Concatenated default → [DE | BP | COV] = 486-dim per trial

Usage
-----
    from src.preprocessing.extract_features import extract_features, build_feature_matrix

    # Single trial
    feat = extract_features(eeg)           # eeg: (12, 17, 14) → (486,)

    # Full dataset (from processed .npy files)
    X_enrol, X_verif = build_feature_matrix()

    # Save
    from src.preprocessing.extract_features import save_features
    save_features()
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
from tqdm import tqdm

from src.utils.config import PROCESSED_PATH   # data/processed/


# ─────────────────────────────────────────────────────────────────────────────
# Band index map
# ARRC's 17 frequency bins are assumed to cover 0–40 Hz in ~2.35 Hz steps.
# Adjust BAND_SLICES if your dataset uses different bin edges.
# ─────────────────────────────────────────────────────────────────────────────

# Approximate mapping of 17 bins → canonical EEG bands
# bin 0  ≈  0 Hz  (DC / below delta — excluded from band-level aggregation)
# bins 1–3  → delta   (1–4 Hz)
# bins 4–6  → theta   (4–8 Hz)
# bins 7–9  → alpha   (8–13 Hz)
# bins 10–13 → beta   (13–30 Hz)
# bins 14–16 → gamma  (30–40 Hz)

BAND_SLICES = {
    "delta": slice(1, 4),
    "theta": slice(4, 7),
    "alpha": slice(7, 10),
    "beta":  slice(10, 14),
    "gamma": slice(14, 17),
}

N_CHANNELS  = 12
N_BANDS     = 17
N_SEGMENTS  = 14
_EPS        = 1e-10     # numerical safety for log


# ─────────────────────────────────────────────────────────────────────────────
# Core per-trial extractors
# ─────────────────────────────────────────────────────────────────────────────

def _de(eeg: np.ndarray) -> np.ndarray:
    """
    Differential Entropy: log-variance across time segments.

    Parameters
    ----------
    eeg : ndarray (12, 17, 14)

    Returns
    -------
    ndarray (204,)  — flattened (channels × bands)
    """
    # var across time axis → (12, 17)
    var = np.var(eeg, axis=2)
    de  = 0.5 * np.log(2 * np.pi * np.e * (var + _EPS))
    return de.ravel()


def _band_power(eeg: np.ndarray) -> np.ndarray:
    """
    Mean band power across time segments.

    Parameters
    ----------
    eeg : ndarray (12, 17, 14)

    Returns
    -------
    ndarray (204,)  — flattened (channels × bands)
    """
    bp = np.mean(eeg, axis=2)   # (12, 17)
    return bp.ravel()


def _covariance(eeg: np.ndarray) -> np.ndarray:
    """
    Upper-triangle of normalised channel covariance matrix.

    The tensor is first unfolded to (channels, bands × segments) = (12, 238),
    then the (12 × 12) covariance is computed and normalised by its trace.

    Parameters
    ----------
    eeg : ndarray (12, 17, 14)

    Returns
    -------
    ndarray (78,)  — upper triangle including diagonal
    """
    # unfold frequency × time into one dimension
    X   = eeg.reshape(N_CHANNELS, -1)          # (12, 238)
    cov = np.cov(X)                             # (12, 12)
    tr  = np.trace(cov) + _EPS
    cov = cov / tr                              # normalise by trace
    idx = np.triu_indices(N_CHANNELS)           # upper triangle indices
    return cov[idx]                             # (78,)


def _band_de(eeg: np.ndarray) -> np.ndarray:
    """
    DE aggregated into 5 canonical bands (delta/theta/alpha/beta/gamma).
    Auxiliary feature — not included in the default 486-dim vector but
    available separately for interpretability analysis.

    Returns
    -------
    ndarray (60,)  — (12 channels × 5 bands)
    """
    out = []
    for band, sl in BAND_SLICES.items():
        segment = eeg[:, sl, :]                  # (12, n_bins, 14)
        var     = np.var(segment, axis=(1, 2))   # (12,)
        de      = 0.5 * np.log(2 * np.pi * np.e * (var + _EPS))
        out.append(de)
    return np.concatenate(out)                   # (60,)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(
    eeg: np.ndarray,
    include_de:  bool = True,
    include_bp:  bool = True,
    include_cov: bool = True,
) -> np.ndarray:
    """
    Extract a flat feature vector from a single EEG trial tensor.

    Parameters
    ----------
    eeg         : ndarray (12, 17, 14)
    include_de  : include Differential Entropy features (204-dim)
    include_bp  : include Band Power features (204-dim)
    include_cov : include Covariance features (78-dim)

    Returns
    -------
    ndarray  shape depends on flags:
        all True  → (486,)
        de + bp   → (408,)
        de only   → (204,)
        etc.
    """
    assert eeg.shape == (N_CHANNELS, N_BANDS, N_SEGMENTS), (
        f"Expected EEG shape (12, 17, 14), got {eeg.shape}"
    )
    parts = []
    if include_de:
        parts.append(_de(eeg))
    if include_bp:
        parts.append(_band_power(eeg))
    if include_cov:
        parts.append(_covariance(eeg))
    if not parts:
        raise ValueError("At least one feature family must be enabled.")
    return np.concatenate(parts)


def build_feature_matrix(
    proc_path:   Optional[Path] = None,
    include_de:  bool = True,
    include_bp:  bool = True,
    include_cov: bool = True,
    verbose:     bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load processed .npy tensors and extract features for every trial.

    Parameters
    ----------
    proc_path   : path to data/processed/  (defaults to config.PROCESSED_PATH)
    include_de  : include DE features
    include_bp  : include band power features
    include_cov : include covariance features
    verbose     : show tqdm progress bars

    Returns
    -------
    X_enrol : ndarray (9800, F)   enrolment feature matrix
    X_verif : ndarray (4628, F)   verification feature matrix
    """
    proc_path = Path(proc_path) if proc_path else PROCESSED_PATH

    enrol_raw = np.load(proc_path / "enrol_eeg.npy")   # (9800, 12, 17, 14)
    verif_raw = np.load(proc_path / "verif_eeg.npy")   # (4628, 12, 17, 14)

    if verbose:
        print(f"Enrolment tensor : {enrol_raw.shape}")
        print(f"Verification tensor: {verif_raw.shape}")

    def _extract_all(tensor: np.ndarray, desc: str) -> np.ndarray:
        feats = []
        for i in tqdm(range(len(tensor)), desc=desc, disable=not verbose):
            feats.append(
                extract_features(
                    tensor[i],
                    include_de=include_de,
                    include_bp=include_bp,
                    include_cov=include_cov,
                )
            )
        return np.stack(feats)

    X_enrol = _extract_all(enrol_raw, "Extracting enrolment features")
    X_verif = _extract_all(verif_raw, "Extracting verification features")

    if verbose:
        print(f"\nEnrolment features : {X_enrol.shape}")
        print(f"Verification features: {X_verif.shape}")

    return X_enrol, X_verif


def save_features(
    proc_path:   Optional[Path] = None,
    include_de:  bool = True,
    include_bp:  bool = True,
    include_cov: bool = True,
    verbose:     bool = True,
) -> None:
    """
    Extract features and save to data/processed/.

    Produces:
        enrol_features.npy   (9800, 486)
        verif_features.npy   (4628, 486)
        feature_info.txt     dimensionality report
    """
    proc_path = Path(proc_path) if proc_path else PROCESSED_PATH

    X_enrol, X_verif = build_feature_matrix(
        proc_path=proc_path,
        include_de=include_de,
        include_bp=include_bp,
        include_cov=include_cov,
        verbose=verbose,
    )

    np.save(proc_path / "enrol_features.npy", X_enrol)
    np.save(proc_path / "verif_features.npy", X_verif)

    # ── feature dimension breakdown ───────────────────────────────────────────
    dim_de  = 204 if include_de  else 0
    dim_bp  = 204 if include_bp  else 0
    dim_cov =  78 if include_cov else 0
    total   = dim_de + dim_bp + dim_cov

    report = [
        "ARRC Feature Extraction Report",
        "=" * 40,
        f"DE  features : {dim_de:>4}  dims  (cols   0 – {dim_de-1})",
        f"BP  features : {dim_bp:>4}  dims  (cols {dim_de} – {dim_de+dim_bp-1})",
        f"COV features : {dim_cov:>4}  dims  (cols {dim_de+dim_bp} – {total-1})",
        f"Total        : {total:>4}  dims",
        "",
        f"enrol_features.npy : {X_enrol.shape}",
        f"verif_features.npy : {X_verif.shape}",
    ]
    (proc_path / "feature_info.txt").write_text("\n".join(report))

    print(f"\n✅ Features saved to {proc_path.resolve()}")
    print(f"   enrol_features.npy  {X_enrol.shape}")
    print(f"   verif_features.npy  {X_verif.shape}")
    print(f"   Total feature dims : {total}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    save_features(verbose=True)
