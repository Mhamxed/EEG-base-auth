"""
normalize.py
------------
Per-subject and global normalization routines for the EEG feature tensors.

The ARRC EEG tensors (12 × 17 × 14) represent spectral-spatial-temporal
features.  Raw values are log-power estimates and vary significantly across
subjects (inter-subject variability).  We apply Z-score normalization
**within each subject** so that the downstream models learn identity-
discriminative structure rather than amplitude differences.

Strategy
--------
1. **Within-subject Z-score**  (default, recommended):
   For each subject, compute mean/std across ALL their trials (pooled from
   all sessions and stimulus types), then standardize.  This removes DC
   bias but preserves intra-subject variation.

2. **Per-channel Z-score**:
   Normalize each of the 12 channels independently (aggregated across
   frequency and time).  Useful when channels have very different amplitude
   ranges.

3. **Global Z-score**:
   Normalize using the full dataset mean/std.  Useful for cross-subject
   comparisons but may leak subject identity via amplitude.

Usage
-----
    from src.preprocessing.normalize import SubjectNormalizer

    norm = SubjectNormalizer(strategy="within_subject")
    X_norm = norm.fit_transform(X, subject_ids)   # X: (N, 12, 17, 14)
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Literal, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Within-Subject Z-score Normalizer
# ─────────────────────────────────────────────────────────────────────────────

class SubjectNormalizer(BaseEstimator, TransformerMixin):
    """
    Z-score normalizer that standardizes each subject's EEG tensors
    independently using statistics computed on the TRAINING split only.

    Parameters
    ----------
    strategy : {'within_subject', 'per_channel', 'global'}
        Normalization scope.
    eps : float
        Small constant to avoid divide-by-zero.
    """

    def __init__(
        self,
        strategy: Literal["within_subject", "per_channel", "global"] = "within_subject",
        eps: float = 1e-8,
    ):
        self.strategy   = strategy
        self.eps        = eps
        self._means: dict = {}   # subject_id → mean array
        self._stds:  dict = {}   # subject_id → std  array

    # ── fit ──────────────────────────────────────────────────────────────────
    def fit(self, X: np.ndarray, subject_ids: np.ndarray) -> "SubjectNormalizer":
        """
        Compute normalization statistics from training data.

        Parameters
        ----------
        X           : ndarray (N, 12, 17, 14)  raw EEG tensors
        subject_ids : ndarray (N,)             integer subject labels
        """
        X           = np.asarray(X, dtype=np.float32)
        subject_ids = np.asarray(subject_ids)

        if self.strategy == "global":
            self._global_mean = X.mean()
            self._global_std  = X.std() + self.eps

        elif self.strategy == "per_channel":
            # Mean / std per channel across all subjects
            # X shape (N, C, F, T) → mean over (N, F, T) → shape (C,)
            self._global_mean = X.mean(axis=(0, 2, 3), keepdims=True)  # (1,C,1,1)
            self._global_std  = X.std (axis=(0, 2, 3), keepdims=True) + self.eps

        else:  # within_subject (default)
            for sid in np.unique(subject_ids):
                mask = subject_ids == sid
                X_sub = X[mask]                             # (n_sub, 12, 17, 14)
                self._means[sid] = X_sub.mean()
                self._stds [sid] = X_sub.std()  + self.eps

        return self

    # ── transform ────────────────────────────────────────────────────────────
    def transform(self, X: np.ndarray, subject_ids: np.ndarray) -> np.ndarray:
        """
        Apply stored normalization statistics.

        Unseen subjects (e.g. open-set authentication) fall back to global
        statistics derived from training subjects.
        """
        X           = np.asarray(X, dtype=np.float32).copy()
        subject_ids = np.asarray(subject_ids)

        if self.strategy in ("global", "per_channel"):
            return (X - self._global_mean) / self._global_std

        # within_subject
        fallback_mean = np.mean(list(self._means.values()))
        fallback_std  = np.mean(list(self._stds.values()))

        for sid in np.unique(subject_ids):
            mask  = subject_ids == sid
            mean  = self._means.get(sid, fallback_mean)
            std   = self._stds .get(sid, fallback_std)
            X[mask] = (X[mask] - mean) / std

        return X

    def fit_transform(self, X: np.ndarray, subject_ids: np.ndarray) -> np.ndarray:
        return self.fit(X, subject_ids).transform(X, subject_ids)


# ─────────────────────────────────────────────────────────────────────────────
# Log-transform helper (for raw power spectra)
# ─────────────────────────────────────────────────────────────────────────────

def log_transform(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply log(1 + X) to stabilize variance of power spectral features.
    Safe for zero/negative values (clips to eps first).
    """
    return np.log1p(np.clip(X, eps, None).astype(np.float32))


# ─────────────────────────────────────────────────────────────────────────────
# Min-Max scaler (per-feature, across training set)
# ─────────────────────────────────────────────────────────────────────────────

class FeatureMinMaxScaler:
    """
    Scales each feature column to [0, 1] using training-set min/max.
    Applied AFTER feature extraction (operates on 2-D feature matrix).
    """

    def __init__(self, feature_range=(0.0, 1.0)):
        self.feature_range = feature_range
        self._min: Optional[np.ndarray] = None
        self._max: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "FeatureMinMaxScaler":
        """X: (N, D) feature matrix."""
        self._min = X.min(axis=0)
        self._max = X.max(axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        lo, hi = self.feature_range
        denom  = (self._max - self._min) + 1e-8
        return lo + (X - self._min) / denom * (hi - lo)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
