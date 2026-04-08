"""
per_stimulus_baseline.py
------------------------
Train one LDA biometric baseline per stimulus type, compute authentication
score matrices, and report Equal Error Rate (EER) per stimulus.

This is the first point in the pipeline where you see *which stimuli carry
biometric signal* — the direct answer to Research Question 1.

Design decisions explained
--------------------------
Classifier choice — LDA
    LDA finds the linear projection that maximises between-subject variance
    relative to within-subject variance.  For EEG biometrics with tens of
    subjects and hundreds of trials it is a strong, fast baseline that is
    both interpretable and well-studied in the literature.  We use it as a
    *baseline* — not necessarily the final system.

    Sample size constraint: with 21 subjects and ~480 total features (after
    PCA), LDA is close to the small-sample problem boundary.  We therefore
    apply PCA first to reduce to N_COMPONENTS << N_samples, then LDA on the
    projected space.  This is the standard "PCA+LDA" or "Fisherface"-style
    approach for high-dimensional small-sample biometrics.

Scoring
    For each test trial we compute the cosine similarity between the trial's
    LDA-projected vector and the *enrolled template* for each subject (mean
    of that subject's enrolment trials in LDA space).  This produces a
    (N_test_trials × N_subjects) score matrix per stimulus.

EER computation
    For each subject s:
        genuine scores  = column s of score matrix, rows where true label == s
        impostor scores = column s of score matrix, rows where true label != s
    We vary the threshold and report the point where FAR ≈ FRR.

Output files (data/processed/results/)
---------------------------------------
    scores/
        scores_<STIMULUS>.npz     score matrix + true labels + subject index
    eer_per_stimulus.csv          EER, AUC, n_genuine, n_impostor per stimulus
    lda_models/
        lda_<STIMULUS>.pkl        fitted PCA+LDA pipeline per stimulus
    baseline_report.txt           ranked summary table

Usage
-----
    python -m src.analysis.per_stimulus_baseline

    from src.analysis.per_stimulus_baseline import run_all_stimuli, load_results
    results_df = run_all_stimuli()
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc

from src.utils.config import PROCESSED_PATH
from src.preprocessing.normalize import load_normalized


# ─────────────────────────────────────────────────────────────────────────────
# Hyper-parameters
# ─────────────────────────────────────────────────────────────────────────────

# PCA: retain enough variance before LDA to avoid singular covariance.
# We cap at (n_subjects - 1) = 20 for LDA dimensionality, so PCA needs to
# produce at least 20 components.  150 is a safe upper bound that retains
# >99% variance in practice while keeping the problem well-conditioned.
PCA_COMPONENTS   = 150

# LDA: number of discriminant components = min(n_classes-1, PCA_COMPONENTS)
# Sklearn handles this automatically; we set n_components=None to let it pick.
LDA_SOLVER       = "svd"      # no matrix inversion needed; stable for small n

# Results subdirectory
RESULTS_DIR_NAME = "results"


# ─────────────────────────────────────────────────────────────────────────────
# EER computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_eer(
    genuine:  np.ndarray,
    impostor: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute Equal Error Rate and ROC-AUC from genuine/impostor score arrays.

    Parameters
    ----------
    genuine  : 1-D array of similarity scores for same-subject pairs
    impostor : 1-D array of similarity scores for different-subject pairs

    Returns
    -------
    eer : float in [0, 1] — lower is better
    auc_score : float in [0, 1] — higher is better
    """
    y_true  = np.concatenate([np.ones(len(genuine)),  np.zeros(len(impostor))])
    y_score = np.concatenate([genuine,                 impostor])

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_score = auc(fpr, tpr)

    # EER: the threshold where FPR ≈ FNR (= 1 - TPR)
    fnr = 1.0 - tpr
    abs_diff = np.abs(fpr - fnr)
    eer_idx  = np.argmin(abs_diff)
    eer      = float(np.mean([fpr[eer_idx], fnr[eer_idx]]))

    return eer, auc_score


# ─────────────────────────────────────────────────────────────────────────────
# Per-stimulus LDA pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_pipeline(n_pca: int = PCA_COMPONENTS) -> Pipeline:
    """
    PCA → LDA pipeline.

    PCA reduces dimensionality to avoid the small-sample singularity that
    plagues LDA when n_features >> n_samples.  SVD-based LDA is then applied
    in the reduced space.
    """
    return Pipeline([
        ("pca", PCA(n_components=n_pca, whiten=True, random_state=42)),
        ("lda", LinearDiscriminantAnalysis(solver=LDA_SOLVER, store_covariance=False)),
    ])


def _enrol_templates(
    pipe:       Pipeline,
    X_enrol:    np.ndarray,
    y_enrol:    np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project enrolment data into LDA space and compute per-subject templates
    as the mean of all enrolment trials for that subject.

    Returns
    -------
    templates   : ndarray (n_subjects, n_lda_dims) — one row per subject
    subject_ids : ndarray (n_subjects,) — subject ID for each template row
    """
    X_proj       = pipe.transform(X_enrol)             # (N_enrol, n_lda)
    subject_ids  = np.unique(y_enrol)
    templates    = np.stack([
        X_proj[y_enrol == s].mean(axis=0)
        for s in subject_ids
    ])
    return templates, subject_ids


def _score_matrix(
    X_verif:    np.ndarray,
    pipe:       Pipeline,
    templates:  np.ndarray,
    subject_ids: np.ndarray,
) -> np.ndarray:
    """
    Compute cosine similarity between each verification trial and each
    enrolled subject template.

    Returns
    -------
    scores : ndarray (N_verif, n_subjects)
        scores[i, j] = cosine similarity of trial i with subject j's template
    """
    X_proj = pipe.transform(X_verif)                   # (N_verif, n_lda)

    # L2-normalise for cosine similarity
    X_norm  = X_proj   / (np.linalg.norm(X_proj,   axis=1, keepdims=True) + 1e-10)
    T_norm  = templates / (np.linalg.norm(templates, axis=1, keepdims=True) + 1e-10)

    return X_norm @ T_norm.T                           # (N_verif, n_subjects)


def run_one_stimulus(
    stimulus:    str,
    X_enrol:     np.ndarray,
    y_enrol:     np.ndarray,
    X_verif:     np.ndarray,
    y_verif:     np.ndarray,
    results_dir: Path,
    verbose:     bool = True,
) -> Dict:
    """
    Full train-score-evaluate cycle for a single stimulus type.

    Returns a dict with EER, AUC, and counts for the results table.
    """
    n_classes = len(np.unique(y_enrol))

    if n_classes < 2:
        print(f"  [{stimulus}] skipped — only {n_classes} subject(s) in enrolment set")
        return None

    if len(X_enrol) < n_classes:
        print(f"  [{stimulus}] skipped — fewer enrolment trials ({len(X_enrol)}) "
              f"than subjects ({n_classes})")
        return None

    # Clip PCA components to what the data can support
    n_pca = min(PCA_COMPONENTS, len(X_enrol) - 1, X_enrol.shape[1])

    # ── Build and fit pipeline ────────────────────────────────────────────────
    pipe = build_pipeline(n_pca=n_pca)
    pipe.fit(X_enrol, y_enrol)

    # ── Enrolment templates ───────────────────────────────────────────────────
    templates, subject_ids = _enrol_templates(pipe, X_enrol, y_enrol)

    # ── Score matrix on verification set ─────────────────────────────────────
    if len(X_verif) == 0:
        print(f"  [{stimulus}] no verification trials — skipped")
        return None

    scores = _score_matrix(X_verif, pipe, templates, subject_ids)

    # ── Genuine / impostor pools ──────────────────────────────────────────────
    genuine_scores  = []
    impostor_scores = []
    subj_to_col     = {s: i for i, s in enumerate(subject_ids)}

    for row_idx, true_subj in enumerate(y_verif):
        if true_subj not in subj_to_col:
            continue                                    # subject not enrolled
        genuine_col  = subj_to_col[true_subj]
        genuine_scores.append(scores[row_idx, genuine_col])
        impostor_cols = [c for c in range(len(subject_ids)) if c != genuine_col]
        impostor_scores.extend(scores[row_idx, impostor_cols].tolist())

    if not genuine_scores or not impostor_scores:
        print(f"  [{stimulus}] empty genuine or impostor pool — skipped")
        return None

    genuine_arr  = np.array(genuine_scores)
    impostor_arr = np.array(impostor_scores)

    eer, auc_score = compute_eer(genuine_arr, impostor_arr)

    # ── Save scores and model ─────────────────────────────────────────────────
    scores_dir = results_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        scores_dir / f"scores_{stimulus}.npz",
        scores       = scores,
        y_verif      = y_verif,
        subject_ids  = subject_ids,
        genuine      = genuine_arr,
        impostor     = impostor_arr,
    )

    model_dir = results_dir / "lda_models"
    model_dir.mkdir(parents=True, exist_ok=True)

    with open(model_dir / f"lda_{stimulus}.pkl", "wb") as fh:
        pickle.dump({"pipe": pipe, "templates": templates, "subject_ids": subject_ids}, fh)

    if verbose:
        print(f"  [{stimulus:<14}]  "
              f"enrol={len(X_enrol):>5,}  verif={len(X_verif):>5,}  "
              f"subjects={n_classes:>2}  "
              f"EER={eer*100:.2f}%  AUC={auc_score:.4f}")

    return {
        "stimulus":    stimulus,
        "n_enrol":     len(X_enrol),
        "n_verif":     len(X_verif),
        "n_subjects":  n_classes,
        "n_genuine":   len(genuine_arr),
        "n_impostor":  len(impostor_arr),
        "eer":         eer,
        "eer_pct":     eer * 100,
        "auc":         auc_score,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Run all stimuli
# ─────────────────────────────────────────────────────────────────────────────

def run_all_stimuli(
    proc_path:    Optional[Path] = None,
    per_stimulus: bool           = True,
    verbose:      bool           = True,
) -> pd.DataFrame:
    """
    Run the LDA baseline for every stimulus type found in the dataset.

    Parameters
    ----------
    proc_path    : path to data/processed/
    per_stimulus : use per-stimulus-normalised features (recommended for
                   fair cross-stimulus comparison — True by default).
    verbose      : print per-stimulus progress

    Returns
    -------
    results_df : DataFrame with EER and AUC per stimulus, sorted by EER
    """
    proc_path   = Path(proc_path) if proc_path else PROCESSED_PATH
    results_dir = proc_path / RESULTS_DIR_NAME
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Load normalised features ──────────────────────────────────────────────
    X_enrol, enrol_meta, X_verif, verif_meta = load_normalized(
        proc_path=proc_path, per_stimulus=per_stimulus
    )

    if verbose:
        norm_label = "per-stimulus" if per_stimulus else "global"
        print(f"\nUsing {norm_label}-normalised features")
        print(f"Enrolment : {X_enrol.shape}")
        print(f"Verification: {X_verif.shape}")
        print()

    # ── Iterate over stimulus types ───────────────────────────────────────────
    stimulus_types = sorted(
        set(enrol_meta["stimuli_type"].unique()) |
        set(verif_meta["stimuli_type"].unique())
    )

    if verbose:
        print(f"Stimulus types found: {stimulus_types}\n")

    rows = []
    for st in stimulus_types:
        e_mask = enrol_meta["stimuli_type"] == st
        v_mask = verif_meta["stimuli_type"] == st

        Xe = X_enrol[e_mask]
        ye = _get_labels(enrol_meta[e_mask])
        Xv = X_verif[v_mask]
        yv = _get_labels(verif_meta[v_mask])

        row = run_one_stimulus(
            stimulus    = st,
            X_enrol     = Xe,
            y_enrol     = ye,
            X_verif     = Xv,
            y_verif     = yv,
            results_dir = results_dir,
            verbose     = verbose,
        )
        if row is not None:
            rows.append(row)

    results_df = pd.DataFrame(rows).sort_values("eer").reset_index(drop=True)

    # ── Save summary table ────────────────────────────────────────────────────
    results_df.to_csv(results_dir / "eer_per_stimulus.csv", index=False)

    # ── Print ranked table ────────────────────────────────────────────────────
    if verbose:
        _print_ranked_table(results_df)

    # ── Write text report ─────────────────────────────────────────────────────
    _write_report(results_df, results_dir)

    return results_df


# ─────────────────────────────────────────────────────────────────────────────
# Load saved results
# ─────────────────────────────────────────────────────────────────────────────

def load_results(
    proc_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Load the EER-per-stimulus CSV produced by run_all_stimuli()."""
    proc_path = Path(proc_path) if proc_path else PROCESSED_PATH
    csv_path  = proc_path / RESULTS_DIR_NAME / "eer_per_stimulus.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run per_stimulus_baseline.py first."
        )
    return pd.read_csv(csv_path)


def load_scores(
    stimulus:  str,
    proc_path: Optional[Path] = None,
) -> Dict[str, np.ndarray]:
    """
    Load saved score arrays for a specific stimulus type.

    Returns dict with keys: scores, y_verif, subject_ids, genuine, impostor
    """
    proc_path  = Path(proc_path) if proc_path else PROCESSED_PATH
    npz_path   = proc_path / RESULTS_DIR_NAME / "scores" / f"scores_{stimulus}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"{npz_path} not found.")
    data = np.load(npz_path)
    return {k: data[k] for k in data.files}


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_labels(meta_subset: pd.DataFrame) -> np.ndarray:
    """
    Return 0-based consecutive integer labels for subject_id, preserving
    the original subject ordering within the full 21-subject set.

    Using subject_id directly (not re-indexed per-stimulus) ensures that
    'subject 5 in IMAGE == subject 5 in COGNITIVE'.
    """
    return meta_subset["subject_id"].values.astype(np.int64)


def _print_ranked_table(df: pd.DataFrame) -> None:
    print("\n" + "═" * 70)
    print("  Per-Stimulus LDA Baseline — EER Ranking")
    print("═" * 70)
    print(f"  {'Rank':<6} {'Stimulus':<16} {'EER %':>7} {'AUC':>8}"
          f"  {'Enrol':>7}  {'Verif':>6}  {'Subjects':>8}")
    print("  " + "─" * 66)
    for rank, row in df.iterrows():
        print(f"  {rank+1:<6} {row['stimulus']:<16} {row['eer_pct']:>6.2f}%"
              f" {row['auc']:>8.4f}"
              f"  {int(row['n_enrol']):>7,}  {int(row['n_verif']):>6,}"
              f"  {int(row['n_subjects']):>8}")
    print("═" * 70)
    best = df.iloc[0]
    worst = df.iloc[-1]
    print(f"\n  Best  stimulus: {best['stimulus']:<14}  EER = {best['eer_pct']:.2f}%")
    print(f"  Worst stimulus: {worst['stimulus']:<14}  EER = {worst['eer_pct']:.2f}%\n")


def _write_report(df: pd.DataFrame, results_dir: Path) -> None:
    lines = [
        "Per-Stimulus LDA Baseline Report",
        "=" * 60,
        "",
        f"{'Rank':<6} {'Stimulus':<16} {'EER %':>7} {'AUC':>8}"
        f"  {'Enrol':>7}  {'Verif':>6}",
        "─" * 60,
    ]
    for i, row in df.iterrows():
        lines.append(
            f"{i+1:<6} {row['stimulus']:<16} {row['eer_pct']:>6.2f}%"
            f" {row['auc']:>8.4f}"
            f"  {int(row['n_enrol']):>7,}  {int(row['n_verif']):>6,}"
        )
    lines += [
        "",
        "Notes:",
        "  • EER computed on session-3 (verification) data.",
        "  • Features normalised per-stimulus before LDA.",
        "  • PCA whitening applied before LDA (n_components=min(150, N-1, F)).",
        "  • Scoring: cosine similarity in LDA space vs. per-subject mean template.",
    ]
    (results_dir / "baseline_report.txt").write_text("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = run_all_stimuli(per_stimulus=True, verbose=True)
    print(f"\nSaved to: {(PROCESSED_PATH / RESULTS_DIR_NAME).resolve()}")
