"""
load_arrc.py
------------
Parses ARRC.mat into structured Python objects.

The ARRC dataset contains 14,876 trials across 21 subjects and 5 stimulus
types.  Each trial stores:

    DATA[0, i].data        → ndarray (12, 17, 14)  EEG feature tensor
    DATA[0, i].y           → uint8                 subject ID (label)
    DATA[0, i].stimuli     → str                   stimulus type
    DATA[0, i].session     → uint8                 session number (1–3)
    DATA[0, i].INFO        → struct
        .STIMULI            → str   ('IMAGE', 'COGNITIVE', 'SSVEP', …)
        .VOTED              → (2,)  [arousal, valence] rated by subject (0–4)
        .EXPECTED_VOTES     → (2,)  normative OASIS/GAPED scores (z-scored)
        .CONTENT            → str   image/task content identifier
        -- or --
        .INPUT              → str   (cognitive task: "91 + 29=120")
        .OPEN_CLOSED        → str   (resting state: 'open' / 'closed')
        .FREQ               → str   (SSVEP stimulus frequency)

Usage
-----
    from src.preprocessing.load_arrc import load_arrc, get_emotion_trials

    all_trials   = load_arrc("data/raw/ARRC.mat")
    emotion_only = get_emotion_trials(all_trials)   # 1,480 IMAGE trials
"""

import scipy.io as sio
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from collections import Counter

from src.utils.config import ARRC_PATH


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_str(arr) -> str:
    """Extract a scalar string from a MATLAB object array."""
    try:
        return str(arr.flat[0])
    except Exception:
        return ""


def _parse_info(info_struct) -> Dict:
    """
    Parse a MATLAB INFO struct (numpy.void) into a plain Python dict.

    INFO field layout varies by stimulus type:
      IMAGE      → STIMULI, VOTED, EXPECTED_VOTES, CONTENT
      COGNITIVE  → STIMULI, INPUT
      SSVEP*     → STIMULI, FREQ
      EYES       → STIMULI, OPEN_CLOSED
    """
    result = {
        "stimuli_type":    "",
        "arousal_voted":   np.nan,
        "valence_voted":   np.nan,
        "arousal_norm":    np.nan,
        "valence_norm":    np.nan,
        "content":         "",
        "cognitive_input": "",
        "open_closed":     "",
        "ssvep_freq":      "",
    }

    fnames = info_struct.dtype.names
    if fnames is None:
        return result

    for fn in fnames:
        val = info_struct[fn]
        fn_up = fn.upper()

        if fn_up == "STIMULI":
            result["stimuli_type"] = _safe_str(val).upper()

        elif fn_up == "VOTED":
            arr = val.flatten()
            if len(arr) >= 2:
                result["arousal_voted"] = float(arr[0])
                result["valence_voted"] = float(arr[1])
            elif len(arr) == 1:
                result["arousal_voted"] = float(arr[0])

        elif fn_up == "EXPECTED_VOTES":
            arr = val.flatten()
            if len(arr) >= 2:
                result["arousal_norm"] = float(arr[0])
                result["valence_norm"] = float(arr[1])
            elif len(arr) == 1:
                result["arousal_norm"] = float(arr[0])

        elif fn_up == "CONTENT":
            result["content"] = _safe_str(val)

        elif fn_up == "INPUT":
            result["cognitive_input"] = _safe_str(val)

        elif fn_up == "OPEN_CLOSED":
            result["open_closed"] = _safe_str(val).lower()

        elif fn_up == "FREQ":
            result["ssvep_freq"] = _safe_str(val)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_arrc(mat_path: Optional[Path] = None, verbose: bool = True) -> pd.DataFrame:
    """
    Load ARRC.mat and return a tidy DataFrame with one row per trial.

    Parameters
    ----------
    mat_path : Path, optional
        Path to ARRC.mat.  Defaults to config.ARRC_PATH.
    verbose : bool
        Show tqdm progress bar.

    Returns
    -------
    pd.DataFrame with columns:
        trial_idx, subject_id, session, stimuli_type,
        arousal_voted, valence_voted, arousal_norm, valence_norm,
        content, cognitive_input, open_closed, ssvep_freq,
        eeg_data   ← numpy array stored as object (shape may vary per trial)
    """
    path = Path(mat_path) if mat_path else ARRC_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"ARRC.mat not found at {path}.\n"
            f"Place the file at data/raw/ARRC.mat (or pass mat_path explicitly)."
        )

    if verbose:
        print(f"Loading {path} …", flush=True)

    mat  = sio.loadmat(str(path))
    DATA = mat["DATA"]          # shape (1, 14876), structured array
    N    = DATA.shape[1]

    rows = []
    iterator = tqdm(range(N), desc="Parsing trials", disable=not verbose)

    for i in iterator:
        entry = DATA[0, i]

        # ── EEG tensor ───────────────────────────────────────────────────────
        eeg = entry["data"][0, 0]               # typically (12, 17, 14)
        # ── Scalar fields ────────────────────────────────────────────────────
        subject_id = int(entry["y"][0, 0])
        session    = int(entry["session"][0, 0])

        # stimuli string lives in two places; prefer top-level field
        stim_raw   = _safe_str(entry["stimuli"]).upper()

        # ── INFO struct ──────────────────────────────────────────────────────
        info_arr  = entry["INFO"]               # ndarray (1,1)
        info_dict = _parse_info(info_arr[0, 0])

        # Resolve stimulus type (top-level 'stimuli' field is authoritative)
        stimuli_type = stim_raw if stim_raw else info_dict["stimuli_type"]

        row = {
            "trial_idx":       i,
            "subject_id":      subject_id,
            "session":         session,
            "stimuli_type":    stimuli_type,
            "arousal_voted":   info_dict["arousal_voted"],
            "valence_voted":   info_dict["valence_voted"],
            "arousal_norm":    info_dict["arousal_norm"],
            "valence_norm":    info_dict["valence_norm"],
            "content":         info_dict["content"],
            "cognitive_input": info_dict["cognitive_input"],
            "open_closed":     info_dict["open_closed"],
            "ssvep_freq":      info_dict["ssvep_freq"],
            "eeg_data":        eeg,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df["subject_id"] = df["subject_id"].astype("int16")
    df["session"]    = df["session"].astype("int8")

    if verbose:
        _print_summary(df)

    return df


def get_emotion_trials(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to IMAGE stimulus trials only (the 1,480 emotion trials).

    These are the trials where subjects viewed affective images from the
    OASIS or GAPED databases and self-reported arousal/valence ratings.

    Returns a copy with reset index.
    """
    mask = df["stimuli_type"].str.upper() == "IMAGE"
    out  = df[mask].copy().reset_index(drop=True)
    return out


def get_eeg_matrix(
    df: pd.DataFrame,
    strategy: str = "filter",
    target_shape: Optional[Tuple] = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Stack all EEG tensors into a single numpy array, handling shape mismatches.

    Parameters
    ----------
    df : DataFrame with 'eeg_data' column
    strategy : str
        'filter'   – keep only trials matching the most common shape (default).
                     Zero data distortion. Recommended for clean pipelines.
        'truncate' – crop every trial to target_shape (or per-dim minimum).
                     Keeps all trials, may lose time samples.
        'pad'      – zero-pad every trial to target_shape (or per-dim maximum).
                     Keeps all trials, introduces synthetic zeros.
    target_shape : tuple, optional
        Explicit target shape (excluding batch N dimension).
        If None, inferred automatically per strategy.

    Returns
    -------
    X       : ndarray, shape (N, ...)   — stacked EEG tensor
    out_df  : DataFrame aligned row-for-row with X
              (may be a filtered subset of df when strategy='filter')
    """
    arrays = df["eeg_data"].values
    shapes = [a.shape for a in arrays]
    shape_counts = Counter(shapes)

    # ── Fast path: all shapes identical ──────────────────────────────────────
    if len(shape_counts) == 1:
        return np.stack(arrays), df.copy().reset_index(drop=True)

    # ── Diagnostic ───────────────────────────────────────────────────────────
    print(f"\n[get_eeg_matrix] WARNING: {len(shape_counts)} distinct EEG shapes found:")
    for sh, cnt in shape_counts.most_common():
        pct = 100.0 * cnt / len(arrays)
        print(f"    {str(sh):<20}  {cnt:>5,} trials  ({pct:.1f}%)")

    most_common_shape = shape_counts.most_common(1)[0][0]

    # ── Strategy: filter ─────────────────────────────────────────────────────
    if strategy == "filter":
        tgt = target_shape or most_common_shape
        mask   = df["eeg_data"].apply(lambda a: a.shape == tgt)
        kept   = df[mask].copy().reset_index(drop=True)
        dropped = len(df) - len(kept)
        print(f"[get_eeg_matrix] strategy=filter  →  target={tgt}  "
              f"kept={len(kept):,}  dropped={dropped:,}\n")
        return np.stack(kept["eeg_data"].values), kept

    # ── Strategy: truncate / pad ──────────────────────────────────────────────
    elif strategy in ("truncate", "pad"):
        all_shapes = np.array(shapes)           # (N, ndim)
        if target_shape is not None:
            tgt = np.array(target_shape)
        elif strategy == "truncate":
            tgt = all_shapes.min(axis=0)        # per-dim minimum
        else:                                   # pad
            tgt = all_shapes.max(axis=0)        # per-dim maximum

        def _resize(a: np.ndarray) -> np.ndarray:
            # 1. Truncate to tgt along every dimension
            slices = tuple(slice(0, int(t)) for t in tgt)
            out = a[slices]
            # 2. Zero-pad if still short (only relevant for 'pad' strategy)
            pad_w = [(0, int(t) - s) for t, s in zip(tgt, out.shape)]
            if any(p[1] > 0 for p in pad_w):
                out = np.pad(out, pad_w, mode="constant", constant_values=0)
            return out

        print(f"[get_eeg_matrix] strategy={strategy}  →  target={tuple(tgt)}  "
              f"trials={len(df):,}\n")
        X = np.stack([_resize(a) for a in arrays])
        return X, df.copy().reset_index(drop=True)

    else:
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            "Choose 'filter', 'truncate', or 'pad'."
        )


def get_labels(df: pd.DataFrame) -> np.ndarray:
    """
    Return subject ID labels as a 1-D integer array.
    Subject IDs are re-indexed to 0-based consecutive integers.
    """
    ids    = df["subject_id"].values
    unique = sorted(np.unique(ids))
    id2idx = {v: i for i, v in enumerate(unique)}
    return np.array([id2idx[x] for x in ids], dtype=np.int64)


def split_sessions(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into enrolment (sessions 1 & 2) and verification (session 3).

    Returns
    -------
    enrol_df  : DataFrame  — sessions 1 and 2  (training / enrolment)
    verif_df  : DataFrame  — session 3          (verification / test)
    """
    enrol = df[df["session"].isin([1, 2])].copy().reset_index(drop=True)
    verif = df[df["session"] == 3].copy().reset_index(drop=True)
    print(f"[split_sessions] Enrolment (sessions 1+2): {len(enrol):,} trials")
    print(f"[split_sessions] Verification (session 3): {len(verif):,} trials")
    return enrol, verif


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary(df: pd.DataFrame) -> None:
    print("\n══════════════════════════════════")
    print("         ARRC Dataset Summary      ")
    print("══════════════════════════════════")
    print(f"  Total trials  : {len(df):,}")
    print(f"  Subjects      : {df['subject_id'].nunique()}")
    print(f"  Sessions      : {sorted(df['session'].unique())}")
    print()
    print("  Stimulus breakdown:")
    for st, cnt in df["stimuli_type"].value_counts().items():
        print(f"    {st:<12} {cnt:>5,} trials")
    print()
    img = df[df["stimuli_type"] == "IMAGE"]
    print(f"  Emotion trials (IMAGE): {len(img):,}")
    if len(img):
        print(f"    Arousal voted range : {img['arousal_voted'].min():.1f} – {img['arousal_voted'].max():.1f}")
        print(f"    Valence voted range : {img['valence_voted'].min():.1f} – {img['valence_voted'].max():.1f}")
    print()
    # EEG shape audit
    shapes = Counter(df["eeg_data"].apply(lambda a: a.shape))
    print(f"  EEG array shapes ({len(shapes)} distinct):")
    for sh, cnt in shapes.most_common():
        print(f"    {str(sh):<20}  {cnt:,} trials")
    print("══════════════════════════════════\n")


# ─────────────────────────────────────────────────────────────────────────────
# Cache helper (save / load processed data)
# ─────────────────────────────────────────────────────────────────────────────

def save_processed(
    df: pd.DataFrame,
    out_path: Path,
    strategy: str = "filter",
) -> None:
    """
    Save parsed DataFrame + EEG tensor to disk, split by session role.

    Produces:
        <out_path>/
            enrol_eeg.npy       EEG tensor  (N_enrol, ...)   sessions 1+2
            enrol_meta.csv      metadata    (N_enrol rows)   sessions 1+2
            verif_eeg.npy       EEG tensor  (N_verif, ...)   session 3
            verif_meta.csv      metadata    (N_verif rows)   session 3
            shape_report.txt    shape audit log

    Parameters
    ----------
    df       : full DataFrame from load_arrc()
    out_path : output directory (created if absent)
    strategy : passed to get_eeg_matrix  ('filter' | 'truncate' | 'pad')
    """
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    # ── 1. Split by session ──────────────────────────────────────────────────
    enrol_df, verif_df = split_sessions(df)

    # ── 2. Stack EEG tensors (handles shape mismatches) ─────────────────────
    print("\n── Enrolment set ──")
    X_enrol, enrol_aligned = get_eeg_matrix(enrol_df, strategy=strategy)

    print("── Verification set ──")
    # Use the same target shape as enrolment to ensure consistency
    enrol_target = X_enrol.shape[1:]
    X_verif, verif_aligned = get_eeg_matrix(
        verif_df, strategy=strategy, target_shape=enrol_target
    )

    # ── 3. Persist ───────────────────────────────────────────────────────────
    np.save(out_path / "enrol_eeg.npy",  X_enrol)
    np.save(out_path / "verif_eeg.npy",  X_verif)

    enrol_aligned.drop(columns=["eeg_data"]).to_csv(
        out_path / "enrol_meta.csv", index=False
    )
    verif_aligned.drop(columns=["eeg_data"]).to_csv(
        out_path / "verif_meta.csv", index=False
    )

    # ── 4. Shape report ──────────────────────────────────────────────────────
    report_lines = [
        "ARRC Processed Data — Shape Report",
        "=" * 40,
        f"strategy           : {strategy}",
        f"enrol_eeg.npy      : {X_enrol.shape}",
        f"enrol_meta.csv     : {len(enrol_aligned):,} rows",
        f"verif_eeg.npy      : {X_verif.shape}",
        f"verif_meta.csv     : {len(verif_aligned):,} rows",
    ]
    (out_path / "shape_report.txt").write_text("\n".join(report_lines))

    print(f"\n✅ Saved to {out_path.resolve()}")
    print(f"   enrol_eeg.npy  {X_enrol.shape}")
    print(f"   verif_eeg.npy  {X_verif.shape}")
    print(f"   enrol_meta.csv {len(enrol_aligned):,} rows")
    print(f"   verif_meta.csv {len(verif_aligned):,} rows")


def load_processed(
    proc_path: Path,
) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame]:
    """
    Load cached processed data.

    Returns
    -------
    X_enrol      : ndarray  (N_enrol, ...)
    enrol_meta   : DataFrame
    X_verif      : ndarray  (N_verif, ...)
    verif_meta   : DataFrame
    """
    proc_path = Path(proc_path)
    X_enrol    = np.load(proc_path / "enrol_eeg.npy")
    X_verif    = np.load(proc_path / "verif_eeg.npy")
    enrol_meta = pd.read_csv(proc_path / "enrol_meta.csv")
    verif_meta = pd.read_csv(proc_path / "verif_meta.csv")
    return X_enrol, enrol_meta, X_verif, verif_meta


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from pathlib import Path

    # 1. Load full dataset
    df = load_arrc(verbose=True)

    # 2. Output directory
    output_dir = Path("data/processed")

    # 3. Save — 'filter' keeps only trials with the dominant EEG shape.
    #    Switch to 'truncate' or 'pad' if you need all trials.
    save_processed(df, output_dir, strategy="filter")

    print(f"\n✅ Done! Files saved in: {output_dir.resolve()}")
