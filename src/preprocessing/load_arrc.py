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
        eeg_data   ← numpy array shape (12, 17, 14) stored as object
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
        eeg = entry["data"][0, 0]               # (12, 17, 14)

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


def get_eeg_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Stack all EEG tensors into a single array.

    Parameters
    ----------
    df : DataFrame with 'eeg_data' column

    Returns
    -------
    X : ndarray, shape (N, 12, 17, 14)
    """
    return np.stack(df["eeg_data"].values)   # (N, 12, 17, 14)


def get_labels(df: pd.DataFrame) -> np.ndarray:
    """
    Return subject ID labels as a 1-D integer array.
    Subject IDs are re-indexed to 0-based consecutive integers.
    """
    ids      = df["subject_id"].values
    unique   = sorted(np.unique(ids))
    id2idx   = {v: i for i, v in enumerate(unique)}
    return np.array([id2idx[x] for x in ids], dtype=np.int64)


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
    print(f"    Arousal voted range : {img['arousal_voted'].min():.1f} – {img['arousal_voted'].max():.1f}")
    print(f"    Valence voted range : {img['valence_voted'].min():.1f} – {img['valence_voted'].max():.1f}")
    print("══════════════════════════════════\n")


# ─────────────────────────────────────────────────────────────────────────────
# Cache helper (save/load processed DataFrame)
# ─────────────────────────────────────────────────────────────────────────────

def save_processed(df: pd.DataFrame, out_path: Path) -> None:
    """Save parsed DataFrame (without eeg_data) + EEG tensor separately."""
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save EEG arrays
    X = get_eeg_matrix(df)
    np.save(out_path / "eeg_tensor.npy", X)

    # Save metadata (drop heavy object column)
    meta = df.drop(columns=["eeg_data"])
    meta.to_csv(out_path / "metadata.csv", index=False)
    print(f"Saved processed data to {out_path}")


def load_processed(proc_path: Path) -> Tuple[np.ndarray, pd.DataFrame]:
    """Load cached processed data. Returns (X, metadata_df)."""
    proc_path = Path(proc_path)
    X    = np.load(proc_path / "eeg_tensor.npy")        # (N, 12, 17, 14)
    meta = pd.read_csv(proc_path / "metadata.csv")
    return X, meta
