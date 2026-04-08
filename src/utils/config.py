from pathlib import Path

# Project root (EEG-base-auth/)
BASE_PATH = Path(__file__).resolve().parents[2]

# Data folders
DATA_PATH = BASE_PATH / "data"
RAW_PATH = DATA_PATH / "raw"
PROCESSED_PATH = DATA_PATH / "processed"

# Dataset file
ARRC_PATH = RAW_PATH / "ARRC.mat"
