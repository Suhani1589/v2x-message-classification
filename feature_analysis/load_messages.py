"""Read existing V2X message text. Does not write to the dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

TEXT_COLUMN = "Message Content"
LABEL_COLUMN = "Message Type"
ID_COLUMN = "Message ID"

LABEL_ORDER = [
    "Emergency",
    "Hazard",
    "Roadwork",
    "Spam",
    "Traffic",
    "Weather",
]

PROCESSED_SPLITS = (
    ("train", ROOT / "data" / "processed" / "train.csv"),
    ("val", ROOT / "data" / "processed" / "val.csv"),
    ("test", ROOT / "data" / "processed" / "test.csv"),
)


def load_messages() -> pd.DataFrame:
    """Load the project's existing message corpus from processed splits.

    Those CSVs are the rows the classification pipeline already prepared.
    Missing required columns raises; the files themselves are not modified.
    """
    frames = []
    missing = [str(path) for _, path in PROCESSED_SPLITS if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Processed message CSVs not found: " + ", ".join(missing)
        )

    for split, path in PROCESSED_SPLITS:
        frame = pd.read_csv(path)
        for column in (ID_COLUMN, TEXT_COLUMN, LABEL_COLUMN):
            if column not in frame.columns:
                raise ValueError(f"{path} is missing column {column!r}")
        part = frame[[ID_COLUMN, TEXT_COLUMN, LABEL_COLUMN]].copy()
        part["split"] = split
        frames.append(part)

    data = pd.concat(frames, ignore_index=True)
    data[TEXT_COLUMN] = data[TEXT_COLUMN].fillna("").astype(str)
    data[LABEL_COLUMN] = data[LABEL_COLUMN].astype(str)
    return data
