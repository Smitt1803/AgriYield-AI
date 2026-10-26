from __future__ import annotations

import os
from pathlib import Path
import tempfile

import pandas as pd
import requests

DATA_URL = (
    "https://raw.githubusercontent.com/jainvandan7/crop-yield-ml/"
    "main/crop_yield_in_indian_states.csv"
)
DEFAULT_PATH = Path("data/raw/crop_yield_in_indian_states.csv")
EXPECTED_COLUMNS = {
    "Crop",
    "Crop_Year",
    "Season",
    "State",
    "Area",
    "Production",
    "Annual_Rainfall",
    "Fertilizer",
    "Pesticide",
    "Yield",
}


def _read_and_validate_csv(source: str | Path | bytes) -> pd.DataFrame:
    """Read a CSV and validate its schema before the data is used or cached."""
    if isinstance(source, bytes):
        from io import BytesIO

        df = pd.read_csv(BytesIO(source))
    else:
        df = pd.read_csv(source)

    df.columns = [str(col).strip() for col in df.columns]
    missing = sorted(EXPECTED_COLUMNS.difference(df.columns))
    if missing:
        raise ValueError(
            "The dataset is missing expected columns: "
            + ", ".join(missing)
            + ". Check the dataset source or download the CSV again."
        )
    if df.empty:
        raise ValueError("The dataset CSV contains no data rows.")
    return df


def load_dataset(path: str | Path = DEFAULT_PATH, download: bool = True) -> pd.DataFrame:
    """Load the source CSV, downloading it atomically if it is not already present."""
    path = Path(path)
    if path.exists():
        return _read_and_validate_csv(path)

    if not download:
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download it from the source linked in README.md."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(DATA_URL, timeout=60)
    response.raise_for_status()

    # Validate the response before writing it, then replace the destination atomically.
    # This prevents an interrupted/invalid response from leaving a partial CSV behind.
    df = _read_and_validate_csv(response.content)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as temp_file:
            temp_file.write(response.content)
            temp_path = Path(temp_file.name)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()

    return df
