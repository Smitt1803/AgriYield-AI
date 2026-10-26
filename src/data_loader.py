from pathlib import Path
import pandas as pd
import requests

DATA_URL = (
    "https://raw.githubusercontent.com/jainvandan7/crop-yield-ml/"
    "main/crop_yield_in_indian_states.csv"
)
DEFAULT_PATH = Path("data/raw/crop_yield_in_indian_states.csv")


def load_dataset(path: str | Path = DEFAULT_PATH, download: bool = True) -> pd.DataFrame:
    """Load the source CSV, downloading it from the documented public repository if absent."""
    path = Path(path)
    if not path.exists():
        if not download:
            raise FileNotFoundError(
                f"Dataset not found at {path}. Download it from the source linked in README.md."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(DATA_URL, timeout=60)
        response.raise_for_status()
        path.write_bytes(response.content)

    df = pd.read_csv(path)
    # Normalize column labels only; retain original values for transparent data handling.
    df.columns = [str(col).strip() for col in df.columns]
    return df
