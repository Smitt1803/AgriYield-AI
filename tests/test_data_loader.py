from pathlib import Path

import pandas as pd
import pytest

from src import data_loader


REQUIRED = {
    "Crop": ["Rice"],
    "Crop_Year": [2020],
    "Season": ["Kharif"],
    "State": ["Gujarat"],
    "Area": [1.0],
    "Production": [2.0],
    "Annual_Rainfall": [500.0],
    "Fertilizer": [10.0],
    "Pesticide": [1.0],
    "Yield": [2.0],
}


class FakeResponse:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self):
        return None


def test_load_existing_dataset_validates_and_reads(tmp_path):
    path = tmp_path / "dataset.csv"
    pd.DataFrame(REQUIRED).to_csv(path, index=False)
    result = data_loader.load_dataset(path, download=False)
    assert len(result) == 1
    assert set(data_loader.EXPECTED_COLUMNS).issubset(result.columns)


def test_load_existing_dataset_rejects_wrong_schema(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"Crop": ["Rice"], "Yield": [1.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing expected columns"):
        data_loader.load_dataset(path, download=False)


def test_download_validates_before_saving(monkeypatch, tmp_path):
    path = tmp_path / "nested" / "dataset.csv"
    content = pd.DataFrame(REQUIRED).to_csv(index=False).encode("utf-8")
    monkeypatch.setattr(data_loader.requests, "get", lambda *args, **kwargs: FakeResponse(content))

    result = data_loader.load_dataset(path)

    assert len(result) == 1
    assert path.exists()
    assert list(path.parent.glob("*.tmp")) == []


def test_invalid_download_is_not_cached(monkeypatch, tmp_path):
    path = tmp_path / "dataset.csv"
    content = b"not_the_expected_schema\n1\n"
    monkeypatch.setattr(data_loader.requests, "get", lambda *args, **kwargs: FakeResponse(content))

    with pytest.raises(ValueError, match="missing expected columns"):
        data_loader.load_dataset(path)

    assert not path.exists()
