import pandas as pd
import pytest
from src.preprocessing import prepare_dataframe, FEATURES

def sample_df():
    return pd.DataFrame({
        "Crop": ["Rice", "Wheat", "Rice"],
        "Crop_Year": [2020, 2021, 2022],
        "Season": ["Kharif", "Rabi", "Kharif"],
        "State": ["Maharashtra", "Punjab", "Maharashtra"],
        "Area": [10, 12, 11],
        "Production": [20, 24, 22],
        "Annual_Rainfall": [800, 450, None],
        "Fertilizer": [100, 120, 110],
        "Pesticide": [2, 3, 2.5],
        "Yield": [2.0, 2.0, 2.0],
    })

def test_preparation_excludes_leakage_columns():
    X, y = prepare_dataframe(sample_df())
    assert list(X.columns) == FEATURES
    assert "Area" not in X.columns
    assert "Production" not in X.columns
    assert len(X) == 3
    assert len(y) == 3

def test_invalid_target_rows_are_removed():
    df = sample_df()
    df.loc[0, "Yield"] = None
    df.loc[1, "Yield"] = -1
    X, y = prepare_dataframe(df)
    assert len(X) == 1
    assert (y >= 0).all()

def test_missing_required_column_raises():
    df = sample_df().drop(columns=["Crop"])
    with pytest.raises(ValueError, match="missing required columns"):
        prepare_dataframe(df)
