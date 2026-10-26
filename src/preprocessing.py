from __future__ import annotations
import pandas as pd

TARGET = "Yield"
CATEGORICAL_FEATURES = ["Crop", "Season", "State"]
NUMERIC_FEATURES = ["Crop_Year", "Annual_Rainfall", "Fertilizer", "Pesticide"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
LEAKAGE_COLUMNS = ["Area", "Production"]


def prepare_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Validate columns, coerce numeric values, and remove unusable target rows."""
    required = FEATURES + [TARGET]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(missing)
            + ". Check the dataset version and its column names."
        )

    work = df[required].copy()
    for col in NUMERIC_FEATURES + [TARGET]:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    # A target cannot be imputed for supervised training. Remove rows without valid targets.
    work = work.dropna(subset=[TARGET])
    work = work[work[TARGET].map(lambda value: pd.notna(value) and value >= 0)]
    if work.empty:
        raise ValueError("No rows with a valid, non-negative Yield target remain after cleaning.")

    # Keep missing predictor values for sklearn's imputation pipeline.
    for col in CATEGORICAL_FEATURES:
        categorical = work[col].astype("string").str.strip()
        categorical = categorical.mask(categorical.eq(""), pd.NA)
        work[col] = categorical.fillna("Unknown").astype(str)

    X = work[FEATURES].copy()
    y = work[TARGET].astype(float).copy()
    return X, y


def dataset_quality_summary(df: pd.DataFrame) -> dict:
    """Return basic, source-data quality statistics for dashboard/reporting."""
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_cells": int(df.isna().sum().sum()),
        "missing_by_column": df.isna().sum().sort_values(ascending=False),
    }
