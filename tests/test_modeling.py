import numpy as np
import pandas as pd
import pytest

from src.modeling import build_models, build_preprocessor, evaluate_regression, train_and_compare
from src.preprocessing import FEATURES


def test_evaluation_metrics_are_reasonable():
    result = evaluate_regression([1, 2, 3], [1, 2, 4])
    assert result["MAE"] > 0
    assert result["RMSE"] > 0
    assert np.isfinite(result["R2"])


def test_model_catalog_has_four_models():
    models = build_models()
    assert len(models) == 4
    assert "Linear Regression" in models
    assert "Random Forest" in models


def test_preprocessor_builds():
    preprocessor = build_preprocessor()
    assert preprocessor is not None


def _small_regression_data(rows=36):
    years = np.tile(np.arange(2000, 2012), 3)[:rows]
    X = pd.DataFrame(
        {
            "Crop": np.where(np.arange(rows) % 2, "Rice", "Wheat"),
            "Season": np.where(np.arange(rows) % 2, "Kharif", "Rabi"),
            "State": np.where(np.arange(rows) % 3, "Gujarat", "Punjab"),
            "Crop_Year": years,
            "Annual_Rainfall": 500 + np.arange(rows) * 4,
            "Fertilizer": 100 + np.arange(rows) * 2,
            "Pesticide": 5 + (np.arange(rows) % 4),
        },
        columns=FEATURES,
    )
    y = pd.Series(10 + (np.arange(rows) % 7) * 2 + (years - 2000) * 0.3, name="Yield")
    return X, y


def test_temporal_training_pipeline_runs_end_to_end():
    X, y = _small_regression_data()
    metrics, best, predictions, importance = train_and_compare(X, y, test_years=3)

    assert len(metrics) == 4
    assert metrics["RMSE"].is_monotonic_increasing
    assert best["validation"] == "temporal holdout (2009–2011)"
    assert set(["actual_yield", "Crop_Year", "Gradient Boosting_prediction"]).issubset(
        predictions.columns
    )
    assert predictions["Crop_Year"].min() >= 2009
    assert np.isfinite(predictions["actual_yield"]).all()
    assert callable(best["pipeline"].predict)


def test_temporal_validation_requires_enough_years():
    X, y = _small_regression_data(rows=6)
    X["Crop_Year"] = [2020, 2020, 2021, 2021, 2022, 2022]
    with pytest.raises(ValueError, match="needs more than 3 distinct valid years"):
        train_and_compare(X, y, test_years=3)


def test_training_rejects_mismatched_row_counts():
    X, y = _small_regression_data()
    with pytest.raises(ValueError, match="same number of rows"):
        train_and_compare(X, y.iloc[:-1])
