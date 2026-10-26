import numpy as np
import pandas as pd
from src.modeling import build_preprocessor, evaluate_regression, build_models
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
