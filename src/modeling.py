from __future__ import annotations

from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

from src.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    return ColumnTransformer(
        [
            ("categorical", categorical, CATEGORICAL_FEATURES),
            ("numeric", numeric, NUMERIC_FEATURES),
        ]
    )


def build_models(random_state: int = 42) -> dict[str, object]:
    """Create models with preprocessing inside each pipeline to prevent preprocessing leakage."""
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=18, min_samples_leaf=3, random_state=random_state
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=160, min_samples_leaf=2, random_state=random_state, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=random_state),
    }


def evaluate_regression(y_true, y_pred) -> dict[str, float]:
    """Compute commonly used regression metrics in target units, except unitless R²."""
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def _make_holdout_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
    split_strategy: Literal["temporal", "random"],
    test_years: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, str]:
    """Create either a random holdout or a chronological holdout by year."""
    if split_strategy == "random":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        return X_train, X_test, y_train, y_test, "random holdout"

    if split_strategy != "temporal":
        raise ValueError("split_strategy must be either 'temporal' or 'random'.")
    if "Crop_Year" not in X.columns:
        raise ValueError("Temporal validation requires a 'Crop_Year' predictor.")
    if test_years < 1:
        raise ValueError("test_years must be at least 1.")

    years = pd.to_numeric(X["Crop_Year"], errors="coerce")
    valid_years = sorted(years.dropna().unique().tolist())
    if len(valid_years) <= test_years:
        raise ValueError(
            f"Temporal validation needs more than {test_years} distinct valid years; "
            f"found {len(valid_years)}. Use fewer test years or a random split."
        )

    # Rows without a year cannot be assigned honestly to a chronological holdout.
    valid_mask = years.notna()
    X_valid = X.loc[valid_mask]
    y_valid = y.loc[valid_mask]
    year_valid = years.loc[valid_mask]
    test_start_year = valid_years[-test_years]
    train_mask = year_valid < test_start_year
    test_mask = year_valid >= test_start_year

    if not train_mask.any() or not test_mask.any():
        raise ValueError("Temporal validation produced an empty training or testing partition.")

    X_train, X_test = X_valid.loc[train_mask], X_valid.loc[test_mask]
    y_train, y_test = y_valid.loc[train_mask], y_valid.loc[test_mask]
    period = f"temporal holdout ({test_start_year:g}–{valid_years[-1]:g})"
    return X_train, X_test, y_train, y_test, period


def train_and_compare(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    split_strategy: Literal["temporal", "random"] = "temporal",
    test_years: int = 3,
) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    """Evaluate models on a shared holdout, select by RMSE, then refit the winner on all data.

    The default is a chronological holdout using the most recent `test_years` observed years,
    which better approximates predicting later years than a random row split. Pass
    split_strategy='random' to reproduce the original random-holdout experiment.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of rows.")
    if len(X) < 5:
        raise ValueError("At least five rows are required to train and evaluate the models.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    X_train, X_test, y_train, y_test, validation_description = _make_holdout_split(
        X, y, test_size, random_state, split_strategy, test_years
    )
    results = []
    fitted: dict[str, Pipeline] = {}
    prediction_rows = X_test.reset_index(drop=True)[
        [column for column in ["Crop", "State", "Season", "Crop_Year"] if column in X_test]
    ].copy()
    prediction_rows.insert(0, "actual_yield", y_test.reset_index(drop=True).to_numpy())

    for name, estimator in build_models(random_state).items():
        pipeline = Pipeline(
            [("preprocessor", build_preprocessor()), ("model", estimator)]
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        results.append({"Model": name, "Validation": validation_description, **evaluate_regression(y_test, predictions)})
        fitted[name] = pipeline
        prediction_rows[f"{name}_prediction"] = predictions

    metrics_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    best_name = str(metrics_df.iloc[0]["Model"])
    best_pipeline = fitted[best_name]

    # Reported metrics remain from the holdout; the saved estimator is refitted for demo use.
    best_pipeline.fit(X, y)
    importance_df = get_feature_importance(best_pipeline)
    best = {
        "name": best_name,
        "pipeline": best_pipeline,
        "validation": validation_description,
    }
    return metrics_df, best, prediction_rows, importance_df


def get_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    """Return tree-based feature importance where the fitted estimator supports it."""
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["Feature", "Importance"])

    preprocessor = pipeline.named_steps["preprocessor"]
    names = preprocessor.get_feature_names_out()
    values = model.feature_importances_
    return (
        pd.DataFrame({"Feature": names, "Importance": values})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )


def save_model(pipeline: Pipeline, path: str | Path = "models/best_model.joblib") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load_model(path: str | Path = "models/best_model.joblib") -> Pipeline:
    return joblib.load(path)
