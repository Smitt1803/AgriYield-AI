from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    return ColumnTransformer([
        ("categorical", categorical, CATEGORICAL_FEATURES),
        ("numeric", numeric, NUMERIC_FEATURES),
    ])


def build_models(random_state: int = 42) -> dict[str, object]:
    """Create models with preprocessing inside each pipeline to avoid preprocessing leakage."""
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
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def train_and_compare(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    """Fit comparable models on the same reproducible holdout set."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    results = []
    fitted = {}
    prediction_rows = pd.DataFrame({"actual_yield": y_test.reset_index(drop=True)})

    for name, estimator in build_models(random_state).items():
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", estimator),
        ])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        metrics = evaluate_regression(y_test, predictions)
        results.append({"Model": name, **metrics})
        fitted[name] = pipeline
        prediction_rows[f"{name}_prediction"] = predictions

    metrics_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    best_name = str(metrics_df.iloc[0]["Model"])
    best_pipeline = fitted[best_name]

    # Refit the selected model on all available data for use in the demo application.
    best_pipeline.fit(X, y)
    importance_df = get_feature_importance(best_pipeline)
    return metrics_df, {"name": best_name, "pipeline": best_pipeline}, prediction_rows, importance_df


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
