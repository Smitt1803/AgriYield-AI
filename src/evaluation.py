from pathlib import Path
import pandas as pd


def save_evaluation_outputs(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    feature_importance: pd.DataFrame,
    reports_dir: str | Path = "reports",
) -> None:
    out = Path(reports_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out / "model_comparison.csv", index=False)
    predictions.to_csv(out / "test_predictions.csv", index=False)
    if not feature_importance.empty:
        feature_importance.to_csv(out / "feature_importance.csv", index=False)
