from src.data_loader import load_dataset
from src.evaluation import save_evaluation_outputs
from src.modeling import save_model, train_and_compare
from src.preprocessing import dataset_quality_summary, prepare_dataframe


def main() -> None:
    print("AgriYield AI — model training and evaluation")
    raw = load_dataset()
    quality = dataset_quality_summary(raw)
    print(f"\nDataset rows: {quality['rows']:,} | columns: {quality['columns']}")
    print(f"Duplicate rows: {quality['duplicate_rows']:,}")
    print(f"Missing cells in source: {quality['missing_cells']:,}")

    X, y = prepare_dataframe(raw)
    print(f"\nUsable labelled rows: {len(X):,}")
    print(f"Features used: {list(X.columns)}")
    print("Excluded from predictors to prevent leakage: Area, Production")

    metrics, best, predictions, importance = train_and_compare(X, y)
    save_model(best["pipeline"])
    save_evaluation_outputs(metrics, predictions, importance)

    print(f"\nEvaluation protocol: {best['validation']}")
    print("Model comparison (lower MAE/RMSE is better; higher R² is generally better):")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSelected by lowest holdout RMSE: {best['name']}")
    print("Saved model: models/best_model.joblib")
    print("Saved metrics: reports/model_comparison.csv")
    print("Saved test predictions: reports/test_predictions.csv")
    if not importance.empty:
        print("Saved feature importance: reports/feature_importance.csv")
    print("\nThe displayed metrics are from the holdout before the selected model was refitted.")
    print("The saved model is refitted on all labelled rows for demo predictions.")


if __name__ == "__main__":
    main()
