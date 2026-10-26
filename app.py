from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from src.data_loader import load_dataset
from src.preprocessing import prepare_dataframe, dataset_quality_summary, FEATURES
from src.modeling import load_model

st.set_page_config(page_title="AgriYield AI", page_icon="🌾", layout="wide")
st.title("🌾 AgriYield AI")
st.caption("Machine Learning-Based Crop Yield Prediction and Agricultural Data Analytics")
st.info(
    "Academic prototype using historical state-level crop data. Predictions are estimates, "
    "not guarantees of future agricultural outcomes."
)

@st.cache_data(show_spinner=False)
def get_data():
    return load_dataset()

try:
    raw = get_data()
except Exception as exc:
    st.error(f"Could not load the dataset. Check your internet connection or place the CSV in data/raw/. Details: {exc}")
    st.markdown("Dataset source: [jainvandan7/crop-yield-ml](https://github.com/jainvandan7/crop-yield-ml)")
    st.stop()

X, y = prepare_dataframe(raw)
quality = dataset_quality_summary(raw)

with st.sidebar:
    st.header("Explore the data")
    crops = sorted(raw["Crop"].dropna().astype(str).unique().tolist())
    states = sorted(raw["State"].dropna().astype(str).unique().tolist())
    selected_crops = st.multiselect("Crop", crops, default=crops[:min(5, len(crops))])
    selected_states = st.multiselect("State", states, default=[])
    st.caption("Leave State empty to include all states.")

filtered = raw.copy()
if selected_crops and "Crop" in filtered:
    filtered = filtered[filtered["Crop"].astype(str).isin(selected_crops)]
if selected_states:
    filtered = filtered[filtered["State"].astype(str).isin(selected_states)]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Source records", f"{quality['rows']:,}")
c2.metric("Columns", f"{quality['columns']}")
c3.metric("Duplicate rows", f"{quality['duplicate_rows']:,}")
c4.metric("Usable labelled records", f"{len(X):,}")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Overview & EDA", "Model Comparison", "Predict Yield", "Dataset & Methodology"]
)

with tab1:
    st.subheader("Yield distribution")
    if "Yield" in filtered.columns:
        yield_values = pd.to_numeric(filtered["Yield"], errors="coerce").dropna()
        fig, ax = plt.subplots(figsize=(9, 4))
        sns.histplot(yield_values, bins=40, kde=True, ax=ax)
        ax.set_xlabel("Yield (source dataset units)")
        ax.set_ylabel("Record count")
        st.pyplot(fig)
        plt.close(fig)

    left, right = st.columns(2)
    with left:
        st.subheader("Average yield by selected crop")
        if not filtered.empty and "Crop" in filtered and "Yield" in filtered:
            crop_avg = (filtered.assign(Yield=pd.to_numeric(filtered["Yield"], errors="coerce"))
                        .groupby("Crop", dropna=True)["Yield"].mean()
                        .sort_values(ascending=False).head(15))
            st.bar_chart(crop_avg)
    with right:
        st.subheader("Average yield over years")
        if not filtered.empty and "Crop_Year" in filtered and "Yield" in filtered:
            trend = filtered.assign(
                Crop_Year=pd.to_numeric(filtered["Crop_Year"], errors="coerce"),
                Yield=pd.to_numeric(filtered["Yield"], errors="coerce")
            ).dropna(subset=["Crop_Year", "Yield"]).groupby("Crop_Year")["Yield"].mean()
            st.line_chart(trend)

    st.subheader("Sample records")
    st.dataframe(filtered.head(100), use_container_width=True)

with tab2:
    st.subheader("Measured model performance")
    metrics_path = Path("reports/model_comparison.csv")
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        st.dataframe(metrics, use_container_width=True, hide_index=True)
        metric_col = st.selectbox("Visualize metric", ["RMSE", "MAE", "R2"])
        chart = metrics.set_index("Model")[[metric_col]].sort_values(metric_col, ascending=(metric_col != "R2"))
        st.bar_chart(chart)
        st.caption("These metrics come from the latest local training run. RMSE/MAE are in target units.")
    else:
        st.warning("No evaluation results yet. Run `python train_models.py` first.")

    importance_path = Path("reports/feature_importance.csv")
    if importance_path.exists():
        importance = pd.read_csv(importance_path).head(15).sort_values("Importance")
        st.subheader("Top feature importances (selected tree model)")
        st.bar_chart(importance.set_index("Feature")["Importance"])
        st.caption("Importance indicates model reliance, not causation.")

with tab3:
    st.subheader("Estimate historical-pattern yield")
    model_path = Path("models/best_model.joblib")
    if not model_path.exists():
        st.warning("Train the model first with `python train_models.py`.")
    else:
        model = load_model(model_path)
        crops_all = sorted(X["Crop"].astype(str).unique())
        states_all = sorted(X["State"].astype(str).unique())
        seasons_all = sorted(X["Season"].astype(str).unique())
        num1, num2 = st.columns(2)
        with num1:
            crop = st.selectbox("Crop", crops_all)
            state = st.selectbox("State", states_all)
            season = st.selectbox("Season", seasons_all)
            year = st.number_input("Crop year", min_value=1990, max_value=2035, value=2020, step=1)
        with num2:
            rainfall = st.number_input("Annual rainfall (dataset units)", min_value=0.0, value=float(X["Annual_Rainfall"].median()) if X["Annual_Rainfall"].notna().any() else 0.0)
            fertilizer = st.number_input("Fertilizer (dataset units)", min_value=0.0, value=float(X["Fertilizer"].median()) if X["Fertilizer"].notna().any() else 0.0)
            pesticide = st.number_input("Pesticide (dataset units)", min_value=0.0, value=float(X["Pesticide"].median()) if X["Pesticide"].notna().any() else 0.0)
        if st.button("Predict yield", type="primary"):
            row = pd.DataFrame([{
                "Crop": crop, "Crop_Year": year, "Season": season, "State": state,
                "Annual_Rainfall": rainfall, "Fertilizer": fertilizer, "Pesticide": pesticide
            }], columns=FEATURES)
            prediction = float(model.predict(row)[0])
            st.success(f"Estimated yield: {prediction:,.4f} (source dataset target units)")
            st.caption("This is a model estimate based on historical patterns, not an agronomic recommendation.")

with tab4:
    st.subheader("Dataset quality")
    st.write({
        "Rows": quality["rows"],
        "Columns": quality["columns"],
        "Duplicate rows": quality["duplicate_rows"],
        "Missing cells": quality["missing_cells"],
    })
    st.write("Missing values by column")
    st.dataframe(quality["missing_by_column"].rename("Missing values"), use_container_width=True)
    st.subheader("Model design")
    st.markdown("""
    - **Task:** supervised regression; the target is continuous `Yield`.
    - **Models:** Linear Regression, Decision Tree, Random Forest, Gradient Boosting.
    - **Metrics:** MAE, RMSE, and R² on the same reproducible holdout split.
    - **Preprocessing:** numeric median imputation; categorical most-frequent imputation and one-hot encoding.
    - **Leakage prevention:** `Area` and `Production` are excluded because the dataset defines yield from production divided by area.
    - **Validation caveat:** a random split may not fully represent forecasting future years or unseen regions. Temporal validation is a recommended extension.
    """)
    st.markdown("Source: [public dataset and project notes](https://github.com/jainvandan7/crop-yield-ml)")
