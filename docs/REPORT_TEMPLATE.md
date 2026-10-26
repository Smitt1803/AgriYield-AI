# AgriYield AI — AUCML ESE Report Template

> Replace every bracketed instruction with verified information from your run. Do not invent metrics.

## Abstract
[Summarize the problem, dataset, methods, best verified model, results, and limitations.]

## 1. Introduction
[Explain crop-yield estimation and why analysis of historical crop data is useful.]

## 2. Problem Statement
[Define the target and intended use. Clarify that this is an academic prototype.]

## 3. Objectives and Scope
[List project objectives, included features, excluded factors, stakeholders, and boundaries.]

## 4. Dataset Description
- Source: https://github.com/jainvandan7/crop-yield-ml
- Rows after cleaning: [fill from run]
- Features: [fill from actual dataset and implementation]
- Target: Yield
- Excluded leakage columns: Area and Production
- Data quality: [missing values, duplicates, invalid target rows]

## 5. Exploratory Data Analysis
[Insert actual plots and describe distributions, crop differences, and trends.]

## 6. Preprocessing
[Explain target cleaning, numeric imputation, categorical imputation/encoding, and why preprocessing is fitted within each model pipeline.]

## 7. Methodology
[Explain why this is regression and describe the chronological holdout: the latest three distinct observed years are reserved for testing and earlier years are used for training. Include the actual years printed by `train_models.py`. The selected model is refitted on all labelled rows after holdout evaluation for interactive predictions.]

## 8. Evaluation Metrics
- MAE: average absolute error in target units.
- RMSE: error metric that penalizes large errors more strongly.
- R²: proportion of target variation explained relative to a mean baseline; it may be negative on held-out data.

## 9. Results and Comparative Analysis
[Insert `reports/model_comparison.csv` values from your run. The CSV includes the validation protocol. Compare models honestly; do not reuse results from an earlier random-split run.]

## 10. Limitations
[Discuss historical data limitations, random split limitations, missing soil/irrigation/local weather variables, and lack of causal interpretation.]

## 11. Conclusion and Future Scope
[Summarize what the implementation achieved. Suggest temporal validation, external data, uncertainty estimates, and domain-expert validation.]

## References
1. Jain, V. (repository author), `crop-yield-ml`, GitHub: https://github.com/jainvandan7/crop-yield-ml
2. Add any course materials or additional papers actually consulted.
