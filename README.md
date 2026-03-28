# Churn Model Streamlit App

Production-ready customer churn project built from the case-study notebooks and packaged as a deployable Streamlit app.

## What this project includes

- `app.py`: Streamlit user interface
- `assets/10alytics_logo.png`: primary 10Alytics logo asset used in the UI
- `assets/10alytics_wordmark.svg`: fallback wordmark asset
- `src/data/`: dataset loading helpers
- `src/features/`: schema definitions and preprocessing logic
- `src/models/`: training, evaluation, artifact saving, and model visual generation
- `src/inference/`: reusable scoring and prediction-explanation helpers
- `data/raw/customer_churn.csv`: runtime dataset used by training and the app
- `artifacts/decision_tree_pipeline.joblib`: saved model pipeline
- `outputs/eda/`: charts and EDA summaries
- `outputs/metrics/`: evaluation reports and model visuals
- `sample_data/customer_churn_batch_template.csv`: batch scoring template
- `tests/`: lightweight verification tests

## Product scope

The app supports:

- project overview and business framing
- dedicated EDA story page with guided visual interpretation
- single-customer churn scoring
- batch CSV scoring with downloadable predictions
- model diagnostics including confusion matrix and feature importances
- readable decision-path explanations for individual predictions
- data exploration backed by the generated EDA outputs

## Run locally

Install runtime dependencies:

```bash
pip install -r requirements.txt
```

Generate EDA outputs:

```bash
python eda.py
```

Train the decision tree and save the artifact:

```bash
python train.py
```

Launch the Streamlit app:

```bash
python -m streamlit run app.py
```

## Batch prediction CLI

```bash
python predict.py --input sample_data/customer_churn_batch_template.csv --output outputs/predictions/customer_churn_predictions.csv
```

## Run tests

```bash
pytest
```

## Deploy on Streamlit Community Cloud

This project is already structured for Streamlit deployment.

Use these settings in Streamlit Community Cloud:

- Main file path: `app.py`
- Python version: `3.12`

The included `runtime.txt` also pins the Streamlit Cloud runtime to Python 3.12.

The app depends on files committed into the project folder, especially:

- `data/raw/customer_churn.csv`
- `artifacts/decision_tree_pipeline.joblib`
- `outputs/eda/*`
- `outputs/metrics/*`

## Modelling notes

- The model is a tuned but still readable decision tree with preprocessing inside a scikit-learn pipeline.
- Numeric features are median-imputed and categorical features are one-hot encoded.
- The tree uses `class_weight="balanced"` so the app can catch more churn cases instead of optimizing only for majority-class accuracy.
- The emphasis stays on interpretability, so the app remains suitable for teaching, demos, and explainable business workflows.

## Branding note

- The UI now uses the 10Alytics logo file stored in `assets/10alytics_logo.png`.
- The visual system is aligned to the logo's orange and charcoal palette.
- If you ever replace the logo asset, keep the same filename or update `app.py`.
