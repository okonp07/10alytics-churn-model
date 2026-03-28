from __future__ import annotations

from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = APP_ROOT / "data" / "raw"
OUTPUT_DIR = APP_ROOT / "outputs"
EDA_DIR = OUTPUT_DIR / "eda"
METRICS_DIR = OUTPUT_DIR / "metrics"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
ARTIFACT_DIR = APP_ROOT / "artifacts"
SAMPLE_DIR = APP_ROOT / "sample_data"

DATA_PATH = DATA_DIR / "customer_churn.csv"
ARTIFACT_PATH = ARTIFACT_DIR / "decision_tree_pipeline.joblib"
EDA_SUMMARY_PATH = EDA_DIR / "eda_summary.json"
EDA_REPORT_PATH = EDA_DIR / "eda_report.md"
MODEL_METRICS_PATH = METRICS_DIR / "model_metrics.json"
CLASSIFICATION_REPORT_PATH = METRICS_DIR / "classification_report.json"
CONFUSION_MATRIX_PATH = METRICS_DIR / "confusion_matrix.json"
FEATURE_IMPORTANCE_PATH = METRICS_DIR / "feature_importance.csv"
TRAINING_SUMMARY_PATH = METRICS_DIR / "training_summary.json"
SAMPLE_BATCH_PATH = SAMPLE_DIR / "customer_churn_batch_template.csv"
DEFAULT_PREDICTIONS_PATH = PREDICTIONS_DIR / "batch_predictions.csv"

EDA_IMAGE_PATHS = {
    "churn_distribution": EDA_DIR / "churn_distribution.png",
    "contract_churn": EDA_DIR / "contract_churn.png",
    "service_billing_churn": EDA_DIR / "service_billing_churn.png",
    "numeric_boxplots": EDA_DIR / "numeric_boxplots.png",
}

MODEL_IMAGE_PATHS = {
    "confusion_matrix": METRICS_DIR / "confusion_matrix.png",
    "feature_importance": METRICS_DIR / "feature_importance.png",
    "decision_tree": METRICS_DIR / "decision_tree.png",
}

ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"
POSITIVE_CLASS = "Yes"
NEGATIVE_CLASS = "No"
RANDOM_STATE = 42


def ensure_project_dirs() -> None:
    for path in (
        DATA_DIR,
        OUTPUT_DIR,
        EDA_DIR,
        METRICS_DIR,
        PREDICTIONS_DIR,
        ARTIFACT_DIR,
        SAMPLE_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
