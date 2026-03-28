from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree

from src.config import (
    ARTIFACT_PATH,
    CLASSIFICATION_REPORT_PATH,
    CONFUSION_MATRIX_PATH,
    FEATURE_IMPORTANCE_PATH,
    ID_COLUMN,
    METRICS_DIR,
    MODEL_IMAGE_PATHS,
    MODEL_METRICS_PATH,
    NEGATIVE_CLASS,
    POSITIVE_CLASS,
    RANDOM_STATE,
    TARGET_COLUMN,
    TRAINING_SUMMARY_PATH,
    ensure_project_dirs,
)
from src.features.preprocessing import build_preprocessor, clean_customer_churn_dataframe, get_modeling_frame


@dataclass
class TrainingResult:
    artifact: dict[str, Any]
    metrics: dict[str, Any]
    classification: dict[str, Any]
    confusion: dict[str, Any]
    feature_importance: pd.DataFrame
    tree_depth: int
    leaf_count: int


def train_decision_tree_model(dataframe: pd.DataFrame) -> TrainingResult:
    cleaned = clean_customer_churn_dataframe(dataframe)
    modeling_frame = get_modeling_frame(cleaned)
    feature_frame = modeling_frame.drop(columns=[TARGET_COLUMN])
    target = modeling_frame[TARGET_COLUMN]

    preprocessor, numeric_features, categorical_features = build_preprocessor(feature_frame)
    classifier = DecisionTreeClassifier(
        max_depth=6,
        min_samples_leaf=80,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        feature_frame,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)
    positive_index = list(pipeline.named_steps["classifier"].classes_).index(POSITIVE_CLASS)
    positive_scores = probabilities[:, positive_index]

    baseline_accuracy = float(y_test.value_counts(normalize=True).max())
    metrics = {
        "baseline_accuracy": baseline_accuracy,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, pos_label=POSITIVE_CLASS)),
        "recall": float(recall_score(y_test, predictions, pos_label=POSITIVE_CLASS)),
        "f1_score": float(f1_score(y_test, predictions, pos_label=POSITIVE_CLASS)),
        "mean_positive_probability": float(positive_scores.mean()),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }

    class_report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, predictions, labels=[NEGATIVE_CLASS, POSITIVE_CLASS])

    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
    importances = pipeline.named_steps["classifier"].feature_importances_
    feature_importance = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    artifact = {
        "pipeline": pipeline,
        "feature_columns": feature_frame.columns.tolist(),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "feature_names": feature_names,
        "class_labels": pipeline.named_steps["classifier"].classes_.tolist(),
        "model_name": "DecisionTreeClassifier",
        "model_params": classifier.get_params(),
        "target_column": TARGET_COLUMN,
        "positive_class": POSITIVE_CLASS,
        "metrics": metrics,
    }

    return TrainingResult(
        artifact=artifact,
        metrics=metrics,
        classification=class_report,
        confusion={
            "labels": [NEGATIVE_CLASS, POSITIVE_CLASS],
            "matrix": matrix.tolist(),
        },
        feature_importance=feature_importance,
        tree_depth=int(pipeline.named_steps["classifier"].get_depth()),
        leaf_count=int(pipeline.named_steps["classifier"].get_n_leaves()),
    )


def save_training_artifact(artifact: dict[str, Any], path: Path | str = ARTIFACT_PATH) -> Path:
    ensure_project_dirs()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path, compress=3)
    return output_path


def save_training_outputs(result: TrainingResult) -> None:
    ensure_project_dirs()

    MODEL_METRICS_PATH.write_text(json.dumps(result.metrics, indent=2), encoding="utf-8")
    CLASSIFICATION_REPORT_PATH.write_text(json.dumps(result.classification, indent=2), encoding="utf-8")
    CONFUSION_MATRIX_PATH.write_text(json.dumps(result.confusion, indent=2), encoding="utf-8")
    FEATURE_IMPORTANCE_PATH.write_text(
        result.feature_importance.to_csv(index=False),
        encoding="utf-8",
    )

    training_summary = {
        "tree_depth": result.tree_depth,
        "leaf_count": result.leaf_count,
        "model_params": result.artifact["model_params"],
        "top_features": result.feature_importance.head(10).to_dict(orient="records"),
        **result.metrics,
    }
    TRAINING_SUMMARY_PATH.write_text(json.dumps(training_summary, indent=2), encoding="utf-8")


def generate_model_visuals(result: TrainingResult) -> None:
    ensure_project_dirs()
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.dpi"] = 180
    plt.rcParams["savefig.dpi"] = 180

    figure, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(
        confusion_matrix=np.array(result.confusion["matrix"]),
        display_labels=result.confusion["labels"],
    ).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Decision Tree Confusion Matrix")
    figure.tight_layout()
    figure.savefig(MODEL_IMAGE_PATHS["confusion_matrix"], bbox_inches="tight", facecolor="white")
    plt.close(figure)

    top_features = result.feature_importance.head(10).sort_values("importance", ascending=True)
    figure, ax = plt.subplots(figsize=(9, 5.5))
    sns.barplot(data=top_features, x="importance", y="feature", hue="feature", legend=False, ax=ax, palette="crest")
    ax.set_title("Top Decision Tree Feature Importances")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    figure.tight_layout()
    figure.savefig(MODEL_IMAGE_PATHS["feature_importance"], bbox_inches="tight", facecolor="white")
    plt.close(figure)

    pipeline = result.artifact["pipeline"]
    tree = pipeline.named_steps["classifier"]
    feature_names = result.artifact["feature_names"]
    figure, ax = plt.subplots(figsize=(16, 8))
    plot_tree(
        tree,
        feature_names=feature_names,
        class_names=[NEGATIVE_CLASS, POSITIVE_CLASS],
        filled=True,
        rounded=True,
        fontsize=7,
        ax=ax,
    )
    ax.set_title("Decision Tree Structure")
    figure.tight_layout()
    figure.savefig(MODEL_IMAGE_PATHS["decision_tree"], bbox_inches="tight", facecolor="white")
    plt.close(figure)
