from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import ARTIFACT_PATH, ID_COLUMN, POSITIVE_CLASS, TARGET_COLUMN
from src.features.preprocessing import align_features_for_inference


def load_artifact(path: Path | str = ARTIFACT_PATH) -> dict[str, Any]:
    return joblib.load(path)


def classify_risk_band(probability: float) -> str:
    if probability >= 0.65:
        return "High"
    if probability >= 0.35:
        return "Moderate"
    return "Low"


def predict_dataframe(dataframe: pd.DataFrame, artifact: dict[str, Any]) -> pd.DataFrame:
    original = dataframe.copy()
    aligned = align_features_for_inference(original, artifact["feature_columns"])
    pipeline = artifact["pipeline"]
    predictions = pipeline.predict(aligned)
    probabilities = pipeline.predict_proba(aligned)
    positive_index = list(artifact["class_labels"]).index(POSITIVE_CLASS)
    churn_probabilities = probabilities[:, positive_index]

    result = original.drop(columns=[TARGET_COLUMN], errors="ignore").copy()
    if ID_COLUMN not in result.columns:
        result.insert(0, ID_COLUMN, [f"ROW-{index + 1:04d}" for index in range(len(result))])
    result["prediction"] = predictions
    result["churn_probability"] = churn_probabilities
    result["risk_band"] = [classify_risk_band(value) for value in churn_probabilities]
    return result


def explain_prediction(
    dataframe: pd.DataFrame,
    artifact: dict[str, Any],
    max_steps: int = 8,
) -> list[str]:
    aligned = align_features_for_inference(dataframe, artifact["feature_columns"])
    pipeline = artifact["pipeline"]
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    transformed = preprocessor.transform(aligned)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    feature_names = artifact["feature_names"]
    node_indicator = classifier.decision_path(transformed)
    leaf_id = classifier.apply(transformed)
    steps: list[str] = []

    node_indices = node_indicator.indices[node_indicator.indptr[0] : node_indicator.indptr[1]]
    for node_id in node_indices:
        if node_id == leaf_id[0]:
            continue

        feature_index = classifier.tree_.feature[node_id]
        if feature_index < 0:
            continue

        threshold = classifier.tree_.threshold[node_id]
        feature_name = feature_names[feature_index]
        feature_value = float(transformed[0, feature_index])
        readable = _humanize_split(feature_name, feature_value, threshold, artifact)
        steps.append(readable)

        if len(steps) >= max_steps:
            break

    return steps


def _humanize_split(
    feature_name: str,
    feature_value: float,
    threshold: float,
    artifact: dict[str, Any],
) -> str:
    if feature_name.startswith("num__"):
        column = feature_name.replace("num__", "", 1)
        direction = "<=" if feature_value <= threshold else ">"
        return f"{column}: {feature_value:.2f} {direction} {threshold:.2f}"

    raw_name = feature_name.replace("cat__", "", 1)
    categorical_features = sorted(artifact["categorical_features"], key=len, reverse=True)

    for column in categorical_features:
        prefix = f"{column}_"
        if raw_name.startswith(prefix):
            category = raw_name[len(prefix) :]
            if feature_value > threshold:
                return f"{column}: is '{category}'"
            return f"{column}: is not '{category}'"

    direction = "<=" if feature_value <= threshold else ">"
    return f"{raw_name}: {feature_value:.2f} {direction} {threshold:.2f}"


def top_global_drivers(artifact: dict[str, Any], limit: int = 5) -> list[dict[str, float]]:
    classifier = artifact["pipeline"].named_steps["classifier"]
    importances = classifier.feature_importances_
    feature_names = artifact["feature_names"]
    pairs = sorted(
        zip(feature_names, importances, strict=False),
        key=lambda item: item[1],
        reverse=True,
    )
    return [
        {"feature": name, "importance": float(score)}
        for name, score in pairs[:limit]
    ]
