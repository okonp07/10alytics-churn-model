from __future__ import annotations

from src.data.loaders import load_dataset
from src.inference.predict import explain_prediction, load_artifact, predict_dataframe


def test_predict_dataframe_outputs_scores_between_zero_and_one():
    dataframe = load_dataset().head(3).drop(columns=["Churn"])
    artifact = load_artifact()
    predictions = predict_dataframe(dataframe, artifact)

    assert "prediction" in predictions.columns
    assert "churn_probability" in predictions.columns
    assert predictions["churn_probability"].between(0.0, 1.0).all()


def test_explain_prediction_returns_tree_steps():
    dataframe = load_dataset().head(1).drop(columns=["Churn"])
    artifact = load_artifact()
    steps = explain_prediction(dataframe, artifact)

    assert steps
    assert all(isinstance(step, str) for step in steps)
