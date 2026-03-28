from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import ID_COLUMN, TARGET_COLUMN


NUMERIC_COLUMNS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]


def clean_customer_churn_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    cleaned = dataframe.copy()

    for column in NUMERIC_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    object_columns = cleaned.select_dtypes(include="object").columns.tolist()
    for column in object_columns:
        series = cleaned[column]
        cleaned[column] = series.where(series.isna(), series.astype(str).str.strip())

    return cleaned


def get_modeling_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    cleaned = clean_customer_churn_dataframe(dataframe)
    return cleaned.drop(columns=[ID_COLUMN], errors="ignore")


def split_feature_types(feature_frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_features = feature_frame.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [column for column in feature_frame.columns if column not in numeric_features]
    return numeric_features, categorical_features


def build_preprocessor(feature_frame: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_features, categorical_features = split_feature_types(feature_frame)

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor, numeric_features, categorical_features


def align_features_for_inference(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    aligned = clean_customer_churn_dataframe(dataframe)
    aligned = aligned.drop(columns=[TARGET_COLUMN], errors="ignore")

    if "SeniorCitizen" in aligned.columns and aligned["SeniorCitizen"].dtype == "object":
        senior_map = {"Yes": 1, "No": 0}
        aligned["SeniorCitizen"] = aligned["SeniorCitizen"].map(senior_map).fillna(aligned["SeniorCitizen"])

    for column in feature_columns:
        if column not in aligned.columns:
            aligned[column] = np.nan

    aligned = aligned[feature_columns].copy()
    return clean_customer_churn_dataframe(aligned)
