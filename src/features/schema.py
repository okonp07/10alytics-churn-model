from __future__ import annotations

from collections import OrderedDict

import pandas as pd

from src.config import ID_COLUMN, TARGET_COLUMN


FIELD_LABELS = {
    "gender": "Gender",
    "SeniorCitizen": "Senior Citizen",
    "Partner": "Has Partner",
    "Dependents": "Has Dependents",
    "tenure": "Tenure (months)",
    "PhoneService": "Phone Service",
    "MultipleLines": "Multiple Lines",
    "InternetService": "Internet Service",
    "OnlineSecurity": "Online Security",
    "OnlineBackup": "Online Backup",
    "DeviceProtection": "Device Protection",
    "TechSupport": "Tech Support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming Movies",
    "Contract": "Contract Type",
    "PaperlessBilling": "Paperless Billing",
    "PaymentMethod": "Payment Method",
    "MonthlyCharges": "Monthly Charges",
    "TotalCharges": "Total Charges",
}

FIELD_HELP = {
    "SeniorCitizen": "Use Yes for senior customers and No otherwise.",
    "tenure": "How many months the customer has stayed with the telecom company.",
    "MonthlyCharges": "Current monthly subscription and service charge.",
    "TotalCharges": "Total amount billed so far. New customers can still be near zero.",
    "Contract": "Contract commitment is one of the strongest churn drivers in this project.",
    "PaymentMethod": "Payment experience can signal convenience or friction for the customer.",
}

FEATURE_GROUPS = OrderedDict(
    {
        "Customer profile": ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure"],
        "Connectivity": ["PhoneService", "MultipleLines", "InternetService"],
        "Value-added services": [
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ],
        "Billing and contract": [
            "Contract",
            "PaperlessBilling",
            "PaymentMethod",
            "MonthlyCharges",
            "TotalCharges",
        ],
    }
)

RISK_BAND_RECOMMENDATIONS = {
    "High": [
        "Offer proactive retention outreach and contract-renewal incentives.",
        "Review service quality, especially for fiber customers with high monthly charges.",
        "Check payment friction and encourage lower-friction payment methods where appropriate.",
    ],
    "Moderate": [
        "Monitor the account for service complaints or sudden price sensitivity.",
        "Offer education on bundled support features and loyalty benefits.",
        "Review early-life onboarding for newer customers with shorter tenure.",
    ],
    "Low": [
        "Maintain service quality and keep engagement strong with light-touch retention campaigns.",
        "Use these customers as a benchmark for healthy long-term customer profiles.",
        "Focus on preserving contract stability and satisfaction rather than urgent intervention.",
    ],
}


def get_feature_columns(dataframe: pd.DataFrame) -> list[str]:
    return [column for column in dataframe.columns if column not in {ID_COLUMN, TARGET_COLUMN}]


def get_feature_options(dataframe: pd.DataFrame) -> dict[str, list[str]]:
    options: dict[str, list[str]] = {}
    for column in get_feature_columns(dataframe):
        if dataframe[column].dtype == "object":
            options[column] = sorted(dataframe[column].dropna().astype(str).unique().tolist())
    options["SeniorCitizen"] = ["No", "Yes"]
    return options


def get_default_record(dataframe: pd.DataFrame) -> dict[str, object]:
    defaults: dict[str, object] = {}
    for column in get_feature_columns(dataframe):
        if column == "SeniorCitizen":
            defaults[column] = "Yes" if dataframe[column].mode(dropna=True).iloc[0] == 1 else "No"
        elif pd.api.types.is_numeric_dtype(dataframe[column]):
            defaults[column] = float(dataframe[column].median(skipna=True))
        else:
            defaults[column] = dataframe[column].mode(dropna=True).iloc[0]
    return defaults
