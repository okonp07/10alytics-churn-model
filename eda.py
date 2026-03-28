from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import EDA_IMAGE_PATHS, EDA_REPORT_PATH, EDA_SUMMARY_PATH, POSITIVE_CLASS, ensure_project_dirs
from src.data.loaders import load_dataset


def main() -> None:
    ensure_project_dirs()
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.dpi"] = 180
    plt.rcParams["savefig.dpi"] = 180

    dataframe = load_dataset()
    dataframe["ChurnFlag"] = dataframe["Churn"].map({"No": 0, "Yes": 1})

    churn_rate = float(dataframe["ChurnFlag"].mean())
    summary = {
        "rows": int(len(dataframe)),
        "columns": int(dataframe.shape[1] - 1),
        "churn_rate": churn_rate,
        "missing_total_charges": int(dataframe["TotalCharges"].isna().sum()),
        "contract_churn_rate": (
            dataframe.groupby("Contract")["ChurnFlag"].mean().mul(100).round(1).to_dict()
        ),
        "internet_churn_rate": (
            dataframe.groupby("InternetService")["ChurnFlag"].mean().mul(100).round(1).to_dict()
        ),
        "payment_churn_rate": (
            dataframe.groupby("PaymentMethod")["ChurnFlag"].mean().mul(100).round(1).to_dict()
        ),
        "monthly_charge_by_churn": (
            dataframe.groupby("Churn")["MonthlyCharges"].mean().round(2).to_dict()
        ),
        "tenure_by_churn": dataframe.groupby("Churn")["tenure"].mean().round(2).to_dict(),
    }
    EDA_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = f"""# Customer Churn EDA Report

## Dataset overview

- Rows: {summary['rows']:,}
- Feature columns: {summary['columns']}
- Churn rate: {summary['churn_rate']:.1%}
- Missing `TotalCharges`: {summary['missing_total_charges']}

## Main insights

1. Month-to-month contracts have the highest churn rate, which signals weaker customer commitment.
2. Fiber-optic customers churn more often than DSL or no-internet customers.
3. Electronic check users churn far more than customers using automatic payment methods.
4. Customers who churn tend to have shorter tenure and higher monthly charges.

## Why this matters

These signals make the case study ideal for teaching. The visual patterns are strong enough to explain clearly, yet realistic enough to discuss customer behavior, pricing, retention, and modelling tradeoffs.
"""
    EDA_REPORT_PATH.write_text(report, encoding="utf-8")

    figure, ax = plt.subplots(figsize=(7, 4.5))
    sns.countplot(data=dataframe, x="Churn", hue="Churn", legend=False, ax=ax, palette=["#4C78A8", "#E45756"])
    ax.set_title("Customer Churn Distribution")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Customers")
    total = len(dataframe)
    for patch in ax.patches:
        height = patch.get_height()
        ax.annotate(
            f"{height / total:.1%}",
            (patch.get_x() + patch.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=10,
        )
    figure.tight_layout()
    figure.savefig(EDA_IMAGE_PATHS["churn_distribution"], bbox_inches="tight", facecolor="white")
    plt.close(figure)

    contract_churn = (
        dataframe.groupby("Contract")["ChurnFlag"].mean().mul(100).sort_values(ascending=False).reset_index()
    )
    figure, ax = plt.subplots(figsize=(7.5, 4.5))
    sns.barplot(data=contract_churn, x="Contract", y="ChurnFlag", hue="Contract", legend=False, ax=ax, palette="Blues_d")
    ax.set_title("Churn Rate by Contract Type")
    ax.set_xlabel("Contract")
    ax.set_ylabel("Churn Rate (%)")
    ax.tick_params(axis="x", rotation=15)
    for patch in ax.patches:
        height = patch.get_height()
        ax.annotate(
            f"{height:.1f}%",
            (patch.get_x() + patch.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    figure.tight_layout()
    figure.savefig(EDA_IMAGE_PATHS["contract_churn"], bbox_inches="tight", facecolor="white")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    internet_churn = (
        dataframe.groupby("InternetService")["ChurnFlag"].mean().mul(100).sort_values(ascending=False).reset_index()
    )
    sns.barplot(
        data=internet_churn,
        x="InternetService",
        y="ChurnFlag",
        hue="InternetService",
        legend=False,
        ax=axes[0],
        palette="Oranges_d",
    )
    axes[0].set_title("Churn Rate by Internet Service")
    axes[0].set_xlabel("Internet Service")
    axes[0].set_ylabel("Churn Rate (%)")

    payment_churn = (
        dataframe.groupby("PaymentMethod")["ChurnFlag"].mean().mul(100).sort_values(ascending=False).reset_index()
    )
    sns.barplot(
        data=payment_churn,
        x="PaymentMethod",
        y="ChurnFlag",
        hue="PaymentMethod",
        legend=False,
        ax=axes[1],
        palette="Greens_d",
    )
    axes[1].set_title("Churn Rate by Payment Method")
    axes[1].set_xlabel("Payment Method")
    axes[1].set_ylabel("Churn Rate (%)")
    axes[1].tick_params(axis="x", rotation=20)
    figure.tight_layout()
    figure.savefig(EDA_IMAGE_PATHS["service_billing_churn"], bbox_inches="tight", facecolor="white")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    sns.boxplot(data=dataframe, x="Churn", y="tenure", hue="Churn", legend=False, ax=axes[0], palette=["#4C78A8", "#E45756"])
    axes[0].set_title("Tenure by Churn")
    axes[0].set_xlabel("Churn")
    axes[0].set_ylabel("Tenure (months)")

    sns.boxplot(
        data=dataframe,
        x="Churn",
        y="MonthlyCharges",
        hue="Churn",
        legend=False,
        ax=axes[1],
        palette=["#4C78A8", "#E45756"],
    )
    axes[1].set_title("Monthly Charges by Churn")
    axes[1].set_xlabel("Churn")
    axes[1].set_ylabel("Monthly Charges")
    figure.tight_layout()
    figure.savefig(EDA_IMAGE_PATHS["numeric_boxplots"], bbox_inches="tight", facecolor="white")
    plt.close(figure)

    print("EDA outputs created successfully.")


if __name__ == "__main__":
    main()
