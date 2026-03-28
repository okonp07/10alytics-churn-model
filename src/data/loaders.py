from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import DATA_PATH, ID_COLUMN, SAMPLE_BATCH_PATH, TARGET_COLUMN, ensure_project_dirs
from src.features.preprocessing import clean_customer_churn_dataframe


def load_dataset(path: Path | str = DATA_PATH) -> pd.DataFrame:
    dataframe = pd.read_csv(path)
    return clean_customer_churn_dataframe(dataframe)


def save_runtime_dataset(dataframe: pd.DataFrame, path: Path | str = DATA_PATH) -> Path:
    ensure_project_dirs()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)
    return output_path


def build_sample_batch_template(
    dataframe: pd.DataFrame,
    path: Path | str = SAMPLE_BATCH_PATH,
    sample_size: int = 8,
) -> Path:
    ensure_project_dirs()
    template = dataframe.head(sample_size).drop(columns=[TARGET_COLUMN], errors="ignore").copy()
    if ID_COLUMN not in template.columns:
        template.insert(0, ID_COLUMN, [f"SAMPLE-{idx + 1:03d}" for idx in range(len(template))])
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(output_path, index=False)
    return output_path


def read_json(path: Path | str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_text(path: Path | str) -> str:
    return Path(path).read_text(encoding="utf-8")
