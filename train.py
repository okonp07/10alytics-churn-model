from __future__ import annotations

from src.config import ARTIFACT_PATH, ensure_project_dirs
from src.data.loaders import build_sample_batch_template, load_dataset
from src.models.training import (
    generate_model_visuals,
    save_training_artifact,
    save_training_outputs,
    train_decision_tree_model,
)


def main() -> None:
    ensure_project_dirs()
    dataframe = load_dataset()
    result = train_decision_tree_model(dataframe)
    save_training_artifact(result.artifact, ARTIFACT_PATH)
    save_training_outputs(result)
    generate_model_visuals(result)
    build_sample_batch_template(dataframe)
    print("Training artifact and outputs created successfully.")


if __name__ == "__main__":
    main()
