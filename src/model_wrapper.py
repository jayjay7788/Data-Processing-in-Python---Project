from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

BUNDLE_NAME = "air_quality_model_bundle.joblib"


def save_model_bundle(
    models: dict[str, Any],
    save_path: str | Path,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> Path:
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    bundle = {
        "models": models,
        "numeric_features": numeric_features or [],
        "categorical_features": categorical_features or [],
        "targets": list(models.keys()),
    }
    bundle_path = save_dir / BUNDLE_NAME
    joblib.dump(bundle, bundle_path)
    return bundle_path


def load_model_bundle(load_path: str | Path) -> dict[str, Any]:
    load_dir = Path(load_path)
    bundle_path = load_dir if load_dir.is_file() else load_dir / BUNDLE_NAME
    return joblib.load(bundle_path)


def prepare_features(data: pd.DataFrame | dict[str, Any], feature_columns: list[str]) -> pd.DataFrame:
    frame = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame([data])

    for column in feature_columns:
        if column not in frame.columns:
            frame[column] = 0.0

    if feature_columns:
        frame = frame[feature_columns]

    return frame


def predict_bundle(bundle: dict[str, Any], data: pd.DataFrame | dict[str, Any]) -> pd.DataFrame:
    models = bundle.get("models", {})
    feature_columns = list(dict.fromkeys(bundle.get("numeric_features", []) + bundle.get("categorical_features", [])))
    features = prepare_features(data, feature_columns)

    predictions = {target: model.predict(features) for target, model in models.items()}
    return pd.DataFrame(predictions, index=features.index)
