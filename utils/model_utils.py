from pathlib import Path
from typing import Any

import joblib
from sklearn.tree import DecisionTreeClassifier

FEATURE_COLUMNS = (
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Type_L",
    "Type_M",
)


class ModelLoadError(RuntimeError):
    """Raised when the configured model artifact cannot be used."""


def find_model_file(models_dir: Path) -> Path | None:
    """Prefer the documented filename, otherwise use the first local .pkl."""
    expected = models_dir / "machine_failure_model.pkl"
    if expected.is_file():
        return expected
    candidates = sorted(models_dir.glob("*.pkl")) if models_dir.is_dir() else []
    return candidates[0] if candidates else None


def load_model(model_path: Path) -> tuple[Any, tuple[str, ...]]:
    """Load a joblib model or {model, features} artifact and validate its schema."""
    try:
        artifact = joblib.load(model_path)
    except Exception as exc:
        raise ModelLoadError(f"Could not load {model_path.name}: {exc}") from exc

    if isinstance(artifact, dict):
        if "model" not in artifact:
            raise ModelLoadError("The model artifact is a dictionary without a 'model' entry.")
        model = artifact["model"]
        supplied_features = artifact.get("features", FEATURE_COLUMNS)
    else:
        model = artifact
        supplied_features = FEATURE_COLUMNS

    if not isinstance(supplied_features, (list, tuple)):
        raise ModelLoadError("The artifact's 'features' entry must be a list or tuple.")
    features = tuple(supplied_features)
    if len(features) != len(FEATURE_COLUMNS) or set(features) != set(FEATURE_COLUMNS):
        raise ModelLoadError(
            "The model feature schema must contain exactly the seven supported columns: "
            + ", ".join(FEATURE_COLUMNS)
        )
    if not callable(getattr(model, "predict", None)):
        raise ModelLoadError("The loaded artifact does not provide a predict() method.")
    if not callable(getattr(model, "predict_proba", None)):
        raise ModelLoadError(
            "The loaded model does not provide predict_proba(); a failure probability "
            "cannot be reported reliably."
        )
    if not isinstance(model, DecisionTreeClassifier) or not hasattr(model, "tree_"):
        raise ModelLoadError("The loaded model must be a fitted scikit-learn DecisionTreeClassifier.")
    if set(model.classes_) != {0, 1} or len(model.classes_) != 2:
        raise ModelLoadError("The Decision Tree must be trained with binary Target classes 0 and 1.")
    return model, features
