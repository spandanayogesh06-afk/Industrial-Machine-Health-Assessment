from typing import Any

import numpy as np
import pandas as pd

from .model_utils import FEATURE_COLUMNS


def build_feature_row(values: dict[str, Any], feature_order: tuple[str, ...]) -> pd.DataFrame:
    """Build the model's one-row input using the documented Type encoding."""
    numeric_fields = FEATURE_COLUMNS[:5]
    parsed: dict[str, float] = {}
    for field in numeric_fields:
        try:
            value = float(values[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Enter a valid value for {field}.") from exc
        if not np.isfinite(value):
            raise ValueError(f"{field} must be a finite number.")
        if value < 0 or (field.endswith("[K]") and value <= 0):
            requirement = "greater than zero" if field.endswith("[K]") else "zero or greater"
            raise ValueError(f"{field} must be {requirement}.")
        parsed[field] = value

    machine_type = values.get("Machine Type")
    if machine_type not in {"H", "M", "L"}:
        raise ValueError("Choose a supported machine type: H, M, or L.")

    parsed["Type_L"] = int(machine_type == "L")
    parsed["Type_M"] = int(machine_type == "M")
    return pd.DataFrame([[parsed[name] for name in feature_order]], columns=feature_order)


def _is_failure_class(label: Any) -> bool:
    if isinstance(label, (int, np.integer, float, np.floating)):
        return float(label) == 1.0
    return str(label).strip().lower() in {"1", "true", "failure", "failed"}


def _decision_path(model: Any, row: pd.DataFrame) -> list[dict[str, Any]]:
    tree = getattr(model, "tree_", None)
    if tree is None:
        return []

    path = model.decision_path(row)
    feature_names = list(row.columns)
    values_by_name = row.iloc[0].to_dict()
    node_ids = path.indices[path.indptr[0] : path.indptr[1]]
    conditions = []
    for node_id in node_ids:
        feature_index = int(tree.feature[node_id])
        if feature_index < 0:
            continue
        name = feature_names[feature_index]
        threshold = float(tree.threshold[node_id])
        value = float(values_by_name[name])
        operator = "<=" if value <= threshold else ">"
        conditions.append(
            {
                "feature": name,
                "value": value,
                "rule": f"{name} {operator} {threshold:.3f}",
            }
        )
    return conditions


def assess_machine(model: Any, feature_order: tuple[str, ...], values: dict[str, Any]) -> dict[str, Any]:
    row = build_feature_row(values, feature_order)
    predicted = model.predict(row)[0]
    classes = list(model.classes_)
    failure_indexes = [index for index, label in enumerate(classes) if _is_failure_class(label)]
    if len(failure_indexes) != 1:
        raise ValueError("The model must identify class 1 as the failure class.")

    probabilities = model.predict_proba(row)
    failure_probability = float(probabilities[0][failure_indexes[0]])
    if not np.isfinite(failure_probability) or not 0 <= failure_probability <= 1:
        raise ValueError("The model returned an invalid failure probability.")

    if failure_probability >= 0.5:
        risk_level = "High"
    elif failure_probability >= 0.2:
        risk_level = "Moderate"
    else:
        risk_level = "Low"

    path = _decision_path(model, row)
    if path:
        factor_rows = path
        explanation_source = "Decision path for this assessment"
    else:
        importances = getattr(model, "feature_importances_", None)
        factor_rows = []
        if importances is not None and len(importances) == len(feature_order):
            factor_rows = [
                {"feature": feature, "importance": float(importance)}
                for feature, importance in sorted(
                    zip(feature_order, importances), key=lambda pair: pair[1], reverse=True
                )
                if float(importance) > 0
            ][:5]
        explanation_source = "Overall model feature importance; not a case-specific causal explanation"

    return {
        "predicted_failure": _is_failure_class(predicted),
        "predicted_class": str(predicted),
        "failure_probability": failure_probability,
        "risk_level": risk_level,
        "factors": factor_rows,
        "explanation_source": explanation_source,
        "feature_values": row.iloc[0].to_dict(),
        "machine_type": values["Machine Type"],
    }


def maintenance_recommendations(assessment: dict[str, Any]) -> list[str]:
    probability = assessment["failure_probability"]
    factor_names = list(dict.fromkeys(factor["feature"] for factor in assessment["factors"]))
    named_factors = ", ".join(factor_names[:3])

    if probability >= 0.5 or assessment["predicted_failure"]:
        recommendations = [
            "Treat this result as a high-priority alert: arrange a qualified inspection before continuing normal production.",
            "Verify the recorded operating measurements and inspect the decision-path parameters shown in Prediction Details.",
            "Follow the equipment manufacturer's shutdown, inspection, and restart procedures; this model is not a substitute for them.",
        ]
    elif probability >= 0.2:
        recommendations = [
            "Schedule a condition check at the next suitable planned maintenance window.",
            "Recheck the entered measurements and inspect the decision-path parameters shown in Prediction Details.",
            "Continue only under the site's normal operating and safety procedures, with closer monitoring until reviewed.",
        ]
    else:
        recommendations = [
            "No elevated model risk was indicated for these inputs; continue the manufacturer's preventive-maintenance schedule.",
            "Keep measurement units and sensor calibration consistent, and reassess after operating conditions change.",
        ]

    if named_factors:
        recommendations.insert(1, f"Prioritize verification of these model-driving input(s): {named_factors}.")
    return recommendations
