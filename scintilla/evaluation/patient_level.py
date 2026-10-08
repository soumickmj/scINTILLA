"""Patient-level evaluation utilities."""

from __future__ import annotations

from typing import Dict, Union

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score


def aggregate_predictions_to_patient(
    predictions_df: pd.DataFrame,
    patient_col: str,
    true_col: str,
    pred_col: str = "predicted",
) -> pd.DataFrame:
    """Aggregate cell-level predictions to patient level via majority vote.

    Parameters
    ----------
    predictions_df:
        DataFrame with columns: patient_col, true_col, pred_col.
    patient_col:
        Column with patient identifiers.
    true_col:
        Column with true labels.
    pred_col:
        Column with predicted labels.

    Returns
    -------
    pd.DataFrame with columns: patient, true_label, predicted_label.
    """
    records = []
    for patient, grp in predictions_df.groupby(patient_col):
        # majority vote for prediction
        pred_counts = grp[pred_col].value_counts()
        pred_label = pred_counts.idxmax()
        # most frequent true label
        true_counts = grp[true_col].value_counts()
        true_label = true_counts.idxmax()
        records.append({patient_col: patient, "true_label": true_label, "predicted_label": pred_label})
    return pd.DataFrame(records)


def patient_confusion_matrix(
    patient_true: np.ndarray,
    patient_pred: np.ndarray,
) -> Dict:
    """Compute confusion matrix and accuracy at patient level.

    Returns
    -------
    dict: confusion_matrix, accuracy, n_patients
    """
    cm = confusion_matrix(patient_true, patient_pred)
    acc = float(accuracy_score(patient_true, patient_pred))
    return {
        "confusion_matrix": cm,
        "accuracy": acc,
        "n_patients": len(patient_true),
    }
