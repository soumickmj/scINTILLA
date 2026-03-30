"""Classification evaluation metrics."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    cohen_kappa_score,
    balanced_accuracy_score,
)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    classes: Optional[np.ndarray] = None,
) -> Dict:
    """Compute comprehensive classification metrics.

    Parameters
    ----------
    y_true:
        Ground-truth labels.
    y_pred:
        Predicted labels.
    y_prob:
        Predicted probability matrix (n_samples x n_classes).  Column order
        must match *classes* when provided.
    classes:
        Class labels corresponding to the columns of *y_prob*, in order.
        Typically ``model.classes_``.  When *None*, ``np.unique(y_true)``
        is assumed — which is only safe when the model's internal class
        ordering happens to match ``np.unique``.

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, fdr, fnr, confusion_matrix,
    auroc, auprc, mcc, cohen_kappa, balanced_accuracy
    FDR = 1 - precision (macro-averaged, zero_division=0)
    FNR = 1 - recall    (macro-averaged, zero_division=0)
    """
    acc = float(accuracy_score(y_true, y_pred))

    # Use zero_division=0 to avoid warnings on unseen classes
    prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    cm = confusion_matrix(y_true, y_pred)

    # FDR = 1 - precision, FNR = 1 - recall (macro-averaged, same class set)
    fdr = float(1.0 - prec)
    fnr = float(1.0 - rec)

    # Additional metrics
    try:
        mcc = float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        mcc = float("nan")

    try:
        kappa = float(cohen_kappa_score(y_true, y_pred))
    except Exception:
        kappa = float("nan")

    try:
        bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    except Exception:
        bal_acc = float("nan")

    auroc = float("nan")
    auprc = float("nan")
    if y_prob is not None:
        # Determine the canonical label ordering for y_prob columns.
        # When *classes* is provided (e.g. model.classes_), reindex y_prob so
        # columns align with np.unique(y_true) — the order expected by
        # sklearn's roc_auc_score.
        unique_labels = np.unique(y_true)
        n_cls = len(unique_labels)
        if classes is not None and n_cls > 2:
            classes = np.asarray(classes)
            # Build a permutation mapping: for each unique_label find its
            # column position in the model's *classes* array.
            col_order = [int(np.where(classes == lbl)[0][0]) for lbl in unique_labels]
            y_prob_aligned = y_prob[:, col_order]
        else:
            y_prob_aligned = y_prob
        try:
            if n_cls == 2:
                if classes is not None:
                    # Find the column index for the positive class
                    pos_idx = int(np.where(np.asarray(classes) == unique_labels[1])[0][0])
                    auroc = float(roc_auc_score(y_true, y_prob[:, pos_idx]))
                    # Macro AUPRC: average over both classes so the result
                    # is independent of which class is lexicographically second.
                    neg_idx = int(np.where(np.asarray(classes) == unique_labels[0])[0][0])
                    y_bin = (y_true == unique_labels[1]).astype(int)
                    ap_pos = float(average_precision_score(y_bin, y_prob[:, pos_idx]))
                    ap_neg = float(average_precision_score(1 - y_bin, y_prob[:, neg_idx]))
                    auprc = (ap_pos + ap_neg) / 2.0
                else:
                    auroc = float(roc_auc_score(y_true, y_prob[:, 1]))
                    y_bin = (y_true == unique_labels[1]).astype(int)
                    ap_pos = float(average_precision_score(y_bin, y_prob[:, 1]))
                    ap_neg = float(average_precision_score(1 - y_bin, y_prob[:, 0]))
                    auprc = (ap_pos + ap_neg) / 2.0
            elif n_cls > 2:
                auroc = float(
                    roc_auc_score(
                        y_true, y_prob_aligned,
                        multi_class="ovr", average="macro",
                        labels=unique_labels,
                    )
                )
                # macro AUPRC: average per-class AP
                ap_list = []
                for i, cls in enumerate(unique_labels):
                    bin_y = (y_true == cls).astype(int)
                    ap_list.append(average_precision_score(bin_y, y_prob_aligned[:, i]))
                auprc = float(np.mean(ap_list))
        except Exception:
            pass

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "fdr": fdr,
        "fnr": fnr,
        "confusion_matrix": cm,
        "auroc": auroc,
        "auprc": auprc,
        "mcc": mcc,
        "cohen_kappa": kappa,
        "balanced_accuracy": bal_acc,
    }
