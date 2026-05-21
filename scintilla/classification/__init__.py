"""Classification sub-package."""

from scintilla.classification.models import (
    lda_classification,
    qda_classification,
    svm_classification,
    random_forest_classification,
    logistic_regression_classification,
    mlp_classification,
    knn_classification,
)
from scintilla.classification.run import supervised_analysis
from scintilla.classification.visualise import plot_classifier_comparison, plot_celltype_label_quality, plot_metric_correlation, plot_celltype_confusion, compute_label_quality_score, plot_celltype_confusion_network

__all__ = [
    "lda_classification",
    "qda_classification",
    "svm_classification",
    "random_forest_classification",
    "logistic_regression_classification",
    "mlp_classification",
    "knn_classification",
    "supervised_analysis",
    "plot_classifier_comparison",
    "plot_celltype_label_quality",
    "plot_metric_correlation",
    "plot_celltype_confusion",
    "compute_label_quality_score",
    "plot_celltype_confusion_network",
]
