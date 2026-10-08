"""Evaluation sub-package."""

from scintilla.batch_correction.metrics import batch_asw
from scintilla.evaluation.batch_metrics import (
    graph_connectivity,
    principal_component_regression,
)
from scintilla.evaluation.classification_metrics import calculate_metrics
from scintilla.evaluation.clustering_metrics import (
    accuracy_from_mapped_labels,
    adjusted_mutual_info,
    adjusted_rand_index,
    calinski_harabasz,
    completeness,
    comprehensive_clustering_metrics,
    cophenetic_correlation_coefficient,
    davies_bouldin,
    fowlkes_mallows,
    homogeneity,
    normalised_mutual_info,
    silhouette,
    v_measure,
)
from scintilla.evaluation.embedding_metrics import knn_preservation, trustworthiness
from scintilla.evaluation.patient_level import (
    aggregate_predictions_to_patient,
    patient_confusion_matrix,
)

__all__ = [
    "adjusted_rand_index",
    "silhouette",
    "cophenetic_correlation_coefficient",
    "accuracy_from_mapped_labels",
    "normalised_mutual_info",
    "adjusted_mutual_info",
    "v_measure",
    "homogeneity",
    "completeness",
    "fowlkes_mallows",
    "calinski_harabasz",
    "davies_bouldin",
    "comprehensive_clustering_metrics",
    "calculate_metrics",
    "aggregate_predictions_to_patient",
    "patient_confusion_matrix",
    "batch_asw",
    "graph_connectivity",
    "principal_component_regression",
    "trustworthiness",
    "knn_preservation",
]
