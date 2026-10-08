"""The array-level statistics layer.

These functions take NumPy arrays or pandas objects, not an AnnData.  They are the
documented, tested building blocks behind the AnnData-level tools in :mod:`scintilla.tl`:
bootstrap intervals, effect sizes, multiple-testing correction, rank aggregation, adaptive
dimensionality and resolution selection, permutation tests, and the clustering,
classification, batch and embedding metrics.
"""

from scintilla.evaluation.batch_metrics import graph_connectivity, principal_component_regression
from scintilla.evaluation.classification_metrics import calculate_metrics
from scintilla.evaluation.clustering_metrics import (
    accuracy_from_mapped_labels,
    adjusted_mutual_info,
    adjusted_rand_index,
    bootstrap_clustering_metrics,
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
from scintilla.evaluation.patient_level import aggregate_predictions_to_patient, patient_confusion_matrix
from scintilla.statistical_tests import *  # noqa: F403
from scintilla.statistical_tests import __all__ as _stat_all

__all__ = list(_stat_all) + [
    "calculate_metrics",
    "adjusted_rand_index",
    "adjusted_mutual_info",
    "normalised_mutual_info",
    "v_measure",
    "homogeneity",
    "completeness",
    "fowlkes_mallows",
    "silhouette",
    "calinski_harabasz",
    "davies_bouldin",
    "cophenetic_correlation_coefficient",
    "accuracy_from_mapped_labels",
    "comprehensive_clustering_metrics",
    "bootstrap_clustering_metrics",
    "graph_connectivity",
    "principal_component_regression",
    "trustworthiness",
    "knn_preservation",
    "aggregate_predictions_to_patient",
    "patient_confusion_matrix",
]
