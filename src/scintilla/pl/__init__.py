"""Plotting.

Every function returns the figure (or a dict of figures) **without closing it**, and accepts
``show`` and ``save`` in the scanpy style.  Importing this package never changes the matplotlib
backend.
"""

from scintilla.classification import visualise as _cv
from scintilla.visualisation import (
    batch_plots,
    benchmark_summary,
    classification_plots,
    clustering_plots,
    de_plots,
    embedding_plots,
    gene_plots,
    pca_plots,
    sankey_plots,
    transformation_plots,
)
from scintilla.visualisation._utils import plot_api as _api

clustering_benchmark = _api(clustering_plots.plot_clustering_benchmark)
ari_benchmark = _api(clustering_plots.ari_benchmark_plot)
dendrogram = _api(clustering_plots.dendrogram_plot)
cophenetic_scatter = _api(clustering_plots.cophenetic_vs_original_scatter)
classification_benchmark = _api(classification_plots.plot_classification_benchmark)
benchmark_bar_chart = _api(classification_plots.benchmark_bar_chart)
confusion_matrix_heatmap = _api(classification_plots.confusion_matrix_heatmap)
classifier_comparison = _api(_cv.plot_classifier_comparison)
celltype_label_quality = _api(_cv.plot_celltype_label_quality)
metric_correlation = _api(_cv.plot_metric_correlation)
celltype_confusion = _api(_cv.plot_celltype_confusion)
celltype_confusion_network = _api(_cv.plot_celltype_confusion_network)
volcano = _api(de_plots.volcano_plot)
ma = _api(de_plots.ma_plot)
embedding = _api(embedding_plots.plot_embedding)
compare_embeddings = _api(embedding_plots.compare_embeddings)
boxplot_genes_by_group = _api(gene_plots.boxplot_genes_by_group)
expression_heatmap = _api(gene_plots.expression_heatmap)
pca_2d = _api(pca_plots.pca_2d_plot)
pca_3d_multiview = _api(pca_plots.pca_3d_multiview)
cumulative_variance = _api(pca_plots.cumulative_variance_plot)
sankey = _api(sankey_plots.cluster_label_sankey)
before_after_distribution = _api(transformation_plots.before_after_distribution_plot)
batch_correction_comparison = _api(batch_plots.plot_batch_correction_comparison)
radar_chart = _api(benchmark_summary.radar_chart)
benchmark_heatmap = _api(benchmark_summary.benchmark_heatmap)

__all__ = [
    "clustering_benchmark", "ari_benchmark", "dendrogram", "cophenetic_scatter",
    "classification_benchmark", "benchmark_bar_chart", "confusion_matrix_heatmap",
    "classifier_comparison", "celltype_label_quality", "metric_correlation",
    "celltype_confusion", "celltype_confusion_network",
    "volcano", "ma", "embedding", "compare_embeddings",
    "boxplot_genes_by_group", "expression_heatmap",
    "pca_2d", "pca_3d_multiview", "cumulative_variance",
    "sankey", "before_after_distribution", "batch_correction_comparison",
    "radar_chart", "benchmark_heatmap",
]
