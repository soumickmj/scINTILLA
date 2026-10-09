"""Tools: embeddings, clustering, differential expression, annotation, batch correction, pipelines.

Tools that compute a per-cell or per-gene result write it into ``adata`` under ``key_added`` and
return ``None`` (or a modified copy with ``copy=True``).  Tools whose natural result is a table
(differential expression, marker genes) return a :class:`pandas.DataFrame`; the clustering
algorithms return the label array and store it too when ``key_added`` is given.
"""

from scintilla.annotation.label_transfer import transfer_labels
from scintilla.annotation.marker_based import annotate_by_markers
from scintilla.annotation.over_representation import ora_test
from scintilla.annotation.rank_genes import find_marker_genes
from scintilla.batch_correction.bbknn import bbknn_correct as bbknn
from scintilla.batch_correction.combat import combat_correct as combat
from scintilla.batch_correction.harmony import harmony_correct as harmony
from scintilla.batch_correction.scanorama import scanorama_correct as scanorama
from scintilla.classification.label_quality import label_quality
from scintilla.classification.label_quality_variants import (
    compute_label_quality_variants,
    review_ranks,
)
from scintilla.classification.run import supervised_analysis
from scintilla.clustering.consensus import consensus_clustering as consensus
from scintilla.clustering.dbscan import dbscan_clustering as dbscan
from scintilla.clustering.dbscan import estimate_eps
from scintilla.clustering.hdbscan import hdbscan_clustering as hdbscan
from scintilla.clustering.hierarchical import hierarchical_clustering as hierarchical
from scintilla.clustering.kmeans import kmeans_clustering as kmeans
from scintilla.clustering.leiden import leiden_clustering as leiden
from scintilla.clustering.louvain import louvain_clustering as louvain
from scintilla.clustering.run import unsupervised_analysis
from scintilla.clustering.spectral import spectral_clustering as spectral
from scintilla.clustering.spectral import spectral_grid_search
from scintilla.differential_expression.permutation import permutation_de as permutation
from scintilla.differential_expression.pseudobulk import pseudobulk_de as pseudobulk
from scintilla.differential_expression.pseudobulk import pseudobulk_de_by_celltype as pseudobulk_by_celltype
from scintilla.differential_expression.rank_genes import rank_genes_groups
from scintilla.differential_expression.ttest import ttest_de as ttest
from scintilla.differential_expression.utils import filter_de_genes, volcano_plot_data
from scintilla.differential_expression.wilcoxon import wilcoxon_de as wilcoxon
from scintilla.dimensionality_reduction.diffusion_map import run_diffusion_map as diffmap
from scintilla.dimensionality_reduction.force_directed import run_force_directed as draw_graph
from scintilla.dimensionality_reduction.tsne import run_tsne as tsne
from scintilla.dimensionality_reduction.umap import run_umap as umap

__all__ = [
    "umap", "tsne", "diffmap", "draw_graph",
    "kmeans", "leiden", "louvain", "dbscan", "estimate_eps", "hdbscan", "hierarchical", "spectral",
    "spectral_grid_search", "consensus",
    "wilcoxon", "ttest", "permutation", "pseudobulk", "pseudobulk_by_celltype", "rank_genes_groups",
    "filter_de_genes", "volcano_plot_data",
    "annotate_by_markers", "transfer_labels", "find_marker_genes", "ora_test",
    "combat", "harmony", "bbknn", "scanorama",
    "unsupervised_analysis", "supervised_analysis", "label_quality",
    "compute_label_quality_variants", "review_ranks",
]
