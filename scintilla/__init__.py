"""scintilla: Single-Cell INTegrated Inference, Labelling, and Landscape Analysis."""

from scintilla.config import (
    RANDOM_SEED,
    DEFAULT_TEST_SIZE,
    DEFAULT_N_PCA_COMPS,
    DEFAULT_VARIANCE_THRESHOLD,
)
from scintilla.io.loaders import load_h5ad, load_csv, ensure_anndata
from scintilla.io.exporters import save_results_csv, save_results_json, save_anndata
from scintilla.preprocessing.transformations import get_all_transformations, TRANSFORM_REGISTRY
from scintilla.preprocessing.normality import check_normality
from scintilla.preprocessing.pca import run_pca
from scintilla.clustering.run import unsupervised_analysis
from scintilla.classification.run import supervised_analysis
from scintilla.differential_expression import (
    wilcoxon_de,
    ttest_de,
    pseudobulk_de,
    permutation_de,
    rank_genes_groups,
    volcano_plot_data,
    filter_de_genes,
)
from scintilla.annotation import (
    annotate_by_markers,
    find_marker_genes,
    ora_test,
    transfer_labels,
)
from scintilla.benchmarking import (
    profile_method,
    BenchmarkResult,
    scalability_sweep,
    BenchmarkReport,
    seed_stability_test,
    pairwise_method_comparison,
    estimate_benchmark_time,
    print_time_budget,
)
from scintilla.batch_correction import (
    combat_correct,
    batch_asw,
    benchmark_batch_correction,
)
from scintilla.dimensionality_reduction import (
    run_umap,
    run_tsne,
    run_diffusion_map,
)
from scintilla.feature_selection import (
    select_hvg,
    mi_feature_selection,
    boruta_selection,
    mrmr_selection,
)
from scintilla.analysis_config import AnalysisConfig, generate_default_yaml
from scintilla.statistical_tests import (
    bca_bootstrap_ci,
    bootstrap_metric_ci,
    paired_bootstrap_test,
    dot632plus_bootstrap,
    jackknife_after_bootstrap,
    bootstrap_resample_metrics,
    borda_count,
    rank_aggregate,
    rank_biserial,
    cohens_d,
    hedges_g,
    cliffs_delta,
    gavish_donoho_threshold,
    marchenko_pastur_cutoff,
    kneedle_elbow,
    adaptive_resolution_search,
    nvi_stability,
    permutation_test_methods,
    mcnemar_test,
)
from scintilla.clustering.dbscan import estimate_eps
from scintilla.classification.benchmark import compare_classifiers
from scintilla.classification.diagnostics import influential_cells
from scintilla.evaluation.clustering_metrics import bootstrap_clustering_metrics
from scintilla.feature_selection.benchmark import hvg_sensitivity_analysis
from scintilla.batch_correction.metrics import bio_conservation_score

__version__ = "0.1.0"
__all__ = [
    "RANDOM_SEED",
    "DEFAULT_TEST_SIZE",
    "DEFAULT_N_PCA_COMPS",
    "DEFAULT_VARIANCE_THRESHOLD",
    "load_h5ad",
    "load_csv",
    "ensure_anndata",
    "save_results_csv",
    "save_results_json",
    "save_anndata",
    "get_all_transformations",
    "TRANSFORM_REGISTRY",
    "check_normality",
    "run_pca",
    "unsupervised_analysis",
    "supervised_analysis",
    "wilcoxon_de",
    "ttest_de",
    "pseudobulk_de",
    "permutation_de",
    "rank_genes_groups",
    "volcano_plot_data",
    "filter_de_genes",
    "annotate_by_markers",
    "find_marker_genes",
    "ora_test",
    "transfer_labels",
    "profile_method",
    "BenchmarkResult",
    "scalability_sweep",
    "BenchmarkReport",
    "seed_stability_test",
    "pairwise_method_comparison",
    "combat_correct",
    "batch_asw",
    "benchmark_batch_correction",
    "run_umap",
    "run_tsne",
    "run_diffusion_map",
    "select_hvg",
    "mi_feature_selection",
    "boruta_selection",
    "mrmr_selection",
    "AnalysisConfig",
    "generate_default_yaml",
    # Robust statistics
    "bca_bootstrap_ci",
    "bootstrap_metric_ci",
    "paired_bootstrap_test",
    "dot632plus_bootstrap",
    "jackknife_after_bootstrap",
    "bootstrap_resample_metrics",
    "borda_count",
    "rank_aggregate",
    "rank_biserial",
    "cohens_d",
    "hedges_g",
    "cliffs_delta",
    "gavish_donoho_threshold",
    "marchenko_pastur_cutoff",
    "kneedle_elbow",
    "adaptive_resolution_search",
    "nvi_stability",
    "permutation_test_methods",
    "mcnemar_test",
    "estimate_eps",
    "compare_classifiers",
    "influential_cells",
    "bootstrap_clustering_metrics",
    "hvg_sensitivity_analysis",
    "bio_conservation_score",
]
