"""scintilla: Single-Cell INTegrated Inference, Labelling, and Landscape Analysis.

The recommended entry points are the scanpy-style namespaces ``pp`` (preprocessing), ``tl``
(tools), ``pl`` (plotting), ``stats`` (array-level statistics) and ``benchmark``, plus ``eda``,
``io`` and ``settings``::

    import scintilla as si

    si.settings.verbosity = "info"
    scores = si.tl.label_quality(adata, cell_type_col="cell_type")

The flat names below (``si.run_pca``, ``si.unsupervised_analysis``, ...) are kept as convenient
aliases of the same functions.
"""

from scintilla.analysis_config import AnalysisConfig, generate_default_yaml
from scintilla.annotation import (
    annotate_by_markers,
    find_marker_genes,
    ora_test,
    transfer_labels,
)
from scintilla.batch_correction import (
    batch_asw,
    benchmark_batch_correction,
    combat_correct,
)
from scintilla.batch_correction.metrics import bio_conservation_score
from scintilla.classification.label_quality import label_quality
from scintilla.classification.label_quality_variants import compute_label_quality_variants
from scintilla.settings import settings

from scintilla import benchmark, eda, io, pl, pp, stats, tl  # isort: skip  (namespaces)
from scintilla.benchmarking import (
    BenchmarkReport,
    BenchmarkResult,
    estimate_benchmark_time,
    pairwise_method_comparison,
    print_time_budget,
    profile_method,
    scalability_sweep,
    seed_stability_test,
)
from scintilla.classification.benchmark import compare_classifiers
from scintilla.classification.diagnostics import influential_cells
from scintilla.classification.run import supervised_analysis
from scintilla.clustering.dbscan import estimate_eps
from scintilla.clustering.run import unsupervised_analysis

__version__ = "0.2.0"

from scintilla.config import (
    DEFAULT_N_PCA_COMPS,
    DEFAULT_TEST_SIZE,
    DEFAULT_VARIANCE_THRESHOLD,
    RANDOM_SEED,
)
from scintilla.differential_expression import (
    filter_de_genes,
    permutation_de,
    pseudobulk_de,
    rank_genes_groups,
    ttest_de,
    volcano_plot_data,
    wilcoxon_de,
)
from scintilla.dimensionality_reduction import (
    run_diffusion_map,
    run_tsne,
    run_umap,
)
from scintilla.evaluation.clustering_metrics import bootstrap_clustering_metrics
from scintilla.feature_selection import (
    boruta_selection,
    mi_feature_selection,
    mrmr_selection,
    select_hvg,
)
from scintilla.feature_selection.benchmark import hvg_sensitivity_analysis
from scintilla.io.exporters import save_anndata, save_results_csv, save_results_json
from scintilla.io.loaders import ensure_anndata, load_csv, load_h5ad
from scintilla.preprocessing.normality import check_normality
from scintilla.preprocessing.pca import run_pca
from scintilla.preprocessing.transformations import TRANSFORM_REGISTRY, get_all_transformations
from scintilla.statistical_tests import (
    adaptive_resolution_search,
    bca_bootstrap_ci,
    bootstrap_metric_ci,
    bootstrap_resample_metrics,
    borda_count,
    cliffs_delta,
    cohens_d,
    dot632plus_bootstrap,
    gavish_donoho_threshold,
    hedges_g,
    jackknife_after_bootstrap,
    kneedle_elbow,
    marchenko_pastur_cutoff,
    mcnemar_test,
    nvi_stability,
    paired_bootstrap_test,
    permutation_test_methods,
    rank_aggregate,
    rank_biserial,
)

__all__ = [
    # Namespaces
    "pp",
    "tl",
    "pl",
    "stats",
    "benchmark",
    "eda",
    "io",
    "settings",
    "label_quality",
    "compute_label_quality_variants",
    "estimate_benchmark_time",
    "print_time_budget",
    "__version__",
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
