"""Statistical tests sub-package."""

from scintilla.statistical_tests.anova import anova_per_gene
from scintilla.statistical_tests.boxm import box_m_test
from scintilla.statistical_tests.kruskal import kruskal_per_gene
from scintilla.statistical_tests.correction import correct_pvalues

# Bootstrap confidence intervals & estimators
from scintilla.statistical_tests.bootstrap import (
    bca_bootstrap_ci,
    bootstrap_metric_ci,
    paired_bootstrap_test,
    dot632plus_bootstrap,
    jackknife_after_bootstrap,
    bootstrap_resample_metrics,
)

# Weight-free rank aggregation
from scintilla.statistical_tests.rank_aggregation import borda_count, rank_aggregate

# Standardised effect sizes
from scintilla.statistical_tests.effect_sizes import (
    rank_biserial,
    cohens_d,
    hedges_g,
    cliffs_delta,
)

# Data-adaptive parameter selection
from scintilla.statistical_tests.adaptive import (
    gavish_donoho_threshold,
    marchenko_pastur_cutoff,
    kneedle_elbow,
    adaptive_resolution_search,
    nvi_stability,
)

# Permutation & McNemar tests
from scintilla.statistical_tests.permutation import (
    permutation_test_methods,
    mcnemar_test,
)

__all__ = [
    "anova_per_gene",
    "box_m_test",
    "kruskal_per_gene",
    "correct_pvalues",
    # bootstrap
    "bca_bootstrap_ci",
    "bootstrap_metric_ci",
    "paired_bootstrap_test",
    "dot632plus_bootstrap",
    "jackknife_after_bootstrap",
    "bootstrap_resample_metrics",
    # rank aggregation
    "borda_count",
    "rank_aggregate",
    # effect sizes
    "rank_biserial",
    "cohens_d",
    "hedges_g",
    "cliffs_delta",
    # adaptive
    "gavish_donoho_threshold",
    "marchenko_pastur_cutoff",
    "kneedle_elbow",
    "adaptive_resolution_search",
    "nvi_stability",
    # permutation
    "permutation_test_methods",
    "mcnemar_test",
]
