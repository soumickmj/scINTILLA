"""Unified analysis configuration for scintilla pipelines.

Provides :class:`AnalysisConfig` – a dataclass that lets users select
exactly which methods to run (clustering, classification, feature
selection, etc.) and tune key hyperparameters.  Configs can be created
programmatically, from presets (``default`` / ``fast``), or loaded from
YAML files.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

from scintilla.config import (
    RANDOM_SEED,
    DEFAULT_TEST_SIZE,
    DEFAULT_N_PCA_COMPS,
    DEFAULT_CV_FOLDS,
)


# ── Defaults ────────────────────────────────────────────────────────────

_DEFAULT_CLUSTERING_METHODS: List[str] = [
    "kmeans", "hierarchical", "dbscan", "leiden",
    "louvain", "hdbscan", "spectral", "consensus",
]
_DEFAULT_CLASSIFIERS: List[str] = [
    "LogReg", "RF", "SVM", "MLP", "LDA", "QDA",
    "kNN", "GradientBoosting", "NaiveBayes", "StackingEnsemble",
]
_DEFAULT_FEATURE_SELECTION_METHODS: List[str] = [
    "pca_loadings", "mutual_information",
]
_FAST_CLUSTERING_METHODS: List[str] = ["kmeans", "leiden"]
_FAST_CLASSIFIERS: List[str] = ["LogReg", "RF", "kNN"]
_FAST_FEATURE_SELECTION_METHODS: List[str] = ["pca_loadings", "mutual_information"]


# ── Dataclass ───────────────────────────────────────────────────────────

@dataclass
class AnalysisConfig:
    """Central configuration for scintilla analysis pipelines.

    Every pipeline function accepts an optional ``config`` parameter.
    When provided the config selects which methods are executed and
    overrides key hyperparameters.  Explicit keyword arguments to the
    function always take precedence over config values.

    Examples
    --------
    >>> cfg = AnalysisConfig.fast()
    >>> result = supervised_analysis(adata, target_col="cell_type", config=cfg)

    >>> cfg = AnalysisConfig.from_yaml("my_config.yaml")
    >>> result = unsupervised_analysis(adata, cell_type_col="cell_type", config=cfg)
    """

    # ── Clustering ──────────────────────────────────────────────────
    clustering_methods: List[str] = field(default_factory=lambda: list(_DEFAULT_CLUSTERING_METHODS))
    leiden_resolutions: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0])
    louvain_resolutions: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0])
    hdbscan_min_cluster_sizes: List[int] = field(default_factory=lambda: [10, 20, 50])
    hdbscan_min_samples: list = field(default_factory=lambda: [None, 5])
    spectral_n_clusters_range: List[int] = field(default_factory=lambda: [2, 3, 4, 5, 6, 7, 8, 9, 10])

    # ── Classification ──────────────────────────────────────────────
    classifiers: List[str] = field(default_factory=lambda: list(_DEFAULT_CLASSIFIERS))
    include_shap: bool = True
    cv_folds: int = DEFAULT_CV_FOLDS

    # ── Feature selection ───────────────────────────────────────────
    feature_selection_methods: List[str] = field(default_factory=lambda: list(_DEFAULT_FEATURE_SELECTION_METHODS))
    n_features: int = 50

    # ── General ─────────────────────────────────────────────────────
    random_seed: int = RANDOM_SEED
    test_size: float = DEFAULT_TEST_SIZE
    n_pca_comps: int = DEFAULT_N_PCA_COMPS
    verbose: bool = True

    # ── Robust statistics ───────────────────────────────────────────
    bootstrap_ci: bool = False
    n_bootstrap: int = 2000
    scoring_method: str = "weighted"
    auto_pca_components: Optional[str] = None
    adaptive_resolution: bool = False
    auto_eps: bool = False
    classification_estimator: str = "cv"
    no_info_method: str = "analytical"
    mp_sigma_method: str = "median"

    # ── Presets ─────────────────────────────────────────────────────

    @classmethod
    def default(cls) -> "AnalysisConfig":
        """Return a config that runs *all* available methods (current behaviour)."""
        return cls()

    @classmethod
    def fast(cls) -> "AnalysisConfig":
        """Return a config with only fast methods – suitable for quick exploration."""
        return cls(
            clustering_methods=list(_FAST_CLUSTERING_METHODS),
            leiden_resolutions=[0.5, 1.0],
            louvain_resolutions=[0.5, 1.0],
            hdbscan_min_cluster_sizes=[20],
            hdbscan_min_samples=[None],
            spectral_n_clusters_range=[],
            classifiers=list(_FAST_CLASSIFIERS),
            include_shap=False,
            feature_selection_methods=list(_FAST_FEATURE_SELECTION_METHODS),
            cv_folds=3,
        )

    @classmethod
    def robust(cls) -> "AnalysisConfig":
        """Return a config that enables all statistically-robust options.

        This activates bootstrap confidence intervals, Borda rank
        aggregation, data-adaptive PCA component selection, adaptive
        Leiden resolution search, and data-driven DBSCAN eps estimation.
        """
        return cls(
            bootstrap_ci=True,
            n_bootstrap=2000,
            scoring_method="borda",
            auto_pca_components="gavish_donoho",
            adaptive_resolution=True,
            auto_eps=True,
            classification_estimator="cv",
            cv_folds=5,
            no_info_method="permutation",
            mp_sigma_method="trimmed_mean",
        )

    # ── YAML I/O ────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str) -> "AnalysisConfig":
        """Load a config from a YAML file.

        Parameters
        ----------
        path:
            Path to a YAML file previously created by :meth:`to_yaml` or
            by ``scintilla generate-config``.
        """
        import yaml  # noqa: PLC0415

        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        if raw is None:
            return cls()
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})

    def to_yaml(self, path: str) -> None:
        """Write the config to a YAML file."""
        import yaml  # noqa: PLC0415

        data = asdict(self)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.dump(data, fh, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> dict:
        """Return a plain dict representation."""
        return asdict(self)

    # ── Helpers ─────────────────────────────────────────────────────

    def copy(self, **overrides) -> "AnalysisConfig":
        """Return a shallow copy, optionally overriding specific fields."""
        d = asdict(self)
        d.update(overrides)
        return AnalysisConfig(**d)


def generate_default_yaml(path: str = "scintilla_config.yaml") -> str:
    """Write a commented default config YAML to *path* and return the text.

    The generated file includes comments explaining every option,
    making it easy for users to customise.
    """
    text = """\
# ── scintilla analysis configuration ───────────────────────────────
# Edit this file then pass it to any pipeline function or CLI command:
#   scintilla run-all data.h5ad --target-col cell_type --config this_file.yaml
#   scintilla cluster data.h5ad --cell-type-col cell_type --config this_file.yaml
#
# Or load in Python:
#   from scintilla import AnalysisConfig
#   cfg = AnalysisConfig.from_yaml("this_file.yaml")
#   result = supervised_analysis(adata, target_col="cell_type", config=cfg)
#
# ── Presets ─────────────────────────────────────────────────────────
# Instead of editing this file you can also use built-in presets:
#   cfg = AnalysisConfig.fast()    # only fast methods
#   cfg = AnalysisConfig.default() # all methods

# ── Clustering ──────────────────────────────────────────────────────
# Which clustering algorithms to run.
# Options: kmeans, hierarchical, dbscan, leiden, louvain, hdbscan,
#          spectral, consensus
clustering_methods:
  - kmeans
  - hierarchical
  - dbscan
  - leiden
  - louvain
  - hdbscan
  - spectral
  - consensus

# Leiden resolution grid (higher = more clusters).
leiden_resolutions: [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]

# Louvain resolution grid.
louvain_resolutions: [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]

# HDBSCAN parameter grids.
hdbscan_min_cluster_sizes: [10, 20, 50]
hdbscan_min_samples: [null, 5]

# Spectral clustering – number-of-clusters grid.
spectral_n_clusters_range: [2, 3, 4, 5, 6, 7, 8, 9, 10]

# ── Classification ──────────────────────────────────────────────────
# Which classifiers to train.
# Options: LogReg, RF, SVM, MLP, LDA, QDA, kNN,
#          GradientBoosting, NaiveBayes, StackingEnsemble,
#          XGBoost, LightGBM  (last two require extra packages)
classifiers:
  - LogReg
  - RF
  - SVM
  - MLP
  - LDA
  - QDA
  - kNN
  - GradientBoosting
  - NaiveBayes
  - StackingEnsemble

# Whether to compute SHAP feature importances for the best model.
include_shap: true

# ── Feature selection ───────────────────────────────────────────────
# Which feature-selection methods to benchmark.
# Options: pca_loadings, mutual_information, boruta, mrmr
# Note: boruta and mrmr are slow on large datasets (>10k cells).
feature_selection_methods:
  - pca_loadings
  - mutual_information

# Number of features to select per method.
n_features: 50

# ── General ─────────────────────────────────────────────────────────
random_seed: 42
test_size: 0.2
n_pca_comps: 30
verbose: true

# ── Robust statistics ───────────────────────────────────────────────
# Enable BCa bootstrap confidence intervals for metrics.
bootstrap_ci: false
# Number of bootstrap replicates.
n_bootstrap: 2000
# Scoring method for benchmarks: "weighted" (default), "borda".
scoring_method: weighted
# Data-adaptive PCA component selection: null, "gavish_donoho", "marchenko_pastur".
auto_pca_components: null
# Adaptive Leiden/Louvain resolution search.
adaptive_resolution: false
# Data-driven DBSCAN epsilon estimation.
auto_eps: false
# Classification estimator: "cv", "holdout", "bootstrap_632plus".
classification_estimator: cv
# No-information rate method for .632+ bootstrap: "analytical", "permutation".
# "analytical" is exact for accuracy; "permutation" is correct for any metric.
no_info_method: analytical
# Marchenko-Pastur sigma estimation: "median" (default), "trimmed_mean".
# "trimmed_mean" iteratively estimates noise variance from the bulk and is
# more accurate when the signal-to-noise ratio is moderate.
mp_sigma_method: median
# Number of cross-validation folds.
cv_folds: 5
"""
    p = Path(path)
    p.write_text(text, encoding="utf-8")
    return text
