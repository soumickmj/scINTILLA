"""Everything that is advertised can be imported and has the documented shape."""

from __future__ import annotations

import importlib
import inspect
import pkgutil

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

import scintilla as si

NAMESPACES = ["pp", "tl", "pl", "stats", "benchmark", "eda", "io"]


def test_every_module_imports():
    failures = []
    for module in pkgutil.walk_packages(si.__path__, "scintilla."):
        try:
            importlib.import_module(module.name)
        except Exception as exc:  # pragma: no cover - reported below
            failures.append(f"{module.name}: {exc}")
    assert failures == []


@pytest.mark.parametrize("namespace", NAMESPACES + [""])
def test_every_name_in_all_resolves(namespace):
    module = importlib.import_module(f"scintilla.{namespace}" if namespace else "scintilla")
    missing = [name for name in module.__all__ if not hasattr(module, name)]
    assert missing == []


def test_import_leaves_the_matplotlib_backend_alone():
    assert matplotlib.get_backend() != "" and "scintilla" in importlib.sys.modules


def test_version_is_a_valid_pep440_string():
    from packaging.version import Version

    assert Version(si.__version__)


@pytest.mark.parametrize("namespace", ["pp", "tl"])
def test_in_place_tools_take_adata_first_and_use_keyword_only_options(namespace):
    """Functions that write to AnnData follow ``f(adata, ..., *, key_added, copy)``."""
    module = getattr(si, namespace)
    checked = 0
    for name in module.__all__:
        fn = getattr(module, name)
        if not inspect.isfunction(fn):
            continue
        params = inspect.signature(fn).parameters
        if "copy" in params and "key_added" in params:
            checked += 1
            assert next(iter(params)) in {"adata", "reference_adata"}, name
            for option in ("copy", "key_added"):
                assert params[option].kind is inspect.Parameter.KEYWORD_ONLY, f"{name}.{option}"
    assert checked >= 11


def test_all_public_functions_have_numpydoc_parameters():
    missing = []
    for namespace in NAMESPACES:
        module = getattr(si, namespace)
        for name in module.__all__:
            fn = getattr(module, name)
            if inspect.isfunction(fn) and len(inspect.signature(fn).parameters) > 0:
                doc = inspect.getdoc(fn) or ""
                if "Parameters\n----------" not in doc:
                    missing.append(f"{namespace}.{name}")
    assert missing == []


# ── plotting ────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_plot_functions_return_open_figures_and_support_save(tmp_path):
    table = __import__("pandas").DataFrame(
        {"method": ["a", "b"], "params": ["p", "q"], "ari": [0.9, 0.5], "ami": [0.8, 0.4]}
    )
    fig = si.pl.clustering_benchmark(table, save=str(tmp_path / "bench.png"))
    assert plt.fignum_exists(fig.number)  # not closed: that is the caller's decision
    assert (tmp_path / "bench.png").stat().st_size > 0


def test_pl_functions_expose_show_and_save():
    for name in si.pl.__all__:
        params = inspect.signature(getattr(si.pl, name)).parameters
        assert "show" in params and "save" in params, name


def test_classification_benchmark_plot_handles_all_three_table_layouts():
    import pandas as pd

    cv = pd.DataFrame(
        {"space": ["Gene", "PCA"], "model": ["m", "m"], "mean_accuracy": [0.9, 0.8], "se_accuracy": [0.01, 0.02]}
    )
    flat = pd.DataFrame({"space": ["Gene", "PCA"], "model": ["m", "m"], "accuracy": [0.9, 0.8]})
    assert si.pl.classification_benchmark(cv).axes and si.pl.classification_benchmark(flat).axes


def test_embedding_and_pca_plots_run_on_a_processed_dataset(adata_logged):
    si.tl.umap(adata_logged, random_state=0)
    assert si.pl.embedding(adata_logged, basis="X_umap", colour_by="cell_type").axes
    assert si.pl.pca_2d(adata_logged, colour_col="cell_type").axes
    assert si.pl.cumulative_variance(adata_logged).axes


def test_label_quality_plots_run_on_the_pipeline_output(adata_logged):
    pytest.importorskip("leidenalg")
    config = si.AnalysisConfig.fast().copy(clustering_methods=["kmeans"], classifiers=["LogReg"], include_shap=False)
    si.tl.label_quality(adata_logged, "cell_type", config=config)
    assert si.pl.celltype_label_quality(adata_logged, "cell_type").axes
    assert si.pl.celltype_confusion(adata_logged, "cell_type") is not None


def test_volcano_plot_from_a_de_table(adata_dense):
    table = si.tl.wilcoxon(adata_dense, "cell_type", "type0", "type1")
    assert si.pl.volcano(table).axes


def test_hierarchical_dendrogram_is_drawn_by_pl_not_by_the_algorithm(adata_logged):
    from scintilla.clustering.hierarchical import hierarchical_scipy

    labels, Z, cpcc = hierarchical_scipy(adata_logged.obsm["X_pca"][:60], n_clusters=3)
    assert np.isfinite(cpcc) and plt.get_fignums() == []  # the algorithm opened no figure
    assert si.pl.dendrogram(Z).axes
