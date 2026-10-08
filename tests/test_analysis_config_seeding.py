from __future__ import annotations

import anndata as ad
import inspect
import numpy as np
import pandas as pd
import pytest

from scintilla.analysis_config import AnalysisConfig


class _Predictor:
    def predict(self, X):
        return np.zeros(len(X), dtype=int)


def _classification_data() -> ad.AnnData:
    adata = ad.AnnData(X=np.arange(48, dtype=float).reshape(12, 4))
    labels = pd.Categorical([0, 1] * 6)
    adata.obs["cell_type"] = labels
    adata.obs["target"] = labels
    return adata


def test_supervised_analysis_uses_config_include_shap_when_omitted(monkeypatch):
    from scintilla.classification import run

    shap_calls = []

    def fake_classifier(X_train, X_test, y_train, y_test, **kwargs):
        return _Predictor(), {"accuracy": 1.0, "f1": 1.0}

    def fake_shap(*args, **kwargs):
        shap_calls.append((args, kwargs))
        return None, None

    monkeypatch.setattr(run, "logistic_regression_classification", fake_classifier)
    monkeypatch.setattr(run, "shap_analysis", fake_shap)

    run.supervised_analysis(
        _classification_data(),
        normality=True,
        models=["LogReg"],
        verbose=False,
        config=AnalysisConfig(include_shap=False),
    )

    assert shap_calls == []


def test_supervised_analysis_explicit_false_overrides_config_include_shap(monkeypatch):
    from scintilla.classification import run

    shap_calls = []

    def fake_classifier(X_train, X_test, y_train, y_test, **kwargs):
        return _Predictor(), {"accuracy": 1.0, "f1": 1.0}

    monkeypatch.setattr(run, "logistic_regression_classification", fake_classifier)
    monkeypatch.setattr(
        run,
        "shap_analysis",
        lambda *args, **kwargs: shap_calls.append((args, kwargs)),
    )

    run.supervised_analysis(
        _classification_data(),
        normality=True,
        include_shap=False,
        models=["LogReg"],
        verbose=False,
        config=AnalysisConfig(include_shap=True),
    )

    assert shap_calls == []


def test_feature_selection_explicit_n_features_overrides_config(monkeypatch):
    from scintilla.feature_selection import benchmark as benchmark_module
    from scintilla.feature_selection import mutual_information

    def fake_mi(X, y, n_features, **kwargs):
        n_selected = min(n_features, X.shape[1])
        return np.arange(n_selected), pd.DataFrame()

    monkeypatch.setattr(mutual_information, "mi_feature_selection", fake_mi)

    result = benchmark_module.benchmark_feature_selection(
        _classification_data(),
        target_col="target",
        methods=["mutual_information"],
        n_features=50,
        config=AnalysisConfig(n_features=1),
    )

    assert result.loc[0, "n_features"] == 4


def test_feature_selection_uses_config_n_features_when_omitted(monkeypatch):
    from scintilla.feature_selection import benchmark as benchmark_module
    from scintilla.feature_selection import mutual_information

    def fake_mi(X, y, n_features, **kwargs):
        return np.arange(n_features), pd.DataFrame()

    monkeypatch.setattr(mutual_information, "mi_feature_selection", fake_mi)

    result = benchmark_module.benchmark_feature_selection(
        _classification_data(),
        target_col="target",
        methods=["mutual_information"],
        config=AnalysisConfig(n_features=2),
    )

    assert result.loc[0, "n_features"] == 2


def test_supervised_analysis_uses_config_random_seed_when_omitted(monkeypatch):
    from scintilla.classification import run

    split_random_states = []
    real_split = run.train_test_split

    def recording_split(*args, **kwargs):
        split_random_states.append(kwargs["random_state"])
        return real_split(*args, **kwargs)

    def fake_classifier(X_train, X_test, y_train, y_test, **kwargs):
        return _Predictor(), {"accuracy": 1.0, "f1": 1.0}

    monkeypatch.setattr(run, "train_test_split", recording_split)
    monkeypatch.setattr(run, "logistic_regression_classification", fake_classifier)

    run.supervised_analysis(
        _classification_data(),
        normality=True,
        include_shap=False,
        models=["LogReg"],
        verbose=False,
        config=AnalysisConfig(random_seed=731),
    )

    assert split_random_states == [731]


def test_supervised_analysis_uses_config_test_size_when_omitted(monkeypatch):
    from scintilla.classification import run

    split_test_sizes = []
    real_split = run.train_test_split

    def recording_split(*args, **kwargs):
        split_test_sizes.append(kwargs["test_size"])
        return real_split(*args, **kwargs)

    def fake_classifier(X_train, X_test, y_train, y_test, **kwargs):
        return _Predictor(), {"accuracy": 1.0, "f1": 1.0}

    monkeypatch.setattr(run, "train_test_split", recording_split)
    monkeypatch.setattr(run, "logistic_regression_classification", fake_classifier)

    run.supervised_analysis(
        _classification_data(),
        normality=True,
        include_shap=False,
        models=["LogReg"],
        verbose=False,
        config=AnalysisConfig(test_size=0.5),
    )

    assert split_test_sizes == [0.5]


def test_supervised_analysis_uses_config_verbose_when_omitted(monkeypatch, capsys):
    from scintilla.classification import run

    def fake_classifier(X_train, X_test, y_train, y_test, **kwargs):
        return _Predictor(), {"accuracy": 1.0, "f1": 1.0}

    monkeypatch.setattr(run, "logistic_regression_classification", fake_classifier)

    run.supervised_analysis(
        _classification_data(),
        normality=True,
        include_shap=False,
        models=["LogReg"],
        config=AnalysisConfig(verbose=False),
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_random_forest_classification_uses_explicit_random_state():
    from scintilla.classification.models import random_forest_classification

    X_train = np.arange(40, dtype=float).reshape(10, 4)
    X_test = np.arange(8, dtype=float).reshape(2, 4)
    y_train = np.array([0, 1] * 5)
    y_test = np.array([0, 1])

    model, _ = random_forest_classification(
        X_train, X_test, y_train, y_test, n_estimators=2, random_state=731
    )

    assert model.random_state == 731


@pytest.mark.parametrize(
    "function_name",
    [
        "svm_classification",
        "logistic_regression_classification",
        "mlp_classification",
        "xgboost_classification",
        "lightgbm_classification",
        "gradient_boosting_classification",
        "stacking_ensemble_classification",
    ],
)
def test_stochastic_classifier_public_api_accepts_random_state(function_name):
    from scintilla.classification import models

    assert "random_state" in inspect.signature(getattr(models, function_name)).parameters


def test_supervised_analysis_passes_random_state_to_stochastic_models(monkeypatch):
    from scintilla.classification import run

    model_random_states = []

    def fake_classifier(X_train, X_test, y_train, y_test, random_state):
        model_random_states.append(random_state)
        return _Predictor(), {"accuracy": 1.0, "f1": 1.0}

    monkeypatch.setattr(run, "random_forest_classification", fake_classifier)

    run.supervised_analysis(
        _classification_data(),
        normality=True,
        include_shap=False,
        models=["RF"],
        verbose=False,
        random_state=731,
    )

    assert model_random_states == [731]


@pytest.mark.parametrize(
    ("module_name", "function_name"),
    [
        ("scintilla.benchmarking.scalability", "scalability_sweep"),
        ("scintilla.benchmarking.time_estimator", "estimate_benchmark_time"),
        ("scintilla.classification.benchmark", "benchmark_models_comprehensive"),
        ("scintilla.classification.diagnostics", "check_normality_for_classifier"),
        ("scintilla.classification.feature_importance", "shap_analysis"),
        ("scintilla.clustering.benchmark", "benchmark_clustering_methods"),
        ("scintilla.clustering.leiden", "leiden_clustering"),
        ("scintilla.clustering.louvain", "louvain_clustering"),
        ("scintilla.clustering.spectral", "spectral_clustering"),
        ("scintilla.clustering.consensus", "consensus_clustering"),
        ("scintilla.clustering.run", "unsupervised_analysis"),
        ("scintilla.clustering.similarity_assembly", "build_per_celltype_clustering"),
        ("scintilla.clustering.similarity_assembly", "patient_level_clustering"),
        ("scintilla.differential_expression.permutation", "permutation_de"),
        ("scintilla.dimensionality_reduction.umap", "run_umap"),
        ("scintilla.dimensionality_reduction.tsne", "run_tsne"),
        ("scintilla.dimensionality_reduction.diffusion_map", "run_diffusion_map"),
        ("scintilla.dimensionality_reduction.force_directed", "run_force_directed"),
        ("scintilla.dimensionality_reduction.benchmark", "benchmark_embeddings"),
        ("scintilla.feature_selection.benchmark", "benchmark_feature_selection"),
        ("scintilla.feature_selection.benchmark", "hvg_sensitivity_analysis"),
        ("scintilla.feature_selection.mutual_information", "mi_feature_selection"),
        ("scintilla.feature_selection.mrmr", "mrmr_selection"),
        ("scintilla.feature_selection.pca_loadings", "extract_top_genes_per_pc"),
        ("scintilla.feature_selection.pca_loadings", "validate_reduced_set"),
        ("scintilla.preprocessing.benchmark", "benchmark_transformations"),
        ("scintilla.preprocessing.normality", "check_normality"),
        ("scintilla.preprocessing.pca", "run_pca"),
        ("scintilla.preprocessing.transformations", "glm_pca_transform"),
        ("scintilla.visualisation.batch_plots", "plot_batch_correction_comparison"),
        ("scintilla.classification.visualise", "plot_celltype_confusion_network"),
    ],
)
def test_public_stochastic_api_accepts_random_state(module_name, function_name):
    import importlib

    function = getattr(importlib.import_module(module_name), function_name)
    assert "random_state" in inspect.signature(function).parameters


def test_unsupervised_analysis_threads_config_pca_fields_and_seed(monkeypatch):
    from scintilla.clustering import run

    pca_kwargs = []
    benchmark_kwargs = []

    def fake_pca(adata, **kwargs):
        pca_kwargs.append(kwargs)
        adata.obsm["X_pca"] = np.asarray(adata.X)
        return adata

    def fake_benchmark(*args, **kwargs):
        benchmark_kwargs.append(kwargs)
        return pd.DataFrame(columns=["method", "params", "ari"]), {}, None

    monkeypatch.setattr(run, "run_pca", fake_pca)
    monkeypatch.setattr(run, "benchmark_clustering_methods", fake_benchmark)

    run.unsupervised_analysis(
        _classification_data(),
        cell_type_col="target",
        config=AnalysisConfig(
            n_pca_comps=7,
            auto_pca_components="marchenko_pastur",
            mp_sigma_method="trimmed_mean",
            random_seed=731,
            verbose=False,
        ),
    )

    assert pca_kwargs == [{
        "n_comps": 7,
        "auto_components": "marchenko_pastur",
        "mp_sigma_method": "trimmed_mean",
        "random_state": 731,
    }]
    assert benchmark_kwargs[0]["random_state"] == 731
    assert benchmark_kwargs[0]["verbose"] is False


def test_unsupervised_analysis_explicit_none_disables_config_pca_adaptation(
    monkeypatch,
):
    """Catch config values overriding explicit ``None`` PCA controls."""
    from scintilla.clustering import run

    pca_kwargs = []

    def fake_pca(adata, **kwargs):
        pca_kwargs.append(kwargs)
        adata.obsm["X_pca"] = np.asarray(adata.X)
        return adata

    monkeypatch.setattr(run, "run_pca", fake_pca)
    monkeypatch.setattr(
        run,
        "benchmark_clustering_methods",
        lambda *args, **kwargs: (
            pd.DataFrame(columns=["method", "params", "ari"]), {}, None,
        ),
    )

    run.unsupervised_analysis(
        _classification_data(),
        cell_type_col="target",
        auto_pca_components=None,
        mp_sigma_method=None,
        verbose=False,
        config=AnalysisConfig(
            auto_pca_components="gavish_donoho",
            mp_sigma_method="trimmed_mean",
        ),
    )

    assert pca_kwargs[0]["auto_components"] is None
    assert pca_kwargs[0]["mp_sigma_method"] is None


def test_classifier_benchmark_threads_config_bootstrap_fields(monkeypatch):
    from scintilla.classification import benchmark

    calls = []

    def fake_bootstrap(*args, **kwargs):
        calls.append(kwargs)
        return pd.DataFrame(), None

    monkeypatch.setattr(benchmark, "_benchmark_632plus", fake_bootstrap)
    config = AnalysisConfig(
        n_pca_comps=9,
        verbose=False,
        bootstrap_ci=True,
        n_bootstrap=17,
        no_info_method="permutation",
        classification_estimator="bootstrap_632plus",
        random_seed=731,
    )

    benchmark.benchmark_models_comprehensive(
        _classification_data(), target_col="target", config=config
    )

    assert calls == [{
        "B": 17,
        "bootstrap_ci": True,
        "no_info_method": "permutation",
        "random_state": 731,
    }]


def test_transformation_benchmark_uses_config_scoring_method():
    from scintilla.preprocessing.benchmark import benchmark_transformations

    adata = _classification_data()
    result, _, _ = benchmark_transformations(
        adata,
        transformations={"identity": lambda value: value.copy()},
        config=AnalysisConfig(scoring_method="borda", verbose=False),
    )

    assert "borda_rank" in result.columns


def test_transformation_benchmark_rejects_unknown_scoring_method():
    """Catch misspelled scoring modes silently using weighted ranking."""
    from scintilla.preprocessing.benchmark import benchmark_transformations

    with pytest.raises(ValueError, match="Unknown transformation scoring_method"):
        benchmark_transformations(
            _classification_data(),
            transformations={"identity": lambda value: value.copy()},
            scoring_method="weigthd",
            verbose=False,
        )


@pytest.mark.parametrize(
    ("module_name", "function_name"),
    [
        ("scintilla.annotation.marker_based", "annotate_by_markers"),
        ("scintilla.annotation.rank_genes", "find_marker_genes"),
        ("scintilla.batch_correction.benchmark", "benchmark_batch_correction"),
        ("scintilla.batch_correction.harmony", "harmony_correct"),
        ("scintilla.batch_correction.bbknn", "bbknn_correct"),
        ("scintilla.batch_correction.scanorama", "scanorama_correct"),
        ("scintilla.differential_expression.rank_genes", "rank_genes_groups"),
        ("scintilla.evaluation.batch_metrics", "graph_connectivity"),
        ("scintilla.evaluation.batch_metrics", "principal_component_regression"),
    ],
)
def test_remaining_public_stochastic_api_accepts_random_state(
    module_name, function_name,
):
    import importlib

    function = getattr(importlib.import_module(module_name), function_name)
    assert "random_state" in inspect.signature(function).parameters


def test_batch_benchmark_uses_config_and_explicit_values_win(monkeypatch):
    from scintilla.batch_correction import benchmark

    calls = []

    def fake_apply(adata, method, batch_key, n_pcs, random_state):
        calls.append((method, n_pcs, random_state))
        return adata

    monkeypatch.setattr(benchmark, "_apply_method", fake_apply)
    monkeypatch.setattr(benchmark, "batch_asw", lambda *args: 0.5)
    monkeypatch.setattr(benchmark, "bio_conservation_score", lambda *args: 0.5)

    config_result = benchmark.benchmark_batch_correction(
        _classification_data(),
        batch_key="target",
        label_key="target",
        methods=["combat"],
        config=AnalysisConfig(
            n_pca_comps=7,
            scoring_method="borda",
            random_seed=731,
        ),
    )
    explicit_result = benchmark.benchmark_batch_correction(
        _classification_data(),
        batch_key="target",
        label_key="target",
        methods=["combat"],
        n_pcs=3,
        scoring_method="single_metric",
        random_state=919,
        config=AnalysisConfig(
            n_pca_comps=7,
            scoring_method="borda",
            random_seed=731,
        ),
    )

    assert calls == [("combat", 7, 731), ("combat", 3, 919)]
    assert "borda_rank" in config_result["leaderboard"].columns
    assert "borda_rank" not in explicit_result["leaderboard"].columns


def test_batch_benchmark_rejects_unknown_scoring_method():
    """Catch unknown batch scoring modes silently using batch ASW."""
    from scintilla.batch_correction.benchmark import benchmark_batch_correction

    with pytest.raises(ValueError, match="Unknown batch scoring_method"):
        benchmark_batch_correction(
            _classification_data(),
            batch_key="target",
            methods=["not-a-real-method"],
            scoring_method="weigthd",
        )


def test_scanpy_annotation_and_logreg_de_forward_random_state(monkeypatch):
    import sys
    from types import SimpleNamespace

    from scintilla.annotation.marker_based import annotate_by_markers
    from scintilla.differential_expression.rank_genes import rank_genes_groups

    score_seeds = []
    rank_seeds = []

    def fake_score_genes(adata, *, score_name, random_state, **kwargs):
        score_seeds.append(random_state)
        adata.obs[score_name] = 1.0

    def fake_rank_genes(adata, *, random_state, **kwargs):
        rank_seeds.append(random_state)
        adata.uns["rank_genes_groups"] = {
            "names": np.rec.fromarrays([["0"]], names=["A"]),
            "scores": np.rec.fromarrays([[1.0]], names=["A"]),
            "pvals": np.rec.fromarrays([[0.1]], names=["A"]),
            "pvals_adj": np.rec.fromarrays([[0.2]], names=["A"]),
            "logfoldchanges": np.rec.fromarrays([[1.5]], names=["A"]),
        }

    fake_scanpy = SimpleNamespace(
        tl=SimpleNamespace(
            score_genes=fake_score_genes,
            rank_genes_groups=fake_rank_genes,
        )
    )
    monkeypatch.setitem(sys.modules, "scanpy", fake_scanpy)

    adata = _classification_data()
    annotate_by_markers(adata, {"A": ["0"]}, random_state=731)
    rank_genes_groups(adata, "target", method="logreg", n_genes=1, random_state=919)

    assert score_seeds == [731]
    assert rank_seeds == [919]


def test_batch_correction_backends_forward_random_state(monkeypatch):
    import sys
    from types import SimpleNamespace

    from scintilla.batch_correction.bbknn import bbknn_correct
    from scintilla.batch_correction.harmony import harmony_correct
    from scintilla.batch_correction.scanorama import scanorama_correct

    calls = []
    monkeypatch.setitem(
        sys.modules,
        "harmonypy",
        SimpleNamespace(
            run_harmony=lambda X, obs, key, **kwargs: (
                calls.append(("harmony", kwargs["random_state"]))
                or SimpleNamespace(Z_corr=X.T)
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "bbknn",
        SimpleNamespace(
            bbknn=lambda adata, **kwargs: calls.append(
                ("bbknn", kwargs["pynndescent_random_state"])
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "scanpy",
        SimpleNamespace(tl=SimpleNamespace(pca=lambda *args, **kwargs: None)),
    )

    def fake_scanorama(adatas, *, seed, **kwargs):
        calls.append(("scanorama", seed))
        for item in adatas:
            item.obsm["X_scanorama"] = np.asarray(item.X)[:, :2]
        return adatas

    monkeypatch.setitem(
        sys.modules, "scanorama", SimpleNamespace(correct_scanpy=fake_scanorama),
    )

    adata = _classification_data()
    adata.obsm["X_pca"] = np.asarray(adata.X)
    harmony_correct(adata, "target", random_state=731)
    bbknn_correct(adata, "target", random_state=919)
    scanorama_correct(adata, "target", random_state=2026)

    assert calls == [("harmony", 731), ("bbknn", 919), ("scanorama", 2026)]


def test_batch_evaluation_helpers_forward_random_state(monkeypatch):
    import sys
    from types import SimpleNamespace

    import scipy.sparse as sp
    from sklearn import decomposition

    from scintilla.evaluation.batch_metrics import (
        graph_connectivity,
        principal_component_regression,
    )

    neighbor_seeds = []
    pca_seeds = []

    def fake_neighbors(adata, *, random_state):
        neighbor_seeds.append(random_state)
        adata.obsp["connectivities"] = sp.eye(adata.n_obs, format="csr")

    class FakePCA:
        def __init__(self, n_components, random_state):
            pca_seeds.append(random_state)
            self.n_components = n_components
            self.explained_variance_ratio_ = np.full(n_components, 1 / n_components)

        def fit_transform(self, X):
            return np.asarray(X)[:, :self.n_components]

    monkeypatch.setitem(
        sys.modules,
        "scanpy",
        SimpleNamespace(pp=SimpleNamespace(neighbors=fake_neighbors)),
    )
    monkeypatch.setattr(decomposition, "PCA", FakePCA)

    graph_connectivity(_classification_data(), "target", random_state=731)
    X = np.arange(48, dtype=float).reshape(12, 4)
    principal_component_regression(
        X, X + 1, np.array([0, 1] * 6), random_state=919,
    )

    assert neighbor_seeds == [731]
    assert pca_seeds == [919, 919]


def test_scalability_sweep_forwards_resolved_seed_and_config_to_method():
    from scintilla.benchmarking.scalability import scalability_sweep

    config = AnalysisConfig(random_seed=731)
    received = []

    def method(adata, *, random_state, config, marker):
        received.append((adata.n_obs, random_state, config, marker))

    scalability_sweep(
        method,
        _classification_data(),
        fractions=[0.5],
        config=config,
        marker="forwarded",
    )

    assert received == [(6, 731, config, "forwarded")]


def test_scalability_sweep_preserves_kwargs_only_method_contract():
    """Catch injected defaults and swallowed explicit ``config=None`` kwargs."""
    from scintilla.benchmarking.scalability import scalability_sweep

    received = []

    def method(adata, **kwargs):
        received.append(kwargs)

    scalability_sweep(method, _classification_data(), fractions=[0.5])
    scalability_sweep(
        method, _classification_data(), fractions=[0.5], config=None,
    )
    scalability_sweep(
        method, _classification_data(), fractions=[0.5], random_state=None,
    )

    assert received == [{}, {"config": None}, {"random_state": None}]


def test_classifier_benchmark_explicit_none_cv_folds_disables_config_cv(
    monkeypatch,
):
    from scintilla.classification import benchmark

    cv_calls = []

    def fake_cv(*args, **kwargs):
        cv_calls.append((args, kwargs))
        return pd.DataFrame(), None

    def fake_classifier(X_train, X_test, y_train, y_test):
        return _Predictor(), {"accuracy": 1.0, "f1": 1.0}

    monkeypatch.setattr(benchmark, "_benchmark_cv", fake_cv)
    monkeypatch.setattr(benchmark, "_MODEL_FNS", {"LDA": fake_classifier})

    result, _ = benchmark.benchmark_models_comprehensive(
        _classification_data(),
        target_col="target",
        cv_folds=None,
        use_pca=False,
        verbose=False,
        config=AnalysisConfig(cv_folds=3),
    )

    assert cv_calls == []
    assert result.loc[0, "model"] == "LDA"


def test_diffusion_map_forwards_random_state_to_scanpy_leaf(monkeypatch):
    import sys
    from types import SimpleNamespace

    from scintilla.dimensionality_reduction.diffusion_map import run_diffusion_map

    diffmap_seeds = []

    def fake_diffmap(adata, *, n_comps, random_state):
        diffmap_seeds.append(random_state)
        adata.obsm["X_diffmap"] = np.zeros((adata.n_obs, n_comps))

    monkeypatch.setitem(
        sys.modules,
        "scanpy",
        SimpleNamespace(
            pp=SimpleNamespace(neighbors=lambda *args, **kwargs: None),
            tl=SimpleNamespace(diffmap=fake_diffmap),
        ),
    )
    adata = _classification_data()
    adata.obsm["X_pca"] = np.asarray(adata.X)

    result = run_diffusion_map(adata, n_comps=3, random_state=731)

    assert diffmap_seeds == [731]
    assert result.obsm["X_diffmap"].shape == (12, 3)


def test_shap_analysis_forwards_random_state_to_explainer(monkeypatch):
    import sys
    from types import SimpleNamespace

    from scintilla.classification.feature_importance import shap_analysis

    explainer_seeds = []

    class FakeExplainer:
        def __init__(self, model, background, *, seed):
            explainer_seeds.append(seed)

        def __call__(self, X):
            return SimpleNamespace(values=np.ones_like(X, dtype=float))

    monkeypatch.setitem(sys.modules, "shap", SimpleNamespace(Explainer=FakeExplainer))
    X = np.arange(12, dtype=float).reshape(4, 3)

    values, importances = shap_analysis(object(), X, X, random_state=731)

    assert explainer_seeds == [731]
    assert values.shape == (4, 3)
    assert importances["importance"].tolist() == [1.0, 1.0, 1.0]


@pytest.mark.filterwarnings("ignore:Observation names are not unique:UserWarning")
def test_large_transformation_benchmark_seeds_all_silhouette_samples(
    monkeypatch,
):
    from scintilla.preprocessing import benchmark

    silhouette_calls = []

    def fake_silhouette(X, labels, *, sample_size, random_state=None):
        silhouette_calls.append((sample_size, random_state))
        return float(np.random.default_rng(random_state).random())

    monkeypatch.setattr(benchmark, "silhouette_score", fake_silhouette)
    monkeypatch.setattr(benchmark, "_knn_overlap", lambda *args, **kwargs: 0.5)
    monkeypatch.setattr(
        benchmark, "_pca_preservation", lambda *args, **kwargs: 0.5,
    )

    adata = ad.AnnData(X=np.arange(3003, dtype=float).reshape(1001, 3))
    adata.obs["target"] = pd.Categorical([0, 1] * 500 + [0])
    options = {
        "transformations": {"identity": lambda value: value.copy()},
        "n_pca_components": 2,
        "verbose": False,
        "random_state": 731,
    }

    labelled, _, _ = benchmark.benchmark_transformations(
        adata,
        cell_type_col="target",
        bootstrap_ci=True,
        n_bootstrap=1,
        **options,
    )
    labelled_repeat, _, _ = benchmark.benchmark_transformations(
        adata,
        cell_type_col="target",
        bootstrap_ci=True,
        n_bootstrap=1,
        **options,
    )
    inferred, _, _ = benchmark.benchmark_transformations(adata, **options)
    inferred_repeat, _, _ = benchmark.benchmark_transformations(adata, **options)

    assert silhouette_calls == [
        (1000, 731), (500, 731),
        (1000, 731), (500, 731),
        (1000, 731), (1000, 731),
    ]
    assert labelled.loc[0, "composite_score"] == labelled_repeat.loc[0, "composite_score"]
    assert labelled.loc[0, "composite_ci_low"] == labelled_repeat.loc[0, "composite_ci_low"]
    assert inferred.loc[0, "composite_score"] == inferred_repeat.loc[0, "composite_score"]
