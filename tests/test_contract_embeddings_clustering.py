"""Embeddings and clustering: only the requested key is written, seeds are honoured."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import scintilla as si

EMBEDDINGS = [
    ("umap", "X_umap", {}),
    ("tsne", "X_tsne", {"perplexity": 10.0}),
    ("diffmap", "X_diffmap", {"n_comps": 5}),
]


@pytest.mark.parametrize(("name", "key", "kwargs"), EMBEDDINGS)
def test_embeddings_write_only_their_obsm_key(adata_logged, name, key, kwargs):
    obsm, obsp, uns, layers = (set(getattr(adata_logged, a)) for a in ("obsm", "obsp", "uns", "layers"))
    assert getattr(si.tl, name)(adata_logged, random_state=1, **kwargs) is None
    assert set(adata_logged.obsm) - obsm == {key}
    assert set(adata_logged.obsp) == obsp  # no neighbour graph leaked
    assert set(adata_logged.uns) - uns == {"scintilla"}
    assert set(adata_logged.layers) == layers


@pytest.mark.parametrize(("name", "key", "kwargs"), EMBEDDINGS)
def test_embeddings_copy_key_added_and_seed(adata_logged, name, key, kwargs):
    fn = getattr(si.tl, name)
    first = fn(adata_logged, random_state=7, key_added="mine", copy=True, **kwargs)
    second = fn(adata_logged, random_state=7, key_added="mine", copy=True, **kwargs)
    assert "mine" in first.obsm and "mine" not in adata_logged.obsm
    np.testing.assert_allclose(first.obsm["mine"], second.obsm["mine"], rtol=1e-5, atol=1e-5)


def test_embeddings_use_a_temporary_pca_without_storing_it(adata_dense):
    si.tl.umap(adata_dense, random_state=0)
    assert "X_pca" not in adata_dense.obsm and "X_umap" in adata_dense.obsm


def test_draw_graph_records_the_layout_actually_used(adata_logged):
    pytest.importorskip("igraph")
    si.tl.draw_graph(adata_logged, random_state=0)
    assert adata_logged.uns["scintilla"]["X_draw_graph_fa"]["params"]["layout"] in {"fa", "fr"}
    assert adata_logged.obsm["X_draw_graph_fa"].shape == (adata_logged.n_obs, 2)


def test_embedding_benchmark_reports_unknown_methods_and_leaves_the_input_alone(adata_logged):
    obsm = set(adata_logged.obsm)
    table = si.benchmark.benchmark_embeddings(adata_logged, methods=["umap", "bogus"])
    assert table.set_index("method").loc["bogus", "status"] == "unknown"
    assert table.set_index("method").loc["umap", "status"] == "ok"
    assert set(adata_logged.obsm) == obsm


# ── clustering ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["leiden", "louvain"])
def test_graph_clustering_does_not_touch_the_input(adata_logged, name):
    pytest.importorskip("leidenalg" if name == "leiden" else "louvain")
    obs, obsp, uns = list(adata_logged.obs.columns), set(adata_logged.obsp), set(adata_logged.uns)
    labels = getattr(si.tl, name)(adata_logged, resolution=0.5, random_state=0)
    assert labels.shape == (adata_logged.n_obs,)
    assert list(adata_logged.obs.columns) == obs and set(adata_logged.obsp) == obsp and set(adata_logged.uns) == uns


def test_leiden_key_added_stores_a_categorical_and_provenance(adata_logged):
    pytest.importorskip("leidenalg")
    labels = si.tl.leiden(adata_logged, resolution=0.5, random_state=0, key_added="clusters")
    assert adata_logged.obs["clusters"].dtype == "category"
    assert adata_logged.obs["clusters"].astype(int).to_numpy().tolist() == labels.tolist()
    assert adata_logged.uns["scintilla"]["clusters"]["params"]["method"] == "leiden"


def test_leiden_finds_the_planted_cell_types(adata_logged):
    pytest.importorskip("leidenalg")
    from sklearn.metrics import adjusted_rand_score

    labels = si.tl.leiden(adata_logged, resolution=0.3, random_state=0)
    assert adjusted_rand_score(adata_logged.obs.cell_type, labels) > 0.9


@pytest.mark.parametrize("name", ["leiden", "louvain"])
def test_graph_clustering_is_reproducible(adata_logged, name):
    pytest.importorskip("leidenalg" if name == "leiden" else "louvain")
    fn = getattr(si.tl, name)
    np.testing.assert_array_equal(fn(adata_logged, random_state=3), fn(adata_logged, random_state=3))


def test_kmeans_recovers_the_cell_types_and_accepts_sparse_input(adata_logged):
    from scipy import sparse
    from sklearn.metrics import adjusted_rand_score

    labels, _, metrics = si.tl.kmeans(adata_logged.obsm["X_pca"], n_clusters=4, random_state=0)
    assert adjusted_rand_score(adata_logged.obs.cell_type, labels) > 0.95
    assert metrics["silhouette"] > 0.3
    sparse_labels, _, _ = si.tl.kmeans(sparse.csr_matrix(adata_logged.obsm["X_pca"]), n_clusters=4, random_state=0)
    assert adjusted_rand_score(labels, sparse_labels) > 0.99


def test_hierarchical_returns_labels_only_and_scipy_mode_is_deprecated(adata_logged):
    X = adata_logged.obsm["X_pca"]
    labels = si.tl.hierarchical(X, n_clusters=4)
    assert labels.shape == (adata_logged.n_obs,)
    with pytest.warns(DeprecationWarning, match="hierarchical_scipy"):
        result = si.tl.hierarchical(X, n_clusters=4, mode="scipy")
    assert len(result) == 3  # labels, linkage matrix, cophenetic correlation: no figure any more


def test_spectral_has_a_stable_return_shape_and_a_separate_grid_search(adata_logged):
    X = adata_logged.obsm["X_pca"][:80]
    labels, model, metrics = si.tl.spectral(X, n_clusters=3, random_state=0)
    assert labels.shape == (80,) and metrics["n_clusters"] == 3
    table, best = si.tl.spectral_grid_search(X, n_clusters_range=[2, 3], random_state=0)
    assert list(table["status"]) == ["ok", "ok"] and best.shape == (80,)
    with pytest.warns(DeprecationWarning, match="spectral_grid_search"):
        si.tl.spectral(X, n_clusters=None, n_clusters_range=[2], random_state=0)


def test_dbscan_and_hdbscan_keep_failures_visible(adata_logged):
    X = adata_logged.obsm["X_pca"]
    eps = si.tl.estimate_eps(X, min_samples=5)
    labels, n_clusters, n_noise, _ = si.tl.dbscan(X, eps=eps * 1.5, min_samples=5)
    assert len(labels) == len(X) and n_clusters >= 1 and n_noise >= 0
    table, best = si.tl.hdbscan(X, min_cluster_size_range=[10], min_samples_range=[None])
    assert list(table["status"]) == ["ok"] and len(best) == len(X)


def test_clustering_benchmark_returns_data_only_and_lists_every_method(adata_logged):
    config = si.AnalysisConfig.fast().copy(clustering_methods=["kmeans"])
    table, labels = si.benchmark.benchmark_clustering(adata_logged, "cell_type", config=config, random_state=0)
    assert isinstance(table, pd.DataFrame) and isinstance(labels, dict)
    assert set(table["status"]) == {"ok"} and table["ari"].max() > 0.95
    fig = si.pl.clustering_benchmark(table)
    assert fig.axes  # plotting is a separate, explicit step


def test_array_level_clustering_accepts_an_anndata_with_use_rep_and_key_added(adata_logged):
    labels, _, _ = si.tl.kmeans(adata_logged, n_clusters=4, use_rep="X_pca", key_added="km", random_state=0)
    assert adata_logged.obs["km"].astype(int).to_numpy().tolist() == labels.tolist()
    assert adata_logged.uns["scintilla"]["km"]["params"]["use_rep"] == "X_pca"
    with pytest.raises(KeyError, match="obsm has no"):
        si.tl.kmeans(adata_logged, n_clusters=4, use_rep="missing")
    with pytest.raises(TypeError, match="requires an AnnData"):
        si.tl.kmeans(adata_logged.obsm["X_pca"], n_clusters=4, key_added="km")
