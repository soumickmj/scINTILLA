"""Contract tests for DE, annotation, batch correction and feature handling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import scintilla as si
from conftest import to_dense

# ── differential expression ─────────────────────────────────────────────


@pytest.mark.parametrize("name", ["wilcoxon", "ttest"])
def test_two_group_tests_agree_between_dense_and_sparse(adata_dense, adata_csr, name):
    fn = getattr(si.tl, name)
    dense = fn(adata_dense, "cell_type", "type0", "type1")
    sp = fn(adata_csr, "cell_type", "type0", "type1")
    pd.testing.assert_frame_equal(dense, sp, rtol=1e-6)


def test_permutation_test_is_seeded_and_agrees_for_sparse(adata_dense, adata_csr):
    kwargs = dict(n_permutations=50, random_state=3)
    a = si.tl.permutation(adata_dense, "cell_type", "type0", "type1", **kwargs)
    b = si.tl.permutation(adata_csr, "cell_type", "type0", "type1", **kwargs)
    pd.testing.assert_frame_equal(a, b, rtol=1e-6)
    assert (a["p_value"] >= 1 / 51).all()


def test_de_finds_the_planted_marker_genes(adata_logged):
    table = si.tl.wilcoxon(adata_logged, "cell_type", "type0", "type1").sort_values("p_value")
    top = set(table.head(20)["gene"])
    assert len(top & {f"gene{i}" for i in range(40)}) >= 18  # blocks 0 and 1 differ between these types


def test_two_group_tests_only_densify_the_two_groups(adata_csr, monkeypatch):
    from scintilla.differential_expression import _common

    seen = []
    original = _common.get_matrix

    def spy(sub, *args, **kwargs):
        seen.append(sub.n_obs)
        return original(sub, *args, **kwargs)

    monkeypatch.setattr(_common, "get_matrix", spy)
    si.tl.wilcoxon(adata_csr, "cell_type", "type0", "type1")
    assert seen == [150]  # two of four equally sized cell types, not all 300 cells


def test_de_validates_groups_and_columns(adata_dense):
    with pytest.raises(ValueError, match="not found"):
        si.tl.ttest(adata_dense, "cell_type", "type0", "nope")
    with pytest.raises(KeyError):
        si.tl.wilcoxon(adata_dense, "missing", "a", "b")


def test_rank_genes_groups_returns_a_table_and_leaves_adata_untouched(adata_logged):
    uns = set(adata_logged.uns)
    table = si.tl.rank_genes_groups(adata_logged, "cell_type", n_genes=10)
    assert list(table.columns) == ["group", "gene", "score", "pval", "pval_adj", "logfoldchange"]
    assert len(table) == 4 * 10 and set(adata_logged.uns) == uns
    si.tl.rank_genes_groups(adata_logged, "cell_type", n_genes=5, key_added="rg")
    assert isinstance(adata_logged.uns["rg"], pd.DataFrame)


def test_pseudobulk_needs_replicates_and_works_with_them(adata_dense):
    adata_dense.obs["condition"] = np.where(adata_dense.obs.cell_type.isin(["type0", "type1"]), "A", "B")
    table = si.tl.pseudobulk(adata_dense, "condition", "sample")
    assert isinstance(table, pd.DataFrame) and len(table) == adata_dense.n_vars
    one_sample_per_condition = adata_dense.copy()
    one_sample_per_condition.obs["sample"] = one_sample_per_condition.obs["condition"]
    with pytest.raises(ValueError, match="at least 2 biological samples"):
        si.tl.pseudobulk(one_sample_per_condition, "condition", "sample")


def test_pseudobulk_agrees_between_dense_and_sparse(adata_dense, adata_csr):
    for a in (adata_dense, adata_csr):
        a.obs["condition"] = np.where(a.obs.cell_type.isin(["type0", "type1"]), "A", "B")
    pd.testing.assert_frame_equal(
        si.tl.pseudobulk(adata_dense, "condition", "sample"), si.tl.pseudobulk(adata_csr, "condition", "sample")
    )


# ── annotation ──────────────────────────────────────────────────────────

MARKERS = {f"type{k}": [f"gene{i}" for i in range(k * 20, k * 20 + 20)] for k in range(4)}


@pytest.mark.parametrize("method", ["score", "threshold"])
def test_annotate_by_markers_labels_cells_in_place(adata_logged, method):
    obs = list(adata_logged.obs.columns)
    assert si.tl.annotate_by_markers(adata_logged, MARKERS, method=method, threshold=1.0, random_state=0) is None
    accuracy = (adata_logged.obs["predicted_cell_type"] == adata_logged.obs["cell_type"]).mean()
    assert accuracy > 0.95
    assert list(adata_logged.obs.columns) == obs + ["predicted_cell_type"]  # no score_* columns leaked
    scores = adata_logged.obsm["predicted_cell_type_scores"]
    assert list(scores.columns) == list(MARKERS) and len(scores) == adata_logged.n_obs


def test_annotate_by_markers_copy_key_added_and_sparse(adata_logged):
    from scipy import sparse

    sp = adata_logged.copy()
    sp.X = sparse.csr_matrix(sp.X)
    out = si.tl.annotate_by_markers(sp, MARKERS, method="threshold", threshold=1.0, key_added="auto", copy=True)
    assert "auto" in out.obs and "auto" not in sp.obs and sparse.issparse(sp.X)


def test_annotate_by_markers_warns_about_unknown_markers_and_rejects_unknown_methods(adata_logged, caplog):
    with caplog.at_level("WARNING", logger="scintilla"):
        si.tl.annotate_by_markers(adata_logged, {"x": ["nope"], **MARKERS})
    assert "no markers of 'x'" in caplog.text
    with pytest.raises(ValueError, match="Unknown method"):
        si.tl.annotate_by_markers(adata_logged, MARKERS, method="bogus")


def test_transfer_labels_maps_a_shuffled_query_correctly(adata_logged):
    ref = adata_logged
    query = ref.copy()
    order = np.random.default_rng(0).permutation(query.n_vars)
    query = query[:, order].copy()  # same genes, different column order
    del query.obs["cell_type"]
    si.tl.transfer_labels(ref, query, "cell_type", n_neighbors=5)
    assert (query.obs["cell_type_transferred"].to_numpy() == ref.obs["cell_type"].to_numpy()).mean() > 0.98


# ── batch correction ────────────────────────────────────────────────────


def test_combat_writes_a_layer_and_keeps_x(adata_logged):
    x = adata_logged.X.copy()
    assert si.tl.combat(adata_logged, "batch") is None
    np.testing.assert_array_equal(adata_logged.X, x)
    assert adata_logged.layers["combat"].shape == adata_logged.shape


def test_combat_accepts_sparse_input_and_removes_a_planted_batch_effect(adata_logged):
    adata_logged.X = adata_logged.X + np.where(adata_logged.obs.batch.to_numpy() == "b2", 2.0, 0.0)[:, None]
    adata_logged.X = sparse.csr_matrix(adata_logged.X)
    si.tl.combat(adata_logged, "batch")
    corrected = to_dense(adata_logged.layers["combat"])
    b = adata_logged.obs.batch.to_numpy()
    assert abs(corrected[b == "b1"].mean() - corrected[b == "b2"].mean()) < 0.05


def test_harmony_stores_the_corrected_embedding(adata_logged):
    pytest.importorskip("harmonypy")
    obsm = set(adata_logged.obsm)
    assert si.tl.harmony(adata_logged, "batch", n_components=10, random_state=0) is None
    assert set(adata_logged.obsm) - obsm == {"X_pca_harmony"}
    assert adata_logged.obsm["X_pca_harmony"].shape == (adata_logged.n_obs, 10)


def test_scanorama_stores_an_embedding_in_the_original_cell_order(adata_logged):
    pytest.importorskip("scanorama")
    si.tl.scanorama(adata_logged, "batch", random_state=0)
    assert adata_logged.obsm["X_scanorama"].shape[0] == adata_logged.n_obs


def test_batch_benchmark_scores_each_method_in_its_own_embedding(adata_logged, monkeypatch):
    """Regression: every method used to be scored on the *uncorrected* ``X_pca``."""
    from scintilla.batch_correction import benchmark

    n = adata_logged.n_obs
    batch = adata_logged.obs.batch.to_numpy()
    separated = np.column_stack([(batch == "b2") * 10.0, np.random.default_rng(0).normal(size=n)])
    adata_logged.obsm["X_pca"] = separated  # the uncorrected embedding: strongly batch-structured
    mixed = np.random.default_rng(1).normal(size=(n, 2))

    def fake_apply(adata, method, batch_key, n_pcs, random_state):
        out = adata.copy()
        out.obsm["X_corrected"] = mixed
        return out, "X_corrected"

    monkeypatch.setattr(benchmark, "_apply_method", fake_apply)
    result = benchmark.benchmark_batch_correction(adata_logged, "batch", "cell_type", methods=["combat"])
    row = result["leaderboard"].iloc[0]
    assert row["embed_key"] == "X_corrected"
    assert row["batch_asw"] > -0.1  # well mixed; scoring X_pca would give about -0.9


def test_batch_benchmark_end_to_end_reports_every_method(adata_logged):
    pytest.importorskip("harmonypy")
    result = si.benchmark.benchmark_batch_correction(
        adata_logged, "batch", "cell_type", methods=["combat", "harmony"], n_pcs=10, random_state=0
    )
    table = result["leaderboard"].set_index("method")
    assert set(table.index) == {"combat", "harmony"} and set(table["status"]) == {"ok"}
    assert table.loc["combat", "embed_key"] == "X_pca_combat" and table.loc["harmony", "embed_key"] == "X_pca_harmony"
    assert set(result["corrected_adatas"]) == {"combat", "harmony"}


def test_bbknn_stores_the_graph_and_the_benchmark_embeds_it(adata_logged):
    pytest.importorskip("bbknn")
    si.tl.bbknn(adata_logged, "batch", n_pcs=10)
    assert "connectivities" in adata_logged.obsp
    result = si.benchmark.benchmark_batch_correction(adata_logged, "batch", "cell_type", methods=["bbknn"], n_pcs=10)
    assert result["leaderboard"].iloc[0]["embed_key"] == "X_bbknn_spectral"
