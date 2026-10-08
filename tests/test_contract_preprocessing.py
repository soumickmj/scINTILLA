"""The API contract for preprocessing: where results land, copy, key_added, sparsity."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from conftest import to_dense
from scipy import sparse

import scintilla as si
from scintilla.preprocessing import transformations as T

ZERO_PRESERVING = [
    "log_shift_size_factor",
    "arcsinh_transform",
    "log_alpha_transform",
    "log_cpm_transform",
    "normalise_scran",
]
LAYER_TRANSFORMS = [
    "log_shift_size_factor",
    "arcsinh_transform",
    "log_alpha_transform",
    "log_cpm_transform",
    "log_shift_scale_by_std",
    "log_shift_size_factor_z",
    "normalise_scran",
    "normalise_tmm",
    "pearson_residuals_transform",
    "sanity_transform",
]
SUBSETTING_TRANSFORMS = ["log_shift_size_factor_hvg", "log_shift_hvg_z"]


# ── PCA ─────────────────────────────────────────────────────────────────


def test_pca_writes_obsm_and_returns_none_in_place(adata_dense):
    assert si.pp.pca(adata_dense, n_comps=10, random_state=0) is None
    assert adata_dense.obsm["X_pca"].shape == (adata_dense.n_obs, 10)
    assert adata_dense.uns["scintilla"]["X_pca"]["params"]["n_comps"] == 10


def test_pca_copy_leaves_the_input_untouched(adata_dense):
    out = si.pp.pca(adata_dense, n_comps=5, copy=True)
    assert out is not adata_dense
    assert "X_pca" in out.obsm and "X_pca" not in adata_dense.obsm


def test_pca_key_added_preserves_an_existing_default_embedding(adata_dense):
    si.pp.pca(adata_dense, n_comps=6, random_state=0)
    before = adata_dense.obsm["X_pca"].copy()
    si.pp.pca(adata_dense, n_comps=3, key_added="X_pca_small", random_state=0)
    np.testing.assert_array_equal(adata_dense.obsm["X_pca"], before)
    assert adata_dense.obsm["X_pca_small"].shape[1] == 3
    assert "X_pca_small_loadings" in adata_dense.varm


def test_pca_is_reproducible_and_matches_for_sparse_input(adata_dense, adata_csr):
    si.pp.pca(adata_dense, n_comps=8, random_state=3)
    si.pp.pca(adata_csr, n_comps=8, random_state=3)
    a, b = adata_dense.obsm["X_pca"], adata_csr.obsm["X_pca"]
    # Principal components are defined up to sign.
    np.testing.assert_allclose(np.abs(a), np.abs(b), rtol=1e-3, atol=1e-3)


def test_pca_of_a_dataframe_returns_the_converted_anndata():
    df = pd.DataFrame(np.random.default_rng(0).normal(size=(30, 6)), columns=list("abcdef"))
    out = si.pp.pca(df, n_comps=3)
    assert isinstance(out, ad.AnnData) and out.obsm["X_pca"].shape == (30, 3)


def test_pca_rejects_degenerate_input():
    with pytest.raises(ValueError, match="at least 2 observations"):
        si.pp.pca(ad.AnnData(X=np.ones((1, 5), dtype=np.float32)))


# ── Transformations ─────────────────────────────────────────────────────


@pytest.mark.parametrize("name", LAYER_TRANSFORMS)
def test_transformations_write_a_layer_and_keep_x(adata_dense, name):
    x_before = adata_dense.X.copy()
    assert getattr(si.pp, name)(adata_dense) is None
    np.testing.assert_array_equal(adata_dense.X, x_before)  # raw counts survive
    assert name in adata_dense.layers
    assert adata_dense.layers[name].shape == adata_dense.shape
    assert name in adata_dense.uns["scintilla"]  # provenance is keyed by the layer written


def test_transformation_key_added_replace_x_copy_and_input_layer(adata_dense):
    si.pp.log_shift_size_factor(adata_dense, key_added="norm")
    assert "norm" in adata_dense.layers and "log_shift_size_factor" not in adata_dense.layers

    copy = si.pp.arcsinh_transform(adata_dense, layer="norm", key_added="arc", copy=True)
    assert "arc" in copy.layers and "arc" not in adata_dense.layers

    expected = np.arcsinh(0.05 * to_dense(adata_dense.layers["norm"]))
    np.testing.assert_allclose(to_dense(copy.layers["arc"]), expected, rtol=1e-5, atol=1e-6)

    si.pp.log_alpha_transform(adata_dense, replace_x=True)
    assert "log_alpha_transform" not in adata_dense.layers
    np.testing.assert_allclose(adata_dense.X.max(), np.log1p(0.05 * _raw_max()), rtol=1e-4)


def _raw_max() -> float:
    from conftest import _make_counts

    return float(_make_counts().X.max())


def test_transformation_rejects_an_unknown_input_layer(adata_dense):
    with pytest.raises(KeyError, match="nope"):
        si.pp.log_cpm_transform(adata_dense, layer="nope")


@pytest.mark.parametrize("name", SUBSETTING_TRANSFORMS)
def test_variable_changing_transformations_return_a_new_anndata(adata_dense, name):
    out = getattr(si.pp, name)(adata_dense)
    assert isinstance(out, ad.AnnData) and out.n_vars < adata_dense.n_vars
    assert adata_dense.n_vars == 200 and not adata_dense.layers.keys() - {None}


@pytest.mark.parametrize("name", sorted(T.TRANSFORM_REGISTRY))
def test_registry_pure_form_agrees_between_dense_and_sparse(adata_dense, adata_csr, name):
    if name == "glm_pca_transform":
        pytest.skip("randomised SVD: compared separately")
    fn = T.TRANSFORM_REGISTRY[name]
    dense, sp = fn(adata_dense), fn(adata_csr)
    np.testing.assert_allclose(to_dense(dense.X), to_dense(sp.X), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("name", ZERO_PRESERVING)
def test_zero_preserving_transformations_keep_csr_input_sparse(adata_csr, name):
    getattr(si.pp, name)(adata_csr)
    assert sparse.issparse(adata_csr.layers[name])


def test_sparse_transformation_never_densifies_the_matrix(adata_csr, monkeypatch):
    def boom(self, *a, **k):
        raise AssertionError("matrix was densified")

    monkeypatch.setattr(sparse.csr_matrix, "toarray", boom)
    for name in ZERO_PRESERVING:
        getattr(si.pp, name)(adata_csr)


def test_log_shift_size_factor_matches_the_formula(adata_dense):
    si.pp.log_shift_size_factor(adata_dense)
    X = adata_dense.X.astype(np.float64)
    expected = np.log1p(X / X.sum(axis=1, keepdims=True))
    np.testing.assert_allclose(adata_dense.layers["log_shift_size_factor"], expected, rtol=1e-5)


def test_public_transformation_signature_documents_the_contract():
    import inspect

    params = inspect.signature(si.pp.arcsinh_transform).parameters
    assert list(params)[:2] == ["adata", "alpha"]
    assert {"layer", "key_added", "replace_x", "copy"} <= set(params)
    assert all(params[p].kind is inspect.Parameter.KEYWORD_ONLY for p in ("layer", "key_added", "replace_x", "copy"))
    assert "Parameters" in si.pp.arcsinh_transform.__doc__


# ── HVG and feature selection ───────────────────────────────────────────


def test_highly_variable_genes_flags_var_and_only_var(adata_dense):
    obs_before, uns_before = list(adata_dense.obs.columns), set(adata_dense.uns)
    assert si.pp.highly_variable_genes(adata_dense, method="cell_ranger", n_top_genes=30, key_added="hv") is None
    assert int(adata_dense.var["hv"].sum()) == 30
    assert list(adata_dense.obs.columns) == obs_before
    assert set(adata_dense.uns) - uns_before == {"scintilla"}  # no scanpy "hvg" entry leaked


def test_highly_variable_genes_subset_returns_a_filtered_copy(adata_dense):
    sub = si.pp.highly_variable_genes(adata_dense, method="cell_ranger", n_top_genes=30, subset=True)
    assert sub.n_vars == 30 and adata_dense.n_vars == 200


def test_highly_variable_genes_rejects_an_unknown_method(adata_dense):
    with pytest.raises(ValueError, match="Unknown HVG method"):
        si.pp.highly_variable_genes(adata_dense, method="nope")


@pytest.mark.parametrize("method", ["mutual_information", "pca_loadings"])
def test_select_features_flags_requested_number_of_genes(adata_logged, method):
    si.pp.select_features(adata_logged, "cell_type", method=method, n_features=15, key_added="sel")
    assert int(adata_logged.var["sel"].sum()) == 15
    assert "sel_score" in adata_logged.var


def test_select_features_finds_the_planted_marker_genes(adata_logged):
    si.pp.select_features(adata_logged, "cell_type", method="mutual_information", n_features=80)
    planted = set(range(80))  # four blocks of 20 up-regulated genes
    chosen = set(np.flatnonzero(adata_logged.var["selected"].to_numpy()))
    assert len(chosen & planted) / 80 > 0.8


def test_select_features_validates_its_arguments(adata_logged):
    with pytest.raises(ValueError, match="Unknown method"):
        si.pp.select_features(adata_logged, "cell_type", method="nope")
    with pytest.raises(KeyError):
        si.pp.select_features(adata_logged, "missing")


# ── Normality and aggregation ───────────────────────────────────────────


def test_check_normality_distinguishes_gaussian_from_counts():
    rng = np.random.default_rng(0)
    gaussian = ad.AnnData(X=rng.normal(size=(200, 10)).astype(np.float32))
    counts = ad.AnnData(X=rng.poisson(0.2, size=(200, 10)).astype(np.float32))
    assert si.pp.check_normality(gaussian, random_state=0)[0] is True
    assert si.pp.check_normality(counts, random_state=0)[0] is False


def test_aggregate_by_patient_celltype_means(adata_dense):
    adata_dense.obs["patient"] = ["p1", "p2"] * (adata_dense.n_obs // 2)
    out = si.pp.aggregate_by_patient_celltype(adata_dense, "patient", "cell_type", method="mean")
    assert out.index.names == ["patient", "cell_type"] and out.shape == (8, adata_dense.n_vars)
    mask = (adata_dense.obs.patient == "p1") & (adata_dense.obs.cell_type == "type0")
    np.testing.assert_allclose(out.loc[("p1", "type0")].to_numpy(), adata_dense.X[mask.to_numpy()].mean(axis=0), rtol=1e-5)
