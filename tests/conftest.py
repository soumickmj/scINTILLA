"""Shared fixtures: small synthetic AnnData objects in dense and CSR form."""

from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).parent))

N_CELLS, N_GENES, N_TYPES = 300, 200, 4


def _make_counts(seed: int = 0) -> ad.AnnData:
    """300 cells x 200 genes of Poisson counts with four well-separated cell types and two batches."""
    rng = np.random.default_rng(seed)
    types = np.repeat([f"type{i}" for i in range(N_TYPES)], N_CELLS // N_TYPES)
    base = rng.gamma(2.0, 1.0, size=N_GENES)
    profiles = np.tile(base, (N_TYPES, 1))
    for k in range(N_TYPES):  # each type up-regulates its own block of 20 genes
        profiles[k, k * 20 : (k + 1) * 20] *= 6.0
    lam = profiles[np.repeat(np.arange(N_TYPES), N_CELLS // N_TYPES)]
    X = rng.poisson(lam).astype(np.float32)
    batch = np.tile(["b1", "b2"], N_CELLS // 2)
    obs = pd.DataFrame(
        {"cell_type": types, "batch": batch, "sample": [f"{b}_{t}" for b, t in zip(batch, types)]},
        index=[f"cell{i}" for i in range(N_CELLS)],
    )
    return ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"gene{i}" for i in range(N_GENES)]))


@pytest.fixture(scope="session")
def _template() -> ad.AnnData:
    return _make_counts()


@pytest.fixture
def adata_dense(_template) -> ad.AnnData:
    return _template.copy()


@pytest.fixture
def adata_csr(_template) -> ad.AnnData:
    out = _template.copy()
    out.X = sparse.csr_matrix(out.X)
    return out


@pytest.fixture(params=["dense", "csr"])
def adata_both(request, adata_dense, adata_csr) -> ad.AnnData:
    """The same data twice, so a test runs once per storage format."""
    return adata_dense if request.param == "dense" else adata_csr


@pytest.fixture
def adata_logged(adata_dense) -> ad.AnnData:
    """Log-normalised copy with a PCA, as most tools expect."""
    import scanpy as sc

    out = adata_dense
    sc.pp.normalize_total(out, target_sum=1e4)
    sc.pp.log1p(out)
    sc.pp.pca(out, n_comps=20, random_state=0)
    return out


def to_dense(X) -> np.ndarray:
    return X.toarray() if sparse.issparse(X) else np.asarray(X)
