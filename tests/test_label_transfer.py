"""Regression tests for label transfer data-integrity checks."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scintilla.annotation.label_transfer import transfer_labels

GENES = [f"gene_{index}" for index in range(10)]


def _reference() -> ad.AnnData:
    reference = ad.AnnData(
        X=np.array([[10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        obs=pd.DataFrame({"cell_type": ["alpha", "beta"]}),
        var=pd.DataFrame(index=GENES),
    )
    reference.obsm["X_pca"] = np.array([[0.0, 100.0], [100.0, 0.0]])
    return reference


def test_transfer_aligns_named_genes_and_ignores_precomputed_pca() -> None:
    """Catch positional-gene matching and incomparable PCA-space transfer."""
    reference = _reference()
    query = ad.AnnData(
        X=np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0]]),
        var=pd.DataFrame(index=list(reversed(GENES))),
    )
    query.obsm["X_pca"] = np.array([[100.0, 0.0], [0.0, 100.0]])

    result = transfer_labels(reference, query, "cell_type", n_neighbors=1)

    assert result.obs["cell_type_transferred"].tolist() == ["alpha", "beta"]
    assert "cell_type_transferred" not in query.obs
    np.testing.assert_array_equal(query.obsm["X_pca"], [[100.0, 0.0], [0.0, 100.0]])


def test_transfer_rejects_zero_or_insufficient_shared_genes() -> None:
    """Catch attempts to transfer labels from biologically meaningless overlaps."""
    reference = _reference()
    no_overlap = ad.AnnData(
        X=np.ones((1, 10)), var=pd.DataFrame(index=[f"other_{i}" for i in range(10)])
    )
    insufficient_overlap = ad.AnnData(
        X=np.ones((1, 9)), var=pd.DataFrame(index=GENES[:9])
    )

    with pytest.raises(ValueError, match="No shared genes"):
        transfer_labels(reference, no_overlap, "cell_type", n_neighbors=1)
    with pytest.raises(ValueError, match="at least 10 shared genes"):
        transfer_labels(reference, insufficient_overlap, "cell_type", n_neighbors=1)


def test_transfer_validates_labels_and_one_cell_reference_neighbors() -> None:
    """Catch invalid label columns and the historical zero-neighbor edge case."""
    reference = _reference()
    query = ad.AnnData(X=np.ones((1, 10)), var=pd.DataFrame(index=GENES))

    with pytest.raises(ValueError, match="label_col.*missing"):
        transfer_labels(reference, query, "missing", n_neighbors=1)

    single_reference = reference[:1].copy()
    result = transfer_labels(single_reference, query, "cell_type", n_neighbors=1)

    assert result.obs["cell_type_transferred"].tolist() == ["alpha"]
