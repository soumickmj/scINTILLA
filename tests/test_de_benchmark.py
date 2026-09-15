"""Regression tests for DE benchmark integrity checks."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scintilla.differential_expression.benchmark import (
    _run_pseudobulk,
    benchmark_de_methods,
)
from scintilla.differential_expression.pseudobulk import pseudobulk_de


def _adata_without_samples() -> ad.AnnData:
    return ad.AnnData(
        X=np.array([[1.0, 4.0], [2.0, 3.0], [6.0, 1.0], [7.0, 2.0]]),
        obs=pd.DataFrame({"condition": ["control", "control", "treated", "treated"]}),
        var=pd.DataFrame(index=["gene_a", "gene_b"]),
    )


def test_pseudobulk_requires_explicit_sample_column() -> None:
    """Catch fabrication of one pseudo-replicate per condition."""
    with pytest.raises(ValueError, match="requires a 'sample' column"):
        _run_pseudobulk(_adata_without_samples(), "condition", "control", "treated")


def test_benchmark_records_pseudobulk_failure_without_dropping_other_methods() -> None:
    """Catch benchmarks that conceal invalid pseudobulk while omitting valid arms."""
    result = benchmark_de_methods(
        _adata_without_samples(), "condition", "control", "treated"
    ).set_index("method")

    assert set(result.index) == {"wilcoxon", "ttest", "pseudobulk"}
    assert result.loc["pseudobulk", "status"] == "pseudobulk requires a 'sample' column"
    assert result.loc["wilcoxon", "status"] == "ok"
    assert result.loc["ttest", "status"] == "ok"


def test_pseudobulk_rejects_fewer_than_two_biological_replicates() -> None:
    """Catch substitution of cells for missing biological replicates."""
    adata = _adata_without_samples()
    adata.obs["sample"] = ["control_1", "control_1", "treated_1", "treated_1"]

    with pytest.raises(
        ValueError,
        match="requires at least 2 biological samples per condition",
    ):
        pseudobulk_de(adata, "condition", "sample")


def test_cell_type_pseudobulk_exposes_insufficient_replicates() -> None:
    """Catch per-cell-type failures returned as unexplained empty results."""
    adata = _adata_without_samples()
    adata.obs["sample"] = ["control_1", "control_1", "treated_1", "treated_1"]
    adata.obs["cell_type"] = ["T", "T", "T", "T"]

    with pytest.warns(
        UserWarning,
        match="Pseudobulk DE failed for cell type 'T'",
    ):
        result = pseudobulk_de(
            adata,
            "condition",
            "sample",
            cell_type_col="cell_type",
        )

    failed = result["T"]
    assert failed.empty
    assert failed.attrs == {
        "status": "failed",
        "failure_reason": (
            "pseudobulk requires at least 2 biological samples per condition; "
            "got control=1, treated=1"
        ),
    }
