"""Regression tests for Scanpy rank-gene result contracts."""

import sys
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd

from scintilla.differential_expression.rank_genes import rank_genes_groups


def test_logreg_preserves_ranked_genes_when_scanpy_omits_inference(monkeypatch) -> None:
    """Catch treating Scanpy's coefficient-only logreg output as empty DE."""
    adata = ad.AnnData(
        X=np.ones((4, 2)),
        obs=pd.DataFrame({"cell_type": ["A", "A", "B", "B"]}),
        var=pd.DataFrame(index=["gene_a", "gene_b"]),
    )

    def fake_rank_genes_groups(target, **_kwargs):
        names = np.zeros(2, dtype=[("A", "O"), ("B", "O")])
        scores = np.zeros(2, dtype=[("A", "f8"), ("B", "f8")])
        names["A"] = ["gene_a", "gene_b"]
        names["B"] = ["gene_b", "gene_a"]
        scores["A"] = [2.5, -0.5]
        scores["B"] = [1.5, -1.0]
        target.uns["rank_genes_groups"] = {
            "params": {"method": "logreg"},
            "names": names,
            "scores": scores,
        }

    fake_scanpy = SimpleNamespace(
        tl=SimpleNamespace(rank_genes_groups=fake_rank_genes_groups),
    )
    monkeypatch.setitem(sys.modules, "scanpy", fake_scanpy)

    result = rank_genes_groups(
        adata,
        groupby="cell_type",
        method="logreg",
        n_genes=2,
    )

    assert result[["group", "gene", "score"]].to_dict("records") == [
        {"group": "A", "gene": "gene_a", "score": 2.5},
        {"group": "A", "gene": "gene_b", "score": -0.5},
        {"group": "B", "gene": "gene_b", "score": 1.5},
        {"group": "B", "gene": "gene_a", "score": -1.0},
    ]
    assert result[["pval", "pval_adj", "logfoldchange"]].isna().all().all()
