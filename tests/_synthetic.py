"""Deterministic synthetic AnnData objects shared by the tests."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd


def make_labelled_blobs(n_per: int = 40, n_genes: int = 60, seed: int = 7) -> ad.AnnData:
    """Four Gaussian cell types where part of ``D`` is mislabelled as ``A``.

    The mislabelling makes ``D`` a low-quality label, so the label-quality
    scorers have something to find. Used by the numerical baseline test.
    """
    rng = np.random.default_rng(seed)
    centres = rng.normal(0, 2.0, size=(4, n_genes))
    X = np.vstack([rng.normal(c, 1.0, size=(n_per, n_genes)) for c in centres]).astype(np.float32)
    labels = np.repeat(["A", "B", "C", "D"], n_per).astype(object)
    labels[3 * n_per : 3 * n_per + n_per // 2] = "A"  # half of D is labelled A
    return ad.AnnData(
        X=X,
        obs=pd.DataFrame({"cell_type": labels}, index=[f"c{i}" for i in range(X.shape[0])]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(n_genes)]),
    )
