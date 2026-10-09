"""Find marker genes for annotation."""

from __future__ import annotations

from typing import Optional, Union

import anndata as ad
import pandas as pd

from scintilla.config import RANDOM_SEED


def find_marker_genes(
    adata: Union[pd.DataFrame, ad.AnnData],
    groupby: str,
    method: str = "wilcoxon",
    n_genes: int = 50,
    random_state: int = RANDOM_SEED,
    *,
    layer: Optional[str] = None,
    key_added: Optional[str] = None,
) -> pd.DataFrame:
    """Find marker genes per group using scanpy.

    Parameters
    ----------
    adata:
        Input data.
    groupby:
        Column in obs with group labels.
    method:
        DE method ('wilcoxon', 't-test', 'logreg').
    n_genes:
        Number of top genes to return per group.
    random_state:
        Random seed used when ``method="logreg"``.
    layer:
        Layer to use; ``None`` uses ``adata.X``.
    key_added:
        If given, also store the table in ``adata.uns[key_added]``.

    Returns
    -------
    pandas.DataFrame
        Columns ``group``, ``gene``, ``score``, ``pval``, ``pval_adj``, ``logfoldchange``.
    """
    from scintilla.differential_expression.rank_genes import rank_genes_groups
    return rank_genes_groups(
        adata,
        groupby=groupby,
        method=method,
        n_genes=n_genes,
        random_state=random_state,
        layer=layer,
        key_added=key_added,
    )
