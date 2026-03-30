"""Over-representation analysis (ORA)."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests


def ora_test(
    gene_list: List[str],
    background: List[str],
    gene_sets: Dict[str, List[str]],
) -> pd.DataFrame:
    """Over-representation analysis using Fisher's exact test.

    Parameters
    ----------
    gene_list:
        Query gene list.
    background:
        Background gene list (all genes).
    gene_sets:
        Dictionary {set_name: list_of_genes}.

    Returns
    -------
    pd.DataFrame  columns=[gene_set, odds_ratio, p_value, p_adjusted, significant,
                            n_overlap, n_query, n_set, n_background]
    """
    query = set(gene_list) & set(background)
    bg = set(background)
    n_bg = len(bg)
    n_query = len(query)

    records = []
    for set_name, genes in gene_sets.items():
        gs = set(genes) & bg
        n_set = len(gs)
        overlap = query & gs
        n_overlap = len(overlap)

        # 2x2 contingency table:
        #              in_set  not_in_set
        # in_query       a        b
        # not_in_query   c        d
        a = n_overlap
        b = n_query - n_overlap
        c = n_set - n_overlap
        d = n_bg - n_query - c

        try:
            odds_ratio, p_val = fisher_exact([[a, b], [c, d]], alternative="greater")
        except Exception:
            odds_ratio, p_val = float("nan"), 1.0

        records.append({
            "gene_set": set_name,
            "odds_ratio": odds_ratio,
            "p_value": p_val,
            "n_overlap": n_overlap,
            "n_query": n_query,
            "n_set": n_set,
            "n_background": n_bg,
        })

    if not records:
        return pd.DataFrame(columns=["gene_set", "odds_ratio", "p_value", "p_adjusted",
                                     "significant", "n_overlap", "n_query", "n_set", "n_background"])

    df = pd.DataFrame(records)
    _, p_adj, _, _ = multipletests(df["p_value"].values, method="fdr_bh")
    df["p_adjusted"] = p_adj
    df["significant"] = p_adj < 0.05
    return df.sort_values("p_value").reset_index(drop=True)
